"""M6 step 3: a small causal transformer over the pitches of an at-bat.

Why a transformer: the logistic model only sees the one previous pitch. Here pitch t attends to
every earlier pitch in the at-bat (type, result, zone), so it can learn things like "after two
sliders away, the fastball comes in" (Vaswani et al. 2017, "Attention Is All You Need"; causal
masking as in GPT, Radford et al. 2018).

Why pooled: one pitcher has 2-4k pitches, too few for even a small network. All pitchers train one
model, with a learned pitcher embedding so each keeps his own tendencies, and the output is masked
to the pitch types that pitcher threw in training. Same fit/predict interface as LogisticSeq.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from pitchsim.validation import split_by_game

# Context of the pitch being predicted, plus what happened on the previous pitch.
TOKENS = ["pitcher", "count", "stand", "tto", "prev_type", "prev_result", "prev_zone"]
MAX_POS = 12   # at-bats longer than this share the last position embedding
ORDER = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]


class _Net(nn.Module):
    def __init__(self, sizes: list[int], n_out: int, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        # Each categorical feature gets its own embedding; a token is their sum (like BERT's
        # token + segment + position embeddings). Index 0 = level unseen in training.
        self.emb = nn.ModuleList(nn.Embedding(n, d) for n in sizes)
        self.pos = nn.Embedding(MAX_POS, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, n_features) ints; pad: (batch, seq) True where padding. -> logits."""
        L = x.shape[1]
        h = sum(e(x[..., i]) for i, e in enumerate(self.emb)) + self.pos(torch.arange(L).clamp(max=MAX_POS - 1))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)   # True = blocked: pitch t sees pitches <= t
        return self.out(self.enc(h, mask=causal, src_key_padding_mask=pad, is_causal=True))


class TransformerSeq:
    """Pooled causal transformer. `types` is the union of every pitcher's pitch types."""

    # Candidate (width, dropout, weight decay), from roomy to heavily regularised. fit() picks one
    # on held-out training games. The first config overfit the two smaller pitchers (2-3k pitches).
    GRID = [(32, 0.1, 0.01), (32, 0.3, 0.1), (16, 0.3, 0.1), (16, 0.5, 0.3)]

    def __init__(self, types: list[str], layers: int = 2, heads: int = 2, lr: float = 3e-3,
                 max_epochs: int = 60, patience: int = 6, batch: int = 64, seed: int = 0,
                 grid: list[tuple[int, float, float]] | None = None):
        self.types, self.layers, self.heads, self.lr = list(types), layers, heads, lr
        self.max_epochs, self.patience, self.batch, self.seed = max_epochs, patience, batch, seed
        self.grid = grid or self.GRID

    # ---- encoding -------------------------------------------------------------------------
    def _encode(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Pad at-bats into (n_ab, L, n_features) ints. Returns x, pad mask, targets (-100 where
        no label), and for each df row its (ab, position) slot, in df's row order."""
        d = df.reset_index(drop=True)
        o = d.sort_values(ORDER).index.to_numpy()
        ab_id = d.loc[o].groupby(ORDER[:3], sort=False).ngroup().to_numpy()
        pos = d.loc[o].groupby(ORDER[:3], sort=False).cumcount().to_numpy()
        n_ab, L = ab_id.max() + 1, pos.max() + 1
        x = torch.zeros(n_ab, L, len(TOKENS), dtype=torch.long)
        for i, c in enumerate(TOKENS):
            codes = d.loc[o, c].astype(str).map(self.vocab[c]).fillna(0).astype(int).to_numpy()
            x[ab_id, pos, i] = torch.from_numpy(np.array(codes))   # writable copy
        pad = torch.ones(n_ab, L, dtype=torch.bool)
        pad[ab_id, pos] = False
        y = torch.full((n_ab, L), -100, dtype=torch.long)
        if "pitch_type" in d:
            y[ab_id, pos] = torch.from_numpy(d.loc[o, "pitch_type"].map(self.tix).fillna(-100).astype(int).to_numpy())
        slot = np.empty((len(d), 2), dtype=int)
        slot[o] = np.c_[ab_id, pos]
        return x, pad, y, slot

    def _mask_logits(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """-inf on pitch types the at-bat's pitcher never threw in training."""
        return logits.masked_fill(~self.arsenal[x[..., 0]], float("-inf"))

    # ---- training -------------------------------------------------------------------------
    def _train(self, train: pd.DataFrame, epochs: int, val: pd.DataFrame | None = None) -> list[float]:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x, pad, y, _ = self._encode(train)
        d, dropout, wd = self.config
        self.net = _Net([len(self.vocab[c]) + 1 for c in TOKENS], len(self.types), d, self.layers, self.heads, dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=wd)
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
        if val is not None:
            vx, vpad, vy, _ = self._encode(val)
        val_loss = []
        for _ in range(epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(len(x)), max(1, len(x) // self.batch)):
                logits = self._mask_logits(self.net(x[b], pad[b]), x[b])
                loss = lossf(logits.reshape(-1, len(self.types)), y[b].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            if val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = self._mask_logits(self.net(vx, vpad), vx).reshape(-1, len(self.types))
                    # guard: a val pitch type its pitcher never threw in training has p = 0
                    vl = torch.nan_to_num(torch.log_softmax(vl, -1), neginf=np.log(1e-4))
                    val_loss.append(nn.functional.nll_loss(vl, vy.reshape(-1), ignore_index=-100).item())
                if len(val_loss) - 1 - int(np.argmin(val_loss)) >= self.patience:
                    break
        return val_loss

    def fit(self, train: pd.DataFrame) -> "TransformerSeq":
        """Pick the config (GRID) and epoch count by early stopping on a 30% game slice of
        `train`, then retrain on all of `train` with them (so the test games never choose anything)."""
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(train[c].astype(str).unique()))} for c in TOKENS}
        self.tix = {t: i for i, t in enumerate(self.types)}
        arsenal = torch.zeros(len(self.vocab["pitcher"]) + 1, len(self.types), dtype=torch.bool)
        for (p, t) in train[["pitcher", "pitch_type"]].astype(str).drop_duplicates().itertuples(index=False):
            arsenal[self.vocab["pitcher"][p], self.tix[t]] = True
        arsenal[0] = True   # unseen pitcher: allow everything
        self.arsenal = arsenal
        fit_part, val_part = split_by_game(train, seed=self.seed)
        best = None
        for self.config in self.grid:
            curve = self._train(fit_part, self.max_epochs, val_part)
            if best is None or min(curve) < best[0]:
                best = (min(curve), self.config, int(np.argmin(curve)) + 1)
        _, self.config, self.epochs = best
        self._train(train, self.epochs)
        return self

    def predict(self, df: pd.DataFrame, types: list[str] | None = None) -> np.ndarray:
        """(len(df), len(types)) probabilities in df's row order. `types` (default: all) picks and
        renormalises columns, with a 1e-4 floor like LogisticSeq."""
        x, pad, _, slot = self._encode(df)
        self.net.eval()
        with torch.no_grad():
            p = torch.softmax(self._mask_logits(self.net(x, pad), x), -1).numpy()
        p = p[slot[:, 0], slot[:, 1]]
        if types is not None:
            p = p[:, [self.tix[t] for t in types]] if all(t in self.tix for t in types) else \
                np.stack([p[:, self.tix[t]] if t in self.tix else np.zeros(len(p)) for t in types], 1)
        p = np.maximum(p, 1e-4)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, df: pd.DataFrame, rng: np.random.Generator) -> list[str]:
        """One pitch type per row, drawn from the distribution (never argmax)."""
        return [str(rng.choice(self.types, p=p)) for p in self.predict(df)]
