# One-shot, idempotent setup of the 3D-lift environment from a clean checkout.  Run from the repo root:
#   powershell -File scripts/setup_pose_env.ps1
# Everything torch-related lives in .venv-pose; the main .venv and src/pitchsim never import torch.
#
# Prerequisite you do by hand (licence): SMPL v1.1.0 .pkl files in third_party/body_models/smpl/
#   (basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl) -- from https://smpl.is.tue.mpg.de (free registration).
#
# Notes on choices:
# - Python 3.12 from python.org: uv-managed pythons are blocked by Application Control on this machine.
# - torch from the cu128 index: the RTX 5060 Ti is Blackwell (sm_120), which only cu128+ wheels support.
# - Lifter = HMR 2.0 (4D-Humans) at a pinned commit plus scripts/patches/4d-humans.patch (cache dir from
#   env HMR2_CACHE_PARENT; no mesh renderer).  The pyrender stand-in lives in scripts/m5_lift.py (committed),
#   not here.  GVHMR/WHAM were not usable, see results/m5/spike.md.
$ErrorActionPreference = "Stop"
$env:UV_HTTP_TIMEOUT = "300"   # the torch wheel is 2.7 GB; the default timeout drops on slow links
$LIFTER_COMMIT = "efe18deff163b29dff87ddbd575fa29b716a356c"   # shubham-goel/4D-Humans the patch applies to
$py = ".venv-pose\Scripts\python.exe"

# 1. venv + dependencies (working order: torch, build tools, chumpy without build isolation, the rest)
if (-not (Test-Path $py)) {
    uv venv .venv-pose --python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
}
uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py pip setuptools wheel
uv pip install --python $py --no-build-isolation chumpy   # needed to unpickle the SMPL .pkl
uv pip install --python $py numpy scipy pandas opencv-python einops timm lightning hydra-core hydra-zen hydra_colorlog rich smplx trimesh gdown pytorch-lightning tqdm joblib scikit-image yacs dill webdataset braceexpand pyrender

# 2. lifter checkout at the pinned commit, patched (skipped if the patch is already applied)
$lifter = "third_party\4D-Humans"
if (-not (Test-Path $lifter)) {
    git clone https://github.com/shubham-goel/4D-Humans $lifter
    git -C $lifter checkout $LIFTER_COMMIT
}
$patch = (Resolve-Path "scripts\patches\4d-humans.patch").Path
git -C $lifter apply --ignore-whitespace --check -R $patch 2>$null
if ($LASTEXITCODE -eq 0) { "4D-Humans patch already applied" }
else { git -C $lifter apply --ignore-whitespace $patch; "4D-Humans patch applied" }

# 3. official checkpoint + SMPL placed where the lifter expects them (HMR2_CACHE_PARENT/4DHumans/...)
New-Item -ItemType Directory -Force third_party\hmr2_cache, third_party\hmr2_cache_parent | Out-Null
$tar = "third_party\hmr2_cache\hmr2_data.tar.gz"   # a plain tar despite the name; 2.7 GB, resumable
if (-not (Test-Path "third_party\hmr2_cache\logs")) {
    curl.exe -L -C - --retry 10 -o $tar https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz
    tar -xf $tar -C third_party\hmr2_cache
}
if (-not (Test-Path "third_party\hmr2_cache_parent\4DHumans")) {
    cmd /c mklink /J third_party\hmr2_cache_parent\4DHumans third_party\hmr2_cache
}
New-Item -ItemType Directory -Force third_party\hmr2_cache\data\smpl | Out-Null
Copy-Item third_party\body_models\smpl\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl third_party\hmr2_cache\data\smpl\SMPL_NEUTRAL.pkl -Force

& $py -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
