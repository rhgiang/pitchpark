import { test } from "node:test";
import assert from "node:assert/strict";
import { showMenu, heightLabel, displayName, pitchMix, cardModels } from "../src/menu.js";

test("the menu shows unless the URL names a pitcher (deep links skip it)", () => {
  assert.equal(showMenu(""), true);
  assert.equal(showMenu("?view=tv"), true);
  assert.equal(showMenu("?pitcher=sale-chris&ab=0"), false);
});

test("height shows as feet and inches; missing is blank", () => {
  assert.equal(heightLabel(76), "6'4\"");
  assert.equal(heightLabel(72), "6'0\"");
  assert.equal(heightLabel(null), "");
});

test("names read first-name-first", () => {
  assert.equal(displayName("Sale, Chris"), "Chris Sale");
  assert.equal(displayName("Ohtani"), "Ohtani");
});

const pitcher = (types) => ({ at_bats: [{ pitches: types.map((pitch_type) => ({ pitch_type })) }] });

test("pitch mix: whole percents summing to 100, biggest first, tail folded into Other", () => {
  const types = [...Array(35).fill("FF"), ...Array(25).fill("ST"), ...Array(20).fill("SI"),
    ...Array(10).fill("CH"), ...Array(5).fill("CU"), ...Array(3).fill("FC"), ...Array(2).fill("SL")];
  const mix = pitchMix(pitcher(types));
  assert.deepEqual(mix.slice(0, 2), [{ type: "FF", pct: 35 }, { type: "ST", pct: 25 }]);
  assert.equal(mix.length, 6);
  assert.equal(mix.at(-1).type, "Other");
  assert.equal(mix.reduce((s, m) => s + m.pct, 0), 100);
  const thirds = pitchMix(pitcher(["FF", "SL", "CH"]));
  assert.equal(thirds.reduce((s, m) => s + m.pct, 0), 100);
});

test("pitch mix ignores rows with no pitch type; empty data gives none", () => {
  assert.deepEqual(pitchMix(pitcher([null, undefined])), []);
  assert.deepEqual(pitchMix({ at_bats: [] }), []);
});

test("card models: dedupe slugs, flag deliveries, keep index order", () => {
  const index = [
    { slug: "a", name: "A, Al", throws: "L", n_at_bats: 3 },
    { slug: "b", name: "B, Bo", throws: "R", n_at_bats: 4 },
    { slug: "a", name: "A, Al", throws: "L", n_at_bats: 5 },
  ];
  const cards = cardModels(index, new Set(["b"]));
  assert.deepEqual(cards.map((c) => c.slug), ["a", "b"]);
  assert.equal(cards[0].nAtBats, 5);
  assert.deepEqual(cards.map((c) => [c.name, c.hand, c.fullDelivery]), [["Al A", "LHP", false], ["Bo B", "RHP", true]]);
});
