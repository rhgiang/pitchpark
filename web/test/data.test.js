import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { loadPitcher, loadDelivery, prefetchPitcher } from "../src/data.js";
import { mixFromCounts } from "../src/menu.js";

const calls = [];
const okJson = (body, type = "application/json") => ({ ok: true, status: 200, headers: { get: () => type }, json: async () => body });
let fail;
beforeEach(() => {
  calls.length = 0;
  fail = false;
  globalThis.fetch = async (url) => {
    calls.push(url);
    await new Promise((r) => setTimeout(r, 5));
    if (url.includes("/deliveries/")) return okJson({ pre_s: 1 });
    if (fail) return { ok: false, status: 500, headers: { get: () => "text/html" } };
    return okJson({ meta: { schema: 3, step_s: 0.005, knot_s: 0.05 }, at_bats: [] });
  };
});

test("a prefetch and the real load are one download per file", async () => {
  prefetchPitcher("a-a");
  prefetchPitcher("a-a"); // hover then focus then select: still one request
  const [data, delivery] = await Promise.all([loadPitcher("a-a"), loadDelivery("a-a")]);
  assert.equal(data.meta.schema, 3);
  assert.equal(delivery.pre_s, 1);
  assert.deepEqual(calls.sort(), ["/data/a-a.json", "/data/deliveries/a-a.json"]);
  await loadPitcher("a-a");
  assert.equal(calls.length, 2);
});

test("a failed load is not cached, so Try again retries; a prefetch failure is swallowed", async () => {
  fail = true;
  prefetchPitcher("b-b"); // must not throw / reject unhandled
  await assert.rejects(loadPitcher("b-b"), /HTTP 500/);
  fail = false;
  assert.equal((await loadPitcher("b-b")).meta.schema, 3);
});

test("an old-schema file rejects (the loader's caller shows the data-error card)", async () => {
  globalThis.fetch = async () => okJson({ meta: { schema: 2 }, at_bats: [] });
  await assert.rejects(loadPitcher("c-c"), /schema 2/);
});

test("mixFromCounts matches pitchMix's rules and tolerates a missing mix", () => {
  assert.deepEqual(mixFromCounts({ FF: 3, SL: 1 }), [{ type: "FF", pct: 75 }, { type: "SL", pct: 25 }]);
  assert.deepEqual(mixFromCounts(undefined), []);
});
