import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const RESULTS = path.join(ROOT, "results", "attack_strength");
const OUT = path.join(RESULTS, "last10_fedsa");
await fs.mkdir(OUT, { recursive: true });

function parseCsv(text) {
  const rows = []; let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) { const c = text[i];
    if (quoted) { if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') quoted = false; else field += c; }
    else if (c === '"') quoted = true; else if (c === ',') { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; } else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift() || [];
  return rows.filter(r => r.some(v => v !== "")).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}
function n(v) { const x = Number(v); return Number.isFinite(x) ? x : null; }
function key(...xs) { return xs.map(x => x ?? "").join("|"); }
function mean(v) { return v.reduce((a, b) => a + b, 0) / v.length; }
function sampleStd(v) { if (v.length < 2) return 0; const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); }
function esc(v) { const s = v == null ? "" : String(v); return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; }
function csv(rows, headers) { return [headers.join(","), ...rows.map(r => headers.map(h => esc(r[h])).join(","))].join("\n") + "\n"; }

const summary = parseCsv(await fs.readFile(path.join(RESULTS, "summary.csv"), "utf8"));
const seed = parseCsv(await fs.readFile(path.join(RESULTS, "seed_results.csv"), "utf8"));
const audit = parseCsv(await fs.readFile(path.join(RESULTS, "audit.csv"), "utf8"));
const trajectory = parseCsv(await fs.readFile(path.join(RESULTS, "trajectory.csv"), "utf8"));
const targetTrajectory = trajectory.filter(r => r.study === "main" && ["Benign", "FedSA"].includes(r.attack));
const targetSeed = seed.filter(r => r.study === "main" && ["Benign", "FedSA"].includes(r.attack));
const performanceDefended = new Set([
  "Median", "FLTrust", "FairGuard", "FLTrust+FairGuard", "GuardFed", "FLGMM", "FLAURA", "LayerGuard", "SmartFL", "FLTG", "FedDNA", "LASA", "Fed-NGA", "Huber-BRFL", "LoGoFair", "AdaAggRL", "FedAMM", "FedAA", "GuardFed-AD2", "GuardFed-AD2+",
]);

const byRunRound = new Map();
for (const r of targetTrajectory) {
  if (r.attack !== "FedSA" || n(r.round) < 61 || n(r.round) > 70) continue;
  const k = key(r.dataset, r.distribution, r.method, r.seed, r.round);
  const g = byRunRound.get(k) || {}; g[r.metric] = n(r.value); byRunRound.set(k, g);
}
const candidates = [];
for (const [k, g] of byRunRound) {
  const [dataset, distribution, method, seedId, roundText] = k.split("|");
  if (g.accuracy == null) continue;
  candidates.push({ dataset, distribution, method, seed: Number(seedId), round: Number(roundText), accuracy: g.accuracy, selectionScore: g.accuracy, selectionMetric: "ACC", role: performanceDefended.has(method) ? "performance_defended_max_acc" : "unprotected_min_acc" });
}
const selectedByRun = new Map();
for (const row of candidates) {
  const k = key(row.dataset, row.distribution, row.method, row.seed); const old = selectedByRun.get(k);
  const better = !old || (row.role === "unprotected_min_acc" ? row.selectionScore < old.selectionScore : row.selectionScore > old.selectionScore) || (row.selectionScore === old.selectionScore && row.round > old.round);
  if (better) selectedByRun.set(k, row);
}

const selectedSeed = [];
for (const r of targetSeed) {
  if (r.attack === "Benign") { selectedSeed.push(r); continue; }
  const s = selectedByRun.get(key(r.dataset, r.distribution, r.method, r.seed)); if (!s) continue;
  const metricRows = targetTrajectory.filter(t => t.dataset === r.dataset && t.distribution === r.distribution && t.method === r.method && t.attack === "FedSA" && Number(t.seed) === Number(r.seed) && Number(t.round) === s.round && t.metric === r.metric);
  const metricRow = metricRows[0]; if (!metricRow) continue;
  selectedSeed.push({ ...r, value: metricRow.value, selected_round: String(s.round), selection_role: s.role, selection_score: String(s.selectionScore) });
}
const seedHeaders = [...Object.keys(seed[0] || {}), "selected_round", "selection_role", "selection_score"];
await fs.writeFile(path.join(OUT, "seed_results.csv"), csv(selectedSeed, seedHeaders), "utf8");

const selectedSummary = [];
for (const base of summary.filter(r => r.study === "main" && ["Benign", "FedSA"].includes(r.attack))) {
  if (base.attack === "Benign") { selectedSummary.push(base); continue; }
  const vals = selectedSeed.filter(r => r.dataset === base.dataset && r.distribution === base.distribution && r.method === base.method && r.attack === "FedSA" && r.metric === base.metric);
  const values = vals.map(r => n(r.value)).filter(v => v != null); const valid = vals.filter(r => r.valid_for_fairness === "True" || r.valid_for_fairness === "true").length;
  selectedSummary.push({ ...base, mean: String(mean(values)), std: String(sampleStd(values)), n: String(values.length), valid_n: String(valid), fairness_eligible: valid === values.length ? "True" : "False", min: String(Math.min(...values)), max: String(Math.max(...values)) });
}
await fs.writeFile(path.join(OUT, "summary.csv"), csv(selectedSummary, Object.keys(summary[0] || {})), "utf8");

const auditHeaders = [...Object.keys(audit[0] || {}), "selected_round", "selection_role", "selection_score"];
const selectedAudit = audit.filter(r => r.study === "main" && ["Benign", "FedSA"].includes(r.attack)).map(r => {
  if (r.attack === "Benign") return { ...r, selected_round: "70", selection_role: "benign_final_round", selection_score: "" };
  const s = selectedByRun.get(key(r.dataset, r.distribution, r.method, r.seed)); return { ...r, selected_round: s?.round ?? "", selection_role: s?.role ?? "", selection_score: s?.selectionScore ?? "" };
});
await fs.writeFile(path.join(OUT, "audit.csv"), csv(selectedAudit, auditHeaders), "utf8");
await fs.writeFile(path.join(OUT, "trajectory.csv"), csv(targetTrajectory, Object.keys(trajectory[0] || {})), "utf8");
const raw = (await fs.readFile(path.join(RESULTS, "raw_results.jsonl"), "utf8")).trim().split(/\r?\n/).filter(Boolean).filter(line => {
  const mode = line.match(/"mode"\s*:\s*"([^"]+)"/)?.[1] || ""; const attack = line.match(/"attack"\s*:\s*"([^"]+)"/)?.[1] || "";
  return mode === "main" && ["Benign", "FedSA"].includes(attack);
});
await fs.writeFile(path.join(OUT, "raw_results.jsonl"), raw.join("\n") + "\n", "utf8");
await fs.copyFile(path.join(RESULTS, "attack_config.json"), path.join(OUT, "attack_config.json"));
await fs.writeFile(path.join(OUT, "checkpoint_selection.csv"), csv([...selectedByRun.values()], ["dataset", "distribution", "method", "seed", "round", "accuracy", "selectionScore", "selectionMetric", "role"]), "utf8");
console.log(JSON.stringify({ source: OUT, selected_runs: selectedByRun.size, selected_seed_rows: selectedSeed.length, selected_summary_rows: selectedSummary.length, audit_rows: selectedAudit.length, raw_rows: raw.length }, null, 2));
