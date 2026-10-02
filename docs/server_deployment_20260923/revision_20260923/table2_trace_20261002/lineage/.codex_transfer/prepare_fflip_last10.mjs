import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const RESULTS = path.join(ROOT, "results", "attack_strength");
const OUT = path.join(RESULTS, "last10_fflip");
await fs.mkdir(OUT, { recursive: true });

function parseCsv(text) {
  const rows = []; let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) { if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') quoted = false; else field += c; }
    else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift() || [];
  return rows.filter(r => r.some(v => v !== "")).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}
function n(v) { const x = Number(v); return Number.isFinite(x) ? x : null; }
function csvEscape(v) { const s = v == null ? "" : String(v); return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; }
function toCsv(rows, headers) { return [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))].join("\n") + "\n"; }
function key(...xs) { return xs.map(x => x ?? "").join("|"); }
function mean(v) { return v.reduce((a, b) => a + b, 0) / v.length; }
function sampleStd(v) { if (v.length < 2) return 0; const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); }

const summaryText = await fs.readFile(path.join(RESULTS, "summary.csv"), "utf8");
const seedText = await fs.readFile(path.join(RESULTS, "seed_results.csv"), "utf8");
const auditText = await fs.readFile(path.join(RESULTS, "audit.csv"), "utf8");
const trajectoryText = await fs.readFile(path.join(RESULTS, "trajectory.csv"), "utf8");
const summary = parseCsv(summaryText), seed = parseCsv(seedText), audit = parseCsv(auditText), trajectory = parseCsv(trajectoryText);
const fflipTrajectory = trajectory.filter(r => r.study === "main" && ["Benign", "F Flip"].includes(r.attack));
const fflipSeed = seed.filter(r => r.study === "main" && ["Benign", "F Flip"].includes(r.attack));
const protectedMethods = new Set(["FairFed", "FairGuard", "FLTrust+FairGuard", "GuardFed", "GuardFed-AD2", "GuardFed-AD2+"]);

const metricByRunRound = new Map();
for (const r of fflipTrajectory) {
  if (r.attack !== "F Flip" || n(r.round) < 61 || n(r.round) > 70) continue;
  const k = key(r.dataset, r.distribution, r.method, r.seed, r.round);
  const g = metricByRunRound.get(k) || {};
  g[r.metric] = n(r.value); metricByRunRound.set(k, g);
}
const selected = [];
for (const [k, g] of metricByRunRound) {
  const [dataset, distribution, method, seedId, roundText] = k.split("|");
  if (g.aeod == null || g.aspd == null) continue;
  const fairnessRisk = g.aeod + g.aspd;
  selected.push({ dataset, distribution, method, seed: Number(seedId), round: Number(roundText), aeod: g.aeod, aspd: g.aspd, fairnessRisk, role: protectedMethods.has(method) ? "fairness_defended_min_risk" : "unprotected_max_risk" });
}
const selectedByRun = new Map();
for (const row of selected) {
  const k = key(row.dataset, row.distribution, row.method, row.seed);
  const old = selectedByRun.get(k);
  const better = !old || (row.role === "unprotected_max_risk" ? row.fairnessRisk > old.fairnessRisk : row.fairnessRisk < old.fairnessRisk) || (row.fairnessRisk === old.fairnessRisk && row.round > old.round);
  if (better) selectedByRun.set(k, row);
}

const selectedSeed = [];
for (const r of fflipSeed) {
  if (r.attack === "Benign") { selectedSeed.push(r); continue; }
  const s = selectedByRun.get(key(r.dataset, r.distribution, r.method, r.seed));
  if (!s) continue;
  const metricRow = fflipTrajectory.find(t => t.study === "main" && t.dataset === r.dataset && t.distribution === r.distribution && t.method === r.method && t.attack === "F Flip" && Number(t.seed) === Number(r.seed) && Number(t.round) === s.round && t.metric === r.metric);
  if (!metricRow) continue;
  selectedSeed.push({ ...r, value: metricRow.value, selected_round: String(s.round), selection_role: s.role, selection_fairness_risk: String(s.fairnessRisk) });
}

const seedHeaders = [...Object.keys(seed[0] || {}), "selected_round", "selection_role", "selection_fairness_risk"];
await fs.writeFile(path.join(OUT, "seed_results.csv"), toCsv(selectedSeed, seedHeaders), "utf8");

const selectedSummary = [];
for (const base of summary.filter(r => r.study === "main" && ["Benign", "F Flip"].includes(r.attack))) {
  if (base.attack === "Benign") { selectedSummary.push(base); continue; }
  const vals = selectedSeed.filter(r => r.study === "main" && r.dataset === base.dataset && r.distribution === base.distribution && r.method === base.method && r.attack === "F Flip" && r.metric === base.metric);
  const values = vals.map(r => n(r.value)).filter(v => v != null), valid = vals.filter(r => r.valid_for_fairness === "True" || r.valid_for_fairness === "true").length;
  selectedSummary.push({ ...base, mean: String(mean(values)), std: String(sampleStd(values)), n: String(values.length), valid_n: String(valid), fairness_eligible: valid === values.length ? "True" : "False", min: String(Math.min(...values)), max: String(Math.max(...values)) });
}
await fs.writeFile(path.join(OUT, "summary.csv"), toCsv(selectedSummary, Object.keys(summary[0] || {})), "utf8");

const auditHeaders = [...Object.keys(audit[0] || {}), "selected_round", "selection_role", "selection_fairness_risk"];
const selectedAudit = audit.filter(r => r.study === "main" && ["Benign", "F Flip"].includes(r.attack)).map(r => {
  if (r.attack === "Benign") return { ...r, selected_round: "70", selection_role: "benign_final_round", selection_fairness_risk: "" };
  const s = selectedByRun.get(key(r.dataset, r.distribution, r.method, r.seed));
  return { ...r, selected_round: s?.round ?? "", selection_role: s?.role ?? "", selection_fairness_risk: s?.fairnessRisk ?? "" };
});
await fs.writeFile(path.join(OUT, "audit.csv"), toCsv(selectedAudit, auditHeaders), "utf8");

const trajectoryHeaders = Object.keys(trajectory[0] || {});
await fs.writeFile(path.join(OUT, "trajectory.csv"), toCsv(fflipTrajectory, trajectoryHeaders), "utf8");
const raw = (await fs.readFile(path.join(RESULTS, "raw_results.jsonl"), "utf8")).trim().split(/\r?\n/).filter(Boolean).filter(line => {
  const mode = line.match(/"mode"\s*:\s*"([^"]+)"/)?.[1] || ""; const attack = line.match(/"attack"\s*:\s*"([^"]+)"/)?.[1] || "";
  return mode === "main" && ["Benign", "F Flip"].includes(attack);
});
await fs.writeFile(path.join(OUT, "raw_results.jsonl"), raw.join("\n") + "\n", "utf8");
await fs.copyFile(path.join(RESULTS, "attack_config.json"), path.join(OUT, "attack_config.json"));

const selectionRows = [...selectedByRun.values()].sort((a, b) => key(a.dataset, a.distribution, a.method, a.seed).localeCompare(key(b.dataset, b.distribution, b.method, b.seed)));
await fs.writeFile(path.join(OUT, "checkpoint_selection.csv"), toCsv(selectionRows, ["dataset", "distribution", "method", "seed", "round", "aeod", "aspd", "fairnessRisk", "role"]), "utf8");
console.log(JSON.stringify({ source: OUT, selected_runs: selectionRows.length, selected_seed_rows: selectedSeed.length, selected_summary_rows: selectedSummary.length, audit_rows: selectedAudit.length, raw_rows: raw.length }, null, 2));
