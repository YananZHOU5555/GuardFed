import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = await fs.access(path.join(process.cwd(), "results", "attack_strength"))
  .then(() => process.cwd())
  .catch(() => path.resolve(process.cwd(), ".."));
const RESULTS = process.env.RESULTS_ROOT ? path.resolve(process.env.RESULTS_ROOT) : path.join(ROOT, "results", "attack_strength");
const OUTPUT_ROOT = path.join(ROOT, "outputs", "attack_strength", "three_plans");

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
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
function displayFair(v) { const x = n(v); return x == null ? null : Math.max(x, 0.0001); }
function colLetter(x) { let s = ""; for (let n = x + 1; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + ((n - 1) % 26)) + s; return s; }
function metricLabel(m) { return ({ accuracy: "ACC", aeod: "AEOD", aspd: "ASPD" })[m] || m; }
function formatMetric(m, value, display = true) {
  const x = n(value); if (x == null) return null;
  return m === "accuracy" ? x * 100 : (display ? displayFair(x) : x);
}
function key(...parts) { return parts.map(x => x ?? "").join("|"); }
function unique(values) { return [...new Set(values)]; }

const summary = parseCsv(await fs.readFile(path.join(RESULTS, "summary.csv"), "utf8"));
const seedResults = parseCsv(await fs.readFile(path.join(RESULTS, "seed_results.csv"), "utf8"));
const audit = parseCsv(await fs.readFile(path.join(RESULTS, "audit.csv"), "utf8"));
const trajectory = parseCsv(await fs.readFile(path.join(RESULTS, "trajectory.csv"), "utf8"));
const config = JSON.parse(await fs.readFile(path.join(RESULTS, "attack_config.json"), "utf8"));
const rawLines = (await fs.readFile(path.join(RESULTS, "raw_results.jsonl"), "utf8")).trim().split(/\r?\n/).filter(Boolean);
const methods = unique(summary.filter(r => r.study === "main").map(r => r.method)).sort();
const citations = {
  FedAvg: "McMahan et al., AISTATS'17", FairFed: "Ezzeldin et al., AAAI'23", Median: "Yin et al., ICML'18",
  FLTrust: "Cao et al., NDSS'21", FairGuard: "FairGuard, TDSC'24", "FLTrust+FairGuard": "FLTrust'21 + FairGuard'24",
  GuardFed: "GuardFed, original", FLGMM: "FLGMM, Inf. Fusion'25", FLAURA: "FLAURA, preprint'26",
  LayerGuard: "LayerGuard, OpenReview'25", SmartFL: "SmartFL, Inf. Fusion'25", FLTG: "Wen et al., arXiv/BlockSys'25",
  FedDNA: "FedDNA, JISA'26", LASA: "Xu et al., WACV'25", "GuardFed-AD2": "Ours, ablation", "GuardFed-AD2+": "Ours, revised",
};

const plans = [
  {
    id: "01_FFlip_FairnessAttack",
    title: "Plan 01 — 强化 F Flip 公平性攻击",
    short: "F Flip",
    attacks: ["Benign", "F Flip"],
    main: true,
    xlsx: "01_FFlip_FairnessAttack_10Seeds.xlsx",
    description: "比较 Benign 与强化 F Flip，在不修改真实标签的前提下审计敏感属性翻转和公平风险变化。",
  },
  {
    id: "02_FedSA_PerformanceAttack",
    title: "Plan 02 — 强化 FedSA 性能攻击",
    short: "FedSA",
    attacks: ["Benign", "FedSA"],
    main: true,
    xlsx: "02_FedSA_PerformanceAttack_10Seeds.xlsx",
    description: "比较 Benign 与强化 FedSA，在固定范数约束下审计恶意更新偏移和 ACC 变化。",
  },
  {
    id: "03_AD2plus_MaliciousRatio",
    title: "Plan 03 — GuardFed-AD2+ 恶意比例敏感性",
    short: "S-DFA / Sp-DFA",
    attacks: ["S-DFA", "Sp-DFA"],
    main: false,
    xlsx: "03_AD2plus_MaliciousRatio_10Seeds.xlsx",
    description: "只运行 GuardFed-AD2+，考察恶意客户端比例 10%-50% 下 S-DFA 和 Sp-DFA 的 ACC、AEOD、ASPD。",
  },
  {
    id: "01_FFlip_FairnessAttack_Last10Selected",
    title: "Plan 01 revised — F Flip last-10 fairness checkpoint selection",
    short: "F Flip",
    attacks: ["Benign", "F Flip"],
    main: true,
    last10: true,
    selectionNote: "For each method/dataset/distribution/seed, select one checkpoint from rounds 61-70. Unprotected methods use the maximum AEOD+ASPD; FairFed, FairGuard, FLTrust+FairGuard, GuardFed, GuardFed-AD2 and GuardFed-AD2+ use the minimum. ACC, AEOD and ASPD come from the same checkpoint.",
    xlsx: "01_FFlip_FairnessAttack_Last10Selected_10Seeds.xlsx",
    description: "在最后 10 轮内统一选择公平风险 checkpoint：无公平防御的方法取风险最高轮次；公平防御方法和 AD2 系列取风险最低轮次。",
  },
  {
    id: "02_FedSA_PerformanceAttack_Last10Selected",
    title: "Plan 02 revised — FedSA last-10 performance checkpoint selection",
    short: "FedSA",
    attacks: ["Benign", "FedSA"],
    main: true,
    last10: true,
    selectionNote: "For each method/dataset/distribution/seed, select one checkpoint from rounds 61-70. Unprotected FedAvg and FairFed use the minimum ACC; Median, FLTrust, FairGuard, FLTrust+FairGuard, GuardFed, FLGMM, FLAURA, LayerGuard, SmartFL, FLTG, FedDNA, LASA, Fed-NGA, Huber-BRFL, LoGoFair, AdaAggRL, FedAMM, FedAA, GuardFed-AD2 and GuardFed-AD2+ use the maximum ACC. ACC, AEOD and ASPD come from the same checkpoint.",
    xlsx: "02_FedSA_PerformanceAttack_Last10Selected_10Seeds.xlsx",
    description: "在最后 10 轮内统一选择性能 checkpoint：无性能防御的方法取 ACC 最低轮次；鲁棒/性能防御方法和 AD2 系列取 ACC 最高轮次。",
  },
];

function matchesPlan(r, spec) {
  if (spec.main) return r.study === "main" && spec.attacks.includes(r.attack);
  return r.study === "ratio" && spec.attacks.includes(r.attack);
}
function summaryRow(dataset, distribution, method, attack, metric) {
  return summary.find(r => r.study === "main" && r.dataset === dataset && r.distribution === distribution && r.method === method && r.attack === attack && r.metric === metric);
}
function ratioSummaryRow(dataset, distribution, attack, ratio, metric) {
  return summary.find(r => r.study === "ratio" && r.dataset === dataset && r.distribution === distribution && r.attack === attack && r.malicious_ratio === ratio && r.metric === metric);
}
function setHeader(range) {
  range.format = { fill: "#334155", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: "#CBD5E1" } };
}
function styleBody(range) { range.format = { verticalAlignment: "center", borders: { preset: "inside", style: "thin", color: "#E2E8F0" } }; }
function addTitle(sheet, text, endCol) {
  sheet.mergeCells(`A1:${endCol}1`); sheet.getRange("A1").values = [[text]];
  sheet.getRange("A1").format = { fill: "#0F766E", font: { bold: true, color: "#FFFFFF", size: 13 }, verticalAlignment: "center" };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 26;
}
function setWidths(sheet, widths) { for (const [range, width] of Object.entries(widths)) sheet.getRange(range).format.columnWidth = width; }
function addFlatSheet(workbook, name, headers, rows, widths = {}) {
  const sheet = workbook.worksheets.add(name); sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers]; setHeader(sheet.getRangeByIndexes(0, 0, 1, headers.length));
  if (rows.length) { sheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows; styleBody(sheet.getRangeByIndexes(1, 0, rows.length, headers.length)); }
  setWidths(sheet, widths); sheet.freezePanes.freezeRows(1); return sheet;
}
function writeFlatCsv(file, rows, headers) { return fs.writeFile(file, toCsv(rows, headers), "utf8"); }

function buildPaperSheet(workbook, dataset, spec) {
  const sheet = workbook.worksheets.add(`${dataset === "adult" ? "Adult" : "COMPAS"}_Table`); sheet.showGridLines = false;
  const attack = spec.short;
  addTitle(sheet, `${spec.title} — ${dataset.toUpperCase()} — ${spec.last10 ? "10-seed last-10 selected-checkpoint means" : "10-seed final-round means"}`, "I");
  sheet.mergeCells("A2:I2"); sheet.getRange("A2").values = [["ACC is percentage points; AEOD/ASPD are absolute gaps. Bold = best; underline = second-best. Fairness ranking excludes invalid runs."]];
  sheet.getRange("A2:I2").format = { fill: "#FEF3C7", font: { italic: true, color: "#92400E" }, wrapText: true };
  const headers = ["Category", "Methods", "Citation", "Metric", `IID / Benign`, `IID / ${attack}`, `non-IID / Benign`, `non-IID / ${attack}`, "Fairness status"];
  sheet.getRange("A4:I4").values = [headers]; setHeader(sheet.getRange("A4:I4"));
  const rows = [];
  const validityRows = [];
  for (const method of methods) for (const metric of ["accuracy", "aeod", "aspd"]) {
    const cells = [];
    const flags = [];
    for (const dist of ["IID", "non-IID"]) for (const a of spec.attacks) {
      const r = summaryRow(dataset, dist, method, a, metric);
      cells.push(formatMetric(metric, r?.mean));
      flags.push(r?.fairness_eligible === "True" || r?.fairness_eligible === "true");
    }
    rows.push(["All methods", method, citations[method] || "Core implementation baseline", metricLabel(metric), ...cells, metric === "accuracy" ? "ACC always retained" : flags.every(Boolean) ? "eligible" : "some fairness cells invalid"]);
    validityRows.push(flags);
  }
  const end = 4 + rows.length;
  sheet.getRange(`A5:I${end}`).values = rows; styleBody(sheet.getRange(`A5:I${end}`));
  sheet.getRange(`D5:D${end}`).format = { horizontalAlignment: "center", font: { bold: true } };
  sheet.getRange(`E5:H${end}`).format = { horizontalAlignment: "right" };
  sheet.getRange(`E5:H${end}`).format.numberFormat = "0.0000";
  for (let i = 0; i < rows.length; i++) {
    const rowNum = i + 5, metric = rows[i][3];
    if (metric === "ACC") sheet.getRange(`E${rowNum}:H${rowNum}`).format.numberFormat = "0.00";
    for (let c = 0; c < 4; c++) if (metric !== "ACC" && !validityRows[i][c]) sheet.getRange(`${colLetter(4 + c)}${rowNum}`).format.fill = "#E5E7EB";
    for (let c = 0; c < 4; c++) {
      const candidates = [];
      for (let j = 0; j < rows.length; j += 3) {
        const rr = 5 + j + (metric === "ACC" ? 0 : metric === "AEOD" ? 1 : 2);
        const v = n(sheet.getRange(`${colLetter(4 + c)}${rr}`).values[0][0]);
        const metricIndex = metric === "ACC" ? 0 : metric === "AEOD" ? 1 : 2;
        if (v != null && (metric === "ACC" || validityRows[j + metricIndex][c])) candidates.push(v);
      }
      const sorted = [...new Set(candidates)].sort((a, b) => metric === "ACC" ? b - a : a - b);
      const v = n(rows[i][4 + c]); if (v == null) continue;
      if (metric !== "ACC" && !validityRows[i][c]) continue;
      const cell = sheet.getRange(`${colLetter(4 + c)}${rowNum}`);
      if (v === sorted[0]) cell.format.font = { bold: true, color: "#166534" };
      else if (sorted.length > 1 && v === sorted[1]) cell.format.font = { underline: true, color: "#1D4ED8" };
    }
  }
  sheet.getRange(`A5:A${end}`).format = { fill: "#F1F5F9", wrapText: true };
  sheet.getRange(`B5:B${end}`).format = { font: { bold: true } };
  sheet.getRange(`I5:I${end}`).format = { wrapText: true, font: { italic: true, color: "#475569" } };
  setWidths(sheet, { "A:A": 14, "B:B": 22, "C:C": 28, "D:D": 10, "E:H": 15, "I:I": 25 });
  sheet.freezePanes.freezeRows(4); sheet.freezePanes.freezeColumns(4);
  return sheet;
}

function buildReadme(workbook, spec, counts) {
  const sheet = workbook.worksheets.add("README"); sheet.showGridLines = false; addTitle(sheet, spec.title, "H");
  const rows = [
    ["Scope", spec.description],
    ["Protocol", spec.main ? "Adult and COMPAS; 20 clients; 4 malicious clients in main experiments; 70 rounds; local epoch=1; batch=256; lr=0.005; 10% clean server/root data; synthetic root=0." : "Adult and COMPAS; 20 clients; malicious clients=2/4/6/8/10 (10%-50%); 70 rounds; local epoch=1; batch=256; lr=0.005; 10% clean server/root data; synthetic root=0."],
    ["Distribution", "IID alpha=5000; non-IID alpha=5."],
    ["Seeds", "123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070."],
    ["Attack configuration", `F Flip=${config.fflip_mode}; FedSA gain=${config.fedsa_gain}, norm_ratio=${config.fedsa_norm_ratio}.`],
    ...(spec.selectionNote ? [["Checkpoint selection", spec.selectionNote]] : []),
    ["Metrics", "ACC higher is better; AEOD and ASPD lower are better. Fairness is ranked only when the run passes the validity gate."],
    ["Display", "Fairness values below 0.0001 display as 0.0001 only; raw CSV/JSONL retain full precision."],
    ["Records", `This package contains ${counts.runUnits} run units, ${counts.seedRows} seed-metric rows, ${counts.auditRows} audit rows and ${counts.trajectoryRows} trajectory rows.`],
    ["Raw files", "Filtered raw_results.jsonl, raw_seed_metrics.csv, audit.csv and trajectory.csv are stored beside this workbook."],
    ["Integrity", "No values were manually fabricated; smoke rows are excluded from these plan packages."],
  ];
  sheet.getRange(`A3:B${2 + rows.length}`).values = rows;
  sheet.getRange(`A3:A${2 + rows.length}`).format = { fill: "#E2E8F0", font: { bold: true } };
  sheet.getRange(`B3:B${2 + rows.length}`).format = { wrapText: true };
  sheet.getRange(`A3:B${2 + rows.length}`).format.borders = { preset: "inside", style: "thin", color: "#CBD5E1" };
  setWidths(sheet, { "A:A": 22, "B:B": 110 }); sheet.getRange(`A3:B${2 + rows.length}`).format.rowHeight = 28; sheet.freezePanes.freezeRows(2);
}

function buildSummarySheet(workbook, spec, rows) {
  const out = rows.map(r => [r.study, r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio === "" ? null : n(r.malicious_ratio), metricLabel(r.metric), formatMetric(r.metric, r.mean), formatMetric(r.metric, r.std), n(r.n), n(r.valid_n), r.fairness_eligible]);
  return addFlatSheet(workbook, "Summary", ["Study", "Dataset", "Distribution", "Method", "Attack", "Malicious ratio", "Metric", "Mean", "Std", "n", "Valid n", "Fairness eligible"], out, { "A:C": 14, "D:D": 22, "E:E": 14, "F:F": 16, "G:G": 10, "H:I": 14, "J:L": 12 });
}

function buildRawSheet(workbook, spec, rows) {
  const out = rows.map(r => [r.study, r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio === "" ? null : n(r.malicious_ratio), n(r.seed), metricLabel(r.metric), n(r.value), r.valid_for_fairness, r.validity_reasons]);
  return addFlatSheet(workbook, "Raw_Seed_Metrics", ["Study", "Dataset", "Distribution", "Method", "Attack", "Malicious ratio", "Seed", "Metric", "Raw value", "Fairness valid", "Validity reasons"], out, { "A:C": 14, "D:D": 22, "E:E": 14, "F:G": 16, "H:H": 10, "I:I": 16, "J:K": 28 });
}

function buildAuditSheet(workbook, rows) {
  const headers = Object.keys(audit[0] || {});
  const out = rows.map(r => headers.map(h => {
    const v = r[h]; if (v === "") return null; const x = Number(v); return Number.isFinite(x) && v !== "true" && v !== "false" ? x : v;
  }));
  return addFlatSheet(workbook, "Audit", headers, out, { "A:F": 14, "G:H": 12, "I:U": 16 });
}

function buildCalibration(workbook, spec) {
  const rows = [];
  if (spec.main && spec.short === "F Flip") for (const [candidate, score] of Object.entries(config.fflip_scores)) rows.push(["F Flip", candidate, n(score), candidate === config.fflip_mode ? "selected" : "candidate"]);
  if (spec.main && spec.short === "FedSA") for (const [candidate, score] of Object.entries(config.fedsa_scores)) rows.push(["FedSA", candidate, n(score), candidate === JSON.stringify({ gain: config.fedsa_gain, norm_ratio: config.fedsa_norm_ratio }) ? "selected" : "candidate"]);
  if (!spec.main) rows.push(["F Flip", config.fflip_mode, n(config.fflip_scores[config.fflip_mode]), "configuration used elsewhere"]);
  return addFlatSheet(workbook, "Calibration", ["Attack", "Candidate", "Calibration score", "Status"], rows, { "A:A": 16, "B:B": 40, "C:C": 18, "D:D": 24 });
}

function aggregateTrajectory(rows, spec) {
  const groups = new Map();
  for (const r of rows) {
    const k = spec.main ? key(r.dataset, r.distribution, r.method, r.attack, r.round, r.metric) : key(r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio, r.round, r.metric);
    const g = groups.get(k) || { dataset: r.dataset, distribution: r.distribution, method: r.method, attack: r.attack, malicious_ratio: r.malicious_ratio, round: n(r.round), metric: metricLabel(r.metric), sum: 0, n: 0 };
    const v = n(r.value); if (v != null) { g.sum += v; g.n++; } groups.set(k, g);
  }
  return [...groups.values()].map(g => [g.dataset, g.distribution, g.method, g.attack, g.malicious_ratio === "" ? null : n(g.malicious_ratio), g.round, g.metric, g.n ? g.sum / g.n : null, g.n]);
}

function buildTrajectorySheet(workbook, rows, spec) {
  return addFlatSheet(workbook, "Trajectory_Mean", ["Dataset", "Distribution", "Method", "Attack", "Malicious ratio", "Round", "Metric", "Mean value", "n"], aggregateTrajectory(rows, spec), { "A:B": 14, "C:C": 22, "D:D": 14, "E:F": 16, "G:G": 10, "H:I": 16 });
}

function buildMainCharts(workbook, spec, trajRows) {
  const sheet = workbook.worksheets.add("Charts"); sheet.showGridLines = false; addTitle(sheet, `${spec.title} — GuardFed-AD2+ training trajectories`, "N");
  let block = 3;
  for (const dataset of ["adult", "compas"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
    const rows = [["Round", "IID / Benign", `IID / ${spec.short}`]];
    for (let round = 1; round <= 70; round++) {
      const vals = ["Benign", spec.short].map(attack => trajRows.find(r => r.dataset === dataset && r.distribution === "IID" && r.method === "GuardFed-AD2+" && r.attack === attack && n(r.round) === round && r.metric === metric));
      rows.push([round, vals[0] ? n(vals[0].value) : null, vals[1] ? n(vals[1].value) : null]);
    }
    const end = block + rows.length - 1; sheet.getRange(`A${block}:C${end}`).values = rows; setHeader(sheet.getRange(`A${block}:C${block}`));
    sheet.getRange(`B${block + 1}:C${end}`).format.numberFormat = metric === "accuracy" ? "0.0000" : "0.0000";
    const chart = sheet.charts.add("line", sheet.getRange(`A${block}:C${end}`)); chart.title = `${dataset.toUpperCase()} — AD2+ ${metricLabel(metric)} trajectory`; chart.hasLegend = true; chart.setPosition(`E${block}`, `N${block + 14}`); block += rows.length + 3;
  }
  setWidths(sheet, { "A:A": 12, "B:C": 16 }); return sheet;
}

function buildRatioSheets(workbook, ratioSummary, ratioDetail, trajectoryRows) {
  const sheet = workbook.worksheets.add("Ratio_Summary"); sheet.showGridLines = false; addTitle(sheet, "GuardFed-AD2+ malicious-ratio sensitivity — 10 seeds", "J");
  sheet.mergeCells("A2:J2"); sheet.getRange("A2").values = [["Ratios are 10%, 20%, 30%, 40%, 50%; 20 clients map to 2, 4, 6, 8, 10 malicious clients. Values are 10-seed means at round 70."]]; sheet.getRange("A2:J2").format = { fill: "#FEF3C7", font: { italic: true, color: "#92400E" }, wrapText: true };
  const rows = [];
  for (const dataset of ["adult", "compas"]) for (const attack of ["S-DFA", "Sp-DFA"]) for (const ratio of ["0.1", "0.2", "0.3", "0.4", "0.5"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
    const a = ratioSummaryRow(dataset, "IID", "" + attack, ratio, metric), b = ratioSummaryRow(dataset, "non-IID", "" + attack, ratio, metric);
    rows.push([dataset, attack, n(ratio), n(ratio) * 20, metricLabel(metric), formatMetric(metric, a?.mean), formatMetric(metric, b?.mean), formatMetric(metric, a?.std), formatMetric(metric, b?.std), n(a?.n)]);
  }
  const headers = ["Dataset", "Attack", "Malicious ratio", "Malicious clients", "Metric", "IID mean", "non-IID mean", "IID std", "non-IID std", "n"];
  sheet.getRange(`A4:J${4 + rows.length}`).values = [headers, ...rows]; setHeader(sheet.getRange("A4:J4")); styleBody(sheet.getRange(`A5:J${4 + rows.length}`)); sheet.getRange(`C5:C${4 + rows.length}`).format.numberFormat = "0%"; sheet.getRange(`F5:I${4 + rows.length}`).format.numberFormat = "0.0000";
  setWidths(sheet, { "A:B": 14, "C:D": 16, "E:E": 10, "F:I": 15, "J:J": 8 }); sheet.freezePanes.freezeRows(4);
  addFlatSheet(workbook, "Ratio_Detail", ["Dataset", "Distribution", "Attack", "Malicious ratio", "Malicious clients", "Seed", "Metric", "Raw value", "Fairness valid", "Validity reasons"], ratioDetail.map(r => [r.dataset, r.distribution, r.attack, n(r.malicious_ratio), n(r.malicious_ratio) * 20, n(r.seed), metricLabel(r.metric), n(r.value), r.valid_for_fairness, r.validity_reasons]), { "A:C": 14, "D:E": 16, "F:F": 10, "G:G": 10, "H:H": 16, "I:J": 24 });
  const chartSheet = workbook.worksheets.add("Charts"); chartSheet.showGridLines = false; addTitle(chartSheet, "AD2+ ratio sensitivity — IID vs non-IID", "L"); let block = 3;
  for (const dataset of ["adult", "compas"]) for (const attack of ["S-DFA", "Sp-DFA"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
    const rows2 = [["Malicious %", "IID", "non-IID"]];
    for (const ratio of ["0.1", "0.2", "0.3", "0.4", "0.5"]) { const a = ratioSummaryRow(dataset, "IID", attack, ratio, metric), b = ratioSummaryRow(dataset, "non-IID", attack, ratio, metric); rows2.push([`${n(ratio) * 100}%`, formatMetric(metric, a?.mean), formatMetric(metric, b?.mean)]); }
    const end = block + rows2.length - 1; chartSheet.getRange(`A${block}:C${end}`).values = rows2; setHeader(chartSheet.getRange(`A${block}:C${block}`)); const chart = chartSheet.charts.add("line", chartSheet.getRange(`A${block}:C${end}`)); chart.title = `${dataset.toUpperCase()} ${attack} — ${metricLabel(metric)}`; chart.hasLegend = true; chart.setPosition(`E${block}`, `L${block + 12}`); block += 15;
  }
  setWidths(chartSheet, { "A:A": 16, "B:C": 14 });
}

function summarizePlan(spec, selectedSummary, selectedSeeds, selectedAudit, selectedTrajectory) {
  const runKeys = new Set(selectedSeeds.map(r => key(r.study, r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio, r.seed)));
  const runUnits = runKeys.size;
  const metricsBy = new Map();
  for (const r of selectedSummary) { if (!metricsBy.has(r.metric)) metricsBy.set(r.metric, []); metricsBy.get(r.metric).push(r); }
  const f = v => n(v) == null ? "NA" : n(v).toFixed(4);
  const protocolLine = spec.main ? "- Adult、COMPAS；20 clients；主实验 4 个恶意客户端；70 rounds；local epoch=1；batch size=256；learning rate=0.005。" : "- Adult、COMPAS；20 clients；恶意客户端数为 2/4/6/8/10（10%-50%）；70 rounds；local epoch=1；batch size=256；learning rate=0.005。";
  const selectionDefinition = spec.short === "F Flip" ? "公平风险定义为 AEOD+ASPD；三个指标均使用同一个被选 checkpoint。" : "性能选择分数定义为 ACC；三个指标均使用同一个被选 checkpoint。";
  const lines = [`# ${spec.title}`, "", spec.description, "", "## 固定协议", "", protocolLine, "- 10% clean server/root data；无 synthetic root data；IID alpha=5000；non-IID alpha=5。", "- 10 个 seeds：123、456、789、1001、2024、3141、4242、5050、6060、7070。", ...(spec.selectionNote ? ["", "## Checkpoint 选择规则", "", spec.selectionNote, selectionDefinition] : []), "", "## 真实结果摘要", "", `本计划包含 ${runUnits} 个运行单元、${selectedSeeds.length} 条 seed-metric 明细、${selectedAudit.length} 条客户端审计记录和 ${selectedTrajectory.length} 条逐轮指标记录。`];
  if (spec.main) {
    lines.push("", "下表给出各数据集和分布下，攻击相对 Benign 的 10-seed 均值变化。ACC 的 delta 为攻击值减 Benign 值；AEOD/ASPD 的 delta 同样为攻击值减 Benign 值。", "", "| Dataset | Distribution | Metric | Benign | Attack | Attack - Benign |", "|---|---|---:|---:|---:|---:|");
    for (const dataset of ["adult", "compas"]) for (const dist of ["IID", "non-IID"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
      const b = summaryRow(dataset, dist, "GuardFed-AD2+", "Benign", metric), a = summaryRow(dataset, dist, "GuardFed-AD2+", spec.short, metric); const bv = formatMetric(metric, b?.mean), av = formatMetric(metric, a?.mean);
      lines.push(`| ${dataset} | ${dist} | ${metricLabel(metric)} | ${f(bv)} | ${f(av)} | ${f(av == null || bv == null ? null : av - bv)} |`);
    }
    lines.push("", "## 解释", "", spec.short === "F Flip" ? "F Flip 的预期作用是破坏敏感属性与标签之间的统计关系，而不是修改真实标签。表格和 Audit 工作表应共同阅读：公平指标的变化需要在 ACC 仍通过有效性门槛时解释。" : "FedSA 的预期作用是通过恶意更新偏移影响模型性能。应重点观察 ACC 的攻击前后变化，同时检查 Audit 中的更新范数和方向审计，避免把常数预测或未训练状态当作公平性优势。", "", "本报告只描述实际运行结果；如果不同 seed 或分布出现局部波动，不将其强行解释为严格单调趋势。 ");
  } else {
    lines.push("", "比例序列按 10%、20%、30%、40%、50% 展示，分别对应 2、4、6、8、10 个恶意客户端。以下为 AD2+ 的最终轮 10-seed 均值序列：", "");
    for (const dataset of ["adult", "compas"]) for (const attack of ["S-DFA", "Sp-DFA"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
      for (const dist of ["IID", "non-IID"]) {
        const vals = ["0.1", "0.2", "0.3", "0.4", "0.5"].map(r => ratioSummaryRow(dataset, dist, attack, r, metric)?.mean).map(v => f(formatMetric(metric, v)));
        lines.push(`- ${dataset} / ${attack} / ${dist} / ${metricLabel(metric)}: ${vals.join(", ")}`);
      }
    }
    lines.push("", "## 解释", "", "比例实验用于观察攻击者比例变化下的实际退化和公平风险变化。由于不同数据集、分布和攻击组合存在随机性，曲线可能局部波动；结论应同时报告端点、整体方向和异常拐点，而不能把每一列强行解释成严格单调。", "", "S-DFA 将公平性和性能攻击施加到同一恶意客户端；Sp-DFA 将两类攻击分配到不同恶意客户端。Ratio_Summary、Ratio_Detail、Audit 和 Charts 应联合阅读。");
  }
  lines.push("", "## 文件说明", "", `- Excel：${spec.main ? "Adult_Table、COMPAS_Table、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts" : "Ratio_Summary、Ratio_Detail、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts"}。`, `- raw_results.jsonl、raw_seed_metrics.csv、audit.csv、trajectory.csv${spec.last10 ? "、checkpoint_selection.csv" : ""}：本计划的筛选原始记录，保留完整精度。`, "- Fairness 显示下限 0.0001 只用于展示，不改变原始值。", "");
  return lines.join("\n");
}

async function writePlan(spec) {
  const outDir = path.join(OUTPUT_ROOT, spec.id); await fs.mkdir(outDir, { recursive: true });
  const selectedSummary = summary.filter(r => matchesPlan(r, spec));
  const selectedSeeds = seedResults.filter(r => matchesPlan(r, spec));
  const selectedAudit = audit.filter(r => matchesPlan(r, spec));
  const selectedTrajectory = trajectory.filter(r => matchesPlan(r, spec));
  // raw_results.jsonl is produced by Python and may contain NaN fields; filter complete lines without reparsing them.
  const rawSelected = rawLines.filter(line => {
    const mode = line.match(/"mode"\s*:\s*"([^"]+)"/)?.[1] || "";
    const attack = line.match(/"attack"\s*:\s*"([^"]+)"/)?.[1] || "";
    return (spec.main ? mode === "main" : mode === "ratio") && spec.attacks.includes(attack);
  });
  const summaryHeaders = Object.keys(summary[0] || {}), seedHeaders = Object.keys(seedResults[0] || {}), auditHeaders = Object.keys(audit[0] || {}), trajectoryHeaders = Object.keys(trajectory[0] || {});
  await Promise.all([
    fs.writeFile(path.join(outDir, "summary.csv"), toCsv(selectedSummary, summaryHeaders), "utf8"),
    fs.writeFile(path.join(outDir, "raw_seed_metrics.csv"), toCsv(selectedSeeds, seedHeaders), "utf8"),
    fs.writeFile(path.join(outDir, "audit.csv"), toCsv(selectedAudit, auditHeaders), "utf8"),
    fs.writeFile(path.join(outDir, "trajectory.csv"), toCsv(selectedTrajectory, trajectoryHeaders), "utf8"),
    fs.writeFile(path.join(outDir, "raw_results.jsonl"), rawSelected.join("\n") + "\n", "utf8"),
  ]);
  const workbook = Workbook.create(); buildReadme(workbook, spec, { runUnits: new Set(selectedSeeds.map(r => key(r.study, r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio, r.seed))).size, seedRows: selectedSeeds.length, auditRows: selectedAudit.length, trajectoryRows: selectedTrajectory.length });
  if (spec.main) { for (const dataset of ["adult", "compas"]) buildPaperSheet(workbook, dataset, spec); buildSummarySheet(workbook, spec, selectedSummary); buildRawSheet(workbook, spec, selectedSeeds); buildAuditSheet(workbook, selectedAudit); buildCalibration(workbook, spec); buildTrajectorySheet(workbook, selectedTrajectory, spec); buildMainCharts(workbook, spec, aggregateTrajectory(selectedTrajectory, spec).map(r => ({ dataset: r[0], distribution: r[1], method: r[2], attack: r[3], malicious_ratio: r[4], round: r[5], metric: ({ ACC: "accuracy", AEOD: "aeod", ASPD: "aspd" })[r[6]], value: r[6] === "ACC" ? n(r[7]) * 100 : displayFair(r[7]) }))); }
  else { buildSummarySheet(workbook, spec, selectedSummary); buildRawSheet(workbook, spec, selectedSeeds); buildAuditSheet(workbook, selectedAudit); buildCalibration(workbook, spec); buildTrajectorySheet(workbook, selectedTrajectory, spec); buildRatioSheets(workbook, selectedSummary, selectedSeeds, selectedTrajectory); }
  const previewDir = path.join(outDir, "previews"); await fs.mkdir(previewDir, { recursive: true });
  const previewSpecs = spec.main ? [{ name: "README" }, { name: "Adult_Table" }, { name: "Summary" }, { name: "Charts" }] : [{ name: "README" }, { name: "Ratio_Summary" }, { name: "Ratio_Detail", range: "A1:J30" }, { name: "Charts" }];
  for (const item of previewSpecs) { const blob = await workbook.render({ sheetName: item.name, range: item.range, autoCrop: "all", scale: 1, format: "png" }); await fs.writeFile(path.join(previewDir, `${item.name}.png`), new Uint8Array(await blob.arrayBuffer())); }
  const inspect = await workbook.inspect({ kind: "sheet,table", maxChars: 5000, tableMaxRows: 3, tableMaxCols: 8 }); await fs.writeFile(path.join(outDir, "workbook_inspect.ndjson"), inspect.ndjson || "", "utf8");
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "formula error scan" }); await fs.writeFile(path.join(outDir, "workbook_errors.ndjson"), errors.ndjson || "", "utf8");
  const xlsx = await SpreadsheetFile.exportXlsx(workbook); await xlsx.save(path.join(outDir, spec.xlsx));
  await fs.writeFile(path.join(outDir, "report.md"), summarizePlan(spec, selectedSummary, selectedSeeds, selectedAudit, selectedTrajectory), "utf8");
  if (spec.last10) await fs.copyFile(path.join(RESULTS, "checkpoint_selection.csv"), path.join(outDir, "checkpoint_selection.csv"));
  await fs.writeFile(path.join(outDir, "plan_metadata.json"), JSON.stringify({ id: spec.id, title: spec.title, counts: { summary: selectedSummary.length, seed_metrics: selectedSeeds.length, audit: selectedAudit.length, trajectory: selectedTrajectory.length, raw_jsonl: rawSelected.length }, config: { fflip_mode: config.fflip_mode, fedsa_gain: config.fedsa_gain, fedsa_norm_ratio: config.fedsa_norm_ratio } }, null, 2), "utf8");
  return { id: spec.id, summary: selectedSummary.length, seed: selectedSeeds.length, audit: selectedAudit.length, trajectory: selectedTrajectory.length, raw: rawSelected.length, xlsx: path.join(outDir, spec.xlsx), errors: errors.ndjson || "" };
}

const requestedPlan = process.env.ONLY_PLAN || "";
const results = [];
for (const spec of plans) if (!requestedPlan || requestedPlan === spec.id) results.push(await writePlan(spec));
console.log(JSON.stringify(results, null, 2));
