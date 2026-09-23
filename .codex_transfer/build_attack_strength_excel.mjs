import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = await fs.access(path.join(process.cwd(), "results", "attack_strength"))
  .then(() => process.cwd())
  .catch(() => path.resolve(process.cwd(), ".."));
const RESULTS = path.join(ROOT, "results", "attack_strength");
const OUTPUT_DIR = path.join(ROOT, "outputs", "attack_strength");
const OUTPUT = path.join(OUTPUT_DIR, "GuardFed_AD2plus_AttackStrength_10Seeds.xlsx");
const PREVIEW_DIR = path.join(OUTPUT_DIR, "previews");

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
  const headers = rows.shift();
  return rows.filter(r => r.some(v => v !== "")).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

function num(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }
function displayFair(v) { const n = num(v); return n == null ? null : Math.max(n, 0.0001); }
function colLetter(n) {
  let s = ""; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + ((n - 1) % 26)) + s; return s;
}

const summary = parseCsv(await fs.readFile(path.join(RESULTS, "summary.csv"), "utf8"));
const seedResults = parseCsv(await fs.readFile(path.join(RESULTS, "seed_results.csv"), "utf8"));
const ratioSummary = summary.filter(r => r.study === "ratio");
const ratioDetail = seedResults.filter(r => r.study === "ratio");
const config = JSON.parse(await fs.readFile(path.join(RESULTS, "attack_config.json"), "utf8"));
const rawRows = seedResults.filter(r => r.study === "main" || r.study === "ratio");
const audit = parseCsv(await fs.readFile(path.join(RESULTS, "audit.csv"), "utf8"));
const citations = {
  "FedAvg": "McMahan et al., AISTATS'17",
  "FairFed": "Ezzeldin et al., AAAI'23",
  "Median": "Yin et al., ICML'18",
  "FLTrust": "Cao et al., NDSS'21",
  "FairGuard": "FairGuard, TDSC'24",
  "FLTrust+FairGuard": "FLTrust'21 + FairGuard'24",
  "GuardFed": "GuardFed, original",
  "FLGMM": "FLGMM, Inf. Fusion'25",
  "FLAURA": "FLAURA, preprint'26",
  "LayerGuard": "LayerGuard, OpenReview'25",
  "SmartFL": "SmartFL, Inf. Fusion'25",
  "FLTG": "Wen et al., arXiv/BlockSys'25",
  "FedDNA": "FedDNA, JISA'26",
  "LASA": "Xu et al., WACV'25",
  "Fed-NGA": "Core implementation baseline",
  "Huber-BRFL": "Core implementation baseline",
  "LoGoFair": "Core implementation baseline",
  "AdaAggRL": "Core implementation baseline",
  "FedAMM": "Core implementation baseline",
  "FedAA": "Core implementation baseline",
  "GuardFed-AD2": "Ours, ablation",
  "GuardFed-AD2+": "Ours, revised",
};
const methods = [...new Set(summary.filter(r => r.study === "main").map(r => r.method))];
const metricLabel = { accuracy: "ACC", aeod: "AEOD", aspd: "ASPD" };
const attacks = ["Benign", "F Flip", "FedSA"];
const distributions = ["IID", "non-IID"];

function findSummary(dataset, method, dist, attack, metric) {
  return summary.find(r => r.study === "main" && r.dataset === dataset && r.method === method && r.distribution === dist && r.attack === attack && r.metric === metric);
}

function setHeader(sheet, range) {
  range.format = {
    fill: "#334155", font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
    borders: { preset: "all", style: "thin", color: "#CBD5E1" },
  };
}
function styleBody(sheet, range) {
  range.format = { verticalAlignment: "center", borders: { preset: "inside", style: "thin", color: "#E2E8F0" } };
}
function title(sheet, text, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[text]];
  sheet.getRange("A1").format = { fill: "#0F766E", font: { bold: true, color: "#FFFFFF", size: 13 }, verticalAlignment: "center" };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 26;
}

function buildPaperSheet(dataset, attack) {
  const sheet = workbook.worksheets.add(`${dataset === "adult" ? "Adult" : "COMPAS"}_${attack === "F Flip" ? "FFlip" : "FedSA"}`);
  sheet.showGridLines = false;
  title(sheet, `TABLE: ${dataset.toUpperCase()} — ${attack} attack; 10-seed final-round means`, "K");
  sheet.mergeCells("A2:K2");
  sheet.getRange("A2").values = [["ACC is percentage points; AEOD/ASPD are absolute gaps. Bold = best; underline = second-best. Fairness ranking uses only valid runs." ]];
  sheet.getRange("A2:K2").format = { fill: "#FEF3C7", font: { italic: true, color: "#92400E" }, wrapText: true };
  const headers = [["Category", "Methods", "Citation", "Metric", "IID / Benign", "IID / F Flip", "IID / FedSA", "non-IID / Benign", "non-IID / F Flip", "non-IID / FedSA", "Fairness status"]];
  sheet.getRange("A4:K4").values = headers; setHeader(sheet, sheet.getRange("A4:K4"));
  const rows = [];
  for (const method of methods) {
    for (const metric of ["accuracy", "aeod", "aspd"]) {
      const vals = [];
      const valid = [];
      for (const dist of distributions) for (const a of attacks) {
        const r = findSummary(dataset, method, dist, a, metric);
        vals.push(metric === "accuracy" ? (num(r?.mean) == null ? null : num(r.mean) * 100) : displayFair(r?.mean));
        valid.push(r?.fairness_eligible === "True" || r?.fairness_eligible === "true");
      }
      const status = metric === "accuracy" ? "ACC always retained" : valid.every(Boolean) ? "eligible (10/10)" : "grey: invalid fairness ranking";
      rows.push(["Main baselines", method, citations[method] || "Core implementation baseline", metricLabel[metric], ...vals, status]);
    }
  }
  const end = 4 + rows.length;
  sheet.getRange(`A5:K${end}`).values = rows; styleBody(sheet, sheet.getRange(`A5:K${end}`));
  sheet.getRange(`D5:D${end}`).format = { horizontalAlignment: "center", font: { bold: true } };
  sheet.getRange(`E5:J${end}`).format = { horizontalAlignment: "right" };
  sheet.getRange(`E5:J${end}`).format.numberFormat = "0.0000";
  for (let i = 0; i < rows.length; i++) {
    const rowNum = 5 + i, metric = rows[i][3];
    if (metric === "ACC") sheet.getRange(`E${rowNum}:J${rowNum}`).format.numberFormat = "0.00";
    if (metric !== "ACC" && rows[i][10].startsWith("grey")) sheet.getRange(`E${rowNum}:J${rowNum}`).format.fill = "#E5E7EB";
    const candidates = rows[i].slice(4, 10).map(v => num(v)).filter(v => v != null);
    const eligible = metric === "ACC" || !rows[i][10].startsWith("grey");
    if (!eligible || !candidates.length) continue;
    const sorted = [...new Set(candidates)].sort((a, b) => metric === "ACC" ? b - a : a - b);
    for (let c = 0; c < 6; c++) {
      const v = num(rows[i][4 + c]);
      if (v == null) continue;
      const cell = sheet.getRange(`${colLetter(4 + c)}${rowNum}`);
      if (v === sorted[0]) cell.format.font = { bold: true, color: "#166534" };
      else if (sorted.length > 1 && v === sorted[1]) cell.format.font = { underline: true, color: "#1D4ED8" };
    }
  }
  sheet.getRange(`A5:A${end}`).format = { fill: "#F1F5F9", wrapText: true };
  sheet.getRange(`B5:B${end}`).format = { font: { bold: true } };
  sheet.getRange(`K5:K${end}`).format = { wrapText: true, font: { italic: true, color: "#475569" } };
  for (const [col, width] of [["A:A", 16], ["B:B", 22], ["C:C", 28], ["D:D", 10], ["E:J", 14], ["K:K", 23]]) sheet.getRange(col).format.columnWidth = width;
  sheet.freezePanes.freezeRows(4); sheet.freezePanes.freezeColumns(4);
  return sheet;
}

function buildRatioSummary() {
  const sheet = workbook.worksheets.add("Ratio_Summary"); sheet.showGridLines = false;
  title(sheet, "GuardFed-AD2+ malicious-ratio sensitivity — 10 seeds", "J");
  sheet.mergeCells("A2:J2"); sheet.getRange("A2").values = [["Ratios are 10%, 20%, 30%, 40%, 50%; 20 clients map to 2, 4, 6, 8, 10 malicious clients. Values are 10-seed means at round 70." ]];
  sheet.getRange("A2:J2").format = { fill: "#FEF3C7", font: { italic: true, color: "#92400E" }, wrapText: true };
  const rows = [["Dataset", "Attack", "Malicious ratio", "Malicious clients", "Metric", "IID mean", "non-IID mean", "IID std", "non-IID std", "n"]];
  for (const dataset of ["adult", "compas"]) for (const attack of ["S-DFA", "Sp-DFA"]) for (const ratio of ["0.1", "0.2", "0.3", "0.4", "0.5"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
    const a = ratioSummary.find(r => r.dataset === dataset && r.attack === attack && r.malicious_ratio === ratio && r.metric === metric && r.distribution === "IID");
    const b = ratioSummary.find(r => r.dataset === dataset && r.attack === attack && r.malicious_ratio === ratio && r.metric === metric && r.distribution === "non-IID");
    rows.push([dataset, attack, Number(ratio), Number(ratio) * 20, metricLabel[metric], metric === "accuracy" ? num(a?.mean) * 100 : displayFair(a?.mean), metric === "accuracy" ? num(b?.mean) * 100 : displayFair(b?.mean), metric === "accuracy" ? num(a?.std) * 100 : displayFair(a?.std), metric === "accuracy" ? num(b?.std) * 100 : displayFair(b?.std), num(a?.n)]);
  }
  sheet.getRange(`A4:J${3 + rows.length}`).values = rows; setHeader(sheet, sheet.getRange("A4:J4")); styleBody(sheet, sheet.getRange(`A5:J${3 + rows.length}`));
  sheet.getRange(`C5:C${3 + rows.length}`).format.numberFormat = "0%"; sheet.getRange(`F5:I${3 + rows.length}`).format.numberFormat = "0.0000";
  for (let r = 5; r <= 3 + rows.length; r++) if (sheet.getRange(`E${r}`).values[0][0] === "ACC") sheet.getRange(`F${r}:I${r}`).format.numberFormat = "0.00";
  for (const [col, width] of [["A:B", 14], ["C:D", 16], ["E:E", 10], ["F:I", 14], ["J:J", 8]]) sheet.getRange(col).format.columnWidth = width;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

function buildFlatSheet(name, headers, rows, widths = {}) {
  const sheet = workbook.worksheets.add(name); sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers]; setHeader(sheet, sheet.getRangeByIndexes(0, 0, 1, headers.length));
  if (rows.length) { sheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows; styleBody(sheet, sheet.getRangeByIndexes(1, 0, rows.length, headers.length)); }
  for (const [col, width] of Object.entries(widths)) sheet.getRange(col).format.columnWidth = width;
  sheet.freezePanes.freezeRows(1); return sheet;
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README"); readme.showGridLines = false;
title(readme, "GuardFed-AD2+ 强化攻击与恶意比例实验", "H");
readme.getRange("A3:B16").values = [
  ["Scope", "Adult and COMPAS; Class-B FL and GuardFed-ACT excluded."],
  ["Protocol", "20 clients, 4 malicious for main table, 70 rounds, 1 local epoch, batch 256, lr 0.005, 10% clean server/root data, no synthetic root data."],
  ["Distributions", "IID alpha=5000; non-IID alpha=5."],
  ["Seeds", "123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070."],
  ["F Flip", `Calibrated global mode: ${config.fflip_mode}; labels unchanged; malicious sensitive-attribute audit is in Audit.`],
  ["FedSA", `Calibrated global gain=${config.fedsa_gain}, norm_ratio=${config.fedsa_norm_ratio}; audit fields retained in Audit.`],
  ["Ratio study", "AD2+ only; S-DFA and Sp-DFA; malicious ratios 10%-50%, corresponding to 2/4/6/8/10 clients."],
  ["Validity", "ACC must exceed majority baseline and positive prediction rate must be 1%-99%; invalid fairness cells are retained but not ranked."],
  ["Display", "ACC is shown in percentage points. Fairness values below 0.0001 display as 0.0001; raw JSONL/CSV retain full precision."],
  ["Ranking", "ACC higher is better; AEOD/ASPD lower is better. Bold is best, underline is second-best per table column."],
  ["Evidence", "Main study has 2640 run units and 7920 three-metric seed rows; ratio study has 400 run units and 1200 three-metric seed rows."],
  ["Raw trajectories", "Complete per-round trajectories are retained in results/attack_strength/trajectory.csv and raw_results.jsonl."],
  ["Calibration", `Selected F Flip=${config.fflip_mode}; FedSA gain=${config.fedsa_gain}, norm_ratio=${config.fedsa_norm_ratio}.`],
  ["Source", "GuardFed repository implementation and local Adult/COMPAS data; no presentation values were manually fabricated."],
];
readme.getRange("A3:A16").format = { fill: "#E2E8F0", font: { bold: true } }; readme.getRange("B3:B16").format = { wrapText: true };
readme.getRange("A3:B16").format.borders = { preset: "inside", style: "thin", color: "#CBD5E1" };
readme.getRange("A:A").format.columnWidth = 20; readme.getRange("B:B").format.columnWidth = 110; readme.getRange("A3:B16").format.rowHeight = 28;
readme.freezePanes.freezeRows(2);

for (const dataset of ["adult", "compas"]) for (const attack of ["F Flip", "FedSA"]) buildPaperSheet(dataset, attack);
buildRatioSummary();

const ratioDetailRows = ratioDetail.map(r => [r.dataset, r.distribution, r.attack, num(r.malicious_ratio), num(r.malicious_ratio) * 20, num(r.seed), metricLabel[r.metric] || r.metric, r.metric === "accuracy" ? num(r.value) * 100 : displayFair(r.value), r.valid_for_fairness, r.validity_reasons]);
buildFlatSheet("Ratio_Detail", ["Dataset", "Distribution", "Attack", "Malicious ratio", "Malicious clients", "Seed", "Metric", "Value", "Fairness valid", "Validity reasons"], ratioDetailRows, { "A:C": 14, "D:E": 16, "F:F": 10, "G:G": 10, "H:H": 14, "I:J": 24 });
const rawRowsOut = rawRows.map(r => [r.study, r.dataset, r.distribution, r.method, r.attack, r.malicious_ratio === "" ? null : num(r.malicious_ratio), r.seed, metricLabel[r.metric] || r.metric, r.value === "" ? null : (r.metric === "accuracy" ? num(r.value) * 100 : displayFair(r.value)), r.valid_for_fairness, r.validity_reasons]);
buildFlatSheet("Raw_Seed_Metrics", ["Study", "Dataset", "Distribution", "Method", "Attack", "Malicious ratio", "Seed", "Metric", "Value", "Fairness valid", "Validity reasons"], rawRowsOut, { "A:C": 14, "D:D": 22, "E:E": 14, "F:G": 15, "H:H": 10, "I:I": 14, "J:K": 24 });
const auditRows = audit.filter(r => r.study === "main" || r.study === "ratio").map(r => Object.values(r).map(v => v === "" ? null : (Number.isFinite(Number(v)) && v !== "true" && v !== "false" ? Number(v) : v)));
buildFlatSheet("Audit", Object.keys(audit[0] || {}), auditRows, { "A:F": 14, "G:H": 10, "I:I": 12, "J:J": 18, "K:U": 16 });
const calRows = Object.entries(config.fflip_scores).map(([candidate, score]) => ["F Flip", candidate, Number(score), candidate === config.fflip_mode ? "selected" : "candidate"]);
for (const [key, score] of Object.entries(config.fedsa_scores)) calRows.push(["FedSA", key, Number(score), key === JSON.stringify({ gain: config.fedsa_gain, norm_ratio: config.fedsa_norm_ratio }) ? "selected" : "candidate"]);
buildFlatSheet("Calibration", ["Attack", "Candidate", "Calibration score", "Status"], calRows, { "A:A": 14, "B:B": 38, "C:C": 18, "D:D": 14 });

const ratioCharts = workbook.worksheets.add("Ratio_Charts"); ratioCharts.showGridLines = false;
title(ratioCharts, "AD2+ ratio sensitivity charts — IID vs non-IID", "R");
let block = 3;
for (const dataset of ["adult", "compas"]) for (const attack of ["S-DFA", "Sp-DFA"]) for (const metric of ["accuracy", "aeod", "aspd"]) {
  const rows = [["Malicious %", "IID", "non-IID"]];
  for (const ratio of ["0.1", "0.2", "0.3", "0.4", "0.5"]) {
    const a = ratioSummary.find(r => r.dataset === dataset && r.attack === attack && r.metric === metric && r.malicious_ratio === ratio && r.distribution === "IID");
    const b = ratioSummary.find(r => r.dataset === dataset && r.attack === attack && r.metric === metric && r.malicious_ratio === ratio && r.distribution === "non-IID");
    rows.push([`${Number(ratio) * 100}%`, metric === "accuracy" ? num(a?.mean) * 100 : displayFair(a?.mean), metric === "accuracy" ? num(b?.mean) * 100 : displayFair(b?.mean)]);
  }
  const end = block + rows.length - 1;
  ratioCharts.getRange(`A${block}:C${end}`).values = rows;
  ratioCharts.getRange(`A${block}:C${block}`).format = { fill: "#64748B", font: { bold: true, color: "#FFFFFF" } };
  ratioCharts.getRange(`B${block + 1}:C${end}`).format.numberFormat = metric === "accuracy" ? "0.00" : "0.0000";
  const chart = ratioCharts.charts.add("line", ratioCharts.getRange(`A${block}:C${end}`));
  chart.title = `${dataset.toUpperCase()} ${attack} — ${metric === "accuracy" ? "ACC (%)" : metric.toUpperCase()}`;
  chart.hasLegend = true; chart.xAxis = { axisType: "textAxis" }; chart.yAxis = { numberFormatCode: metric === "accuracy" ? "0.00" : "0.0000" };
  chart.setPosition(`E${block}`, `L${block + 12}`);
  block += 15;
}
ratioCharts.getRange("A:A").format.columnWidth = 16; ratioCharts.getRange("B:C").format.columnWidth = 14; ratioCharts.freezePanes.freezeRows(2);

await fs.mkdir(OUTPUT_DIR, { recursive: true }); await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (const sheetName of ["README", "Adult_FFlip", "Adult_FedSA", "COMPAS_FFlip", "COMPAS_FedSA", "Ratio_Summary", "Ratio_Charts"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const inspect = await workbook.inspect({ kind: "sheet,table", maxChars: 4000, tableMaxRows: 3, tableMaxCols: 8 });
await fs.writeFile(path.join(OUTPUT_DIR, "workbook_inspect.ndjson"), inspect.ndjson || "", "utf8");
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.writeFile(path.join(OUTPUT_DIR, "workbook_errors.ndjson"), errors.ndjson || "", "utf8");
const xlsx = await SpreadsheetFile.exportXlsx(workbook); await xlsx.save(OUTPUT);
console.log(JSON.stringify({ output: OUTPUT, sheets: workbook.worksheets.items.map(s => s.name), errorScan: errors.ndjson || "" }, null, 2));
