import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "excel_delivery");

const COLORS = {
  title: "#17365D",
  header: "#D9EAF7",
  subheader: "#EAF3F8",
  note: "#FFF2CC",
  good: "#D9EAD3",
  second: "#DDEBFF",
  warn: "#FCE4D6",
  grid: "#D9E2F3",
  darkText: "#1F2937",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cur += ch;
      }
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ",") {
        row.push(cur);
        cur = "";
      } else if (ch === "\n") {
        row.push(cur);
        rows.push(row);
        row = [];
        cur = "";
      } else if (ch !== "\r") {
        cur += ch;
      }
    }
  }
  if (cur.length || row.length) {
    row.push(cur);
    rows.push(row);
  }
  if (!rows.length) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).filter((r) => r.some((v) => v !== "")).map((r) => {
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = r[i] ?? "";
    });
    return obj;
  });
}

async function readCsv(relPath) {
  const txt = await fs.readFile(path.join(ROOT, relPath), "utf8");
  return parseCsv(txt);
}

function n(v) {
  if (v === null || v === undefined || v === "") return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function s(v) {
  return v === null || v === undefined ? "" : String(v);
}

function round(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return null;
  const p = 10 ** digits;
  return Math.round(Number(v) * p) / p;
}

function mean(xs) {
  const vals = xs.map(Number).filter(Number.isFinite);
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function std(xs) {
  const vals = xs.map(Number).filter(Number.isFinite);
  if (vals.length < 2) return null;
  const m = mean(vals);
  return Math.sqrt(vals.reduce((acc, x) => acc + (x - m) ** 2, 0) / (vals.length - 1));
}

function groupBy(rows, keyFn) {
  const map = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(r);
  }
  return map;
}

function styleTitle(sheet, title, subtitle, width = 8) {
  const titleRange = sheet.getRangeByIndexes(0, 0, 1, width);
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format = {
    fill: COLORS.title,
    font: { bold: true, color: "#FFFFFF", size: 14 },
    horizontalAlignment: "left",
  };
  const sub = sheet.getRangeByIndexes(1, 0, 1, width);
  sub.merge();
  sub.values = [[subtitle]];
  sub.format = {
    fill: COLORS.note,
    font: { italic: true, color: COLORS.darkText },
    wrapText: true,
  };
}

function writeMatrix(sheet, startRow, startCol, matrix, opts = {}) {
  if (!matrix.length || !matrix[0].length) return null;
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, matrix[0].length);
  range.values = matrix;
  if (opts.border !== false) {
    range.format.borders = { preset: "all", style: "thin", color: "#E5E7EB" };
  }
  if (opts.headerRows) {
    const head = sheet.getRangeByIndexes(startRow, startCol, opts.headerRows, matrix[0].length);
    head.format = {
      fill: opts.headerFill || COLORS.header,
      font: { bold: true, color: "#111827" },
      wrapText: true,
    };
  }
  if (opts.autofit !== false) {
    range.format.autofitColumns();
    range.format.autofitRows();
  }
  return range;
}

function writeRows(sheet, rows, headers, startRow = 0, startCol = 0, opts = {}) {
  const headerLabels = headers.map((h) => h.label || h.key || h);
  const keys = headers.map((h) => h.key || h);
  const matrix = [headerLabels, ...rows.map((r) => keys.map((k) => r[k]))];
  const range = writeMatrix(sheet, startRow, startCol, matrix, { headerRows: 1, ...opts });
  for (let c = 0; c < headers.length; c++) {
    const fmt = headers[c].format;
    if (fmt && rows.length) {
      sheet.getRangeByIndexes(startRow + 1, startCol + c, rows.length, 1).format.numberFormat = fmt;
    }
  }
  if (headers.length) {
    sheet.freezePanes.freezeRows(startRow + 1);
  }
  return range;
}

function addNotesSheet(wb, title, rows) {
  const sh = wb.worksheets.add("Overview");
  sh.showGridLines = false;
  styleTitle(sh, title, "All numeric values are copied from completed experiment CSV/raw outputs. No table values are manually edited.", 8);
  const matrix = [["Item", "Detail"], ...rows];
  writeMatrix(sh, 3, 0, matrix, { headerRows: 1 });
  sh.getRange("A:A").format.columnWidth = 28;
  sh.getRange("B:B").format.columnWidth = 110;
  sh.getRange("B:B").format.wrapText = true;
  return sh;
}

async function exportAndVerify(wb, filename, previewPrefix) {
  const err = await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
    maxChars: 2000,
  });
  console.log(`${filename} formula scan: ${err.ndjson}`);
  const sheetInfo = await wb.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
  console.log(`${filename} sheets: ${sheetInfo.ndjson}`);
  await fs.mkdir(OUT, { recursive: true });
  const sheetNames = wb.worksheets.items.map((ws) => ws.name);
  for (const name of sheetNames) {
    const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, `${previewPrefix}_${name.replace(/[^A-Za-z0-9]+/g, "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  const output = await SpreadsheetFile.exportXlsx(wb);
  const xlsxPath = path.join(OUT, filename);
  await output.save(xlsxPath);
  return xlsxPath;
}

function metricHeaders(extra = []) {
  return [
    ...extra,
    { key: "n", label: "n", format: "0" },
    { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Joint score", format: "0.0000" },
  ];
}

function normalizeMetricRows(rows) {
  return rows.map((r) => {
    const acc = n(r.ACC ?? r.acc_mean ?? r.accuracy);
    const aeod = n(r.AEOD ?? r.aeod_mean);
    const aspd = n(r.ASPD ?? r.aspd_mean);
    const score = n(r.score ?? r.score_mean);
    return {
      ...r,
      ACC: acc,
      ACC_pct: acc === null ? null : acc * 100,
      AEOD: aeod,
      ASPD: aspd,
      FairAvg: aeod === null || aspd === null ? null : (aeod + aspd) / 2,
      Score: score === null && acc !== null && aeod !== null && aspd !== null ? acc - 0.5 * (aeod + aspd) : score,
    };
  });
}

async function buildAblationWorkbook() {
  const pilot = normalizeMetricRows(await readCsv("01_ablation/revised_ablation_pilot_raw.csv"));
  const adultRaw = normalizeMetricRows(await readCsv("01_ablation/adult_ad2plus_ablation_70r_raw.csv"));
  const adultSummaryCsv = await readCsv("01_ablation/adult_ablation_by_tag_summary_70r.csv");
  const adultSummary = adultSummaryCsv.map((r) => ({
    tag: r.tag,
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    ACC_min_pct: n(r.acc_min) * 100,
    ACC_max_pct: n(r.acc_max) * 100,
    AEOD: n(r.aeod_mean),
    ASPD: n(r.aspd_mean),
    FairAvg: n(r.fair_mean),
    Score: n(r.score_mean),
    score_rank: n(r.score_rank),
    acc_rank: n(r.acc_rank),
    fair_rank: n(r.fair_rank),
  }));

  const profileNames = {
    full: "Full AD2+",
    no_utility_U: "w/o utility U",
    no_fairness_FV: "w/o fairness risk+violation",
    no_geometry_CA: "w/o centrality+alignment",
    utility_only_U: "Utility only",
    fairness_only_FV: "Fairness only",
  };
  const fullBySlice = new Map();
  for (const r of pilot) {
    if (r.profile === "full") {
      fullBySlice.set([r.dataset, r.distribution, r.attack].join("|"), r);
    }
  }
  const deltas = [];
  for (const r of pilot) {
    const base = fullBySlice.get([r.dataset, r.distribution, r.attack].join("|"));
    if (!base) continue;
    const fair = r.FairAvg;
    const baseFair = base.FairAvg;
    const dAcc = (r.ACC - base.ACC) * 100;
    const dAeod = r.AEOD - base.AEOD;
    const dAspd = r.ASPD - base.ASPD;
    const dFair = fair - baseFair;
    let verdict = "Reference full model";
    if (r.profile === "no_utility_U") verdict = dAcc < -0.5 ? "Utility removal hurts ACC" : "Utility removal has mild ACC effect in this slice";
    if (r.profile === "no_fairness_FV") verdict = dFair > 0.005 ? "Fairness terms reduce fairness loss" : "Fairness signal is masked in this slice";
    if (r.profile === "no_geometry_CA") verdict = dFair > 0.005 ? "Geometry terms help fairness/stability" : "Geometry removal is not harmful in this slice";
    if (r.profile === "utility_only_U") verdict = dFair > 0.005 ? "Utility-only worsens fairness" : "Utility-only keeps utility but fairness varies";
    if (r.profile === "fairness_only_FV") verdict = dAcc < -1 ? "Fairness-only collapse: lower ACC" : "Fairness-only does not collapse here";
    deltas.push({
      Dataset: r.dataset,
      Distribution: r.distribution,
      Attack: r.attack,
      Profile: profileNames[r.profile] || r.profile,
      ACC_pct: r.ACC_pct,
      AEOD: r.AEOD,
      ASPD: r.ASPD,
      FairAvg: fair,
      Score: r.Score,
      Delta_ACC_pp_vs_Full: dAcc,
      Delta_AEOD_vs_Full: dAeod,
      Delta_ASPD_vs_Full: dAspd,
      Delta_FairAvg_vs_Full: dFair,
      Verdict: verdict,
    });
  }

  const collapsed = [];
  for (const [key, rs] of groupBy(deltas, (r) => [r.Dataset, r.Attack, r.Profile].join("|"))) {
    const [Dataset, Attack, Profile] = key.split("|");
    collapsed.push({
      Dataset,
      Attack,
      Profile,
      n: rs.length,
      ACC_pct: mean(rs.map((r) => r.ACC_pct)),
      AEOD: mean(rs.map((r) => r.AEOD)),
      ASPD: mean(rs.map((r) => r.ASPD)),
      FairAvg: mean(rs.map((r) => r.FairAvg)),
      Score: mean(rs.map((r) => r.Score)),
      Delta_ACC_pp_vs_Full: mean(rs.map((r) => r.Delta_ACC_pp_vs_Full)),
      Delta_FairAvg_vs_Full: mean(rs.map((r) => r.Delta_FairAvg_vs_Full)),
    });
  }
  collapsed.sort((a, b) => s(a.Dataset).localeCompare(s(b.Dataset)) || s(a.Attack).localeCompare(s(b.Attack)) || s(a.Profile).localeCompare(s(b.Profile)));

  const wb = Workbook.create();
  addNotesSheet(wb, "Experiment 01: GuardFed-AD2+ Ablation", [
    ["Primary purpose", "Separate AD2+ into utility, fairness-risk/violation, and geometry terms; show how removing each component changes ACC, AEOD and ASPD."],
    ["Main raw source", "01_ablation/adult_ad2plus_ablation_70r_raw.csv; 01_ablation/revised_ablation_pilot_raw.csv"],
    ["Protocol note", "Adult 70-round ablation is the full long run. Revised two-dataset diagnostic is a saved 20-round pilot used to expose component trends across Adult/COMPAS and FedSA/F Flip."],
    ["Score", "Joint score = ACC - 0.5 * (AEOD + ASPD). ACC is reported in percentage points; AEOD/ASPD are lower-is-better."],
    ["Important interpretation", "Rows labeled fairness-only collapse are not counted as better just because fairness is low; the workbook keeps ACC and fairness separate."],
  ]);

  const sh1 = wb.worksheets.add("Pilot_Deltas");
  sh1.showGridLines = false;
  styleTitle(sh1, "Revised AD2+ Component Deltas", "Delta columns are relative to Full AD2+ within the same dataset/distribution/attack slice.", 14);
  writeRows(sh1, deltas, [
    "Dataset", "Distribution", "Attack", "Profile",
    { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Joint score", format: "0.0000" },
    { key: "Delta_ACC_pp_vs_Full", label: "ΔACC pp vs Full", format: "0.00" },
    { key: "Delta_AEOD_vs_Full", label: "ΔAEOD", format: "0.0000" },
    { key: "Delta_ASPD_vs_Full", label: "ΔASPD", format: "0.0000" },
    { key: "Delta_FairAvg_vs_Full", label: "ΔFairAvg", format: "0.0000" },
    "Verdict",
  ], 3, 0);
  sh1.getRange("N:N").format.columnWidth = 42;

  const sh2 = wb.worksheets.add("Pilot_Profile_Mean");
  sh2.showGridLines = false;
  styleTitle(sh2, "Component Means By Dataset And Attack", "Means are averaged over IID and non-IID in the revised diagnostic pilot.", 12);
  writeRows(sh2, collapsed, [
    "Dataset", "Attack", "Profile", { key: "n", label: "n", format: "0" },
    { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Joint score", format: "0.0000" },
    { key: "Delta_ACC_pp_vs_Full", label: "Mean ΔACC pp", format: "0.00" },
    { key: "Delta_FairAvg_vs_Full", label: "Mean ΔFairAvg", format: "0.0000" },
  ], 3, 0);

  const sh3 = wb.worksheets.add("Adult_70r_Summary");
  sh3.showGridLines = false;
  styleTitle(sh3, "Adult 70-Round Ablation Summary", "Full long-run Adult ablation; this is the strongest source for final paper claims on Adult.", 12);
  writeRows(sh3, adultSummary, [
    "tag", { key: "n", label: "n", format: "0" },
    { key: "ACC_pct", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC_min_pct", label: "ACC min (%)", format: "0.00" },
    { key: "ACC_max_pct", label: "ACC max (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD mean", format: "0.0000" },
    { key: "ASPD", label: "ASPD mean", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Joint score", format: "0.0000" },
    { key: "score_rank", label: "Score rank", format: "0" },
    { key: "acc_rank", label: "ACC rank", format: "0" },
    { key: "fair_rank", label: "Fair rank", format: "0" },
  ], 3, 0);

  const sh4 = wb.worksheets.add("Adult_70r_Raw");
  sh4.showGridLines = false;
  writeRows(sh4, adultRaw, Object.keys(adultRaw[0]).map((k) => ({ key: k, label: k, format: k.includes("ACC") ? "0.00" : (["AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);

  const sh5 = wb.worksheets.add("Pilot_Raw");
  sh5.showGridLines = false;
  writeRows(sh5, pilot, Object.keys(pilot[0]).map((k) => ({ key: k, label: k, format: k === "ACC_pct" ? "0.00" : (["ACC", "AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);

  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "Ablation Visual Diagnostics", "Bars below use the profile-mean table. Negative ΔACC indicates lost performance; positive ΔFairAvg indicates worse fairness.", 12);
  const chartRows = collapsed.filter((r) => ["w/o utility U", "w/o fairness risk+violation", "Utility only", "Fairness only"].includes(r.Profile));
  writeRows(ch, chartRows, [
    "Dataset", "Attack", "Profile",
    { key: "Delta_ACC_pp_vs_Full", label: "ΔACC pp", format: "0.00" },
    { key: "Delta_FairAvg_vs_Full", label: "ΔFairAvg", format: "0.0000" },
  ], 4, 0);
  try {
    const chart = ch.charts.add("bar", ch.getRangeByIndexes(4, 2, chartRows.length + 1, 2));
    chart.title = "Component removal: ΔACC pp";
    chart.hasLegend = false;
    chart.setPosition("G4", "N18");
    const fairChart = ch.charts.add("bar", { chartType: "bar", title: "Component removal: ΔFairAvg", hasLegend: false });
    const fairSeries = fairChart.series.add("ΔFairAvg");
    fairSeries.categoryFormula = "'Charts'!$C$6:$C$21";
    fairSeries.formula = "'Charts'!$E$6:$E$21";
    fairChart.yAxis = { numberFormatCode: "0.0000" };
    fairChart.setPosition("G20", "N34");
  } catch (e) {
    ch.getRange("G4").values = [[`Chart render skipped: ${e.message}`]];
  }

  return exportAndVerify(wb, "GuardFed_AD2plus_01_Ablation_Revised.xlsx", "01_ablation");
}

async function buildServerWorkbook() {
  const byDataset = (await readCsv("02_server_distribution/server_dist30_strongfloor_v4_by_alpha_dataset.csv")).map((r) => ({
    dataset: r.dataset,
    server_alpha: n(r.server_alpha),
    log10_alpha: Math.log10(n(r.server_alpha)),
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    ACC_std_pct: n(r.acc_std) * 100,
    AEOD: n(r.aeod_mean),
    AEOD_std: n(r.aeod_std),
    ASPD: n(r.aspd_mean),
    ASPD_std: n(r.aspd_std),
    FairAvg: (n(r.aeod_mean) + n(r.aspd_mean)) / 2,
    Score: n(r.score_mean),
    Score_std: n(r.score_std),
    round_mean: n(r.round_mean),
  })).sort((a, b) => a.dataset.localeCompare(b.dataset) || a.server_alpha - b.server_alpha);
  const bySlice = (await readCsv("02_server_distribution/server_dist30_strongfloor_v4_by_alpha_slice.csv")).map((r) => ({
    ...r,
    server_alpha: n(r.server_alpha),
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    AEOD: n(r.aeod_mean),
    ASPD: n(r.aspd_mean),
    FairAvg: (n(r.aeod_mean) + n(r.aspd_mean)) / 2,
    Score: n(r.score_mean),
  }));
  const diag = (await readCsv("02_server_distribution/server_dist30_strongfloor_v4_curve_diagnostics.csv")).map((r) => ({
    ...r,
    num_alpha: n(r.num_alpha),
    acc_min_pct: n(r.acc_min) * 100,
    acc_max_pct: n(r.acc_max) * 100,
    acc_range_pp: n(r.acc_range) * 100,
    max_adjacent_acc_jump_pp: n(r.max_adjacent_acc_jump) * 100,
    max_adjacent_score_jump: n(r.max_adjacent_score_jump),
    best_alpha_by_score: n(r.best_alpha_by_score),
    best_score: n(r.best_score),
  }));
  const wb = Workbook.create();
  addNotesSheet(wb, "Experiment 02: Clean Server/Root Distribution", [
    ["Primary purpose", "Check how AD2+ changes when the 10% clean server/root set is sampled with different Dirichlet alpha values."],
    ["Raw source", "02_server_distribution/server_dist30_strongfloor_v4_by_alpha_dataset.csv and *_by_alpha_slice.csv"],
    ["Protocol", "30 alpha values, 3 seeds, Adult/COMPAS, IID/non-IID, Benign/FedSA, server sampling = dirichlet_label_preserved_strong_floor."],
    ["Reading rule", "Larger alpha means the clean server/root set is closer to the global distribution. The real result is a stable plateau with small ACC range, not a manually forced monotonic curve."],
  ]);
  const sh1 = wb.worksheets.add("By_Dataset");
  sh1.showGridLines = false;
  styleTitle(sh1, "Server Distribution Curve By Dataset", "Alpha values are real completed runs; ACC is percentage, fairness gaps are raw AEOD/ASPD.", 14);
  writeRows(sh1, byDataset, [
    "dataset", { key: "server_alpha", label: "alpha", format: "0.00" }, { key: "log10_alpha", label: "log10(alpha)", format: "0.00" },
    { key: "n", label: "n", format: "0" }, { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "ACC_std_pct", label: "ACC std pp", format: "0.00" }, { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" }, { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Joint score", format: "0.0000" }, { key: "Score_std", label: "Score std", format: "0.0000" },
    { key: "round_mean", label: "Best round mean", format: "0.0" },
  ], 3, 0);
  const sh2 = wb.worksheets.add("By_Slice");
  sh2.showGridLines = false;
  writeRows(sh2, bySlice, Object.keys(bySlice[0]).map((k) => ({ key: k, label: k, format: k === "ACC_pct" ? "0.00" : (["AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);
  const sh3 = wb.worksheets.add("Diagnostics");
  sh3.showGridLines = false;
  styleTitle(sh3, "Curve Diagnostics", "Range and adjacent jump values are used to judge whether the server/root data effect is stable.", 12);
  writeRows(sh3, diag, [
    "level", "dataset", "distribution", "attack", { key: "num_alpha", label: "# alpha", format: "0" },
    { key: "acc_min_pct", label: "ACC min (%)", format: "0.00" }, { key: "acc_max_pct", label: "ACC max (%)", format: "0.00" },
    { key: "acc_range_pp", label: "ACC range pp", format: "0.00" }, { key: "max_adjacent_acc_jump_pp", label: "Max adj ACC jump pp", format: "0.00" },
    { key: "max_adjacent_score_jump", label: "Max adj score jump", format: "0.0000" }, { key: "best_alpha_by_score", label: "Best alpha", format: "0.00" }, { key: "best_score", label: "Best score", format: "0.0000" },
  ], 3, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "Server Distribution Figures", "Line charts use dataset-level means over distributions/attacks/seeds.", 10);
  const alphaVals = [...new Set(byDataset.map((r) => r.server_alpha))].sort((a, b) => a - b);
  const chartRows = alphaVals.map((alpha) => {
    const a = byDataset.find((r) => r.dataset === "adult" && r.server_alpha === alpha);
    const c = byDataset.find((r) => r.dataset === "compas" && r.server_alpha === alpha);
    return {
      alpha,
      adult_ACC_pct: a?.ACC_pct ?? null,
      compas_ACC_pct: c?.ACC_pct ?? null,
      adult_Score: a?.Score ?? null,
      compas_Score: c?.Score ?? null,
      adult_FairAvg: a?.FairAvg ?? null,
      compas_FairAvg: c?.FairAvg ?? null,
    };
  });
  writeRows(ch, chartRows, [
    { key: "alpha", label: "alpha", format: "0.00" },
    { key: "adult_ACC_pct", label: "Adult ACC (%)", format: "0.00" },
    { key: "compas_ACC_pct", label: "COMPAS ACC (%)", format: "0.00" },
    { key: "adult_Score", label: "Adult score", format: "0.0000" },
    { key: "compas_Score", label: "COMPAS score", format: "0.0000" },
    { key: "adult_FairAvg", label: "Adult FairAvg", format: "0.0000" },
    { key: "compas_FairAvg", label: "COMPAS FairAvg", format: "0.0000" },
  ], 4, 0);
  ch.getRange("A:A").format.columnWidth = 11;
  try {
    const c1 = ch.charts.add("line", ch.getRangeByIndexes(4, 0, chartRows.length + 1, 3));
    c1.title = "ACC vs server alpha";
    c1.hasLegend = true;
    c1.yAxis = { numberFormatCode: "0.00" };
    c1.setPosition("I4", "P20");
    const c2 = ch.charts.add("line", { chartType: "line", title: "Joint score vs server alpha", hasLegend: true });
    const s1 = c2.series.add("Adult score");
    s1.categoryFormula = "'Charts'!$A$6:$A$35";
    s1.formula = "'Charts'!$D$6:$D$35";
    const s2 = c2.series.add("COMPAS score");
    s2.categoryFormula = "'Charts'!$A$6:$A$35";
    s2.formula = "'Charts'!$E$6:$E$35";
    c2.title = "Joint score vs server alpha";
    c2.hasLegend = true;
    c2.yAxis = { numberFormatCode: "0.000" };
    c2.setPosition("I22", "P38");
  } catch (e) {
    ch.getRange("I4").values = [[`Chart render skipped: ${e.message}`]];
  }
  return exportAndVerify(wb, "GuardFed_AD2plus_02_Server_Distribution_Revised.xlsx", "02_server");
}

async function buildSyntheticWorkbook() {
  const byDataset = (await readCsv("03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv")).map((r) => ({
    dataset: r.dataset,
    setting: r.setting,
    server_ratio: n(r.server_ratio),
    server_sampling: r.server_sampling,
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    ACC_std_pct: n(r.acc_std) * 100,
    AEOD: n(r.aeod_mean),
    AEOD_std: n(r.aeod_std),
    ASPD: n(r.aspd_mean),
    ASPD_std: n(r.aspd_std),
    FairAvg: (n(r.aeod_mean) + n(r.aspd_mean)) / 2,
    Score: n(r.score_mean),
    Score_std: n(r.score_std),
    score_rank: n(r.score_rank),
    acc_rank: n(r.acc_rank),
    aeod_rank: n(r.aeod_rank),
    aspd_rank: n(r.aspd_rank),
    target_10pct: n(r.server_ratio) === 0.1 ? "YES" : "",
  })).sort((a, b) => a.dataset.localeCompare(b.dataset) || a.server_ratio - b.server_ratio);
  const bySlice = (await readCsv("03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_slice.csv")).map((r) => ({
    ...r,
    server_ratio: n(r.server_ratio),
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    AEOD: n(r.aeod_mean),
    ASPD: n(r.aspd_mean),
    FairAvg: (n(r.aeod_mean) + n(r.aspd_mean)) / 2,
    Score: n(r.score_mean),
  }));
  const raw = normalizeMetricRows(await readCsv("03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_joint_raw.csv"));
  const wb = Workbook.create();
  addNotesSheet(wb, "Experiment 03: 10% Clean Server Data / Synthetic Generation", [
    ["Primary purpose", "Validate the selected 10% clean server/root data setting against 1%-10% clean-data ratios."],
    ["Raw source", "03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_*.csv"],
    ["Protocol", "Stratified sensitive clean-root sampling, Adult/COMPAS, IID/non-IID, attack slices, 3 seeds where available."],
    ["Target setting", "10% real clean is marked YES in the summary table. The result is ranked by joint score, ACC, AEOD and ASPD separately."],
  ]);
  const sh1 = wb.worksheets.add("By_Dataset");
  sh1.showGridLines = false;
  styleTitle(sh1, "Clean Server Ratio Summary", "10% real clean is retained as the final AD2+ setting; ranks are computed within each dataset.", 18);
  writeRows(sh1, byDataset, [
    "dataset", "setting", { key: "server_ratio", label: "server ratio", format: "0.00%" }, "server_sampling", { key: "n", label: "n", format: "0" },
    { key: "ACC_pct", label: "ACC (%)", format: "0.00" }, { key: "ACC_std_pct", label: "ACC std pp", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" }, { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" }, { key: "Score", label: "Joint score", format: "0.0000" },
    { key: "Score_std", label: "Score std", format: "0.0000" }, { key: "score_rank", label: "Score rank", format: "0" },
    { key: "acc_rank", label: "ACC rank", format: "0" }, { key: "aeod_rank", label: "AEOD rank", format: "0" }, { key: "aspd_rank", label: "ASPD rank", format: "0" }, "target_10pct",
  ], 3, 0);
  for (let i = 0; i < byDataset.length; i++) {
    if (byDataset[i].target_10pct === "YES") sh1.getRangeByIndexes(4 + i, 0, 1, 17).format.fill = COLORS.good;
  }
  const sh2 = wb.worksheets.add("By_Slice");
  sh2.showGridLines = false;
  writeRows(sh2, bySlice, Object.keys(bySlice[0]).map((k) => ({ key: k, label: k, format: k === "server_ratio" ? "0.00%" : (k === "ACC_pct" ? "0.00" : (["AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined)) })), 0, 0);
  const sh3 = wb.worksheets.add("Raw");
  sh3.showGridLines = false;
  writeRows(sh3, raw, Object.keys(raw[0]).map((k) => ({ key: k, label: k, format: k === "ACC_pct" ? "0.00" : (["ACC", "AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "10% Clean Server Data Figures", "Line charts compare 1%-10% real clean server/root data by dataset.", 10);
  const ratios = [...new Set(byDataset.map((r) => r.server_ratio))].sort((a, b) => a - b);
  const chartRows = ratios.map((ratio) => {
    const a = byDataset.find((r) => r.dataset === "adult" && r.server_ratio === ratio);
    const c = byDataset.find((r) => r.dataset === "compas" && r.server_ratio === ratio);
    return {
      ratio,
      adult_ACC_pct: a?.ACC_pct ?? null,
      compas_ACC_pct: c?.ACC_pct ?? null,
      adult_Score: a?.Score ?? null,
      compas_Score: c?.Score ?? null,
      adult_FairAvg: a?.FairAvg ?? null,
      compas_FairAvg: c?.FairAvg ?? null,
    };
  });
  writeRows(ch, chartRows, [
    { key: "ratio", label: "clean ratio", format: "0%" },
    { key: "adult_ACC_pct", label: "Adult ACC (%)", format: "0.00" },
    { key: "compas_ACC_pct", label: "COMPAS ACC (%)", format: "0.00" },
    { key: "adult_Score", label: "Adult score", format: "0.0000" },
    { key: "compas_Score", label: "COMPAS score", format: "0.0000" },
    { key: "adult_FairAvg", label: "Adult FairAvg", format: "0.0000" },
    { key: "compas_FairAvg", label: "COMPAS FairAvg", format: "0.0000" },
  ], 4, 0);
  try {
    const c1 = ch.charts.add("line", ch.getRangeByIndexes(4, 0, chartRows.length + 1, 3));
    c1.title = "ACC vs clean server ratio";
    c1.hasLegend = true;
    c1.setPosition("I4", "P20");
    const c2 = ch.charts.add("line", ch.getRangeByIndexes(4, 0, chartRows.length + 1, 5));
    c2.title = "Joint score vs clean server ratio";
    c2.hasLegend = true;
    c2.setPosition("I22", "P38");
  } catch (e) {
    ch.getRange("I4").values = [[`Chart render skipped: ${e.message}`]];
  }
  return exportAndVerify(wb, "GuardFed_AD2plus_03_Synthetic_10pct_Revised.xlsx", "03_synthetic");
}

async function buildFedSAWorkbook() {
  const paperRaw = await readCsv("04_new_performance_attack_FedSA/fedsa_paper_style_selected_ad2plus_raw.csv");
  const summary = (await readCsv("04_new_performance_attack_FedSA/fedsa_all_methods_summary.csv")).map((r) => ({
    ...r,
    n: n(r.n),
    ACC_pct: n(r.acc_mean) * 100,
    ACC_std_pct: n(r.acc_std) * 100,
    AEOD: n(r.aeod_mean),
    AEOD_std: n(r.aeod_std),
    ASPD: n(r.aspd_mean),
    ASPD_std: n(r.aspd_std),
    FairAvg: n(r.fair_mean),
    Score: n(r.score_mean),
    score_rank: n(r.score_rank),
    acc_rank: n(r.acc_rank),
    fair_rank: n(r.fair_rank),
  }));
  const raw = normalizeMetricRows(await readCsv("04_new_performance_attack_FedSA/fedsa_all_methods_raw.csv"));
  const candidates = await readCsv("04_new_performance_attack_FedSA/fedsa_ad2plus_candidate_compact_ranking.csv");
  const wb = Workbook.create();
  addNotesSheet(wb, "Experiment 04: New Performance Attack FedSA", [
    ["Primary purpose", "Replace FOE with a new performance attack setting and compare AD2+ with all listed baselines."],
    ["Raw source", "04_new_performance_attack_FedSA/fedsa_all_methods_summary.csv and fedsa_all_methods_raw.csv"],
    ["Protocol", "70 rounds, 3 seeds where shown by n=3, Adult/COMPAS, IID/non-IID, attack = FedSA."],
    ["Ranking rule", "ACC ranks high-is-best. AEOD/ASPD/fairness ranks low-is-best. Joint score = ACC - 0.5*(AEOD+ASPD)."],
  ]);
  const paper = paperRaw.map((r) => {
    const out = { Method: r.Method, Citation: r.Citation, Metric: r.Metric };
    for (const col of ["ADULT IID", "ADULT non-IID", "COMPAS IID", "COMPAS non-IID"]) {
      const val = n(r[col]);
      out[col] = r.Metric === "ACC" ? val * 100 : val;
    }
    return out;
  });
  const paperCols = ["ADULT IID", "ADULT non-IID", "COMPAS IID", "COMPAS non-IID"];
  const methods = [...new Set(paper.map((r) => r.Method))];
  const metricMap = new Map();
  for (const r of paper) metricMap.set(`${r.Method}|${r.Metric}`, r);
  function rankValue(values, value, highBest) {
    const sorted = [...new Set(values.filter(Number.isFinite))].sort((a, b) => highBest ? b - a : a - b);
    return sorted.findIndex((v) => v === value) + 1;
  }
  const ad2 = paperCols.map((col) => {
    const accVals = methods.map((m) => metricMap.get(`${m}|ACC`)?.[col]).filter(Number.isFinite);
    const aeodVals = methods.map((m) => metricMap.get(`${m}|AEOD`)?.[col]).filter(Number.isFinite);
    const aspdVals = methods.map((m) => metricMap.get(`${m}|ASPD`)?.[col]).filter(Number.isFinite);
    const fairVals = methods.map((m) => {
      const a = metricMap.get(`${m}|AEOD`)?.[col];
      const b = metricMap.get(`${m}|ASPD`)?.[col];
      return Number.isFinite(a) && Number.isFinite(b) ? (a + b) / 2 : null;
    }).filter(Number.isFinite);
    const scoreVals = methods.map((m) => {
      const a = metricMap.get(`${m}|ACC`)?.[col];
      const e = metricMap.get(`${m}|AEOD`)?.[col];
      const p = metricMap.get(`${m}|ASPD`)?.[col];
      return Number.isFinite(a) && Number.isFinite(e) && Number.isFinite(p) ? (a / 100) - 0.5 * (e + p) : null;
    }).filter(Number.isFinite);
    const acc = metricMap.get("GuardFed-AD2+|ACC")?.[col];
    const aeod = metricMap.get("GuardFed-AD2+|AEOD")?.[col];
    const aspd = metricMap.get("GuardFed-AD2+|ASPD")?.[col];
    const fair = (aeod + aspd) / 2;
    const score = (acc / 100) - 0.5 * (aeod + aspd);
    const [datasetRaw, distRaw] = col.split(" ");
    return {
      dataset: datasetRaw.toLowerCase(),
      distribution: distRaw,
      attack: "FedSA",
      ACC_pct: acc,
      AEOD: aeod,
      ASPD: aspd,
      FairAvg: fair,
      Score: score,
      score_rank: rankValue(scoreVals, score, true),
      acc_rank: rankValue(accVals, acc, true),
      aeod_rank: rankValue(aeodVals, aeod, false),
      aspd_rank: rankValue(aspdVals, aspd, false),
      fair_rank: rankValue(fairVals, fair, false),
    };
  });

  const sh0 = wb.worksheets.add("AD2plus_Ranks");
  sh0.showGridLines = false;
  styleTitle(sh0, "AD2+ FedSA Rank Summary", "Ranks are recomputed from the final PaperStyle_Table values, so this sheet matches the reported AD2+ row.", 13);
  writeRows(sh0, ad2, [
    "dataset", "distribution", "attack", { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" }, { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" }, { key: "Score", label: "Joint score", format: "0.0000" },
    { key: "score_rank", label: "Score rank", format: "0" }, { key: "acc_rank", label: "ACC rank", format: "0" },
    { key: "aeod_rank", label: "AEOD rank", format: "0" }, { key: "aspd_rank", label: "ASPD rank", format: "0" }, { key: "fair_rank", label: "FairAvg rank", format: "0" },
  ], 3, 0);

  const sh1 = wb.worksheets.add("PaperStyle_Table");
  sh1.showGridLines = false;
  styleTitle(sh1, "FedSA Paper-Style Table", "Bold = best; blue fill = second. ACC is percentage; AEOD/ASPD are lower-is-better.", 7);
  const paperHeaders = ["Method", "Citation", "Metric", "ADULT IID", "ADULT non-IID", "COMPAS IID", "COMPAS non-IID"];
  writeRows(sh1, paper, paperHeaders.map((h) => ({ key: h, label: h, format: ["ADULT IID", "ADULT non-IID", "COMPAS IID", "COMPAS non-IID"].includes(h) ? "0.0000" : undefined })), 3, 0);
  sh1.getRange("A:C").format.columnWidth = 24;
  sh1.getRange("B:B").format.columnWidth = 34;
  const metricRows = groupBy(paper, (r) => r.Metric);
  for (const metric of ["ACC", "AEOD", "ASPD"]) {
    const rows = metricRows.get(metric) || [];
    for (const colName of ["ADULT IID", "ADULT non-IID", "COMPAS IID", "COMPAS non-IID"]) {
      const vals = rows.map((r) => r[colName]).filter((v) => Number.isFinite(v));
      const sorted = [...new Set(vals)].sort((a, b) => metric === "ACC" ? b - a : a - b);
      const best = sorted[0], second = sorted[1];
      for (let i = 0; i < paper.length; i++) {
        if (paper[i].Metric !== metric) continue;
        const v = paper[i][colName];
        const c = paperHeaders.indexOf(colName);
        const cell = sh1.getRangeByIndexes(4 + i, c, 1, 1);
        if (v === best) cell.format = { fill: COLORS.good, font: { bold: true } };
        else if (v === second) cell.format = { fill: COLORS.second, font: { color: "#1155CC", underline: true } };
      }
    }
  }
  for (let i = 0; i < paper.length; i++) {
    if (paper[i].Metric === "ACC") sh1.getRangeByIndexes(4 + i, 3, 1, 4).format.numberFormat = "0.00";
    else sh1.getRangeByIndexes(4 + i, 3, 1, 4).format.numberFormat = "0.0000";
  }

  const sh2 = wb.worksheets.add("All_Methods_Summary");
  sh2.showGridLines = false;
  writeRows(sh2, summary, Object.keys(summary[0]).map((k) => ({ key: k, label: k, format: k === "ACC_pct" || k === "ACC_std_pct" ? "0.00" : (["AEOD", "AEOD_std", "ASPD", "ASPD_std", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);
  const sh3 = wb.worksheets.add("AD2plus_Candidates");
  sh3.showGridLines = false;
  writeRows(sh3, candidates, Object.keys(candidates[0]).map((k) => ({ key: k, label: k })), 0, 0);
  const sh4 = wb.worksheets.add("Raw");
  sh4.showGridLines = false;
  writeRows(sh4, raw, Object.keys(raw[0]).map((k) => ({ key: k, label: k, format: k === "ACC_pct" ? "0.00" : (["ACC", "AEOD", "ASPD", "FairAvg", "Score"].includes(k) ? "0.0000" : undefined) })), 0, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "FedSA AD2+ Vs Baselines", "Top methods by joint score in each FedSA slice.", 10);
  const chartRows = [];
  for (const [key, rows] of groupBy(summary, (r) => [r.dataset, r.distribution].join("|"))) {
    rows.sort((a, b) => a.score_rank - b.score_rank);
    for (const r of rows.slice(0, 8)) {
      chartRows.push({ Slice: key.replace("|", " "), Method: r.method, Score: r.Score, ACC_pct: r.ACC_pct, FairAvg: r.FairAvg });
    }
  }
  writeRows(ch, chartRows, ["Slice", "Method", { key: "Score", label: "Joint score", format: "0.0000" }, { key: "ACC_pct", label: "ACC (%)", format: "0.00" }, { key: "FairAvg", label: "FairAvg", format: "0.0000" }], 4, 0);
  try {
    const c1 = ch.charts.add("bar", ch.getRangeByIndexes(4, 1, chartRows.length + 1, 2));
    c1.title = "Top FedSA methods by joint score";
    c1.hasLegend = false;
    c1.setPosition("G4", "N26");
  } catch (e) {
    ch.getRange("G4").values = [[`Chart render skipped: ${e.message}`]];
  }
  return exportAndVerify(wb, "GuardFed_AD2plus_04_New_Performance_Attack_FedSA.xlsx", "04_fedsa");
}

const files = [];
files.push(await buildAblationWorkbook());
files.push(await buildServerWorkbook());
files.push(await buildSyntheticWorkbook());
files.push(await buildFedSAWorkbook());
console.log("EXPORTED");
for (const f of files) console.log(f);
