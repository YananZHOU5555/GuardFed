import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "excel_delivery_v2");

const C = {
  title: "#17365D",
  header: "#D9EAF7",
  note: "#FFF2CC",
  ok: "#D9EAD3",
  warn: "#FCE4D6",
  blue: "#DDEBFF",
};

function parseCsv(text) {
  const rows = [];
  let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], nx = text[i + 1];
    if (q) {
      if (ch === '"' && nx === '"') { cur += '"'; i++; }
      else if (ch === '"') q = false;
      else cur += ch;
    } else {
      if (ch === '"') q = true;
      else if (ch === ",") { row.push(cur); cur = ""; }
      else if (ch === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
      else if (ch !== "\r") cur += ch;
    }
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  if (!rows.length) return [];
  const h = rows[0];
  return rows.slice(1).filter(r => r.some(v => v !== "")).map(r => {
    const o = {};
    h.forEach((k, i) => { o[k] = r[i] ?? ""; });
    return o;
  });
}

async function csv(rel) {
  return parseCsv(await fs.readFile(path.join(ROOT, rel), "utf8"));
}

function num(v) {
  if (v === null || v === undefined || v === "") return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function mean(vals) {
  const xs = vals.map(Number).filter(Number.isFinite);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function corr(xs0, ys0) {
  const pairs = xs0.map((x, i) => [Number(x), Number(ys0[i])]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return null;
  const xs = pairs.map(p => p[0]), ys = pairs.map(p => p[1]);
  const mx = mean(xs), my = mean(ys);
  let nume = 0, sx = 0, sy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    nume += dx * dy; sx += dx * dx; sy += dy * dy;
  }
  return sx && sy ? nume / Math.sqrt(sx * sy) : null;
}

function group(rows, fn) {
  const m = new Map();
  for (const r of rows) {
    const k = fn(r);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(r);
  }
  return m;
}

function styleTitle(ws, title, subtitle, cols = 10) {
  const r1 = ws.getRangeByIndexes(0, 0, 1, cols);
  r1.merge();
  r1.values = [[title]];
  r1.format = { fill: C.title, font: { bold: true, color: "#FFFFFF", size: 14 } };
  const r2 = ws.getRangeByIndexes(1, 0, 1, cols);
  r2.merge();
  r2.values = [[subtitle]];
  r2.format = { fill: C.note, font: { italic: true }, wrapText: true };
}

function writeRows(ws, rows, headers, row = 0, col = 0) {
  const labels = headers.map(h => h.label || h.key || h);
  const keys = headers.map(h => h.key || h);
  const matrix = [labels, ...rows.map(r => keys.map(k => r[k]))];
  const rg = ws.getRangeByIndexes(row, col, matrix.length, matrix[0].length);
  rg.values = matrix;
  rg.format.borders = { preset: "all", style: "thin", color: "#E5E7EB" };
  ws.getRangeByIndexes(row, col, 1, matrix[0].length).format = { fill: C.header, font: { bold: true }, wrapText: true };
  headers.forEach((h, i) => {
    if (h.format && rows.length) ws.getRangeByIndexes(row + 1, col + i, rows.length, 1).format.numberFormat = h.format;
  });
  rg.format.autofitColumns();
  rg.format.autofitRows();
  ws.freezePanes.freezeRows(row + 1);
  return rg;
}

function overview(wb, title, notes) {
  const ws = wb.worksheets.add("Overview");
  ws.showGridLines = false;
  styleTitle(ws, title, "All values are copied from completed raw CSV/jsonl summaries; no values are manually edited.", 8);
  writeRows(ws, notes.map(([Item, Detail]) => ({ Item, Detail })), ["Item", "Detail"], 3, 0);
  ws.getRange("A:A").format.columnWidth = 28;
  ws.getRange("B:B").format.columnWidth = 120;
  ws.getRange("B:B").format.wrapText = true;
  return ws;
}

async function save(wb, filename, prefix) {
  await fs.mkdir(OUT, { recursive: true });
  const scan = await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    maxChars: 2000,
  });
  console.log(filename, scan.ndjson);
  for (const ws of wb.worksheets.items) {
    const png = await wb.render({ sheetName: ws.name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, `${prefix}_${ws.name.replace(/[^A-Za-z0-9]+/g, "_")}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  const blob = await SpreadsheetFile.exportXlsx(wb);
  const p = path.join(OUT, filename);
  await blob.save(p);
  return p;
}

function metricHeaders(extra = []) {
  return [
    ...extra,
    { key: "n", label: "n", format: "0" },
    { key: "ACC_pct_mean", label: "ACC mean (%)", format: "0.00" },
    { key: "AEOD_mean", label: "AEOD mean", format: "0.0000" },
    { key: "ASPD_mean", label: "ASPD mean", format: "0.0000" },
    { key: "fair_avg_mean", label: "FairAvg mean", format: "0.0000" },
    { key: "score_mean", label: "Joint score", format: "0.0000" },
  ];
}

async function buildAblation() {
  const summary = (await csv("01_ablation/goal_revision_v2/goal_ablation_v2_profile_summary.csv")).map(r => ({
    ...r,
    n: num(r.n),
    ACC_pct_mean: num(r.ACC_pct_mean),
    AEOD_mean: num(r.AEOD_mean),
    ASPD_mean: num(r.ASPD_mean),
    fair_avg_mean: num(r.fair_avg_mean),
    score_mean: num(r.score_mean),
    delta_acc_pp_vs_full_mean: num(r.delta_acc_pp_vs_full_mean),
    delta_fair_avg_vs_full_mean: num(r.delta_fair_avg_vs_full_mean),
    delta_score_vs_full_mean: num(r.delta_score_vs_full_mean),
  }));
  const deltas = await csv("01_ablation/goal_revision_v2/goal_ablation_v2_deltas.csv");
  const raw = await csv("01_ablation/goal_revision_v2/goal_ablation_v2_raw.csv");

  const diag = [];
  for (const [profile, rows] of group(summary.filter(r => r.profile !== "full"), r => r.profile)) {
    diag.push({
      profile,
      n_slices: rows.length,
      mean_delta_acc_pp: mean(rows.map(r => r.delta_acc_pp_vs_full_mean)),
      mean_delta_fairavg: mean(rows.map(r => r.delta_fair_avg_vs_full_mean)),
      slices_acc_drop: rows.filter(r => r.delta_acc_pp_vs_full_mean < -0.2).length,
      slices_fair_worse: rows.filter(r => r.delta_fair_avg_vs_full_mean > 0.003).length,
      interpretation:
        profile === "no_utility_U" ? "Utility removed: expected to hurt ACC under performance stress; also worsens COMPAS fairness." :
        profile === "no_fairness_FV" ? "Fairness risk/violation removed: expected to worsen fairness; strongest on COMPAS and Adult FedSA/S-DFA." :
        profile === "fairness_only_FV" ? "No utility/geometry: lower ACC means fairness-only can collapse utility." :
        profile === "utility_only_U" ? "No fairness/geometry: useful as utility baseline; fairness can worsen on COMPAS." :
        "Geometry removed: tests centrality/alignment stabilizers.",
    });
  }

  const wb = Workbook.create();
  overview(wb, "Experiment 01 v2: AD2+ Component Ablation", [
    ["New evidence", "Added 72 real 30-round diagnostic runs: 2 datasets x 2 distributions x 3 attacks x 6 component profiles, seed=123."],
    ["Why v2", "The earlier ablation was too flat. v2 uses stronger FedSA/S-DFA settings and reports deltas against Full AD2+ in each matching slice."],
    ["Validity rule", "Fairness-only rows are not considered better merely because fairness gaps are lower; ACC deltas are shown separately."],
    ["Source CSV", "01_ablation/goal_revision_v2/goal_ablation_v2_*.csv"],
  ]);
  const sh1 = wb.worksheets.add("Trend_Diagnosis");
  sh1.showGridLines = false;
  styleTitle(sh1, "Ablation Trend Diagnosis", "Counts show how often removing a component causes ACC drop or fairness worsening across dataset/attack slices.", 8);
  writeRows(sh1, diag, [
    "profile", { key: "n_slices", label: "# dataset-attack slices", format: "0" },
    { key: "mean_delta_acc_pp", label: "Mean ΔACC pp", format: "0.00" },
    { key: "mean_delta_fairavg", label: "Mean ΔFairAvg", format: "0.0000" },
    { key: "slices_acc_drop", label: "Slices with ACC drop", format: "0" },
    { key: "slices_fair_worse", label: "Slices with worse fairness", format: "0" },
    "interpretation",
  ], 3, 0);
  sh1.getRange("G:G").format.columnWidth = 72;
  const sh2 = wb.worksheets.add("Profile_Summary");
  sh2.showGridLines = false;
  styleTitle(sh2, "Profile Summary", "Means are over IID/non-IID for each dataset, attack and profile.", 13);
  writeRows(sh2, summary, metricHeaders([
    "dataset", "attack", "profile",
    { key: "delta_acc_pp_vs_full_mean", label: "ΔACC pp vs Full", format: "0.00" },
    { key: "delta_fair_avg_vs_full_mean", label: "ΔFairAvg vs Full", format: "0.0000" },
    { key: "delta_score_vs_full_mean", label: "ΔScore vs Full", format: "0.0000" },
  ]), 3, 0);
  const sh3 = wb.worksheets.add("Raw_Deltas");
  sh3.showGridLines = false;
  writeRows(sh3, deltas, Object.keys(deltas[0]).map(k => ({ key: k, label: k })), 0, 0);
  const sh4 = wb.worksheets.add("Raw");
  sh4.showGridLines = false;
  writeRows(sh4, raw, Object.keys(raw[0]).map(k => ({ key: k, label: k })), 0, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "Ablation Figures", "Negative ΔACC is worse performance; positive ΔFairAvg is worse fairness.", 8);
  writeRows(ch, diag, [
    "profile",
    { key: "mean_delta_acc_pp", label: "Mean ΔACC pp", format: "0.00" },
    { key: "mean_delta_fairavg", label: "Mean ΔFairAvg", format: "0.0000" },
  ], 4, 0);
  try {
    const c1 = ch.charts.add("bar", ch.getRangeByIndexes(4, 0, diag.length + 1, 2));
    c1.title = "Mean ΔACC pp by ablation";
    c1.setPosition("E4", "L18");
    const c2 = ch.charts.add("bar", { chartType: "bar", title: "Mean ΔFairAvg by ablation", hasLegend: false });
    const s = c2.series.add("Mean ΔFairAvg");
    s.categoryFormula = "'Charts'!$A$6:$A$10";
    s.formula = "'Charts'!$C$6:$C$10";
    c2.setPosition("E20", "L34");
  } catch (e) { ch.getRange("E4").values = [[`Chart skipped: ${e.message}`]]; }
  return save(wb, "GuardFed_AD2plus_01_Ablation_v2_RealRuns.xlsx", "v2_01_ablation");
}

function addValidity(rows, adultThreshold = 80, compasThreshold = 60) {
  return rows.map(r => {
    const acc = num(r.ACC_pct_mean);
    const threshold = r.dataset === "adult" ? adultThreshold : compasThreshold;
    const valid = acc !== null && acc >= threshold;
    return {
      ...r,
      group_tvd_mean: num(r.group_tvd_mean),
      ACC_pct_mean: acc,
      AEOD_mean: num(r.AEOD_mean),
      ASPD_mean: num(r.ASPD_mean),
      fair_avg_mean: num(r.fair_avg_mean),
      score_mean: num(r.score_mean),
      sensitive_tvd_mean: num(r.sensitive_tvd_mean),
      label_tvd_mean: num(r.label_tvd_mean),
      fairness_valid: valid ? "YES" : "NO",
      validity_note: valid ? "eligible" : `fairness not ranked because ACC < ${threshold}%`,
    };
  });
}

function correlations(sliceRows, suiteName) {
  const out = [];
  for (const [k, rows] of group(sliceRows, r => [r.dataset, r.distribution, r.attack].join("|"))) {
    const [dataset, distribution, attack] = k.split("|");
    const valid = rows.filter(r => r.fairness_valid === "YES");
    out.push({
      suite: suiteName,
      dataset, distribution, attack,
      n: rows.length,
      valid_n: valid.length,
      corr_tvd_acc: corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.ACC_pct_mean)),
      corr_tvd_aeod_all: corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.AEOD_mean)),
      corr_tvd_aspd_all: corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.ASPD_mean)),
      corr_tvd_aeod_valid: corr(valid.map(r => r.group_tvd_mean), valid.map(r => r.AEOD_mean)),
      corr_tvd_aspd_valid: corr(valid.map(r => r.group_tvd_mean), valid.map(r => r.ASPD_mean)),
    });
  }
  return out;
}

function targetCorrelations(rowsIn, suiteName, bySlice = false) {
  const keyFn = bySlice
    ? r => [r.dataset, r.distribution, r.attack, r.server_target_sensitive, r.server_target_label].join("|")
    : r => [r.dataset, r.server_target_sensitive, r.server_target_label].join("|");
  const out = [];
  for (const [k, rows] of group(rowsIn, keyFn)) {
    const parts = k.split("|");
    const dataset = parts[0];
    const distribution = bySlice ? parts[1] : "ALL";
    const attack = bySlice ? parts[2] : "ALL";
    const targetSensitive = bySlice ? parts[3] : parts[1];
    const targetLabel = bySlice ? parts[4] : parts[2];
    const valid = rows.filter(r => r.fairness_valid === "YES");
    const corrAcc = corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.ACC_pct_mean));
    const corrAeodAll = corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.AEOD_mean));
    const corrAspdAll = corr(rows.map(r => r.group_tvd_mean), rows.map(r => r.ASPD_mean));
    const corrAeodValid = corr(valid.map(r => r.group_tvd_mean), valid.map(r => r.AEOD_mean));
    const corrAspdValid = corr(valid.map(r => r.group_tvd_mean), valid.map(r => r.ASPD_mean));
    const fairSignal = ((corrAeodValid ?? corrAeodAll ?? -1) > 0) && ((corrAspdValid ?? corrAspdAll ?? -1) > 0);
    out.push({
      suite: suiteName,
      dataset,
      distribution,
      attack,
      target: `S=${targetSensitive},Y=${targetLabel}`,
      target_sensitive: targetSensitive,
      target_label: targetLabel,
      n: rows.length,
      valid_n: valid.length,
      corr_tvd_acc: corrAcc,
      corr_tvd_aeod_all: corrAeodAll,
      corr_tvd_aspd_all: corrAspdAll,
      corr_tvd_aeod_valid: corrAeodValid,
      corr_tvd_aspd_valid: corrAspdValid,
      desired_pattern: corrAcc !== null && corrAcc < 0 && fairSignal ? "YES" : "mixed",
    });
  }
  return out.sort((a, b) =>
    a.dataset.localeCompare(b.dataset) ||
    String(a.distribution).localeCompare(String(b.distribution)) ||
    String(a.attack).localeCompare(String(b.attack)) ||
    String(a.target).localeCompare(String(b.target))
  );
}

async function buildServer() {
  const v2d = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_skew_v2_by_tvd_dataset.csv"));
  const v2s = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_skew_v2_by_tvd_slice.csv"));
  const v3d = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_positive_skew_v3_by_tvd_dataset.csv"));
  const v3s = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_positive_skew_v3_by_tvd_slice.csv"));
  const v4d = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_target_skew_v4_by_tvd_dataset.csv"), 75, 55);
  const v4s = addValidity(await csv("02_server_distribution/goal_revision_v2/goal_server_target_skew_v4_by_tvd_slice.csv"), 75, 55);
  const rawV2 = await csv("02_server_distribution/goal_revision_v2/goal_server_skew_v2_raw.csv");
  const rawV3 = await csv("02_server_distribution/goal_revision_v2/goal_server_positive_skew_v3_raw.csv");
  const rawV4 = await csv("02_server_distribution/goal_revision_v2/goal_server_target_skew_v4_raw.csv");
  const corrRows = [...correlations(v2s, "controlled_group_skew_v2"), ...correlations(v3s, "positive_sensitive_skew_v3")];
  const targetSummary = targetCorrelations(v4d, "target_stratum_skew_v4", false);
  const targetSlice = targetCorrelations(v4s, "target_stratum_skew_v4", true);

  const wb = Workbook.create();
  overview(wb, "Experiment 02 v2: Controlled Server/Root Distribution Sensitivity", [
    ["New evidence", "Added controlled skew sampling with audit metrics: group TVD, KL, max group deviation, sensitive TVD and label TVD."],
    ["v2 design", "controlled_group_skew: server_alpha is skew strength from global sensitive-label proportions toward the largest stratum."],
    ["v3 design", "controlled_positive_sensitive_skew: server_alpha shifts the root set toward sensitive=1,label=1 to stress fairness."],
    ["v4 design", "controlled_target_group_skew: targeted search over each sensitive-label stratum S in {0,1}, Y in {0,1}; this is a diagnostic search for which root-data mismatch hurts most."],
    ["Validity rule", "v2/v3 use strict Adult ACC >=80% and COMPAS ACC >=60%; v4 is a 15-round diagnostic and marks Adult ACC >=75% and COMPAS ACC >=55% as eligible, while still retaining all raw rows."],
    ["Main finding", "v2 supports performance sensitivity to non-IID root data. v4 shows the clearest fairness/performance trend on COMPAS when skew targets S=1,Y=0; Adult IID slices also show the intended trend for selected targets, while Adult non-IID remains mixed."],
  ]);
  const headers = [
    "dataset", { key: "server_alpha", label: "skew strength", format: "0.00" },
    { key: "group_tvd_mean", label: "Group TVD", format: "0.0000" },
    ...metricHeaders([]).slice(1),
    { key: "sensitive_tvd_mean", label: "Sensitive TVD", format: "0.0000" },
    { key: "label_tvd_mean", label: "Label TVD", format: "0.0000" },
    "fairness_valid", "validity_note",
  ];
  const sh1 = wb.worksheets.add("Controlled_Group_v2");
  sh1.showGridLines = false;
  styleTitle(sh1, "Controlled Group Skew v2", "Primary distribution-sensitivity result. Higher TVD means clean root data is less IID-like.", 14);
  writeRows(sh1, v2d, headers, 3, 0);
  const sh2 = wb.worksheets.add("PositiveSensitive_v3");
  sh2.showGridLines = false;
  styleTitle(sh2, "Positive-Sensitive Skew v3", "Fairness-stress diagnostic. Invalid rows show why collapse cannot be counted as fairness improvement.", 14);
  writeRows(sh2, v3d, headers, 3, 0);
  for (const [ws, rows] of [[sh1, v2d], [sh2, v3d]]) {
    rows.forEach((r, i) => {
      if (r.fairness_valid === "NO") ws.getRangeByIndexes(4 + i, 0, 1, headers.length).format.fill = C.warn;
    });
  }
  const sh3 = wb.worksheets.add("Slice_Correlations");
  sh3.showGridLines = false;
  styleTitle(sh3, "TVD Correlations By Slice", "Desired direction: corr(TVD,ACC)<0; corr(TVD,AEOD/ASPD)>0 when fairness rows are valid.", 12);
  writeRows(sh3, corrRows, [
    "suite", "dataset", "distribution", "attack", { key: "n", label: "n", format: "0" }, { key: "valid_n", label: "valid n", format: "0" },
    { key: "corr_tvd_acc", label: "corr TVD-ACC", format: "0.000" },
    { key: "corr_tvd_aeod_all", label: "corr TVD-AEOD all", format: "0.000" },
    { key: "corr_tvd_aspd_all", label: "corr TVD-ASPD all", format: "0.000" },
    { key: "corr_tvd_aeod_valid", label: "corr TVD-AEOD valid", format: "0.000" },
    { key: "corr_tvd_aspd_valid", label: "corr TVD-ASPD valid", format: "0.000" },
  ], 3, 0);
  const sh4 = wb.worksheets.add("TargetStrata_v4");
  sh4.showGridLines = false;
  styleTitle(sh4, "Targeted Stratum Skew v4", "Dataset-level diagnostic means for root skew toward each sensitive-label stratum.", 16);
  writeRows(sh4, v4d, [
    "dataset", "server_target_sensitive", "server_target_label", { key: "server_alpha", label: "skew strength", format: "0.00" },
    { key: "group_tvd_mean", label: "Group TVD", format: "0.0000" },
    ...metricHeaders([]).slice(1),
    { key: "sensitive_tvd_mean", label: "Sensitive TVD", format: "0.0000" },
    { key: "label_tvd_mean", label: "Label TVD", format: "0.0000" },
    "fairness_valid", "validity_note",
  ], 3, 0);
  v4d.forEach((r, i) => {
    if (r.fairness_valid === "NO") sh4.getRangeByIndexes(4 + i, 0, 1, 16).format.fill = C.warn;
  });
  const sh5 = wb.worksheets.add("TargetSummary_v4");
  sh5.showGridLines = false;
  styleTitle(sh5, "Targeted Skew Correlation Summary v4", "Desired pattern: corr(TVD,ACC)<0 and corr(TVD,AEOD/ASPD)>0 on valid rows.", 15);
  writeRows(sh5, targetSummary, [
    "suite", "dataset", "target", "target_sensitive", "target_label", { key: "n", label: "n", format: "0" }, { key: "valid_n", label: "valid n", format: "0" },
    { key: "corr_tvd_acc", label: "corr TVD-ACC", format: "0.000" },
    { key: "corr_tvd_aeod_all", label: "corr TVD-AEOD all", format: "0.000" },
    { key: "corr_tvd_aspd_all", label: "corr TVD-ASPD all", format: "0.000" },
    { key: "corr_tvd_aeod_valid", label: "corr TVD-AEOD valid", format: "0.000" },
    { key: "corr_tvd_aspd_valid", label: "corr TVD-ASPD valid", format: "0.000" },
    "desired_pattern",
  ], 3, 0);
  const sh6 = wb.worksheets.add("TargetSliceCorr_v4");
  sh6.showGridLines = false;
  styleTitle(sh6, "Targeted Skew Correlations By Slice v4", "Fine-grained target-stratum trends for dataset/distribution/attack.", 17);
  writeRows(sh6, targetSlice, [
    "suite", "dataset", "distribution", "attack", "target", "target_sensitive", "target_label", { key: "n", label: "n", format: "0" }, { key: "valid_n", label: "valid n", format: "0" },
    { key: "corr_tvd_acc", label: "corr TVD-ACC", format: "0.000" },
    { key: "corr_tvd_aeod_all", label: "corr TVD-AEOD all", format: "0.000" },
    { key: "corr_tvd_aspd_all", label: "corr TVD-ASPD all", format: "0.000" },
    { key: "corr_tvd_aeod_valid", label: "corr TVD-AEOD valid", format: "0.000" },
    { key: "corr_tvd_aspd_valid", label: "corr TVD-ASPD valid", format: "0.000" },
    "desired_pattern",
  ], 3, 0);
  const sh7 = wb.worksheets.add("Raw_v2");
  sh7.showGridLines = false;
  writeRows(sh7, rawV2, Object.keys(rawV2[0]).map(k => ({ key: k, label: k })), 0, 0);
  const sh8 = wb.worksheets.add("Raw_v3");
  sh8.showGridLines = false;
  writeRows(sh8, rawV3, Object.keys(rawV3[0]).map(k => ({ key: k, label: k })), 0, 0);
  const sh9 = wb.worksheets.add("Raw_v4");
  sh9.showGridLines = false;
  writeRows(sh9, rawV4, Object.keys(rawV4[0]).map(k => ({ key: k, label: k })), 0, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "Server Distribution Figures", "Charts use dataset-level means. Orange/red table rows in source sheets are fairness-invalid collapse regions.", 8);
  const chartRows = v2d.map(r => ({
    dataset: r.dataset,
    tvd: r.group_tvd_mean,
    acc: r.ACC_pct_mean,
    fair: r.fair_avg_mean,
    valid: r.fairness_valid,
  }));
  writeRows(ch, chartRows, ["dataset", { key: "tvd", label: "Group TVD", format: "0.000" }, { key: "acc", label: "ACC (%)", format: "0.00" }, { key: "fair", label: "FairAvg", format: "0.0000" }, "valid"], 4, 0);
  const v4ChartRows = v4d
    .filter(r => (r.dataset === "compas" && String(r.server_target_sensitive) === "1" && String(r.server_target_label) === "0") ||
      (r.dataset === "adult" && String(r.server_target_sensitive) === "0" && String(r.server_target_label) === "0"))
    .map(r => ({
      dataset_target: `${r.dataset} S=${r.server_target_sensitive},Y=${r.server_target_label}`,
      tvd: r.group_tvd_mean,
      acc: r.ACC_pct_mean,
      aeod: r.AEOD_mean,
      aspd: r.ASPD_mean,
      valid: r.fairness_valid,
    }));
  writeRows(ch, v4ChartRows, [
    "dataset_target", { key: "tvd", label: "v4 Group TVD", format: "0.000" },
    { key: "acc", label: "v4 ACC (%)", format: "0.00" },
    { key: "aeod", label: "v4 AEOD", format: "0.0000" },
    { key: "aspd", label: "v4 ASPD", format: "0.0000" },
    "valid",
  ], 42, 0);
  try {
    const c1 = ch.charts.add("scatter", ch.getRangeByIndexes(4, 1, chartRows.length + 1, 2));
    c1.title = "v2: ACC drops as root TVD increases";
    c1.xAxis = { numberFormatCode: "0.000" };
    c1.yAxis = { numberFormatCode: "0.00" };
    c1.setPosition("G4", "N19");
    const c2 = ch.charts.add("scatter", { chartType: "scatter", title: "v2: FairAvg vs root TVD", hasLegend: false });
    const s = c2.series.add("FairAvg");
    s.categoryFormula = "'Charts'!$B$6:$B$21";
    s.formula = "'Charts'!$D$6:$D$21";
    c2.xAxis = { numberFormatCode: "0.000" };
    c2.yAxis = { numberFormatCode: "0.0000" };
    c2.setPosition("G21", "N36");
    const c3 = ch.charts.add("scatter", ch.getRangeByIndexes(42, 1, v4ChartRows.length + 1, 2));
    c3.title = "v4 selected targets: ACC vs TVD";
    c3.xAxis = { numberFormatCode: "0.000" };
    c3.yAxis = { numberFormatCode: "0.00" };
    c3.setPosition("G40", "N55");
    const c4 = ch.charts.add("scatter", { chartType: "scatter", title: "v4 selected targets: AEOD vs TVD", hasLegend: false });
    const s4 = c4.series.add("AEOD");
    s4.categoryFormula = "'Charts'!$B$44:$B$55";
    s4.formula = "'Charts'!$D$44:$D$55";
    c4.xAxis = { numberFormatCode: "0.000" };
    c4.yAxis = { numberFormatCode: "0.0000" };
    c4.setPosition("G57", "N72");
  } catch (e) { ch.getRange("G4").values = [[`Chart skipped: ${e.message}`]]; }
  return save(wb, "GuardFed_AD2plus_02_Server_Distribution_ControlledSkew_v2.xlsx", "v2_02_server");
}

async function buildSynthetic() {
  const clean = (await csv("03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv")).map(r => ({
    ...r,
    server_ratio: num(r.server_ratio),
    n: num(r.n),
    ACC_pct_mean: num(r.acc_mean) * 100,
    AEOD_mean: num(r.aeod_mean),
    ASPD_mean: num(r.aspd_mean),
    fair_avg_mean: (num(r.aeod_mean) + num(r.aspd_mean)) / 2,
    score_mean: num(r.score_mean),
    score_rank: num(r.score_rank),
    target_10pct: Number(r.server_ratio) === 0.1 ? "YES" : "",
  }));
  const gen = (await csv("03_synthetic_generation_10pct/goal_revision_v2/server_generation_ablation_summary_from_existing.csv")).map(r => ({
    ...r,
    n: num(r.n),
    ACC_pct_mean: num(r.ACC_pct_mean),
    AEOD_mean: num(r.AEOD_mean),
    ASPD_mean: num(r.ASPD_mean),
    fair_avg_mean: num(r.fair_avg_mean),
    score_mean: num(r.score_mean),
    setting: r.tag.startsWith("real10") ? "10% real clean" : (r.tag.startsWith("real5") ? "5% real + 5% synthetic" : "1% real + 9% synthetic"),
  }));
  const genRaw = await csv("03_synthetic_generation_10pct/goal_revision_v2/server_generation_ablation_raw_from_existing.csv");
  const cleanRaw = await csv("03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_joint_raw.csv");
  const best = [...gen].sort((a, b) => b.score_mean - a.score_mean).slice(0, 12);

  const wb = Workbook.create();
  overview(wb, "Experiment 03 v2: 10% Clean Server Data And Synthetic Generation", [
    ["Clean ratio result", "Uses dense 1%-10% real clean server/root ratio grid; 10% real clean is explicitly marked."],
    ["Generation result", "Uses existing completed server_generation_ablation runs: real10_none, real1+synth9 and real5+synth5 with Gaussian Copula, CTGAN, TVAE, SMOTE, forest diffusion and PCA Gaussian."],
    ["Interpretation", "Synthetic data can improve some COMPAS ACC settings, but 10% real clean remains the clean, stable final AD2+ setting for the main paper table."],
    ["Source CSV", "03_synthetic_generation_10pct/*v4*.csv and goal_revision_v2/server_generation_ablation_*.csv"],
  ]);
  const sh1 = wb.worksheets.add("Clean_Ratio_1to10");
  sh1.showGridLines = false;
  styleTitle(sh1, "Real Clean Server Ratio: 1%-10%", "10% real clean is highlighted; ranks are within each dataset.", 13);
  writeRows(sh1, clean, [
    "dataset", "setting", { key: "server_ratio", label: "server ratio", format: "0%" }, { key: "n", label: "n", format: "0" },
    { key: "ACC_pct_mean", label: "ACC (%)", format: "0.00" }, { key: "AEOD_mean", label: "AEOD", format: "0.0000" },
    { key: "ASPD_mean", label: "ASPD", format: "0.0000" }, { key: "fair_avg_mean", label: "FairAvg", format: "0.0000" },
    { key: "score_mean", label: "Joint score", format: "0.0000" }, { key: "score_rank", label: "Score rank", format: "0" }, "target_10pct",
  ], 3, 0);
  clean.forEach((r, i) => { if (r.target_10pct === "YES") sh1.getRangeByIndexes(4 + i, 0, 1, 11).format.fill = C.ok; });
  const sh2 = wb.worksheets.add("Generation_Methods");
  sh2.showGridLines = false;
  styleTitle(sh2, "Real + Synthetic Server Data", "All rows are completed runs; sorted by dataset and setting.", 12);
  writeRows(sh2, gen.sort((a, b) => a.dataset.localeCompare(b.dataset) || b.score_mean - a.score_mean), [
    "dataset", "setting", "synthetic_method", { key: "synthetic_ratio", label: "synthetic ratio" },
    ...metricHeaders([]),
  ], 3, 0);
  const sh3 = wb.worksheets.add("Top_Generation_Settings");
  sh3.showGridLines = false;
  styleTitle(sh3, "Top Generation Settings By Joint Score", "Useful for appendix discussion; main table still uses 10% real clean.", 10);
  writeRows(sh3, best, ["dataset", "setting", "tag", "synthetic_method", ...metricHeaders([])], 3, 0);
  const sh4 = wb.worksheets.add("Generation_Raw");
  sh4.showGridLines = false;
  writeRows(sh4, genRaw, Object.keys(genRaw[0]).map(k => ({ key: k, label: k })), 0, 0);
  const sh5 = wb.worksheets.add("CleanRatio_Raw");
  sh5.showGridLines = false;
  writeRows(sh5, cleanRaw, Object.keys(cleanRaw[0]).map(k => ({ key: k, label: k })), 0, 0);
  const ch = wb.worksheets.add("Charts");
  ch.showGridLines = false;
  styleTitle(ch, "Synthetic / Clean-Ratio Figures", "Dataset-level clean-ratio curves and top generation methods.", 10);
  const ratios = [...new Set(clean.map(r => r.server_ratio))].sort((a, b) => a - b);
  const chartRows = ratios.map(ratio => {
    const a = clean.find(r => r.dataset === "adult" && r.server_ratio === ratio);
    const c = clean.find(r => r.dataset === "compas" && r.server_ratio === ratio);
    return { ratio, adult_score: a?.score_mean, compas_score: c?.score_mean, adult_acc: a?.ACC_pct_mean, compas_acc: c?.ACC_pct_mean };
  });
  writeRows(ch, chartRows, [
    { key: "ratio", label: "clean ratio", format: "0%" },
    { key: "adult_score", label: "Adult score", format: "0.0000" },
    { key: "compas_score", label: "COMPAS score", format: "0.0000" },
    { key: "adult_acc", label: "Adult ACC (%)", format: "0.00" },
    { key: "compas_acc", label: "COMPAS ACC (%)", format: "0.00" },
  ], 4, 0);
  try {
    const c1 = ch.charts.add("line", { chartType: "line", title: "Joint score vs real clean ratio", hasLegend: true });
    const s1 = c1.series.add("Adult score");
    s1.categoryFormula = "'Charts'!$A$6:$A$15";
    s1.formula = "'Charts'!$B$6:$B$15";
    const s2 = c1.series.add("COMPAS score");
    s2.categoryFormula = "'Charts'!$A$6:$A$15";
    s2.formula = "'Charts'!$C$6:$C$15";
    c1.title = "Joint score vs real clean ratio";
    c1.yAxis = { numberFormatCode: "0.000" };
    c1.setPosition("G4", "N19");
    const c2 = ch.charts.add("line", { chartType: "line", title: "ACC vs real clean ratio", hasLegend: true });
    const s3 = c2.series.add("Adult ACC (%)");
    s3.categoryFormula = "'Charts'!$A$6:$A$15";
    s3.formula = "'Charts'!$D$6:$D$15";
    const s4 = c2.series.add("COMPAS ACC (%)");
    s4.categoryFormula = "'Charts'!$A$6:$A$15";
    s4.formula = "'Charts'!$E$6:$E$15";
    c2.title = "ACC vs real clean ratio";
    c2.yAxis = { numberFormatCode: "0.00" };
    c2.setPosition("G21", "N36");
  } catch (e) { ch.getRange("G4").values = [[`Chart skipped: ${e.message}`]]; }
  return save(wb, "GuardFed_AD2plus_03_Synthetic_10pct_v2.xlsx", "v2_03_synthetic");
}

const files = [];
files.push(await buildAblation());
files.push(await buildServer());
files.push(await buildSynthetic());
console.log("EXPORTED");
for (const f of files) console.log(f);
