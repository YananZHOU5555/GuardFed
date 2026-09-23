import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "detailed_delivery_v2");

const SOURCES = {
  ablationStress: path.join(ROOT, "01_ablation/goal_revision_v3/goal_ablation_v3_stress_raw.csv"),
  ablationBenign: path.join(ROOT, "01_ablation/goal_revision_v4/goal_ablation_v4_benign_raw.csv"),
  serverByLevel: path.join(ROOT, "02_server_distribution/goal_revision_v6/goal_server_iid_sensitivity_v6_by_level.csv"),
  serverCorr: path.join(ROOT, "02_server_distribution/goal_revision_v6/goal_server_iid_sensitivity_v6_correlations.csv"),
  cleanRatioDataset: path.join(ROOT, "03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv"),
  cleanRatioSlice: path.join(ROOT, "03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_slice.csv"),
  generationSummary: path.join(ROOT, "03_synthetic_generation_10pct/goal_revision_v2/server_generation_ablation_summary_from_existing.csv"),
  fedsaPaperStyle: path.join(ROOT, "04_new_performance_attack_FedSA/fedsa_paper_style_selected_ad2plus_raw.csv"),
  fedsaAllMethods: path.join(ROOT, "04_new_performance_attack_FedSA/fedsa_all_methods_summary.csv"),
  fedsaCandidates: path.join(ROOT, "04_new_performance_attack_FedSA/fedsa_ad2plus_candidate_joint_summary.csv"),
};

const C = {
  title: "#17365D",
  header: "#D9EAF7",
  note: "#FFF2CC",
  section: "#EAF3F8",
  ok: "#D9EAD3",
  warn: "#FCE4D6",
  grid: "#D9E2EC",
  ours: "#E2F0D9",
};

const profileConfig = {
  full: ["Full AD2+", "完整算法", 3.0, 1.4, 0.7, 0.8, 1.8, "on"],
  no_performance_UCA: ["No performance U/C/A", "去掉性能防御项", 0.0, 1.4, 0.7, 0.0, 0.0, "on"],
  no_fairness_FC: ["No fairness F/C", "去掉公平防御项", 3.0, 0.0, 0.0, 0.8, 1.8, "off"],
  no_geometry_CA: ["No geometry C/A", "去掉几何一致性", 3.0, 1.4, 0.7, 0.0, 0.0, "on"],
  utility_only_U: ["Utility only", "只保留 utility", 3.0, 0.0, 0.0, 0.0, 0.0, "off"],
  fairness_only_FC: ["Fairness only F/C", "只保留公平项", 0.0, 1.4, 0.7, 0.0, 0.0, "on"],
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

async function csv(file) {
  return parseCsv(await fs.readFile(file, "utf8"));
}

function num(v) {
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function mean(vals) {
  const xs = vals.map(Number).filter(Number.isFinite);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function styleTitle(ws, title, subtitle, cols) {
  ws.showGridLines = false;
  const r1 = ws.getRangeByIndexes(0, 0, 1, cols);
  r1.merge();
  r1.values = [[title]];
  r1.format = { fill: C.title, font: { bold: true, color: "#FFFFFF", size: 14 } };
  r1.format.rowHeight = 24;
  const r2 = ws.getRangeByIndexes(1, 0, 1, cols);
  r2.merge();
  r2.values = [[subtitle]];
  r2.format = { fill: C.note, font: { italic: true }, wrapText: true };
  r2.format.rowHeight = 32;
}

function writeRows(ws, rows, headers, row, col, tableName = null) {
  const labels = headers.map(h => h.label || h.key || h);
  const keys = headers.map(h => h.key || h);
  const matrix = [labels, ...rows.map(r => keys.map(k => r[k]))];
  const range = ws.getRangeByIndexes(row, col, matrix.length, matrix[0].length);
  range.values = matrix;
  range.format.borders = { preset: "all", style: "thin", color: C.grid };
  const headerRange = ws.getRangeByIndexes(row, col, 1, labels.length);
  headerRange.format = { fill: C.header, font: { bold: true }, wrapText: true };
  headerRange.format.rowHeight = 34;
  headers.forEach((h, i) => {
    if (h.format && rows.length) {
      ws.getRangeByIndexes(row + 1, col + i, rows.length, 1).format.numberFormat = h.format;
    }
  });
  range.format.autofitColumns();
  range.format.autofitRows();
  if (tableName) {
    const table = ws.tables.add(range, true, tableName);
    table.style = "TableStyleLight9";
  }
  return range;
}

function setWidths(ws, widths) {
  widths.forEach((w, i) => {
    ws.getRangeByIndexes(0, i, 1, 1).format.columnWidth = w;
  });
}

function normalizeAblation(rows) {
  return rows.map(r => ({
    dataset: r.dataset,
    distribution: r.distribution,
    attack: r.attack,
    profile: r.profile,
    seed: num(r.seed),
    rounds: num(r.rounds),
    ACC_pct: num(r.ACC_pct),
    AEOD: num(r.AEOD),
    ASPD: num(r.ASPD),
    FairAvg: num(r.fair_avg),
    Score: num(r.score),
  }));
}

function aggregate(rows, keys) {
  const groups = new Map();
  for (const r of rows) {
    const key = keys.map(k => r[k]).join("||");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return Array.from(groups.values()).map(vals => {
    const out = {};
    keys.forEach(k => { out[k] = vals[0][k]; });
    out.n = vals.length;
    out.rounds_mean = mean(vals.map(v => v.rounds));
    out.ACC_pct = mean(vals.map(v => v.ACC_pct));
    out.AEOD = mean(vals.map(v => v.AEOD));
    out.ASPD = mean(vals.map(v => v.ASPD));
    out.FairAvg = mean(vals.map(v => v.FairAvg));
    out.Score = mean(vals.map(v => v.Score));
    return out;
  });
}

function makeAblationRows(agg) {
  return agg
    .sort((a, b) =>
      a.dataset.localeCompare(b.dataset) ||
      a.distribution.localeCompare(b.distribution) ||
      a.attack.localeCompare(b.attack) ||
      a.profile.localeCompare(b.profile)
    )
    .map(r => {
      const cfg = profileConfig[r.profile] || [r.profile, "", null, null, null, null, null, ""];
      return {
        Dataset: r.dataset,
        Distribution: r.distribution,
        Attack: r.attack,
        Profile: cfg[0],
        "Ablation meaning": cfg[1],
        n: r.n,
        "Rounds mean": r.rounds_mean,
        "Utility weight": cfg[2],
        "Fair risk": cfg[3],
        "Fair violation": cfg[4],
        Centrality: cfg[5],
        Alignment: cfg[6],
        Calibration: cfg[7],
        "ACC (%)": r.ACC_pct,
        AEOD: r.AEOD,
        ASPD: r.ASPD,
        FairAvg: r.FairAvg,
        Score: r.Score,
      };
    });
}

function lookup(agg, dataset, attack, profile) {
  return agg.find(r => r.dataset === dataset && r.attack === attack && r.profile === profile);
}

function makeAblationEffectRows(agg) {
  const out = [];
  for (const dataset of ["adult", "compas"]) {
    for (const attack of ["FedSA", "S-DFA"]) {
      const full = lookup(agg, dataset, attack, "full");
      const noPerf = lookup(agg, dataset, attack, "no_performance_UCA");
      out.push({
        Dataset: dataset,
        Test: "性能防御项",
        Attack: attack,
        Metric: "ACC (%)",
        Full: full?.ACC_pct ?? null,
        Ablated: noPerf?.ACC_pct ?? null,
        Delta: full && noPerf ? noPerf.ACC_pct - full.ACC_pct : null,
        Expected: "下降",
      });
    }
    for (const attack of ["F Flip", "S-DFA"]) {
      const full = lookup(agg, dataset, attack, "full");
      const noFair = lookup(agg, dataset, attack, "no_fairness_FC");
      out.push({
        Dataset: dataset,
        Test: "公平防御项",
        Attack: attack,
        Metric: "FairAvg",
        Full: full?.FairAvg ?? null,
        Ablated: noFair?.FairAvg ?? null,
        Delta: full && noFair ? noFair.FairAvg - full.FairAvg : null,
        Expected: "上升",
      });
    }
  }
  return out;
}

function makeServerRows(rows) {
  const clientAlpha = { IID: 5000, "non-IID": 5 };
  const rootSkew = {
    "IID server/root": "stratified clean 10%; controlled skew=0.00",
    "Mild non-IID server/root": "stratified clean 10%; controlled skew=0.05",
    "Moderate non-IID server/root": "stratified clean 10%; controlled skew=0.10",
  };
  const baseline = new Map();
  for (const r of rows) {
    if (r.level === "IID server/root") {
      baseline.set(`${r.dataset}||${r.distribution}||${r.attack}`, r);
    }
  }
  return rows.map(r => {
    const b = baseline.get(`${r.dataset}||${r.distribution}||${r.attack}`);
    const acc = num(r.ACC_pct_mean);
    const fair = num(r.FairAvg_mean);
    const baseAcc = num(b?.ACC_pct_mean);
    const baseFair = num(b?.FairAvg_mean);
    const dAcc = Number.isFinite(acc) && Number.isFinite(baseAcc) ? acc - baseAcc : null;
    const dFair = Number.isFinite(fair) && Number.isFinite(baseFair) ? fair - baseFair : null;
    const alpha = clientAlpha[r.distribution] ?? null;
    const skew = rootSkew[r.level] ?? r.level;
    return {
      Dataset: r.dataset,
      "Client distribution": `${r.distribution} client split`,
      "Client Dirichlet alpha": alpha,
      Attack: r.attack,
      "Server/root level": r.level,
      "Root skew setting": skew,
      "Level index": num(r.level_index),
      n: num(r.n),
      "Group TVD": num(r.group_tvd_mean),
      "Sensitive TVD": num(r.sensitive_tvd_mean),
      "Label TVD": num(r.label_tvd_mean),
      "ACC (%)": acc,
      AEOD: num(r.AEOD_mean),
      ASPD: num(r.ASPD_mean),
      FairAvg: fair,
      Score: num(r.Score_mean),
      "ΔACC vs IID root": dAcc,
      "ΔFairAvg vs IID root": dFair,
      "Numeric reading": `client alpha=${alpha}; ${skew}; Group/Sensitive/Label TVD=${num(r.group_tvd_mean).toFixed(4)}/${num(r.sensitive_tvd_mean).toFixed(4)}/${num(r.label_tvd_mean).toFixed(4)}; ΔACC=${dAcc == null ? "" : dAcc.toFixed(2)} pp; ΔFairAvg=${dFair == null ? "" : dFair.toFixed(4)}`,
    };
  });
}

function makeServerCorrRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Distribution: r.distribution,
    Attack: r.attack,
    "n levels": num(r.n_levels),
    "corr(level, ACC)": num(r.corr_level_acc),
    "corr(level, AEOD)": num(r.corr_level_aeod),
    "corr(level, ASPD)": num(r.corr_level_aspd),
    "corr(level, FairAvg)": num(r.corr_level_fairavg),
    "Moderate-IID ACC delta": num(r.delta_acc_moderate_minus_iid),
    "Moderate-IID FairAvg delta": num(r.delta_fairavg_moderate_minus_iid),
  }));
}

function cleanDatasetRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Setting: r.setting,
    "Server ratio": num(r.server_ratio),
    "Server sampling": r.server_sampling,
    n: num(r.n),
    "ACC mean": num(r.acc_mean) * 100,
    "ACC std": num(r.acc_std) * 100,
    "AEOD mean": num(r.aeod_mean),
    "AEOD std": num(r.aeod_std),
    "ASPD mean": num(r.aspd_mean),
    "ASPD std": num(r.aspd_std),
    FairAvg: (num(r.aeod_mean) + num(r.aspd_mean)) / 2,
    "Score mean": num(r.score_mean),
    "Score std": num(r.score_std),
    "Round mean": num(r.round_mean),
    "Score rank": num(r.score_rank),
    "ACC rank": num(r.acc_rank),
    "AEOD rank": num(r.aeod_rank),
    "ASPD rank": num(r.aspd_rank),
  }));
}

function cleanSliceRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Distribution: r.distribution,
    Attack: r.attack,
    Setting: r.setting,
    "Server ratio": num(r.server_ratio),
    "Server sampling": r.server_sampling,
    n: num(r.n),
    "ACC mean": num(r.acc_mean) * 100,
    "ACC std": num(r.acc_std) * 100,
    "AEOD mean": num(r.aeod_mean),
    "AEOD std": num(r.aeod_std),
    "ASPD mean": num(r.aspd_mean),
    "ASPD std": num(r.aspd_std),
    FairAvg: (num(r.aeod_mean) + num(r.aspd_mean)) / 2,
    "Score mean": num(r.score_mean),
    "Score std": num(r.score_std),
    "Round mean": num(r.round_mean),
    "Score rank": num(r.score_rank),
    "ACC rank": num(r.acc_rank),
    "AEOD rank": num(r.aeod_rank),
    "ASPD rank": num(r.aspd_rank),
  }));
}

function generationRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Tag: r.tag,
    Generator: r.synthetic_method,
    "Synthetic ratio": num(r.synthetic_ratio),
    n: num(r.n),
    "ACC mean (%)": num(r.ACC_pct_mean),
    "ACC min (%)": num(r.ACC_pct_min),
    "ACC max (%)": num(r.ACC_pct_max),
    "AEOD mean": num(r.AEOD_mean),
    "AEOD min": num(r.AEOD_min),
    "AEOD max": num(r.AEOD_max),
    "ASPD mean": num(r.ASPD_mean),
    "ASPD min": num(r.ASPD_min),
    "ASPD max": num(r.ASPD_max),
    "FairAvg mean": num(r.fair_avg_mean),
    "FairAvg min": num(r.fair_avg_min),
    "FairAvg max": num(r.fair_avg_max),
    "Score mean": num(r.score_mean),
    "Score min": num(r.score_min),
    "Score max": num(r.score_max),
  }));
}

function fedsaPaperRows(rows) {
  return rows.map(r => {
    const isAcc = r.Metric === "ACC";
    const conv = v => isAcc ? num(v) * 100 : num(v);
    return {
      Method: r.Method,
      Citation: r.Citation,
      Metric: r.Metric,
      "ADULT IID": conv(r["ADULT IID"]),
      "ADULT non-IID": conv(r["ADULT non-IID"]),
      "COMPAS IID": conv(r["COMPAS IID"]),
      "COMPAS non-IID": conv(r["COMPAS non-IID"]),
    };
  });
}

function fedsaAllMethodRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Distribution: r.distribution,
    Attack: r.attack,
    Method: r.method,
    n: num(r.n),
    "ACC mean (%)": num(r.acc_mean) * 100,
    "ACC std (%)": num(r.acc_std) * 100,
    "AEOD mean": num(r.aeod_mean),
    "AEOD std": num(r.aeod_std),
    "ASPD mean": num(r.aspd_mean),
    "ASPD std": num(r.aspd_std),
    "FairAvg mean": num(r.fair_mean),
    "FairAvg std": num(r.fair_std),
    "Score mean": num(r.score_mean),
    "Score std": num(r.score_std),
    "ACC rank": num(r.acc_rank),
    "Fair rank": num(r.fair_rank),
    "Score rank": num(r.score_rank),
  }));
}

function fedsaCandidateRows(rows) {
  return rows.map(r => ({
    Dataset: r.dataset,
    Distribution: r.distribution,
    Attack: r.attack,
    Method: r.method,
    Suite: r.suite,
    Tag: r.tag,
    "Base method": r.base_method,
    "Server ratio": num(r.server_ratio),
    "Server sampling": r.server_sampling,
    Mode: r.ad2_plus_mode,
    Budget: num(r.ad2_calibration_budget),
    "Max acc drop": num(r.ad2_calibration_max_acc_drop),
    "Norm clip": num(r.ad2_norm_clip_scale),
    "Norm mode": r.ad2_norm_mode,
    Objective: r.ad2_calibration_objective,
    n: num(r.n),
    "ACC mean (%)": num(r.acc_mean) * 100,
    "ACC std (%)": num(r.acc_std) * 100,
    "AEOD mean": num(r.aeod_mean),
    "AEOD std": num(r.aeod_std),
    "ASPD mean": num(r.aspd_mean),
    "ASPD std": num(r.aspd_std),
    "Score mean": num(r.score_mean),
    "Score std": num(r.score_std),
    "Round mean": num(r.round_mean),
    "Score rank": num(r.score_rank),
    "ACC rank": num(r.acc_rank),
    "AEOD rank": num(r.aeod_rank),
    "ASPD rank": num(r.aspd_rank),
  }));
}

function markdownReadme(counts) {
  return `# GuardFed-AD2+ Detailed Four Experiment Tables

这个版本不是 compact 摘要，而是把之前做过的四组大表重新整理成可读的完整表。Excel 文件名：

\`GuardFed_AD2plus_Four_Experiments_Detailed_Tables.xlsx\`

## 工作簿结构

- \`01_Ablation_FULL\`：完整消融表，包含 dataset、distribution、attack、profile、五类参数、ACC、AEOD、ASPD、FairAvg、Score。共 ${counts.ablationDetail} 行明细，顶部另有 8 行关键效应摘要。
- \`02_ServerRoot_FULL\`：完整 server/root 分布表，包含 Adult 和 COMPAS、IID/non-IID、Benign/FedSA、三种 server/root 分布层级，以及 client alpha、root skew、Group/Sensitive/Label TVD、ACC、AEOD、ASPD、FairAvg、Score、相对 IID root 的 ΔACC/ΔFairAvg。共 ${counts.serverDetail} 行明细，底部附相关性/差值摘要。
- \`03_Synthetic_FULL\`：完整 10% clean 与 synthetic generation 表，包含 1%-10% clean ratio 的 dataset 汇总、distribution/attack slice 明细，以及所有 generator 的汇总。三块分别有 ${counts.cleanDataset}、${counts.cleanSlice}、${counts.generation} 行。
- \`04_FedSA_FULL\`：完整 FedSA 新性能攻击表，包含 paper-style baseline 大表、所有方法汇总、AD2+ 参数候选 ranking。三块分别有 ${counts.fedsaPaper}、${counts.fedsaMethods}、${counts.fedsaCandidates} 行。

## 看表方法

- ACC 越高越好。
- AEOD/ASPD 越低越好。
- FairAvg 不是算法，是辅助指标，等于 \`(AEOD + ASPD) / 2\`。
- 如果某个方法 ACC 已经明显塌陷，那么它的低 AEOD/ASPD 不能作为有效公平性胜出。
- 这版表保留完整维度；论文正文可从这些大表中再抽主表或 appendix 表。

## 四组实验的预期结论

1. 消融：去掉性能项后，性能攻击下 ACC 应下降；去掉公平项后，公平攻击或双重攻击下 AEOD/ASPD/FairAvg 应上升。
2. Server/root 分布：server/root 越接近 IID，root reference 越可靠；偏移越大时，Adult 上 ACC 下降、公平风险上升更明显。COMPAS 保留为完整结果，但不建议强行声称严格单调。
   - Client IID = Dirichlet alpha=5000；Client non-IID = Dirichlet alpha=5。
   - IID server/root = 10% clean stratified root, controlled skew=0.00。
   - Mild non-IID server/root = controlled skew=0.05。
   - Moderate non-IID server/root = controlled skew=0.10。
3. 10% clean 与 synthetic：10% real clean 稳定、透明，适合作为主实验设定；少量 real + synthetic 在部分 generator 下能降低公平风险，但 generator-dependent。
4. FedSA 新性能攻击：AD2+ 在 joint score 上表现最强，重点是兼顾 ACC、AEOD、ASPD，而不是只追求某一个单指标第一。
`;
}

async function build() {
  await fs.mkdir(OUT, { recursive: true });

  const ablationRaw = [
    ...normalizeAblation(await csv(SOURCES.ablationBenign)),
    ...normalizeAblation(await csv(SOURCES.ablationStress)),
  ];
  const ablationAggMain = aggregate(ablationRaw, ["dataset", "attack", "profile"]);
  const ablationAggDetail = aggregate(ablationRaw, ["dataset", "distribution", "attack", "profile"]);
  const ablationEffect = makeAblationEffectRows(ablationAggMain);
  const ablationDetail = makeAblationRows(ablationAggDetail);

  const serverDetail = makeServerRows(await csv(SOURCES.serverByLevel));
  const serverCorr = makeServerCorrRows(await csv(SOURCES.serverCorr));

  const cleanDataset = cleanDatasetRows(await csv(SOURCES.cleanRatioDataset));
  const cleanSlice = cleanSliceRows(await csv(SOURCES.cleanRatioSlice));
  const generation = generationRows(await csv(SOURCES.generationSummary));

  const fedsaPaper = fedsaPaperRows(await csv(SOURCES.fedsaPaperStyle));
  const fedsaMethods = fedsaAllMethodRows(await csv(SOURCES.fedsaAllMethods));
  const fedsaCandidates = fedsaCandidateRows(await csv(SOURCES.fedsaCandidates));

  const wb = Workbook.create();

  const readme = wb.worksheets.add("README");
  styleTitle(readme, "GuardFed-AD2+ Four Detailed Experiment Tables", "这版保留完整大表维度；不是摘要表。", 5);
  writeRows(readme, [
    { Sheet: "01_Ablation_FULL", Content: "完整消融明细 + 效应摘要", Rows: ablationDetail.length, Notes: "按 dataset/distribution/attack/profile 展开。" },
    { Sheet: "02_ServerRoot_FULL", Content: "完整 server/root 分布明细 + trend/correlation 摘要", Rows: serverDetail.length, Notes: "Adult 和 COMPAS 都保留。" },
    { Sheet: "03_Synthetic_FULL", Content: "1%-10% clean ratio、slice 明细、synthetic generator 明细", Rows: cleanDataset.length + cleanSlice.length + generation.length, Notes: "三块表在同一个 sheet。" },
    { Sheet: "04_FedSA_FULL", Content: "完整 FedSA paper-style baseline、大方法汇总、AD2+候选参数", Rows: fedsaPaper.length + fedsaMethods.length + fedsaCandidates.length, Notes: "保留所有 recent baselines 和 AD2+候选。" },
  ], ["Sheet", "Content", { key: "Rows", label: "Detail rows", format: "0" }, "Notes"], 3, 0, "ReadmeSummary");
  setWidths(readme, [26, 58, 12, 80]);
  readme.getRange("B:D").format.wrapText = true;

  const s1 = wb.worksheets.add("01_Ablation_FULL");
  styleTitle(s1, "01 消融实验完整大表", "顶部是 8 行关键效应摘要；下面保留所有 dataset/distribution/attack/profile 的完整消融明细。", 18);
  writeRows(s1, ablationEffect, [
    "Dataset", "Test", "Attack", "Metric",
    { key: "Full", label: "Full", format: "0.0000" },
    { key: "Ablated", label: "Ablated", format: "0.0000" },
    { key: "Delta", label: "Delta", format: "0.0000" },
    "Expected",
  ], 3, 0, "AblationEffect");
  const startAblation = 15;
  writeRows(s1, ablationDetail, [
    "Dataset", "Distribution", "Attack", "Profile", "Ablation meaning",
    { key: "n", label: "n", format: "0" },
    { key: "Rounds mean", label: "Rounds", format: "0" },
    { key: "Utility weight", label: "wU", format: "0.0" },
    { key: "Fair risk", label: "lambda risk", format: "0.0" },
    { key: "Fair violation", label: "lambda viol", format: "0.0" },
    { key: "Centrality", label: "wC", format: "0.0" },
    { key: "Alignment", label: "wA", format: "0.0" },
    "Calibration",
    { key: "ACC (%)", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Score", format: "0.0000" },
  ], startAblation, 0, "AblationFullDetail");
  setWidths(s1, [12, 12, 12, 24, 28, 7, 9, 8, 12, 12, 8, 8, 11, 11, 10, 10, 10, 10]);
  s1.getRange("E:E").format.wrapText = true;
  s1.freezePanes.freezeRows(startAblation + 1);

  const s2 = wb.worksheets.add("02_ServerRoot_FULL");
  styleTitle(s2, "02 Server/Root 分布完整大表", "完整保留 Adult/COMPAS、client IID/non-IID 的 alpha 数值、server/root skew 数值、TVD、ACC/AEOD/ASPD/FairAvg/Score，以及相对 IID root 的差值。", 19);
  writeRows(s2, serverDetail, [
    "Dataset", "Client distribution",
    { key: "Client Dirichlet alpha", label: "Client alpha", format: "0" },
    "Attack", "Server/root level", "Root skew setting",
    { key: "Level index", label: "Level", format: "0" },
    { key: "n", label: "n", format: "0" },
    { key: "Group TVD", label: "Group TVD", format: "0.0000" },
    { key: "Sensitive TVD", label: "Sensitive TVD", format: "0.0000" },
    { key: "Label TVD", label: "Label TVD", format: "0.0000" },
    { key: "ACC (%)", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Score", format: "0.0000" },
    { key: "ΔACC vs IID root", label: "ΔACC vs IID root", format: "0.00" },
    { key: "ΔFairAvg vs IID root", label: "ΔFairAvg vs IID root", format: "0.0000" },
    "Numeric reading",
  ], 3, 0, "ServerRootFullDetail");
  const corrStart = serverDetail.length + 7;
  writeRows(s2, serverCorr, [
    "Dataset", "Distribution", "Attack",
    { key: "n levels", label: "n levels", format: "0" },
    { key: "corr(level, ACC)", label: "corr ACC", format: "0.000" },
    { key: "corr(level, AEOD)", label: "corr AEOD", format: "0.000" },
    { key: "corr(level, ASPD)", label: "corr ASPD", format: "0.000" },
    { key: "corr(level, FairAvg)", label: "corr FairAvg", format: "0.000" },
    { key: "Moderate-IID ACC delta", label: "Moderate-IID ACC delta", format: "0.00" },
    { key: "Moderate-IID FairAvg delta", label: "Moderate-IID FairAvg delta", format: "0.0000" },
  ], corrStart, 0, "ServerRootTrendFull");
  setWidths(s2, [12, 24, 12, 12, 28, 44, 8, 7, 12, 14, 12, 10, 10, 10, 10, 10, 14, 18, 90]);
  s2.getRange("F:F").format.wrapText = true;
  s2.getRange("S:S").format.wrapText = true;
  s2.freezePanes.freezeRows(4);

  const s3 = wb.worksheets.add("03_Synthetic_FULL");
  styleTitle(s3, "03 10% Clean 与 Synthetic Generation 完整大表", "依次保留 clean ratio dataset summary、distribution/attack slice detail、synthetic generator summary。", 21);
  writeRows(s3, cleanDataset, [
    "Dataset", "Setting", { key: "Server ratio", label: "Server ratio", format: "0%" }, "Server sampling",
    { key: "n", label: "n", format: "0" },
    { key: "ACC mean", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC std", label: "ACC std", format: "0.00" },
    { key: "AEOD mean", label: "AEOD mean", format: "0.0000" },
    { key: "AEOD std", label: "AEOD std", format: "0.0000" },
    { key: "ASPD mean", label: "ASPD mean", format: "0.0000" },
    { key: "ASPD std", label: "ASPD std", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score mean", label: "Score mean", format: "0.0000" },
    { key: "Score std", label: "Score std", format: "0.0000" },
    { key: "Round mean", label: "Round", format: "0.0" },
    { key: "Score rank", label: "Score rank", format: "0" },
    { key: "ACC rank", label: "ACC rank", format: "0" },
    { key: "AEOD rank", label: "AEOD rank", format: "0" },
    { key: "ASPD rank", label: "ASPD rank", format: "0" },
  ], 3, 0, "CleanRatioDatasetFull");
  const sliceStart = cleanDataset.length + 7;
  writeRows(s3, cleanSlice, [
    "Dataset", "Distribution", "Attack", "Setting", { key: "Server ratio", label: "Server ratio", format: "0%" }, "Server sampling",
    { key: "n", label: "n", format: "0" },
    { key: "ACC mean", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC std", label: "ACC std", format: "0.00" },
    { key: "AEOD mean", label: "AEOD mean", format: "0.0000" },
    { key: "AEOD std", label: "AEOD std", format: "0.0000" },
    { key: "ASPD mean", label: "ASPD mean", format: "0.0000" },
    { key: "ASPD std", label: "ASPD std", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score mean", label: "Score mean", format: "0.0000" },
    { key: "Score std", label: "Score std", format: "0.0000" },
    { key: "Round mean", label: "Round", format: "0.0" },
    { key: "Score rank", label: "Score rank", format: "0" },
    { key: "ACC rank", label: "ACC rank", format: "0" },
    { key: "AEOD rank", label: "AEOD rank", format: "0" },
    { key: "ASPD rank", label: "ASPD rank", format: "0" },
  ], sliceStart, 0, "CleanRatioSliceFull");
  const genStart = sliceStart + cleanSlice.length + 4;
  writeRows(s3, generation, [
    "Dataset", "Tag", "Generator", { key: "Synthetic ratio", label: "Synthetic ratio", format: "0%" },
    { key: "n", label: "n", format: "0" },
    { key: "ACC mean (%)", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC min (%)", label: "ACC min (%)", format: "0.00" },
    { key: "ACC max (%)", label: "ACC max (%)", format: "0.00" },
    { key: "AEOD mean", label: "AEOD mean", format: "0.0000" },
    { key: "AEOD min", label: "AEOD min", format: "0.0000" },
    { key: "AEOD max", label: "AEOD max", format: "0.0000" },
    { key: "ASPD mean", label: "ASPD mean", format: "0.0000" },
    { key: "ASPD min", label: "ASPD min", format: "0.0000" },
    { key: "ASPD max", label: "ASPD max", format: "0.0000" },
    { key: "FairAvg mean", label: "FairAvg mean", format: "0.0000" },
    { key: "FairAvg min", label: "FairAvg min", format: "0.0000" },
    { key: "FairAvg max", label: "FairAvg max", format: "0.0000" },
    { key: "Score mean", label: "Score mean", format: "0.0000" },
    { key: "Score min", label: "Score min", format: "0.0000" },
    { key: "Score max", label: "Score max", format: "0.0000" },
  ], genStart, 0, "GenerationSummaryFull");
  setWidths(s3, [12, 18, 12, 18, 10, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10]);
  s3.freezePanes.freezeRows(4);

  const s4 = wb.worksheets.add("04_FedSA_FULL");
  styleTitle(s4, "04 FedSA 新性能攻击完整大表", "第一块是论文样式完整 baseline 表；第二块是所有方法汇总；第三块是 AD2+ 参数候选 ranking。", 30);
  writeRows(s4, fedsaPaper, [
    "Method", "Citation", "Metric",
    { key: "ADULT IID", label: "ADULT IID", format: "0.0000" },
    { key: "ADULT non-IID", label: "ADULT non-IID", format: "0.0000" },
    { key: "COMPAS IID", label: "COMPAS IID", format: "0.0000" },
    { key: "COMPAS non-IID", label: "COMPAS non-IID", format: "0.0000" },
  ], 3, 0, "FedSAPaperStyleFull");
  const methodStart = fedsaPaper.length + 7;
  writeRows(s4, fedsaMethods, [
    "Dataset", "Distribution", "Attack", "Method",
    { key: "n", label: "n", format: "0" },
    { key: "ACC mean (%)", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC std (%)", label: "ACC std (%)", format: "0.00" },
    { key: "AEOD mean", label: "AEOD mean", format: "0.0000" },
    { key: "AEOD std", label: "AEOD std", format: "0.0000" },
    { key: "ASPD mean", label: "ASPD mean", format: "0.0000" },
    { key: "ASPD std", label: "ASPD std", format: "0.0000" },
    { key: "FairAvg mean", label: "FairAvg mean", format: "0.0000" },
    { key: "FairAvg std", label: "FairAvg std", format: "0.0000" },
    { key: "Score mean", label: "Score mean", format: "0.0000" },
    { key: "Score std", label: "Score std", format: "0.0000" },
    { key: "ACC rank", label: "ACC rank", format: "0" },
    { key: "Fair rank", label: "Fair rank", format: "0" },
    { key: "Score rank", label: "Score rank", format: "0" },
  ], methodStart, 0, "FedSAAllMethodsFull");
  const candidateStart = methodStart + fedsaMethods.length + 4;
  writeRows(s4, fedsaCandidates, [
    "Dataset", "Distribution", "Attack", "Method", "Suite", "Tag", "Base method",
    { key: "Server ratio", label: "Server ratio", format: "0%" },
    "Server sampling", "Mode",
    { key: "Budget", label: "Budget", format: "0.000" },
    { key: "Max acc drop", label: "Max acc drop", format: "0.000" },
    { key: "Norm clip", label: "Norm clip", format: "0.0" },
    "Norm mode", "Objective",
    { key: "n", label: "n", format: "0" },
    { key: "ACC mean (%)", label: "ACC mean (%)", format: "0.00" },
    { key: "ACC std (%)", label: "ACC std (%)", format: "0.00" },
    { key: "AEOD mean", label: "AEOD mean", format: "0.0000" },
    { key: "AEOD std", label: "AEOD std", format: "0.0000" },
    { key: "ASPD mean", label: "ASPD mean", format: "0.0000" },
    { key: "ASPD std", label: "ASPD std", format: "0.0000" },
    { key: "Score mean", label: "Score mean", format: "0.0000" },
    { key: "Score std", label: "Score std", format: "0.0000" },
    { key: "Round mean", label: "Round", format: "0.0" },
    { key: "Score rank", label: "Score rank", format: "0" },
    { key: "ACC rank", label: "ACC rank", format: "0" },
    { key: "AEOD rank", label: "AEOD rank", format: "0" },
    { key: "ASPD rank", label: "ASPD rank", format: "0" },
  ], candidateStart, 0, "FedSACandidatesFull");
  setWidths(s4, [16, 24, 10, 28, 22, 20, 16, 12, 20, 12, 10, 12, 10, 12, 14, 8, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 10]);
  s4.getRange("A:AD").format.wrapText = false;
  s4.freezePanes.freezeRows(4);

  const sources = wb.worksheets.add("Sources");
  styleTitle(sources, "Source Files", "Detailed workbook 的输入文件；表内只重排和格式化，不手工改实验结果。", 3);
  writeRows(sources, Object.entries(SOURCES).map(([Name, Path]) => ({ Name, Path })), ["Name", "Path"], 3, 0, "DetailedSources");
  setWidths(sources, [32, 160]);

  const xlsxPath = path.join(OUT, "GuardFed_AD2plus_Four_Experiments_Detailed_Tables.xlsx");
  const readmePath = path.join(OUT, "README_detailed_four_tables.md");
  await fs.writeFile(readmePath, markdownReadme({
    ablationDetail: ablationDetail.length,
    serverDetail: serverDetail.length,
    cleanDataset: cleanDataset.length,
    cleanSlice: cleanSlice.length,
    generation: generation.length,
    fedsaPaper: fedsaPaper.length,
    fedsaMethods: fedsaMethods.length,
    fedsaCandidates: fedsaCandidates.length,
  }), "utf8");

  for (const sheetName of ["01_Ablation_FULL", "02_ServerRoot_FULL", "03_Synthetic_FULL", "04_FedSA_FULL"]) {
    const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, `preview_${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const output = await SpreadsheetFile.exportXlsx(wb);
  await output.save(xlsxPath);

  console.log(JSON.stringify({
    xlsxPath,
    readmePath,
    counts: {
      ablationDetail: ablationDetail.length,
      serverDetail: serverDetail.length,
      cleanDataset: cleanDataset.length,
      cleanSlice: cleanSlice.length,
      generation: generation.length,
      fedsaPaper: fedsaPaper.length,
      fedsaMethods: fedsaMethods.length,
      fedsaCandidates: fedsaCandidates.length,
    },
  }, null, 2));
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
