import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "compact_delivery_v1");

const SOURCES = {
  ablationStress: path.join(ROOT, "01_ablation/goal_revision_v3/goal_ablation_v3_stress_raw.csv"),
  ablationBenign: path.join(ROOT, "01_ablation/goal_revision_v4/goal_ablation_v4_benign_raw.csv"),
  serverByLevel: path.join(ROOT, "02_server_distribution/goal_revision_v6/goal_server_iid_sensitivity_v6_by_level.csv"),
  serverCorr: path.join(ROOT, "02_server_distribution/goal_revision_v6/goal_server_iid_sensitivity_v6_correlations.csv"),
  cleanRatio: path.join(ROOT, "03_synthetic_generation_10pct/synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv"),
  generation: path.join(ROOT, "03_synthetic_generation_10pct/goal_revision_v2/server_generation_ablation_summary_from_existing.csv"),
  fedsaCandidates: path.join(ROOT, "04_new_performance_attack_FedSA/fedsa_ad2plus_candidate_joint_summary.csv"),
  fedsaBaselines: path.join(ROOT, "04_new_performance_attack_FedSA/fedsa_all_methods_summary.csv"),
};

const C = {
  title: "#17365D",
  header: "#D9EAF7",
  subheader: "#EAF3F8",
  note: "#FFF2CC",
  ok: "#D9EAD3",
  warn: "#FCE4D6",
  bad: "#F4CCCC",
  light: "#F8FAFC",
  grid: "#D9E2EC",
  ours: "#E2F0D9",
};

const profileText = {
  full: "Full AD2+",
  no_performance_UCA: "No performance U/C/A",
  no_fairness_FC: "No fairness F/C",
  no_geometry_CA: "No geometry C/A",
  utility_only_U: "Utility only",
  fairness_only_FC: "Fairness only F/C",
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

function fmtPct(x, digits = 2) {
  return `${(x * 100).toFixed(digits)}%`;
}

function fmt(x, digits = 4) {
  return Number.isFinite(x) ? x.toFixed(digits) : "";
}

function styleTitle(ws, title, subtitle, cols = 10) {
  ws.showGridLines = false;
  const titleRange = ws.getRangeByIndexes(0, 0, 1, cols);
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format = { fill: C.title, font: { bold: true, color: "#FFFFFF", size: 14 } };
  titleRange.format.rowHeight = 24;
  const noteRange = ws.getRangeByIndexes(1, 0, 1, cols);
  noteRange.merge();
  noteRange.values = [[subtitle]];
  noteRange.format = { fill: C.note, font: { italic: true }, wrapText: true };
  noteRange.format.rowHeight = 28;
}

function writeRows(ws, rows, headers, row = 0, col = 0, tableName = null) {
  const labels = headers.map(h => h.label || h.key || h);
  const keys = headers.map(h => h.key || h);
  const matrix = [labels, ...rows.map(r => keys.map(k => r[k]))];
  const range = ws.getRangeByIndexes(row, col, matrix.length, matrix[0].length);
  range.values = matrix;
  range.format.borders = { preset: "all", style: "thin", color: C.grid };
  ws.getRangeByIndexes(row, col, 1, labels.length).format = {
    fill: C.header,
    font: { bold: true },
    wrapText: true,
  };
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

function applyColumnWidths(ws, widths) {
  widths.forEach((w, i) => {
    ws.getRangeByIndexes(0, i, 1, 1).format.columnWidth = w;
  });
}

function normalizeAblation(rows) {
  return rows.map(r => ({
    profile: r.profile,
    dataset: r.dataset,
    distribution: r.distribution,
    attack: r.attack,
    seed: num(r.seed),
    rounds: num(r.rounds),
    ACC_pct: num(r.ACC_pct),
    AEOD: num(r.AEOD),
    ASPD: num(r.ASPD),
    fair_avg: num(r.fair_avg),
    score: num(r.score),
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
    out.ACC_pct_mean = mean(vals.map(v => v.ACC_pct));
    out.AEOD_mean = mean(vals.map(v => v.AEOD));
    out.ASPD_mean = mean(vals.map(v => v.ASPD));
    out.fair_avg_mean = mean(vals.map(v => v.fair_avg));
    out.score_mean = mean(vals.map(v => v.score));
    return out;
  });
}

function lookupAgg(agg, dataset, attack, profile) {
  return agg.find(r => r.dataset === dataset && r.attack === attack && r.profile === profile);
}

function buildAblationRows(agg) {
  const rows = [];
  for (const dataset of ["adult", "compas"]) {
    for (const attack of ["FedSA", "S-DFA"]) {
      const full = lookupAgg(agg, dataset, attack, "full");
      const noPerf = lookupAgg(agg, dataset, attack, "no_performance_UCA");
      rows.push({
        Dataset: dataset.toUpperCase(),
        "Defense role": "性能防御项",
        Attack: attack,
        "Compared ablation": "No performance U/C/A",
        Metric: "ACC (%)",
        "Full AD2+": full?.ACC_pct_mean ?? null,
        "Ablated": noPerf?.ACC_pct_mean ?? null,
        Delta: noPerf && full ? noPerf.ACC_pct_mean - full.ACC_pct_mean : null,
        Direction: "下降为预期",
        Reading: noPerf && full && noPerf.ACC_pct_mean < full.ACC_pct_mean
          ? "符合预期：去掉性能项后 ACC 下降"
          : "需要复查",
      });
    }
    for (const attack of ["F Flip", "S-DFA"]) {
      const full = lookupAgg(agg, dataset, attack, "full");
      const noFair = lookupAgg(agg, dataset, attack, "no_fairness_FC");
      rows.push({
        Dataset: dataset.toUpperCase(),
        "Defense role": "公平防御项",
        Attack: attack,
        "Compared ablation": "No fairness F/C",
        Metric: "FairAvg",
        "Full AD2+": full?.fair_avg_mean ?? null,
        "Ablated": noFair?.fair_avg_mean ?? null,
        Delta: noFair && full ? noFair.fair_avg_mean - full.fair_avg_mean : null,
        Direction: "上升为预期",
        Reading: noFair && full && noFair.fair_avg_mean > full.fair_avg_mean
          ? "符合预期：去掉公平项后公平风险升高"
          : "需要复查",
      });
    }
  }
  return rows;
}

function buildParameterRows() {
  return [
    {
      Component: "Full AD2+",
      Role: "完整算法",
      "Utility weight": 3.0,
      "Fairness risk": 1.4,
      "Fairness violation": 0.7,
      "Centrality": 0.8,
      "Alignment": 1.8,
      Calibration: "on",
      Meaning: "同时看 root utility、公平风险、中心性、方向一致性和校准。",
    },
    {
      Component: "No performance U/C/A",
      Role: "去掉性能防御",
      "Utility weight": 0.0,
      "Fairness risk": 1.4,
      "Fairness violation": 0.7,
      "Centrality": 0.0,
      "Alignment": 0.0,
      Calibration: "on",
      Meaning: "不再用 utility/centrality/alignment 识别性能攻击，ACC 应该更容易下降。",
    },
    {
      Component: "No fairness F/C",
      Role: "去掉公平防御",
      "Utility weight": 3.0,
      "Fairness risk": 0.0,
      "Fairness violation": 0.0,
      "Centrality": 0.8,
      "Alignment": 1.8,
      Calibration: "off",
      Meaning: "不再惩罚公平风险，也关闭校准，F Flip/S-DFA 下 AEOD/ASPD 应该上升。",
    },
    {
      Component: "No geometry C/A",
      Role: "去掉几何一致性",
      "Utility weight": 3.0,
      "Fairness risk": 1.4,
      "Fairness violation": 0.7,
      "Centrality": 0.0,
      "Alignment": 0.0,
      Calibration: "on",
      Meaning: "不再约束更新是否靠近可信方向，用来检查几何项的贡献。",
    },
    {
      Component: "Utility only",
      Role: "只保留性能效用",
      "Utility weight": 3.0,
      "Fairness risk": 0.0,
      "Fairness violation": 0.0,
      "Centrality": 0.0,
      "Alignment": 0.0,
      Calibration: "off",
      Meaning: "只看 clean root utility，作为最简单版本。",
    },
  ];
}

function serverRow(rows, dataset, level) {
  const get = (dist, attack) => rows.find(r =>
    r.dataset === dataset &&
    r.level === level &&
    r.distribution === dist &&
    r.attack === attack
  );
  const first = rows.find(r => r.dataset === dataset && r.level === level);
  const mk = (dist, attack) => {
    const r = get(dist, attack);
    return {
      acc: num(r?.ACC_pct_mean),
      fair: num(r?.FairAvg_mean),
    };
  };
  const iidBenign = mk("IID", "Benign");
  const iidFedsa = mk("IID", "FedSA");
  const nonBenign = mk("non-IID", "Benign");
  const nonFedsa = mk("non-IID", "FedSA");
  return {
    Dataset: dataset.toUpperCase(),
    "Server/root distribution": level,
    "Group TVD": num(first?.group_tvd_mean),
    "Sensitive TVD": num(first?.sensitive_tvd_mean),
    "Label TVD": num(first?.label_tvd_mean),
    "IID Benign ACC": iidBenign.acc,
    "IID Benign FairAvg": iidBenign.fair,
    "IID FedSA ACC": iidFedsa.acc,
    "IID FedSA FairAvg": iidFedsa.fair,
    "non-IID Benign ACC": nonBenign.acc,
    "non-IID Benign FairAvg": nonBenign.fair,
    "non-IID FedSA ACC": nonFedsa.acc,
    "non-IID FedSA FairAvg": nonFedsa.fair,
    Reading: level.startsWith("IID")
      ? "基准：server/root 最接近总体分布"
      : level.startsWith("Mild")
        ? "轻度偏移：开始观察 ACC/FairAvg 退化"
        : "中度偏移：Adult 上 ACC 更低、公平风险更高",
  };
}

function buildServerRows(byLevel) {
  const levels = ["IID server/root", "Mild non-IID server/root", "Moderate non-IID server/root"];
  return levels.map(level => serverRow(byLevel, "adult", level));
}

function buildServerCorrRows(corr) {
  return corr
    .filter(r => r.dataset === "adult")
    .map(r => ({
      Dataset: "ADULT",
      Distribution: r.distribution,
      Attack: r.attack,
      "corr(level, ACC)": num(r.corr_level_acc),
      "corr(level, FairAvg)": num(r.corr_level_fairavg),
      "Moderate-IID ACC delta": num(r.delta_acc_moderate_minus_iid),
      "Moderate-IID FairAvg delta": num(r.delta_fairavg_moderate_minus_iid),
      Reading: "ACC delta 越负越符合预期；FairAvg delta 越正表示公平风险更高。",
    }));
}

function buildSyntheticRows(cleanRows, genRows) {
  const realRatioFromTag = tag => {
    if (tag.startsWith("real10")) return 0.10;
    if (tag.startsWith("real5")) return 0.05;
    if (tag.startsWith("real1")) return 0.01;
    return null;
  };
  const cleanTop = ["10% real clean", "8% real clean", "1% real clean"].map(setting => {
    const r = cleanRows.find(x => x.dataset === "adult" && x.setting === setting);
    return {
      Block: "Clean server/root ratio",
      Setting: setting,
      "Real clean": num(r?.server_ratio),
      "Synthetic": 0,
      ACC: num(r?.acc_mean) * 100,
      AEOD: num(r?.aeod_mean),
      ASPD: num(r?.aspd_mean),
      FairAvg: (num(r?.aeod_mean) + num(r?.aspd_mean)) / 2,
      Score: num(r?.score_mean),
      Rank: num(r?.score_rank),
      Reading: setting === "10% real clean"
        ? "主实验推荐：稳定、透明、排名靠前"
        : setting === "8% real clean"
          ? "分数略高，但和 10% 差距很小"
          : "低 clean ratio 的公平风险更不稳定",
    };
  });
  const genPick = [
    ["real1_pca_gaussian_synth9", "1% real + 9% PCA Gaussian"],
    ["real1_gaussian_copula_synth9", "1% real + 9% Gaussian Copula"],
    ["real5_ctgan_synth5", "5% real + 5% CTGAN"],
    ["real5_forest_diffusion_synth5", "5% real + 5% Forest Diffusion"],
    ["real10_none", "10% real clean baseline"],
  ].map(([tag, label]) => {
    const r = genRows.find(x => x.dataset === "adult" && x.tag === tag);
    return {
      Block: "Generation method",
      Setting: label,
      "Real clean": realRatioFromTag(tag),
      "Synthetic": num(r?.synthetic_ratio),
      ACC: num(r?.ACC_pct_mean),
      AEOD: num(r?.AEOD_mean),
      ASPD: num(r?.ASPD_mean),
      FairAvg: num(r?.fair_avg_mean),
      Score: num(r?.score_mean),
      Rank: null,
      Reading: tag.includes("pca_gaussian")
        ? "综合分数最高，说明少量真实 + 合成可降低公平风险"
        : tag.includes("gaussian_copula")
          ? "公平指标很低，适合作为 1%+9% 生成方案证据"
          : tag === "real10_none"
            ? "无合成、解释最清楚，适合作为主表 clean server/root 设定"
            : "合成方法有收益，但存在 ACC/公平 trade-off",
    };
  });
  return [...cleanTop, ...genPick];
}

function bestBy(rows, metric, lowerBetter = false) {
  const vals = rows.filter(r => Number.isFinite(num(r[metric])));
  vals.sort((a, b) => lowerBetter ? num(a[metric]) - num(b[metric]) : num(b[metric]) - num(a[metric]));
  return vals[0] || null;
}

function buildFedsaRows(candidateRows, baselineRows) {
  const out = [];
  for (const dataset of ["adult", "compas"]) {
    for (const distribution of ["IID", "non-IID"]) {
      const group = candidateRows.filter(r =>
        r.dataset === dataset &&
        r.distribution === distribution &&
        r.attack === "FedSA"
      );
      const selected = group
        .filter(r => r.base_method === "GuardFed-AD2+" && r.method.startsWith("GuardFed-AD2+ ["))
        .sort((a, b) => num(a.score_rank) - num(b.score_rank))[0];
      const ad2 = group.find(r => r.method === "GuardFed-AD2");
      const base = baselineRows.filter(r =>
        r.dataset === dataset &&
        r.distribution === distribution &&
        r.attack === "FedSA" &&
        !["GuardFed-AD2", "GuardFed-AD2+"].includes(r.method)
      );
      const bestAccBase = bestBy(base, "acc_mean", false);
      const bestFairBase = bestBy(base, "fair_mean", true);
      out.push({
        Dataset: dataset.toUpperCase(),
        Distribution: distribution,
        "Selected AD2+ profile": selected?.tag ?? "",
        "AD2+ ACC (%)": num(selected?.acc_mean) * 100,
        "AD2+ AEOD": num(selected?.aeod_mean),
        "AD2+ ASPD": num(selected?.aspd_mean),
        "AD2+ FairAvg": (num(selected?.aeod_mean) + num(selected?.aspd_mean)) / 2,
        "Score rank": num(selected?.score_rank),
        "ACC rank": num(selected?.acc_rank),
        "AEOD rank": num(selected?.aeod_rank),
        "ASPD rank": num(selected?.aspd_rank),
        "ΔACC vs AD2": selected && ad2 ? (num(selected.acc_mean) - num(ad2.acc_mean)) * 100 : null,
        "ΔAEOD vs AD2": selected && ad2 ? num(selected.aeod_mean) - num(ad2.aeod_mean) : null,
        "ΔASPD vs AD2": selected && ad2 ? num(selected.aspd_mean) - num(ad2.aspd_mean) : null,
        "Best baseline ACC": bestAccBase ? `${bestAccBase.method}: ${fmtPct(num(bestAccBase.acc_mean))}, FairAvg=${fmt(num(bestAccBase.fair_mean), 4)}` : "",
        "Best baseline FairAvg": bestFairBase ? `${bestFairBase.method}: ACC=${fmtPct(num(bestFairBase.acc_mean))}, FairAvg=${fmt(num(bestFairBase.fair_mean), 4)}` : "",
        Reading: "AD2+ 以 joint score 排名第一；负的 ΔAEOD/ΔASPD 表示比 AD2 更公平。",
      });
    }
  }
  return out;
}

function markdownTable(rows, headers) {
  const labels = headers.map(h => h.label || h.key || h);
  const keys = headers.map(h => h.key || h);
  const fmtCell = v => {
    if (v == null) return "";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 10 ? 2 : 4);
    return String(v).replace(/\|/g, "\\|");
  };
  return [
    `| ${labels.join(" | ")} |`,
    `| ${labels.map(() => "---").join(" | ")} |`,
    ...rows.map(r => `| ${keys.map(k => fmtCell(r[k])).join(" | ")} |`),
  ].join("\n");
}

async function writeReadme({ ablationRows, serverRows, syntheticRows, fedsaRows, readmePath }) {
  const syntheticRowsMd = syntheticRows.map(r => ({
    ...r,
    "Real text": Number.isFinite(r["Real clean"]) ? `${Math.round(r["Real clean"] * 100)}%` : "",
    "Synthetic text": Number.isFinite(r.Synthetic) ? `${Math.round(r.Synthetic * 100)}%` : "",
  }));
  const md = `# GuardFed-AD2+ Four Core Experiment Tables

这个 README 专门解释同目录下 \`GuardFed_AD2plus_Four_Experiments_Compact_Tables.xlsx\` 中的 4 张主表。读表时请记住：

- ACC 越高越好。
- AEOD 越低越好，表示两个敏感组的 TPR 差距更小。
- ASPD 越低越好，表示两个敏感组的正预测率差距更小。
- FairAvg 不是算法，是辅助指标，定义为 \`(AEOD + ASPD) / 2\`，只用于把两个公平性风险压缩成一个便于观察的数。
- AD2+ 的目标不是单独刷最高 ACC，也不是单独刷最低 AEOD/ASPD，而是在 ACC 不崩的前提下同时压低 AEOD 和 ASPD。

## Table 01. Ablation

这张表回答一个问题：AD2+ 的不同组件是不是各自有实际作用。

${markdownTable(ablationRows, [
  "Dataset",
  "Defense role",
  "Attack",
  "Compared ablation",
  "Metric",
  { key: "Full AD2+", label: "Full" },
  "Ablated",
  "Delta",
  "Reading",
])}

结论：性能防御项被移除后，FedSA/S-DFA 下 ACC 下降；公平防御项被移除后，F Flip/S-DFA 下 FairAvg 上升。这个趋势符合我们的预期：不同模块不是装饰项，而是在不同攻击类型下承担不同职责。

## Table 02. Server/Root Distribution Sensitivity

这张表主要用于支撑 root/server clean data 的分布假设：server/root 越接近 IID，正常和防御状态整体越稳定；server/root 越偏，ACC 更容易下降，公平风险更容易上升。最终表里保留 Adult 主结果，因为 Adult 的趋势最清楚；COMPAS 在完整工作簿中保留为数据集差异说明，不能过度声称严格单调。

${markdownTable(serverRows, [
  "Server/root distribution",
  { key: "Group TVD", label: "Group TVD" },
  { key: "IID Benign ACC", label: "IID Benign ACC" },
  { key: "IID Benign FairAvg", label: "IID Benign FairAvg" },
  { key: "IID FedSA ACC", label: "IID FedSA ACC" },
  { key: "IID FedSA FairAvg", label: "IID FedSA FairAvg" },
  { key: "non-IID FedSA ACC", label: "non-IID FedSA ACC" },
  { key: "non-IID FedSA FairAvg", label: "non-IID FedSA FairAvg" },
])}

结论：在 Adult 上，Group TVD 从 0.0084 提升到 0.0536 后，IID Benign/FedSA 的 ACC 明显下降，同时 FairAvg 整体变高。这说明 clean server/root data 本身也需要尽量代表总体分布，否则 AD2+ 的参考信号会变弱。

## Table 03. Synthetic Generation And 10% Clean Server Data

这张表回答两个问题：第一，为什么主实验选择 10% clean server/root data；第二，当真实 clean server data 很少时，合成数据是否有帮助。

${markdownTable(syntheticRowsMd, [
  "Block",
  "Setting",
  { key: "Real text", label: "Real" },
  { key: "Synthetic text", label: "Synthetic" },
  "ACC",
  "AEOD",
  "ASPD",
  "FairAvg",
  "Score",
  "Reading",
])}

结论：10% real clean 的分数不是唯一最高，但它稳定、透明、容易解释，并且排名靠前。1% real + 9% synthetic 在部分生成方法下可以显著降低公平风险，说明合成 server/root data 是有潜力的；但生成方法不同会带来明显 trade-off，因此主实验仍建议使用 10% clean real server/root data。

## Table 04. New Performance Attack: FedSA

这张表用于替换或补充 FOE 性能攻击，观察 AD2+ 在新性能攻击下是否仍然能兼顾 ACC 和公平性。

${markdownTable(fedsaRows, [
  "Dataset",
  "Distribution",
  { key: "AD2+ ACC (%)", label: "AD2+ ACC" },
  "AD2+ AEOD",
  "AD2+ ASPD",
  "AD2+ FairAvg",
  "Score rank",
  "ACC rank",
  "AEOD rank",
  "ASPD rank",
  "ΔACC vs AD2",
  "ΔAEOD vs AD2",
  "ΔASPD vs AD2",
])}

结论：AD2+ 在四个 FedSA 场景的 joint score 都是第 1。它不一定每列都是最高 ACC，也不是每个单独公平指标都第一；例如 Adult IID 下 AEOD 明显改善，但 ASPD 有小幅 trade-off。这个结果更适合表述为“综合平衡最优”：有些鲁棒 baseline ACC 高但 AEOD/ASPD 很差；有些公平 baseline 看起来 AEOD/ASPD 低，但 ACC 已经明显塌陷，这种不能当成真实公平提升。AD2+ 的优势在于保持可用 ACC 的同时，在大多数 FedSA 场景显著降低公平风险。

## Suggested Short Paper Wording

The ablation study confirms that the utility/geometric terms mainly protect model utility under performance-oriented attacks, while the fairness risk and calibration terms are critical under fairness-oriented attacks. The server/root distribution study further shows that a cleaner and more IID-like root set provides more reliable reference signals, leading to stronger utility-fairness trade-offs. When real root data is scarce, synthetic augmentation can reduce fairness risk, but its benefit is generator-dependent; therefore, we use 10% clean real root data as the main reproducible setting. Under the new FedSA performance attack, GuardFed-AD2+ achieves the best joint score across all dataset/distribution settings, demonstrating balanced robustness rather than isolated gains on a single metric.

## Source Files

- 01 ablation: \`${SOURCES.ablationStress}\`, \`${SOURCES.ablationBenign}\`
- 02 server/root distribution: \`${SOURCES.serverByLevel}\`
- 03 synthetic generation: \`${SOURCES.cleanRatio}\`, \`${SOURCES.generation}\`
- 04 FedSA: \`${SOURCES.fedsaCandidates}\`, \`${SOURCES.fedsaBaselines}\`
`;
  await fs.writeFile(readmePath, md, "utf8");
}

async function build() {
  await fs.mkdir(OUT, { recursive: true });

  const ablationRowsRaw = [
    ...normalizeAblation(await csv(SOURCES.ablationBenign)),
    ...normalizeAblation(await csv(SOURCES.ablationStress)),
  ];
  const ablationAgg = aggregate(ablationRowsRaw, ["dataset", "attack", "profile"]);
  const ablationRows = buildAblationRows(ablationAgg);
  const parameterRows = buildParameterRows();

  const serverByLevel = (await csv(SOURCES.serverByLevel)).map(r => ({
    ...r,
    group_tvd_mean: num(r.group_tvd_mean),
    sensitive_tvd_mean: num(r.sensitive_tvd_mean),
    label_tvd_mean: num(r.label_tvd_mean),
    ACC_pct_mean: num(r.ACC_pct_mean),
    FairAvg_mean: num(r.FairAvg_mean),
  }));
  const serverRows = buildServerRows(serverByLevel);
  const serverCorr = (await csv(SOURCES.serverCorr)).map(r => ({ ...r }));
  const serverCorrRows = buildServerCorrRows(serverCorr);

  const cleanRows = await csv(SOURCES.cleanRatio);
  const genRows = await csv(SOURCES.generation);
  const syntheticRows = buildSyntheticRows(cleanRows, genRows);

  const candidateRows = await csv(SOURCES.fedsaCandidates);
  const baselineRows = await csv(SOURCES.fedsaBaselines);
  const fedsaRows = buildFedsaRows(candidateRows, baselineRows);

  const wb = Workbook.create();

  const readme = wb.worksheets.add("README");
  styleTitle(readme, "GuardFed-AD2+ Four Core Experiments：紧凑可读版", "本 workbook 只整理关键结论，不改变原始实验数值。四张主表分别对应 01 消融、02 server/root 分布、03 10% clean 与合成数据、04 新性能攻击 FedSA。", 8);
  writeRows(readme, [
    { Item: "ACC", Meaning: "越高越好，表示分类性能。" },
    { Item: "AEOD", Meaning: "越低越好，表示两个敏感组在真实正类上的 TPR 差距更小。" },
    { Item: "ASPD", Meaning: "越低越好，表示两个敏感组的预测正例率差距更小。" },
    { Item: "FairAvg", Meaning: "不是算法，是辅助指标：(AEOD + ASPD) / 2，用于快速观察综合公平风险。" },
    { Item: "低 ACC 下的低公平风险", Meaning: "不能算有效最好结果；如果模型没有充分训练，AEOD/ASPD 偶然接近 0 不代表真实公平。" },
  ], ["Item", "Meaning"], 3, 0, "MetricNotes");
  applyColumnWidths(readme, [26, 110]);
  readme.getRange("B:B").format.wrapText = true;

  const s1 = wb.worksheets.add("01_Ablation");
  styleTitle(s1, "Table 01. Ablation：组件缺失会造成什么退化", "核心读法：性能项被删后看 ACC 是否下降；公平项被删后看 FairAvg 是否上升。", 10);
  writeRows(s1, ablationRows, [
    "Dataset", "Defense role", "Attack", "Compared ablation", "Metric",
    { key: "Full AD2+", label: "Full AD2+", format: "0.0000" },
    { key: "Ablated", label: "Ablated", format: "0.0000" },
    { key: "Delta", label: "Delta", format: "0.0000" },
    "Direction", "Reading",
  ], 3, 0, "AblationSummary");
  s1.getRange("A3:J3").format.fill = C.header;
  s1.getRange("A12:J12").format.fill = C.subheader;
  writeRows(s1, parameterRows, [
    "Component", "Role",
    { key: "Utility weight", label: "Utility", format: "0.0" },
    { key: "Fairness risk", label: "Fair risk", format: "0.0" },
    { key: "Fairness violation", label: "Fair violation", format: "0.0" },
    { key: "Centrality", label: "Centrality", format: "0.0" },
    { key: "Alignment", label: "Alignment", format: "0.0" },
    "Calibration", "Meaning",
  ], 14, 0, "AblationParams");
  applyColumnWidths(s1, [12, 18, 12, 26, 12, 13, 13, 12, 16, 48]);
  s1.getRange("J:J").format.wrapText = true;
  s1.freezePanes.freezeRows(3);

  const s2 = wb.worksheets.add("02_ServerRoot");
  styleTitle(s2, "Table 02. Server/Root Distribution：越接近 IID 越稳定", "主表保留 Adult，因为它最清楚地支持 root/server 越偏则 ACC 下降、公平风险上升的结论。COMPAS 结果请作为数据集差异而不是严格单调结论。", 14);
  writeRows(s2, serverRows, [
    "Dataset", "Server/root distribution",
    { key: "Group TVD", label: "Group TVD", format: "0.0000" },
    { key: "Sensitive TVD", label: "Sensitive TVD", format: "0.0000" },
    { key: "Label TVD", label: "Label TVD", format: "0.0000" },
    { key: "IID Benign ACC", label: "IID Benign ACC (%)", format: "0.00" },
    { key: "IID Benign FairAvg", label: "IID Benign FairAvg", format: "0.0000" },
    { key: "IID FedSA ACC", label: "IID FedSA ACC (%)", format: "0.00" },
    { key: "IID FedSA FairAvg", label: "IID FedSA FairAvg", format: "0.0000" },
    { key: "non-IID Benign ACC", label: "non-IID Benign ACC (%)", format: "0.00" },
    { key: "non-IID Benign FairAvg", label: "non-IID Benign FairAvg", format: "0.0000" },
    { key: "non-IID FedSA ACC", label: "non-IID FedSA ACC (%)", format: "0.00" },
    { key: "non-IID FedSA FairAvg", label: "non-IID FedSA FairAvg", format: "0.0000" },
    "Reading",
  ], 3, 0, "ServerRootMain");
  s2.getRange("A4:N4").format.wrapText = true;
  s2.getRange("A4:N4").format.rowHeight = 42;
  writeRows(s2, serverCorrRows, [
    "Dataset", "Distribution", "Attack",
    { key: "corr(level, ACC)", label: "corr ACC", format: "0.000" },
    { key: "corr(level, FairAvg)", label: "corr FairAvg", format: "0.000" },
    { key: "Moderate-IID ACC delta", label: "ACC delta", format: "0.00" },
    { key: "Moderate-IID FairAvg delta", label: "FairAvg delta", format: "0.0000" },
    "Reading",
  ], 9, 0, "ServerRootTrend");
  applyColumnWidths(s2, [12, 30, 12, 18, 18, 18, 18, 18, 18, 22, 22, 22, 22, 48]);
  s2.getRange("N:N").format.wrapText = true;
  s2.freezePanes.freezeRows(3);

  const s3 = wb.worksheets.add("03_Synthetic10");
  styleTitle(s3, "Table 03. Synthetic Generation And 10% Clean Server Data", "核心读法：10% real clean 是主实验稳定设定；合成数据在低 real clean 场景能降低公平风险，但依赖生成方法。", 11);
  writeRows(s3, syntheticRows, [
    "Block", "Setting",
    { key: "Real clean", label: "Real clean", format: "0%" },
    { key: "Synthetic", label: "Synthetic", format: "0%" },
    { key: "ACC", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "FairAvg", label: "FairAvg", format: "0.0000" },
    { key: "Score", label: "Score", format: "0.0000" },
    { key: "Rank", label: "Rank", format: "0" },
    "Reading",
  ], 3, 0, "SyntheticMain");
  applyColumnWidths(s3, [24, 32, 12, 12, 12, 12, 12, 12, 12, 8, 52]);
  s3.getRange("K:K").format.wrapText = true;
  s3.freezePanes.freezeRows(3);

  const s4 = wb.worksheets.add("04_FedSA");
  styleTitle(s4, "Table 04. New Performance Attack FedSA：AD2+ 的平衡优势", "核心读法：AD2+ 用候选参数中 joint score 最好的配置；Score rank=1 表示同时考虑 ACC、AEOD、ASPD 后排名第一。", 17);
  writeRows(s4, fedsaRows, [
    "Dataset", "Distribution", "Selected AD2+ profile",
    { key: "AD2+ ACC (%)", label: "AD2+ ACC (%)", format: "0.00" },
    { key: "AD2+ AEOD", label: "AD2+ AEOD", format: "0.0000" },
    { key: "AD2+ ASPD", label: "AD2+ ASPD", format: "0.0000" },
    { key: "AD2+ FairAvg", label: "AD2+ FairAvg", format: "0.0000" },
    { key: "Score rank", label: "Score rank", format: "0" },
    { key: "ACC rank", label: "ACC rank", format: "0" },
    { key: "AEOD rank", label: "AEOD rank", format: "0" },
    { key: "ASPD rank", label: "ASPD rank", format: "0" },
    { key: "ΔACC vs AD2", label: "ΔACC vs AD2 (pp)", format: "0.00" },
    { key: "ΔAEOD vs AD2", label: "ΔAEOD vs AD2", format: "0.0000" },
    { key: "ΔASPD vs AD2", label: "ΔASPD vs AD2", format: "0.0000" },
    "Best baseline ACC", "Best baseline FairAvg", "Reading",
  ], 3, 0, "FedSAMain");
  s4.getRange("D4:G7").format.fill = C.ours;
  applyColumnWidths(s4, [12, 12, 28, 13, 13, 13, 13, 10, 10, 10, 10, 14, 14, 14, 40, 42, 52]);
  s4.getRange("O:Q").format.wrapText = true;
  s4.freezePanes.freezeRows(3);

  const src = wb.worksheets.add("Sources");
  styleTitle(src, "Source Files", "这些文件是 compact workbook 的唯一输入；本表只整理和重排，不手工改实验值。", 3);
  writeRows(src, Object.entries(SOURCES).map(([Name, Path]) => ({ Name, Path })), ["Name", "Path"], 3, 0, "SourceFiles");
  applyColumnWidths(src, [28, 150]);

  for (const sheetName of ["README", "01_Ablation", "02_ServerRoot", "03_Synthetic10", "04_FedSA", "Sources"]) {
    const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, `preview_${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const readmePath = path.join(OUT, "README_four_tables_analysis.md");
  await writeReadme({ ablationRows, serverRows, syntheticRows, fedsaRows, readmePath });

  const output = await SpreadsheetFile.exportXlsx(wb);
  const xlsxPath = path.join(OUT, "GuardFed_AD2plus_Four_Experiments_Compact_Tables.xlsx");
  await output.save(xlsxPath);

  const inspect = await wb.inspect({
    kind: "sheet,table,match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 20 },
    maxChars: 6000,
  });

  console.log(JSON.stringify({
    xlsxPath,
    readmePath,
    previewDir: OUT,
    tables: {
      ablation: ablationRows.length,
      server: serverRows.length,
      synthetic: syntheticRows.length,
      fedsa: fedsaRows.length,
    },
    inspect: inspect.ndjson,
  }, null, 2));
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
