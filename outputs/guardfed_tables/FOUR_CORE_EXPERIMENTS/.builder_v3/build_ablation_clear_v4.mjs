import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "excel_delivery_v4");
const OLD_ATTACK_RAW = path.join(ROOT, "01_ablation/goal_revision_v3/goal_ablation_v3_stress_raw.csv");
const BENIGN_RAW = path.join(ROOT, "01_ablation/goal_revision_v4/goal_ablation_v4_benign_raw.csv");

const C = {
  title: "#17365D",
  header: "#D9EAF7",
  subheader: "#EAF3F8",
  note: "#FFF2CC",
  ok: "#D9EAD3",
  warn: "#FCE4D6",
  full: "#E2F0D9",
  grid: "#E5E7EB",
};

const attacks = ["Benign", "FedSA", "F Flip", "S-DFA"];
const profiles = [
  {
    profile: "full",
    name: "Full AD2+",
    cn: "完整模型：性能项 + 公平项 + 几何项",
    lambdaRisk: 1.4,
    lambdaViolation: 0.7,
    wUtility: 3.0,
    wCentrality: 0.8,
    wAlignment: 1.8,
    calibration: "on",
  },
  {
    profile: "no_performance_UCA",
    name: "No performance U/C/A",
    cn: "去掉性能防御：不看 root utility、centrality、alignment",
    lambdaRisk: 1.4,
    lambdaViolation: 0.7,
    wUtility: 0.0,
    wCentrality: 0.0,
    wAlignment: 0.0,
    calibration: "on",
  },
  {
    profile: "no_fairness_FC",
    name: "No fairness F/C",
    cn: "去掉公平防御：不看 fairness risk/violation，且关闭校准",
    lambdaRisk: 0.0,
    lambdaViolation: 0.0,
    wUtility: 3.0,
    wCentrality: 0.8,
    wAlignment: 1.8,
    calibration: "off",
  },
  {
    profile: "no_geometry_CA",
    name: "No geometry C/A",
    cn: "去掉几何一致性：不看 centrality 和 alignment",
    lambdaRisk: 1.4,
    lambdaViolation: 0.7,
    wUtility: 3.0,
    wCentrality: 0.0,
    wAlignment: 0.0,
    calibration: "on",
  },
  {
    profile: "utility_only_U",
    name: "Utility only",
    cn: "只保留 root utility，关闭公平与几何项",
    lambdaRisk: 0.0,
    lambdaViolation: 0.0,
    wUtility: 3.0,
    wCentrality: 0.0,
    wAlignment: 0.0,
    calibration: "off",
  },
];

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
  const headers = rows[0];
  return rows.slice(1).filter(r => r.some(v => v !== "")).map(r => {
    const o = {};
    headers.forEach((h, i) => { o[h] = r[i] ?? ""; });
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

function styleTitle(ws, title, subtitle, cols = 8) {
  ws.showGridLines = false;
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
  return range;
}

function normalizeRows(rows) {
  return rows.map(r => ({
    source_suite: r.suite,
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
    for (const k of keys) out[k] = vals[0][k];
    out.n = vals.length;
    out.ACC_pct_mean = mean(vals.map(v => v.ACC_pct));
    out.AEOD_mean = mean(vals.map(v => v.AEOD));
    out.ASPD_mean = mean(vals.map(v => v.ASPD));
    out.fair_avg_mean = mean(vals.map(v => v.fair_avg));
    out.score_mean = mean(vals.map(v => v.score));
    return out;
  });
}

function lookup(agg, dataset, attack, profile, distribution = null) {
  return agg.find(r =>
    r.dataset === dataset &&
    r.attack === attack &&
    r.profile === profile &&
    (distribution == null || r.distribution === distribution)
  );
}

function datasetTableRows(agg, dataset) {
  return profiles.map(p => {
    const row = {
      Setting: p.name,
      "Chinese meaning": p.cn,
      "lambda risk": p.lambdaRisk,
      "lambda violation": p.lambdaViolation,
      "w utility": p.wUtility,
      "w centrality": p.wCentrality,
      "w alignment": p.wAlignment,
      "calibration": p.calibration,
    };
    for (const attack of attacks) {
      const r = lookup(agg, dataset, attack, p.profile);
      row[`${attack} ACC`] = r?.ACC_pct_mean ?? null;
      row[`${attack} AEOD`] = r?.AEOD_mean ?? null;
      row[`${attack} ASPD`] = r?.ASPD_mean ?? null;
    }
    return row;
  });
}

function effectRows(agg) {
  const out = [];
  for (const dataset of ["adult", "compas"]) {
    for (const attack of ["FedSA", "S-DFA"]) {
      const full = lookup(agg, dataset, attack, "full");
      const noPerf = lookup(agg, dataset, attack, "no_performance_UCA");
      out.push({
        Dataset: dataset,
        "What is tested": "性能防御项是否有用",
        Attack: attack,
        Compare: "No performance U/C/A vs Full",
        Metric: "ACC (%)",
        "Full value": full?.ACC_pct_mean ?? null,
        "Ablated value": noPerf?.ACC_pct_mean ?? null,
        "Delta": noPerf && full ? noPerf.ACC_pct_mean - full.ACC_pct_mean : null,
        "Expected direction": "Delta ACC < 0",
        Result: noPerf && full && noPerf.ACC_pct_mean < full.ACC_pct_mean ? "OK" : "CHECK",
      });
    }
    for (const attack of ["F Flip", "S-DFA"]) {
      const full = lookup(agg, dataset, attack, "full");
      const noFair = lookup(agg, dataset, attack, "no_fairness_FC");
      out.push({
        Dataset: dataset,
        "What is tested": "公平防御项是否有用",
        Attack: attack,
        Compare: "No fairness F/C vs Full",
        Metric: "FairAvg",
        "Full value": full?.fair_avg_mean ?? null,
        "Ablated value": noFair?.fair_avg_mean ?? null,
        "Delta": noFair && full ? noFair.fair_avg_mean - full.fair_avg_mean : null,
        "Expected direction": "Delta FairAvg > 0",
        Result: noFair && full && noFair.fair_avg_mean > full.fair_avg_mean ? "OK" : "CHECK",
      });
    }
  }
  return out;
}

function mainHeaders() {
  const base = [
    "Setting",
    "Chinese meaning",
    { key: "lambda risk", label: "lambda_risk", format: "0.00" },
    { key: "lambda violation", label: "lambda_violation", format: "0.00" },
    { key: "w utility", label: "w_utility", format: "0.00" },
    { key: "w centrality", label: "w_centrality", format: "0.00" },
    { key: "w alignment", label: "w_alignment", format: "0.00" },
    "calibration",
  ];
  for (const attack of attacks) {
    base.push({ key: `${attack} ACC`, label: `${attack} ACC (%)`, format: "0.00" });
    base.push({ key: `${attack} AEOD`, label: `${attack} AEOD`, format: "0.0000" });
    base.push({ key: `${attack} ASPD`, label: `${attack} ASPD`, format: "0.0000" });
  }
  return base;
}

async function build() {
  const attacked = normalizeRows(await csv(OLD_ATTACK_RAW));
  const benign = normalizeRows(await csv(BENIGN_RAW));
  const rows = [...benign, ...attacked];
  const aggMain = aggregate(rows, ["dataset", "attack", "profile"]);
  const aggDist = aggregate(rows, ["dataset", "distribution", "attack", "profile"]);
  const wb = Workbook.create();

  const intro = wb.worksheets.add("读我");
  styleTitle(intro, "01 消融实验：看得懂版", "这版只回答：五个参数/组件打开或关闭后，在正常和攻击场景下 ACC、AEOD、ASPD 怎么变。", 8);
  writeRows(intro, [
    { Item: "数据集", Detail: "Adult 和 COMPAS。" },
    { Item: "场景", Detail: "Benign 是正常无攻击；FedSA 是性能攻击；F Flip 是公平攻击；S-DFA 是性能+公平混合攻击。" },
    { Item: "主表读法", Detail: "每一行是一个参数配置；每三列是一个场景下的 ACC / AEOD / ASPD。ACC 越高越好，AEOD/ASPD 越低越好。" },
    { Item: "五个参数", Detail: "lambda_risk、lambda_violation 是公平防御；w_utility 是 root clean data 性能效用；w_centrality 和 w_alignment 是更新几何一致性。" },
    { Item: "真实性", Detail: "Attack 结果来自 v3 已完成 raw runs；Benign 是本次补跑的真实 40-round runs。没有手工改数值。" },
  ], ["Item", "Detail"], 3, 0);
  intro.getRange("A:A").format.columnWidth = 24;
  intro.getRange("B:B").format.columnWidth = 120;
  intro.getRange("B:B").format.wrapText = true;

  const params = wb.worksheets.add("参数配置");
  styleTitle(params, "五个参数和五种配置", "这里把代码 profile 翻译成中文配置。读主表前先看这一页。", 9);
  writeRows(params, profiles.map(p => ({
    Profile: p.profile,
    Name: p.name,
    "中文含义": p.cn,
    "lambda_risk": p.lambdaRisk,
    "lambda_violation": p.lambdaViolation,
    "w_utility": p.wUtility,
    "w_centrality": p.wCentrality,
    "w_alignment": p.wAlignment,
    "calibration": p.calibration,
  })), [
    "Profile", "Name", "中文含义",
    { key: "lambda_risk", label: "lambda_risk", format: "0.00" },
    { key: "lambda_violation", label: "lambda_violation", format: "0.00" },
    { key: "w_utility", label: "w_utility", format: "0.00" },
    { key: "w_centrality", label: "w_centrality", format: "0.00" },
    { key: "w_alignment", label: "w_alignment", format: "0.00" },
    "calibration",
  ], 3, 0);

  for (const dataset of ["adult", "compas"]) {
    const ws = wb.worksheets.add(dataset === "adult" ? "Adult主表" : "COMPAS主表");
    styleTitle(ws, `${dataset.toUpperCase()} 消融主表`, "每个数值是 IID + non-IID、3 个 seed 的平均值。", 20);
    writeRows(ws, datasetTableRows(aggMain, dataset), mainHeaders(), 3, 0);
    ws.getRangeByIndexes(4, 0, 1, 20).format.fill = C.full;
    ws.getRange("B:B").format.columnWidth = 42;
    ws.getRange("B:B").format.wrapText = true;
    ws.freezePanes.freezeRows(4);
    ws.freezePanes.freezeColumns(2);
  }

  const effects = wb.worksheets.add("效果总结");
  styleTitle(effects, "最核心的消融结论", "这里只看最重要的问题：去掉性能项是否让性能攻击下 ACC 变差；去掉公平项是否让公平攻击下 FairAvg 变差。", 10);
  const eRows = effectRows(aggMain);
  writeRows(effects, eRows, [
    "Dataset", "What is tested", "Attack", "Compare",
    "Metric",
    { key: "Full value", label: "Full value", format: "0.0000" },
    { key: "Ablated value", label: "Ablated value", format: "0.0000" },
    { key: "Delta", label: "Delta", format: "0.0000" },
    "Expected direction", "Result",
  ], 3, 0);
  eRows.forEach((r, i) => {
    effects.getRangeByIndexes(4 + i, 0, 1, 10).format.fill = r.Result === "OK" ? C.ok : C.warn;
  });

  const byDist = wb.worksheets.add("IID_nonIID细节");
  styleTitle(byDist, "IID / non-IID 分布细节", "主表是二者平均；这里保留 IID 和 non-IID 分开后的结果，便于审稿追溯。", 13);
  const detail = aggDist
    .filter(r => profiles.some(p => p.profile === r.profile))
    .sort((a, b) => a.dataset.localeCompare(b.dataset) || a.distribution.localeCompare(b.distribution) || attacks.indexOf(a.attack) - attacks.indexOf(b.attack) || profiles.findIndex(p => p.profile === a.profile) - profiles.findIndex(p => p.profile === b.profile));
  writeRows(byDist, detail, [
    "dataset", "distribution", "attack", "profile", { key: "n", label: "n", format: "0" },
    { key: "ACC_pct_mean", label: "ACC (%)", format: "0.00" },
    { key: "AEOD_mean", label: "AEOD", format: "0.0000" },
    { key: "ASPD_mean", label: "ASPD", format: "0.0000" },
    { key: "fair_avg_mean", label: "FairAvg", format: "0.0000" },
    { key: "score_mean", label: "Score", format: "0.0000" },
  ], 3, 0);

  const rawAgg = wb.worksheets.add("Raw汇总");
  writeRows(rawAgg, rows.sort((a, b) => a.dataset.localeCompare(b.dataset) || attacks.indexOf(a.attack) - attacks.indexOf(b.attack)), [
    "source_suite", "profile", "dataset", "distribution", "attack",
    { key: "seed", label: "seed", format: "0" },
    { key: "rounds", label: "rounds", format: "0" },
    { key: "ACC_pct", label: "ACC (%)", format: "0.00" },
    { key: "AEOD", label: "AEOD", format: "0.0000" },
    { key: "ASPD", label: "ASPD", format: "0.0000" },
    { key: "fair_avg", label: "FairAvg", format: "0.0000" },
    { key: "score", label: "Score", format: "0.0000" },
  ], 0, 0);

  for (const ws of wb.worksheets.items) {
    const used = ws.getUsedRange();
    if (used) used.format.autofitColumns();
  }

  await fs.mkdir(OUT, { recursive: true });
  const scan = await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 200 },
    maxChars: 2000,
  });
  console.log(scan.ndjson);
  for (const ws of wb.worksheets.items) {
    const png = await wb.render({ sheetName: ws.name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, `ablation_clear_v4_${ws.name.replace(/[^A-Za-z0-9一-龥]+/g, "_")}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  const blob = await SpreadsheetFile.exportXlsx(wb);
  const out = path.join(OUT, "GuardFed_AD2plus_01_Ablation_v4_Clear_Table.xlsx");
  await blob.save(out);
  console.log(out);
}

await build();
