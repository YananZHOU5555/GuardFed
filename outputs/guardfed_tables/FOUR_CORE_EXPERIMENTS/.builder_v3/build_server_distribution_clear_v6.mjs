import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS";
const OUT = path.join(ROOT, "excel_delivery_v6");
const DATA_DIR = path.join(ROOT, "02_server_distribution/goal_revision_v6");
const BY_LEVEL = path.join(DATA_DIR, "goal_server_iid_sensitivity_v6_by_level.csv");
const CORR = path.join(DATA_DIR, "goal_server_iid_sensitivity_v6_correlations.csv");
const RAW = path.join(DATA_DIR, "goal_server_iid_sensitivity_v6_raw.csv");

const C = {
  title: "#17365D",
  header: "#D9EAF7",
  note: "#FFF2CC",
  ok: "#D9EAD3",
  warn: "#FCE4D6",
  grid: "#E5E7EB",
  iid: "#E2F0D9",
};

const levels = ["IID server/root", "Mild non-IID server/root", "Moderate non-IID server/root"];
const slices = [
  ["IID", "Benign"],
  ["IID", "FedSA"],
  ["non-IID", "Benign"],
  ["non-IID", "FedSA"],
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
  const h = rows[0];
  return rows.slice(1).filter(r => r.some(v => v !== "")).map(r => {
    const o = {};
    h.forEach((k, i) => { o[k] = r[i] ?? ""; });
    return o;
  });
}
async function csv(file) { return parseCsv(await fs.readFile(file, "utf8")); }
function num(v) { const x = Number(v); return Number.isFinite(x) ? x : null; }

function styleTitle(ws, title, subtitle, cols = 10) {
  ws.showGridLines = false;
  const r1 = ws.getRangeByIndexes(0, 0, 1, cols);
  r1.merge(); r1.values = [[title]];
  r1.format = { fill: C.title, font: { bold: true, color: "#FFFFFF", size: 14 } };
  const r2 = ws.getRangeByIndexes(1, 0, 1, cols);
  r2.merge(); r2.values = [[subtitle]];
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
    if (h.format && rows.length) ws.getRangeByIndexes(row + 1, col + i, rows.length, 1).format.numberFormat = h.format;
  });
  range.format.autofitColumns();
  range.format.autofitRows();
  return range;
}

function lookup(rows, dataset, distribution, attack, level) {
  return rows.find(r =>
    r.dataset === dataset &&
    r.distribution === distribution &&
    r.attack === attack &&
    r.level === level
  );
}

function tableRows(rows, dataset) {
  return levels.map(level => {
    const first = rows.find(r => r.dataset === dataset && r.level === level);
    const out = {
      "Server/root distribution": level,
      "n": first?.n ?? null,
      "Group TVD": first?.group_tvd_mean ?? null,
      "Sensitive TVD": first?.sensitive_tvd_mean ?? null,
      "Label TVD": first?.label_tvd_mean ?? null,
    };
    for (const [distribution, attack] of slices) {
      const r = lookup(rows, dataset, distribution, attack, level);
      const prefix = `${distribution} ${attack}`;
      out[`${prefix} ACC`] = r?.ACC_pct_mean ?? null;
      out[`${prefix} AEOD`] = r?.AEOD_mean ?? null;
      out[`${prefix} ASPD`] = r?.ASPD_mean ?? null;
      out[`${prefix} FairAvg`] = r?.FairAvg_mean ?? null;
    }
    return out;
  });
}

function mainHeaders() {
  const h = [
    "Server/root distribution",
    { key: "n", label: "n", format: "0" },
    { key: "Group TVD", label: "Group TVD", format: "0.0000" },
    { key: "Sensitive TVD", label: "Sensitive TVD", format: "0.0000" },
    { key: "Label TVD", label: "Label TVD", format: "0.0000" },
  ];
  for (const [distribution, attack] of slices) {
    const prefix = `${distribution} ${attack}`;
    h.push({ key: `${prefix} ACC`, label: `${prefix} ACC (%)`, format: "0.00" });
    h.push({ key: `${prefix} AEOD`, label: `${prefix} AEOD`, format: "0.0000" });
    h.push({ key: `${prefix} ASPD`, label: `${prefix} ASPD`, format: "0.0000" });
    h.push({ key: `${prefix} FairAvg`, label: `${prefix} FairAvg`, format: "0.0000" });
  }
  return h;
}

async function build() {
  const byLevel = (await csv(BY_LEVEL)).map(r => ({
    ...r,
    level_index: num(r.level_index),
    n: num(r.n),
    group_tvd_mean: num(r.group_tvd_mean),
    sensitive_tvd_mean: num(r.sensitive_tvd_mean),
    label_tvd_mean: num(r.label_tvd_mean),
    ACC_pct_mean: num(r.ACC_pct_mean),
    AEOD_mean: num(r.AEOD_mean),
    ASPD_mean: num(r.ASPD_mean),
    FairAvg_mean: num(r.FairAvg_mean),
    Score_mean: num(r.Score_mean),
  }));
  const corr = (await csv(CORR)).map(r => ({
    ...r,
    n_levels: num(r.n_levels),
    corr_level_acc: num(r.corr_level_acc),
    corr_level_aeod: num(r.corr_level_aeod),
    corr_level_aspd: num(r.corr_level_aspd),
    corr_level_fairavg: num(r.corr_level_fairavg),
    delta_acc_moderate_minus_iid: num(r.delta_acc_moderate_minus_iid),
    delta_fairavg_moderate_minus_iid: num(r.delta_fairavg_moderate_minus_iid),
  }));
  const raw = await csv(RAW);

  const wb = Workbook.create();
  const intro = wb.worksheets.add("读我");
  styleTitle(intro, "02 Server/Root Distribution Sensitivity：清晰版", "主表只展示稳定工作区间：IID server/root、mild non-IID、moderate non-IID。所有指标为 70 轮训练后最后 10 轮均值，3 seeds 平均。", 10);
  writeRows(intro, [
    { Item: "为什么不用旧表", Detail: "旧表使用 Dirichlet alpha 搜索，alpha 与实际分布偏移不单调，极端 alpha 会抽到偏但容易的 root set，导致 ACC 回升或公平指标异常变低。" },
    { Item: "这版怎么定义 server 分布", Detail: "第一行是 stratified clean 10% server/root data，作为 IID-like baseline；后两行把 clean server/root data 受控偏向 majority sensitive-label stratum。" },
    { Item: "读法", Detail: "Group/Sensitive/Label TVD 越大，server/root 越 non-IID。ACC 越高越好，AEOD/ASPD/FairAvg 越低越好。" },
    { Item: "核心 insight", Detail: "server/root 越接近 IID，正常训练和 FedSA 防御下总体表现越稳定；server/root 偏移后，ACC 或 FairAvg 会退化。" },
  ], ["Item", "Detail"], 3, 0);
  intro.getRange("A:A").format.columnWidth = 26;
  intro.getRange("B:B").format.columnWidth = 120;
  intro.getRange("B:B").format.wrapText = true;

  for (const dataset of ["adult", "compas"]) {
    const ws = wb.worksheets.add(dataset === "adult" ? "Adult主表" : "COMPAS主表");
    styleTitle(ws, `${dataset.toUpperCase()} Server Distribution Effect Table`, "每行是 server/root 分布条件；每个场景直接给 ACC / AEOD / ASPD / FairAvg。", 21);
    const rows = tableRows(byLevel, dataset);
    writeRows(ws, rows, mainHeaders(), 3, 0);
    ws.getRangeByIndexes(4, 0, 1, 21).format.fill = C.iid;
    ws.freezePanes.freezeRows(4);
    ws.freezePanes.freezeColumns(1);
  }

  const trend = wb.worksheets.add("趋势总结");
  styleTitle(trend, "Trend Summary", "corr_level_acc < 0 表示 server/root 越 non-IID，ACC 越低；delta_acc_moderate_minus_iid < 0 表示 moderate non-IID 相比 IID 降低。", 11);
  writeRows(trend, corr, [
    "dataset", "distribution", "attack", { key: "n_levels", label: "n levels", format: "0" },
    { key: "corr_level_acc", label: "corr level-ACC", format: "0.000" },
    { key: "corr_level_aeod", label: "corr level-AEOD", format: "0.000" },
    { key: "corr_level_aspd", label: "corr level-ASPD", format: "0.000" },
    { key: "corr_level_fairavg", label: "corr level-FairAvg", format: "0.000" },
    { key: "delta_acc_moderate_minus_iid", label: "Delta ACC: moderate-IID", format: "0.00" },
    { key: "delta_fairavg_moderate_minus_iid", label: "Delta FairAvg: moderate-IID", format: "0.0000" },
  ], 3, 0);
  corr.forEach((r, i) => {
    const ok = r.delta_acc_moderate_minus_iid < 0 || r.delta_fairavg_moderate_minus_iid > 0;
    trend.getRangeByIndexes(4 + i, 0, 1, 10).format.fill = ok ? C.ok : C.warn;
  });

  const by = wb.worksheets.add("ByLevel长表");
  writeRows(by, byLevel.sort((a, b) => a.dataset.localeCompare(b.dataset) || a.distribution.localeCompare(b.distribution) || a.attack.localeCompare(b.attack) || a.level_index - b.level_index), [
    "dataset", "distribution", "attack", "level", { key: "level_index", label: "level index", format: "0" }, { key: "n", label: "n", format: "0" },
    { key: "group_tvd_mean", label: "Group TVD", format: "0.0000" },
    { key: "sensitive_tvd_mean", label: "Sensitive TVD", format: "0.0000" },
    { key: "label_tvd_mean", label: "Label TVD", format: "0.0000" },
    { key: "ACC_pct_mean", label: "ACC (%)", format: "0.00" },
    { key: "AEOD_mean", label: "AEOD", format: "0.0000" },
    { key: "ASPD_mean", label: "ASPD", format: "0.0000" },
    { key: "FairAvg_mean", label: "FairAvg", format: "0.0000" },
    { key: "Score_mean", label: "Score", format: "0.0000" },
  ], 0, 0);

  const rawWs = wb.worksheets.add("Raw");
  writeRows(rawWs, raw, Object.keys(raw[0]).map(k => ({ key: k, label: k })), 0, 0);

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
    await fs.writeFile(path.join(OUT, `server_distribution_v6_${ws.name.replace(/[^A-Za-z0-9一-龥]+/g, "_")}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  const blob = await SpreadsheetFile.exportXlsx(wb);
  const out = path.join(OUT, "GuardFed_AD2plus_02_Server_Distribution_v6_Clear_Table.xlsx");
  await blob.save(out);
  console.log(out);
}

await build();
