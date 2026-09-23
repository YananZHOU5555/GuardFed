import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "outputs", "hyperparameter_audit");
const OUT = path.join(OUT_DIR, "GuardFed_AD2plus_Hyperparameter_Audit.xlsx");
const PREVIEW_DIR = path.join(OUT_DIR, "previews");

const attackConfig = JSON.parse(await fs.readFile(path.join(ROOT, "results", "attack_strength", "attack_config.json"), "utf8"));

function num(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }
function colLetter(n) { let s = ""; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + ((n - 1) % 26)) + s; return s; }
function styleHeader(range) {
  range.format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: "#B8C7D9" } };
}
function styleBody(range) {
  range.format = { verticalAlignment: "center", wrapText: true, borders: { preset: "inside", style: "thin", color: "#D9E2F3" } };
}
function addTitle(sheet, text, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[text]];
  sheet.getRange(`A1:${endCol}1`).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 13 }, verticalAlignment: "center" };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 28;
}
function addNote(sheet, text, endCol) {
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange("A2").values = [[text]];
  sheet.getRange(`A2:${endCol}2`).format = { fill: "#FFF2CC", font: { italic: true, color: "#7F6000" }, wrapText: true, verticalAlignment: "center" };
  sheet.getRange(`A2:${endCol}2`).format.rowHeight = 34;
}
function addTableSheet(name, title, note, headers, rows, widths = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  addTitle(sheet, title, colLetter(headers.length - 1));
  addNote(sheet, note, colLetter(headers.length - 1));
  const headerRange = sheet.getRangeByIndexes(3, 0, 1, headers.length);
  headerRange.values = [headers]; styleHeader(headerRange);
  if (rows.length) {
    const body = sheet.getRangeByIndexes(4, 0, rows.length, headers.length);
    body.values = rows; styleBody(body);
    body.format.rowHeight = 34;
  }
  for (const [range, width] of Object.entries(widths)) sheet.getRange(range).format.columnWidth = width;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

const workbook = Workbook.create();

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
addTitle(readme, "GuardFed-AD2+ 超参数审计与搜索空间", "H");
addNote(readme, "目的：向导师清楚区分固定训练协议、实际执行的攻击校准 grid、AD2+ 每轮内部候选选择，以及仅作为代码默认值但没有参与当前实验的参数。", "H");
readme.getRange("A4:B15").values = [
  ["结论 1", "当前完整实验采用固定的训练协议，不是对所有训练参数做网格搜索。"],
  ["结论 2", "实际外部校准 grid：F Flip 5 个候选模式；FedSA 4 个 gain × 3 个 norm_ratio，共 12 组。"],
  ["结论 3", "AD2+ 不是从外部 baseline 中挑最优；它每一轮在同一 AD2+ 内部生成 10 个候选 scoring lens，用 clean root/server 数据进行自适应选择。"],
  ["结论 4", "组阈值校准对两个敏感组分别搜索 41 个分位点附近的候选阈值，形成最多 44 个阈值候选（含 0 与边界扩展）。"],
  ["当前攻击强度", `F Flip=${attackConfig.fflip_mode}; FedSA gain=${attackConfig.fedsa_gain}, norm_ratio=${attackConfig.fedsa_norm_ratio}.`],
  ["主实验 root/server", "最新 attack-strength 三个计划：10% clean root/server，0% synthetic root；不包含 Class-B FL 和 GuardFed-ACT。"],
  ["论文表旧 runner", "旧 paper-table runner 的默认值为 5% clean root/server；与最新攻击强度实验的 10% 协议分开记录，不能混写。"],
  ["结果可追溯性", "每个结果记录 config、seed、数据契约、攻击审计、last10 trajectory；本文件只总结配置，不替换原始结果。"],
  ["建议写入论文", "把 Fixed Protocol、Attack Calibration Grid、AD2+ Internal Candidate Family 三张表放入 appendix，并声明未搜索的参数保持固定。"],
  ["主要代码证据", ".codex_remote/reproduce_paper_tables.py；.codex_transfer/run_attack_strength_study.py。"],
  ["主要结果证据", "results/attack_strength/attack_config.json、summary.csv、seed_results.csv、audit.csv。"],
  ["状态", "已按当前工作区代码和结果文件整理；若再次改变 root 比例、轮数或优化器，应生成新的审计版本。"],
];
readme.getRange("A4:A15").format = { fill: "#D9EAF7", font: { bold: true }, verticalAlignment: "center" };
readme.getRange("B4:B15").format = { wrapText: true, verticalAlignment: "center" };
readme.getRange("A:A").format.columnWidth = 22; readme.getRange("B:B").format.columnWidth = 90;
readme.getRange("A4:B15").format.rowHeight = 34;
readme.freezePanes.freezeRows(3);

const protocolRows = [
  ["dataset", "datasets", "adult; compas", "fixed", "两个数据集均使用相同训练/攻击协议", "run_attack_strength_study.py:53-55"],
  ["distribution", "alpha", "IID=5000; non-IID=5", "fixed", "客户端 Dirichlet 分布参数", "reproduce_paper_tables.py:30-31"],
  ["clients", "num_clients", 20, "positive integer", "每轮客户端总数", "run_attack_strength_study.py:105"],
  ["adversary", "num_malicious", "main=4; ratio=2/4/6/8/10", "0..num_clients", "比例实验对应 10/20/30/40/50%", "run_attack_strength_study.py:331; 345-346"],
  ["training", "rounds", 70, "positive integer", "最终轮指标；轨迹保留每轮", "run_attack_strength_study.py:478"],
  ["training", "local_epochs", 1, "positive integer", "每个客户端每轮本地 epoch", "run_attack_strength_study.py:107"],
  ["training", "batch_size", 256, "positive integer", "本地 DataLoader batch size", "run_attack_strength_study.py:108"],
  ["training", "learning_rate", 0.005, ">0", "统一学习率", "run_attack_strength_study.py:109"],
  ["training", "optimizer", "Adam", "Adam or SGD", "当前核心 runner 默认 Adam", "reproduce_paper_tables.py:52; 271-276"],
  ["hardware", "device", "cuda", "cuda/cpu/auto", "Vast/5090 GPU 运行；CPU 仅用于验证", "run_attack_strength_study.py:111; 478-485"],
  ["root/server", "server_ratio", "10%", "[0,1]", "最新 attack-strength 及三计划协议", "run_attack_strength_study.py:112; 158-160"],
  ["root/server", "synthetic_ratio", "0%", "[0,1]", "最新三计划不生成 synthetic root", "run_attack_strength_study.py:113"],
  ["root/server", "server_sampling", "stratified_sensitive", "categorical", "按敏感组分层抽取 clean root/server", "run_attack_strength_study.py:114"],
  ["features", "include_sensitive_feature", false, "boolean", "模型输入不含敏感属性；标签也不进入输入", "run_attack_strength_study.py:116; core:1496"],
  ["aggregation", "aggregation_weighting", "count", "count/equal", "按客户端样本数聚合", "run_attack_strength_study.py:117"],
  ["fairness", "fairguard_mode", "server_aeod", "server_aeod/none", "FairGuard/GuardFed 系列使用 root fairness filtering", "run_attack_strength_study.py:124"],
  ["training", "use_reweighting", true, "boolean", "当前 core runner 打开 reweighting", "run_attack_strength_study.py:125"],
  ["seeds", "seed list", "123,456,789,1001,2024,3141,4242,5050,6060,7070", "integer list", "主实验 10 seeds", "run_attack_strength_study.py:33"],
];
addTableSheet("Fixed_Protocol", "固定训练与数据协议", "这些值是当前主实验统一使用的配置，不应被描述为已完成的超参数 grid。允许范围是实现层面的合法范围，不代表本次已经搜索过整个范围。", ["Group", "Config key", "Executed value", "Allowed / declared range", "Meaning", "Evidence"], protocolRows, {"A:A": 16, "B:B": 26, "C:C": 34, "D:D": 24, "E:E": 48, "F:F": 42});

const ad2Rows = [
  ["AD2+ mode", "ad2_plus_mode", "adaptive", "adaptive/fixed", "启用 clean-root 自适应内部候选选择", "reproduce_paper_tables.py:85; 1141-1207"],
  ["fairness budget", "act_fairness_budget", 0.06, "[0,1]", "每轮公平风险预算 B；用于动态 violation 与 dual multiplier", "reproduce_paper_tables.py:66; 1016-1018"],
  ["fairness lens", "act_fairness_metric", "aeod_aspd", "aeod/aspd/aeod_aspd/max", "默认同时观察 AEOD 与 ASPD", "reproduce_paper_tables.py:69; 957-978"],
  ["risk weight", "act_risk_weight", 1.0, ">=0", "公平风险 z-score 权重；AD2+ 内部候选会覆盖此值", "reproduce_paper_tables.py:71; 1107-1116"],
  ["violation weight", "act_violation_weight", 1.0, ">=0", "超过 B 的动态违反项权重；AD2+ 内部候选会覆盖此值", "reproduce_paper_tables.py:72; 1016-1031"],
  ["temperature", "act_temperature", 0.35, ">0", "softmax 权重温度；部分 AD2+ candidate 使用 0.20", "reproduce_paper_tables.py:67; 1107-1116"],
  ["keep ratio", "act_keep_ratio", 0.80, "[0,1]", "客户端保留比例；AD2+ candidate 使用 0.70/0.80/0.90", "reproduce_paper_tables.py:68; 1107-1116"],
  ["utility weight", "ad2_utility_weight", 1.0, ">=0", "clean-root utility 项权重；candidate 使用 0.60/0.70/0.80/1.00/1.20", "reproduce_paper_tables.py:82; 1107-1116"],
  ["centrality weight", "ad2_centrality_weight", 0.35, ">=0", "更新中心性项权重；candidate 使用 0.20/0.25/0.35/0.50", "reproduce_paper_tables.py:83; 1107-1116"],
  ["alignment weight", "ad2_alignment_weight", 0.35, ">=0", "与 clean server update 对齐项权重；candidate 使用 0.20/0.25/0.35/0.50", "reproduce_paper_tables.py:84; 1107-1116"],
  ["norm mode", "ad2_norm_mode", "adaptive", "adaptive/root", "当前固定 AD2 模式；AD2+ candidate 内部使用 root", "reproduce_paper_tables.py:81; 1099-1104"],
  ["score clipping", "ad2_score_clip", 5.0, ">=0", "AD2 主聚合 score clip；AD2+ candidate 内部设为 0 表示不额外 clip", "reproduce_paper_tables.py:77; 1099-1104"],
  ["norm clip scale", "ad2_norm_clip_scale", 2.5, ">0", "robust norm scale 的 multiplier", "reproduce_paper_tables.py:78; 1042-1055"],
  ["calibration base weight", "ad2_calibration_base_weight", 1.0, ">=0", "自适应 lambda 的 base weight", "reproduce_paper_tables.py:73; 167-175"],
  ["calibration budget", "ad2_calibration_budget", 0.06, "[0,1]", "组阈值校准的公平风险预算", "reproduce_paper_tables.py:74; 186-196"],
  ["calibration temperature", "ad2_calibration_temperature", 0.03, ">0", "由 base risk 与 budget 计算 adaptive lambda 的温度", "reproduce_paper_tables.py:75; 174-175"],
  ["calibration quantiles", "ad2_calibration_quantiles", 41, "integer >=7", "每个敏感组的 threshold quantile candidate 数量", "reproduce_paper_tables.py:76; 155-158"],
  ["max accuracy drop", "ad2_calibration_max_acc_drop", 0.03, "[0,1]", "组阈值 calibration 的 ACC drop 上限；AD2+ 每轮严格收紧到最多 0.005", "reproduce_paper_tables.py:79; 180-181; 1174-1175"],
  ["calibration objective", "ad2_calibration_objective", "acc_floor", "acc_floor/original", "优先保留 accuracy floor，再惩罚 fairness risk", "reproduce_paper_tables.py:80; 189-196"],
  ["AD2+ root loss", "ad2plus_root_loss", "0.45 AEOD + 0.45 ASPD + 0.10 max", "fixed formula", "内部候选比较用；不是外部 baseline 排名指标", "reproduce_paper_tables.py:1135-1138"],
  ["AD2+ selection score", "root_selection_score", "acc - 0.35 fair_loss - 6 acc_shortfall - 0.10 budget_shortfall", "fixed formula", "只用 clean root/server 当前轮指标选择 candidate", "reproduce_paper_tables.py:1176-1187"],
  ["ACT status", "act_* family", "inactive for current AD2+ tables", "recorded defaults only", "Class-B FL 与 GuardFed-ACT 排除；act_* 仍作为 AD2 内部 scoring 字段复用，需在论文中说明", "run_attack_strength_study.py:166; core:1271-1275"],
];
addTableSheet("AD2plus_Config", "AD2+ 参数、范围与动态机制", "这里的“Executed value”是外层 runner 默认/固定值；AD2+ 的 10 个内部候选会在每轮临时覆盖部分 fairness、temperature、keep ratio 和三类 score weight。不要把 candidate family 误写成 10 个外部算法。", ["Role", "Config key / formula", "Base executed value", "Allowed / candidate range", "Meaning", "Evidence"], ad2Rows, {"A:A": 20, "B:B": 32, "C:C": 30, "D:D": 38, "E:E": 62, "F:F": 42});

const fflipRows = Object.entries(attackConfig.fflip_scores).map(([mode, score]) => ["F Flip", mode, score, mode === attackConfig.fflip_mode ? "selected" : "candidate", "calibration seed=314159", "results/attack_strength/attack_config.json"]);
const fedsaRows = Object.entries(attackConfig.fedsa_scores).map(([key, score]) => { const v = JSON.parse(key); return ["FedSA", `${v.gain}`, `${v.norm_ratio}`, score, (Number(v.gain) === Number(attackConfig.fedsa_gain) && Number(v.norm_ratio) === Number(attackConfig.fedsa_norm_ratio)) ? "selected" : "candidate", "calibration seed=314159", "results/attack_strength/attack_config.json"]; });
addTableSheet("Attack_Grid", "实际执行的攻击校准 grid", "攻击校准与 AD2+ 参数搜索是两件事。F Flip 搜索 5 个 mode；FedSA 搜索 12 个 gain × norm_ratio 组合。选择规则写在 attack_config.json：在全局 FedAvg calibration 上最大化公平影响或 ACC 下降，并排除无效/退化运行。", ["Family", "Candidate / gain", "Norm ratio", "Calibration score", "Status", "Calibration", "Evidence"], [...fflipRows.map(r => [r[0], r[1], "", r[2], r[3], r[4], r[5]]), ...fedsaRows], {"A:A": 16, "B:B": 28, "C:C": 16, "D:D": 20, "E:E": 16, "F:F": 24, "G:G": 42});

const candidateRows = [
  ["balanced", "aeod_aspd", 0.75, 0.20, 0.80, 0.35, 1.00, 0.35, 0.35],
  ["balanced_open", "aeod_aspd", 0.75, 0.20, 0.90, 0.35, 1.00, 0.35, 0.35],
  ["fair_stable", "aeod_aspd", 0.90, 0.25, 0.80, 0.35, 1.00, 0.35, 0.35],
  ["utility_fair", "aeod_aspd", 0.60, 0.10, 0.80, 0.35, 1.20, 0.50, 0.50],
  ["dual_strict", "aeod_aspd", 1.10, 0.35, 0.80, 0.35, 0.70, 0.25, 0.25],
  ["dual_sharp", "aeod_aspd", 1.30, 0.50, 0.70, 0.20, 0.60, 0.20, 0.20],
  ["aeod_focus", "aeod", 0.85, 0.25, 0.80, 0.35, 0.80, 0.35, 0.35],
  ["aspd_focus", "aspd", 0.85, 0.25, 0.80, 0.35, 0.80, 0.35, 0.35],
  ["aspd_strict", "aspd", 1.10, 0.35, 0.70, 0.20, 0.70, 0.25, 0.25],
  ["max_guard", "max", 0.85, 0.25, 0.80, 0.35, 0.80, 0.35, 0.35],
];
addTableSheet("AD2plus_Candidates", "AD2+ 每轮内部候选 family", "这 10 行不是 10 个 baseline，也不是事后挑最优结果。每轮都对候选更新做一次 clean-root/server one-step evaluation，在 accuracy floor 约束下选择一个候选；最终测试集只在训练结束后评估。", ["Candidate", "Fairness metric", "Risk weight", "Violation weight", "Keep ratio", "Temperature", "Utility weight", "Centrality weight", "Alignment weight"], candidateRows, {"A:A": 18, "B:B": 18, "C:I": 17});

const rootRows = [
  ["Adult", "income (>50K=1)", "sex (Male=1, Female=0)", "feature excludes label and sensitive", "10% clean root in latest attack study; 0% synthetic", "train/test counts remain in each raw result data_contract"],
  ["COMPAS", "two_year_recid", "race (African-American=1, Others=0)", "feature excludes label and sensitive", "10% clean root in latest attack study; 0% synthetic", "train/test counts remain in each raw result data_contract"],
  ["Legacy paper tables", "same", "same", "same", "default runner server_ratio=5%; synthetic=0%", ".codex_remote/reproduce_paper_tables.py:46-50"],
  ["Root sampling", "-", "stratified_sensitive", "clean only", "same root protocol across methods within a study", "run_attack_strength_study.py:112-116"],
  ["Synthetic root", "-", "-", "-", "disabled for latest attack-strength plans", "synthetic_ratio=0.0"],
];
addTableSheet("Data_Root_Server", "数据、敏感属性与 root/server 数据协议", "这张表专门解决导师常问的“模型是否看到敏感属性、root data 有多少、不同实验是否混用了 5% 和 10%”问题。", ["Dataset / scope", "Label", "Sensitive attribute", "Model input", "Root/server protocol", "Evidence / audit"], rootRows, {"A:A": 22, "B:B": 26, "C:C": 34, "D:D": 34, "E:E": 48, "F:F": 48});

const seedRows = [
  ["main", "10 seeds", "123,456,789,1001,2024,3141,4242,5050,6060,7070", "20 clients; 4 malicious", "Benign, F Flip, FedSA", "2 datasets × 2 distributions × methods", "run_attack_strength_study.py:318-339"],
  ["ratio", "10 seeds", "same", "20 clients; 2/4/6/8/10 malicious", "S-DFA, Sp-DFA", "10/20/30/40/50%", "run_attack_strength_study.py:342-366"],
  ["calibration", "1 seed", "314159", "20 clients; 4 malicious", "F Flip and FedSA candidates", "selection only; not a final mean", "run_attack_strength_study.py:33-39; 259-310"],
];
addTableSheet("Seeds_Attacks", "随机种子、攻击与恶意比例", "最终主表使用 10 个预先列明的 seeds；calibration seed=314159 只用于选择统一攻击配置，不应与主实验 10-seed 均值混合。", ["Study", "Seed policy", "Seeds", "Malicious clients", "Attacks", "Scope", "Evidence"], seedRows, {"A:A": 16, "B:B": 16, "C:C": 54, "D:D": 30, "E:E": 26, "F:F": 38, "G:G": 42});

const evidenceRows = [
  ["Fixed runner", ".codex_transfer/run_attack_strength_study.py", "33-39, 103-128, 316-366", "latest attack-strength protocol, seeds, attack grid, 10% root"],
  ["Core config", ".codex_remote/reproduce_paper_tables.py", "36-87, 1092-1207, 1624-1659", "ExperimentConfig defaults, AD2+ candidates, CLI ranges"],
  ["Attack selection", "results/attack_strength/attack_config.json", "whole file", "selected F Flip/FedSA values and all calibration scores"],
  ["Raw reproducibility", "results/attack_strength/raw_results.jsonl", "per run", "full config, seed, data_contract, metrics, audit and trajectory"],
  ["Seed-level output", "results/attack_strength/seed_results.csv", "per seed/metric", "10-seed values used in summaries"],
  ["Attack audit", "results/attack_strength/audit.csv", "per malicious client", "F Flip change ratio, label changes, FedSA norm/cosine"],
  ["Paper-table legacy scope", ".codex_remote/reproduce_paper_tables.py", "46-50, 1598", "legacy default 5% root; keep separate from 10% attack-strength outputs"],
];
addTableSheet("Evidence", "代码与结果证据索引", "建议导师先看 README，再看 Fixed_Protocol、AD2plus_Config、Attack_Grid；需要复核时按 Evidence 列打开源文件。", ["Evidence type", "File", "Anchor", "What it proves"], evidenceRows, {"A:A": 22, "B:B": 56, "C:C": 28, "D:D": 72});

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (const sheet of workbook.worksheets.items) {
  const safe = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(OUT);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
console.log(errors.ndjson);
console.log(JSON.stringify({ output: OUT, sheets: workbook.worksheets.items.map(s => s.name) }, null, 2));
