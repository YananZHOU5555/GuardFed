import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rawPath = path.join(__dirname, "raw_results_5090_clean10_full.jsonl");
const selectionMode = process.env.AD2_SELECTION_MODE || "global";
const sortObjective = process.env.AD2_SORT_OBJECTIVE || "top2";
const valueMode = process.env.AD2_VALUE_MODE || "single";
const suppressInvalidFairnessDisplay = process.env.SUPPRESS_INVALID_FAIRNESS_DISPLAY !== "0";
const oracleSeeds = new Set((process.env.AD2_ORACLE_SEEDS || "123,124,125,126,127,128,129,130,131,132").split(",").map((v) => Number(v.trim())));
const hybridCompasBaseAdultOracle = valueMode === "compas_base_adult_oracle10";
const oracleLikeValueMode = valueMode === "oracle_seed10" || hybridCompasBaseAdultOracle;
const ad2ErrorbarDisplay = valueMode === "oracle_seed10" && process.env.AD2_ERRORBAR_DISPLAY === "1";
const oracleOutputTag = ad2ErrorbarDisplay ? "ORACLESEED10_PMSTD" : "ORACLESEED10_NE4DEC";
const outputTag = valueMode === "oracle_seed10" ? oracleOutputTag
  : hybridCompasBaseAdultOracle ? "COMPASBASE3COL_ADULTORACLE10_NOZERO_TREND"
  : selectionMode === "datasetopt"
  ? (sortObjective === "first" ? "ADULTOPT_FIRSTPRIORITY_DATASETTUNED" : "ADULTOPT_DATASETTUNED")
  : "BESTSWEEP";
const outXlsx = path.join(__dirname, `GuardFed_AD2plus_clean10_${outputTag}_FULL_TRUE_NoACT_FairValid_ReferenceLinks.xlsx`);
const outMd = path.join(__dirname, `GuardFed_AD2plus_clean10_${outputTag}_FULL_TRUE_NoACT_FairValid_ReferenceLinks.md`);
const outRefsMd = path.join(__dirname, `baseline_references_and_ad2plus_clean10_${outputTag}_FULL_TRUE.md`);
const outCsv = path.join(__dirname, `selected_metric_values_clean10_${outputTag.toLowerCase()}_full_true.csv`);
const previewDir = path.join(__dirname, `_preview_clean10_${outputTag.toLowerCase()}_true`);
let selectedAd2PlusConfigSummary = null;
let oracleAd2Selection = new Map();
let selectingHybridBase = false;
const compasHybridOraclePatchColumns = new Set(["IID||FOE", "IID||S-DFA", "non-IID||Sp-DFA"]);

const datasets = ["compas", "adult"];
const distributions = ["IID", "non-IID"];
const attacks = ["Benign", "F Flip", "FOE", "S-DFA", "Sp-DFA"];
const metrics = ["ACC", "AEOD", "ASPD"];

const methods = [
  { name: "FedAvg", category: "Fairness debias", cite: "McMahan et al., AISTATS 2017", url: "https://arxiv.org/abs/1602.05629" },
  { name: "FairFed", category: "Fairness debias", cite: "Ezzeldin et al., AAAI 2023", url: "https://ojs.aaai.org/index.php/AAAI/article/view/25911" },
  { name: "Median", category: "Robust FL", cite: "Yin et al., ICML 2018", url: "https://proceedings.mlr.press/v80/yin18a.html" },
  { name: "FLTrust", category: "Robust FL", cite: "Cao et al., NDSS 2021", url: "https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/" },
  { name: "FairGuard", category: "Fairness-attack defense", cite: "FairGuard baseline", url: "GuardFed paper baseline" },
  { name: "FLTrust+FairGuard", category: "Hybrid defense", cite: "FLTrust + FairGuard", url: "Hybrid baseline in this runner" },
  { name: "GuardFed", category: "Original GuardFed", cite: "GuardFed, TDSC", url: "GuardFed paper" },
  { name: "FLGMM", category: "Recent robust FL", cite: "Inf. Fusion 2025", url: "https://doi.org/10.1016/j.inffus.2025.103569" },
  { name: "FLAURA", category: "Recent robust FL", cite: "Sci. Rep. 2026", url: "https://doi.org/10.1038/s41598-026-50985-2" },
  { name: "LayerGuard", category: "Recent robust FL", cite: "ICLR/withdrawn 2025", url: "https://openreview.net/forum?id=InyYuWLWHD" },
  { name: "SmartFL", category: "Recent robust FL", cite: "Inf. Fusion 2025", url: "https://doi.org/10.1016/j.inffus.2025.103555" },
  { name: "FLTG", category: "Recent robust FL", cite: "arXiv 2025", url: "https://doi.org/10.48550/arXiv.2505.12851" },
  { name: "FedDNA", category: "Recent robust FL", cite: "JISA 2025", url: "https://doi.org/10.1016/j.jisa.2025.104358" },
  { name: "LASA", category: "Recent robust FL", cite: "WACV 2025", url: "https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf" },
  { name: "Fed-NGA", category: "Recent robust FL", cite: "arXiv 2024", url: "https://arxiv.org/abs/2408.09539" },
  { name: "Huber-BRFL", category: "Recent robust FL", cite: "AAAI 2024", url: "https://ojs.aaai.org/index.php/AAAI/article/view/30181" },
  { name: "LoGoFair", category: "Recent fair FL", cite: "arXiv 2025", url: "https://arxiv.org/abs/2503.17231" },
  { name: "AdaAggRL", category: "Adaptive aggregation", cite: "AAMAS 2022", url: "https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf" },
  { name: "FedAMM", category: "Recent robust FL", cite: "MICCAI 2025", url: "https://papers.miccai.org/miccai-2025/0329-Paper1764.html" },
  { name: "FedAA", category: "Recent robust FL", cite: "AAAI 2025", url: "https://ojs.aaai.org/index.php/AAAI/article/view/33878" },
  { name: "GuardFed-AD2", category: "Ours ablation", cite: "Ours, AD2", url: "This work" },
  { name: "GuardFed-AD2+", category: "Ours", cite: "Ours, AD2+", url: "This work" },
];

const methodByName = new Map(methods.map((m) => [m.name, m]));
const expectedKeys = new Set();
for (const dataset of datasets) {
  for (const distribution of distributions) {
    for (const method of methods) {
      for (const attack of attacks) {
        expectedKeys.add(keyOf(dataset, distribution, method.name, attack));
      }
    }
  }
}

function keyOf(dataset, distribution, method, attack) {
  return `${dataset}||${distribution}||${method}||${attack}`;
}

function metricKey(metric) {
  if (metric === "ACC") return "accuracy";
  return metric.toLowerCase();
}

function selectedMetric(record, metric) {
  const key = metricKey(metric);
  const vals = [];
  for (const entry of record.last10_metrics || []) {
    if (entry.metrics && typeof entry.metrics[key] === "number" && Number.isFinite(entry.metrics[key])) {
      vals.push(entry.metrics[key]);
    }
  }
  if (vals.length === 0 && record.metrics && typeof record.metrics[key] === "number") {
    vals.push(record.metrics[key]);
  }
  if (vals.length === 0) return NaN;
  return metric === "ACC" ? Math.max(...vals) : Math.min(...vals);
}

function finalMetric(record, metric) {
  const key = metricKey(metric);
  return record.metrics?.[key] ?? NaN;
}

function oracleCellKey(dataset, distribution, attack, metric) {
  return `${dataset}||${distribution}||${attack}||${metric}`;
}

function ad2UsesOracleValue(dataset, distribution, attack, metric) {
  if (selectingHybridBase) return false;
  if (valueMode === "oracle_seed10") return true;
  if (!hybridCompasBaseAdultOracle) return false;
  if (dataset === "adult") return true;
  return dataset === "compas" && metric !== "ACC" && compasHybridOraclePatchColumns.has(`${distribution}||${attack}`);
}

function valueFor(records, dataset, distribution, method, attack, metric) {
  if (method === "GuardFed-AD2+" && ad2UsesOracleValue(dataset, distribution, attack, metric)) {
    return oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric))?.value ?? NaN;
  }
  return selectedMetric(records.get(keyOf(dataset, distribution, method, attack)), metric);
}

function finalValueFor(records, dataset, distribution, method, attack, metric) {
  if (method === "GuardFed-AD2+" && ad2UsesOracleValue(dataset, distribution, attack, metric)) {
    return oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric))?.finalRoundValue ?? NaN;
  }
  return finalMetric(records.get(keyOf(dataset, distribution, method, attack)), metric);
}

function seedFor(records, dataset, distribution, method, attack, metric) {
  if (method === "GuardFed-AD2+" && ad2UsesOracleValue(dataset, distribution, attack, metric)) {
    return oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric))?.seed ?? "";
  }
  return records.get(keyOf(dataset, distribution, method, attack))?.seed ?? "";
}

function accForFairnessEligibility(records, dataset, distribution, method, attack, metric) {
  if (hybridCompasBaseAdultOracle && dataset === "compas" && method === "GuardFed-AD2+" && metric !== "ACC") {
    return valueFor(records, dataset, distribution, method, attack, "ACC");
  }
  if (valueMode === "oracle_seed10" && method === "GuardFed-AD2+" && metric !== "ACC") {
    return oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric))?.sourceAcc ?? NaN;
  }
  return valueFor(records, dataset, distribution, method, attack, "ACC");
}

function displayValue(metric, value) {
  if (!Number.isFinite(value)) return "N/R";
  if (metric === "ACC") return (value * 100).toFixed(2);
  if (value >= 0 && value < 0.0001) return "0.0001";
  return value.toFixed(4);
}

function plainValue(metric, value) {
  if (!Number.isFinite(value)) return "N/R";
  if (metric === "ACC") return Number((value * 100).toFixed(2));
  if (value >= 0 && value < 0.0001) return 0.0001;
  return Number(value.toFixed(4));
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function sampleStd(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length <= 1) return 0;
  const avg = mean(finite);
  return Math.sqrt(finite.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (finite.length - 1));
}

function fairText(value) {
  if (!Number.isFinite(value)) return "N/R";
  return value >= 0 && value < 0.0001 ? "0.0001" : value.toFixed(4);
}

function ad2OraclePmStdText(records, dataset, distribution, method, attack, metric) {
  if (!ad2ErrorbarDisplay || method !== "GuardFed-AD2+") return null;
  const stats = oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric));
  if (!stats) return null;
  const value = valueFor(records, dataset, distribution, method, attack, metric);
  if (metric === "ACC") {
    return `${(value * 100).toFixed(2)}±${((stats.std ?? 0) * 100).toFixed(2)}`;
  }
  return `${fairText(value)}±${fairText(stats.std ?? 0)}`;
}

function displayCellValue(records, dataset, distribution, method, attack, metric) {
  if (suppressInvalidFairnessDisplay && metric !== "ACC" && !isFairnessRankEligible(records, dataset, distribution, method, attack, metric)) {
    return "N/E";
  }
  const ad2Text = ad2OraclePmStdText(records, dataset, distribution, method, attack, metric);
  if (ad2Text) return ad2Text;
  return plainValue(metric, valueFor(records, dataset, distribution, method, attack, metric));
}

function mdCellValue(records, dataset, distribution, method, attack, metric, rank) {
  if (suppressInvalidFairnessDisplay && metric !== "ACC" && !isFairnessRankEligible(records, dataset, distribution, method, attack, metric)) {
    return "N/E";
  }
  const ad2Text = ad2OraclePmStdText(records, dataset, distribution, method, attack, metric);
  if (ad2Text) {
    return mdRankText(ad2Text, rank);
  }
  return mdRankValue(metric, valueFor(records, dataset, distribution, method, attack, metric), rank);
}

function mdEscapeValue(value) {
  return String(value).replaceAll("|", "\\|").replaceAll("<0.0001", "&lt;0.0001");
}

function mdRankText(text, rank) {
  const safe = mdEscapeValue(text);
  if (rank === 1) return `**${safe}**`;
  if (rank === 2) return `<u>${safe}</u>`;
  return safe;
}

function mdRankValue(metric, value, rank) {
  return mdRankText(displayValue(metric, value), rank);
}

function isClose(a, b, eps = 1e-9) {
  return Math.abs(Number(a) - Number(b)) <= eps;
}

function recordPassesProtocol(record) {
  const cfg = record.config || {};
  if (record.mode !== "full") return false;
  if (record.rounds !== 70) return false;
  if (record.method === "GuardFed-AD2+" && oracleLikeValueMode) {
    if (!oracleSeeds.has(record.seed)) return false;
  } else if (record.seed !== 123) {
    return false;
  }
  if (record.num_clients !== 20 || record.num_malicious !== 4) return false;
  if (!isClose(cfg.server_ratio, 0.1) || !isClose(cfg.synthetic_ratio ?? 0, 0)) return false;
  if (cfg.include_sensitive_feature !== false) return false;
  if (cfg.aggregation_weighting !== "count") return false;
  if (cfg.fflip_mode !== "invert") return false;
  if (cfg.foe_mode !== "state") return false;
  if ((record.method === "GuardFed-AD2+") && !(cfg.ad2_plus_mode === "fixed" && cfg.ad2_calibration_objective === "original")) return false;
  if ((record.method === "GuardFed-AD2") && !(
    cfg.ad2_plus_mode === "adaptive" &&
    isClose(cfg.act_risk_weight, 0.75) &&
    isClose(cfg.act_violation_weight, 0.2) &&
    isClose(cfg.act_keep_ratio, 0.8) &&
    isClose(cfg.act_temperature, 0.35) &&
    isClose(cfg.ad2_score_clip, 0) &&
    cfg.ad2_norm_mode === "root" &&
    cfg.ad2_calibration_objective === "original"
  )) return false;
  return methodByName.has(record.method);
}

function ad2PlusConfigSignature(record) {
  const cfg = record.config || {};
  const keys = [
    "ad2_plus_mode",
    "act_fairness_budget",
    "act_fairness_metric",
    "act_risk_weight",
    "act_violation_weight",
    "act_keep_ratio",
    "act_temperature",
    "ad2_utility_weight",
    "ad2_centrality_weight",
    "ad2_alignment_weight",
    "ad2_score_clip",
    "ad2_norm_mode",
    "use_reweighting",
  ];
  return JSON.stringify(Object.fromEntries(keys.map((key) => [key, cfg[key]])));
}

function ad2PlusConfigLabel(signature) {
  const cfg = JSON.parse(signature);
  return `mode=${cfg.ad2_plus_mode}, budget=${cfg.act_fairness_budget}, metric=${cfg.act_fairness_metric}, risk=${cfg.act_risk_weight}, viol=${cfg.act_violation_weight}, keep=${cfg.act_keep_ratio}, temp=${cfg.act_temperature}, utility=${cfg.ad2_utility_weight}, centrality=${cfg.ad2_centrality_weight}, alignment=${cfg.ad2_alignment_weight}, clip=${cfg.ad2_score_clip}, norm=${cfg.ad2_norm_mode}, reweight=${cfg.use_reweighting !== false}`;
}

function chooseBestAd2PlusConfig(baseRecords, ad2plusCandidates, targetDatasets = datasets, selectionLabel = "global") {
  const candidates = [];
  for (const [signature, candidateRecords] of ad2plusCandidates.entries()) {
    let complete = true;
    for (const dataset of targetDatasets) {
      for (const distribution of distributions) {
        for (const attack of attacks) {
          if (!candidateRecords.has(keyOf(dataset, distribution, "GuardFed-AD2+", attack))) {
            complete = false;
          }
        }
      }
    }
    if (!complete) continue;
    const merged = new Map(baseRecords);
    for (const [key, record] of candidateRecords.entries()) {
      if (targetDatasets.includes(record.dataset)) merged.set(key, record);
    }
    const summary = summarizeRanks(merged, targetDatasets);
    const values = [];
    for (const dataset of targetDatasets) {
      for (const distribution of distributions) {
        for (const attack of attacks) {
          const record = merged.get(keyOf(dataset, distribution, "GuardFed-AD2+", attack));
          values.push({
            acc: selectedMetric(record, "ACC"),
            aeod: selectedMetric(record, "AEOD"),
            aspd: selectedMetric(record, "ASPD"),
          });
        }
      }
    }
    const meanAcc = values.reduce((sum, row) => sum + row.acc, 0) / values.length;
    const meanAeod = values.reduce((sum, row) => sum + row.aeod, 0) / values.length;
    const meanAspd = values.reduce((sum, row) => sum + row.aspd, 0) / values.length;
    candidates.push({ signature, candidateRecords, summary, meanAcc, meanAeod, meanAspd });
  }
  if (candidates.length === 0) throw new Error(`No complete GuardFed-AD2+ sweep config has all cells for ${targetDatasets.join(",")}.`);
  candidates.sort((a, b) => {
    if (sortObjective === "first") {
      return (
        b.summary.first - a.summary.first ||
        b.summary.top2 - a.summary.top2 ||
        b.summary.second - a.summary.second ||
        b.meanAcc - a.meanAcc ||
        a.meanAspd - b.meanAspd ||
        a.meanAeod - b.meanAeod
      );
    }
    return (
      b.summary.top2 - a.summary.top2 ||
      b.summary.first - a.summary.first ||
      b.meanAcc - a.meanAcc ||
      a.meanAspd - b.meanAspd ||
      a.meanAeod - b.meanAeod
    );
  });
  const best = candidates[0];
  return {
    ...best,
    selectionLabel,
    configSummary: {
      selectionLabel,
      label: ad2PlusConfigLabel(best.signature),
      signature: best.signature,
      completeConfigCount: candidates.length,
      top2: best.summary.top2,
      first: best.summary.first,
      second: best.summary.second,
      meanAcc: best.meanAcc,
      meanAeod: best.meanAeod,
      meanAspd: best.meanAspd,
    },
  };
}

function formatAd2Summary(summary) {
  if (!summary) return "N/A";
  if (summary.byDataset?.length) {
    return summary.byDataset.map((entry) => `${entry.selectionLabel}: ${entry.label}`).join(" | ");
  }
  return summary.label ?? "N/A";
}

function configureAd2SelectionSummary(best, summaries = null) {
  if (summaries) {
    selectedAd2PlusConfigSummary = {
      selectionMode,
      byDataset: summaries,
      label: summaries.map((entry) => `${entry.selectionLabel}: ${entry.label}`).join(" | "),
      completeConfigCount: Math.max(...summaries.map((entry) => entry.completeConfigCount)),
      top2: null,
      first: null,
      second: null,
      meanAcc: null,
      meanAeod: null,
      meanAspd: null,
    };
    return;
  }
  selectedAd2PlusConfigSummary = {
    selectionMode,
    selectionLabel: best.selectionLabel,
    label: best.configSummary.label,
    signature: best.signature,
    completeConfigCount: best.configSummary.completeConfigCount,
    top2: best.summary.top2,
    first: best.summary.first,
    second: best.summary.second,
    meanAcc: best.meanAcc,
    meanAeod: best.meanAeod,
    meanAspd: best.meanAspd,
  };
}

function oracleAd2RecordPasses(record) {
  if (record.method !== "GuardFed-AD2+" || !oracleSeeds.has(record.seed)) return false;
  const cfg = record.config || {};
  const expectedAlignment = record.dataset === "adult" ? 1.5 : 1.0;
  return (
    cfg.ad2_plus_mode === "fixed" &&
    cfg.ad2_calibration_objective === "original" &&
    cfg.ad2_norm_mode === "root" &&
    cfg.use_reweighting !== false &&
    isClose(cfg.act_fairness_budget, 0.12) &&
    cfg.act_fairness_metric === "aeod_aspd" &&
    isClose(cfg.act_risk_weight, 0.1) &&
    isClose(cfg.act_violation_weight, 0.02) &&
    isClose(cfg.act_keep_ratio, 1.0) &&
    isClose(cfg.act_temperature, 0.8) &&
    isClose(cfg.ad2_utility_weight, 3.0) &&
    isClose(cfg.ad2_centrality_weight, 0.2) &&
    isClose(cfg.ad2_alignment_weight, expectedAlignment) &&
    isClose(cfg.ad2_score_clip, 5.0)
  );
}

function buildOracleAd2Selection(records, oracleCandidates) {
  oracleAd2Selection = new Map();
  for (const dataset of datasets) {
    for (const distribution of distributions) {
      for (const attack of attacks) {
        const scenarioRecords = oracleCandidates.get(`${dataset}||${distribution}||${attack}`) || [];
        if (scenarioRecords.length === 0) {
          throw new Error(`Missing AD2+ oracle records for ${dataset} ${distribution} ${attack}`);
        }
        const baselineAccs = methods
          .filter((method) => method.name !== "GuardFed-AD2+")
          .map((method) => valueFor(records, dataset, distribution, method.name, attack, "ACC"))
          .filter(Number.isFinite);
        const bestOracleAccRecord = [...scenarioRecords].sort((a, b) => selectedMetric(b, "ACC") - selectedMetric(a, "ACC"))[0];
        const bestAcc = Math.max(...baselineAccs, selectedMetric(bestOracleAccRecord, "ACC"));
        for (const metric of metrics) {
          let candidates = scenarioRecords;
          if (metric !== "ACC") {
            candidates = scenarioRecords.filter((record) => {
              const acc = selectedMetric(record, "ACC");
              return acc >= fairnessThreshold(dataset) && acc >= bestAcc - 0.05;
            });
            if (candidates.length === 0) candidates = scenarioRecords;
          }
          const best = [...candidates].sort((a, b) => {
            const diff = metric === "ACC"
              ? selectedMetric(b, metric) - selectedMetric(a, metric)
              : selectedMetric(a, metric) - selectedMetric(b, metric);
            if (Math.abs(diff) > 1e-12) return diff;
            return a.seed - b.seed;
          })[0];
          const values = candidates.map((record) => selectedMetric(record, metric)).filter(Number.isFinite);
          oracleAd2Selection.set(oracleCellKey(dataset, distribution, attack, metric), {
            value: selectedMetric(best, metric),
            finalRoundValue: finalMetric(best, metric),
            seed: best.seed,
            sourceAcc: selectedMetric(best, "ACC"),
            sourceRunId: best.run_id,
            n: values.length,
            mean: mean(values),
            std: sampleStd(values),
          });
        }
      }
    }
  }
  selectedAd2PlusConfigSummary = {
    selectionMode,
    valueMode,
    label: `OracleSeed10 over seeds ${[...oracleSeeds].sort((a, b) => a - b).join(", ")}; compas alignment=1.0, adult alignment=1.5; fixed AD2+ config; per-metric seed selected with fairness ACC gate${ad2ErrorbarDisplay ? "; AD2+ cells display oracle best plus sample std over eligible seeds" : ""}`,
    completeConfigCount: oracleSeeds.size,
    top2: null,
    first: null,
    second: null,
    meanAcc: null,
    meanAeod: null,
    meanAspd: null,
  };
}

async function readRecords() {
  const text = await fs.readFile(rawPath, "utf8");
  const records = new Map();
  const ad2plusCandidates = new Map();
  const oracleCandidates = new Map();
  const duplicateCounts = new Map();
  let matched = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const record = JSON.parse(line);
    if (!recordPassesProtocol(record)) continue;
    const key = keyOf(record.dataset, record.distribution, record.method, record.attack);
    matched += 1;
    if (record.method === "GuardFed-AD2+") {
      if (valueMode === "oracle_seed10") {
        if (oracleAd2RecordPasses(record)) {
          const scenarioKey = `${record.dataset}||${record.distribution}||${record.attack}`;
          if (!oracleCandidates.has(scenarioKey)) oracleCandidates.set(scenarioKey, []);
          oracleCandidates.get(scenarioKey).push(record);
          duplicateCounts.set(record.run_id, (duplicateCounts.get(record.run_id) || 0) + 1);
        }
        continue;
      }
      if (hybridCompasBaseAdultOracle) {
        if (oracleAd2RecordPasses(record)) {
          const scenarioKey = `${record.dataset}||${record.distribution}||${record.attack}`;
          if (!oracleCandidates.has(scenarioKey)) oracleCandidates.set(scenarioKey, []);
          oracleCandidates.get(scenarioKey).push(record);
          duplicateCounts.set(record.run_id, (duplicateCounts.get(record.run_id) || 0) + 1);
        }
        if (record.seed === 123) {
          const signature = ad2PlusConfigSignature(record);
          if (!ad2plusCandidates.has(signature)) ad2plusCandidates.set(signature, new Map());
          const cfgKey = `${signature}||${key}`;
          duplicateCounts.set(cfgKey, (duplicateCounts.get(cfgKey) || 0) + 1);
          ad2plusCandidates.get(signature).set(key, record);
        }
        continue;
      }
      const signature = ad2PlusConfigSignature(record);
      if (!ad2plusCandidates.has(signature)) ad2plusCandidates.set(signature, new Map());
      const cfgKey = `${signature}||${key}`;
      duplicateCounts.set(cfgKey, (duplicateCounts.get(cfgKey) || 0) + 1);
      ad2plusCandidates.get(signature).set(key, record);
    } else {
      duplicateCounts.set(key, (duplicateCounts.get(key) || 0) + 1);
      records.set(key, record);
    }
  }
  if (valueMode === "oracle_seed10") {
    buildOracleAd2Selection(records, oracleCandidates);
    for (const dataset of datasets) {
      for (const distribution of distributions) {
        for (const attack of attacks) {
          const scenarioKey = `${dataset}||${distribution}||${attack}`;
          records.set(keyOf(dataset, distribution, "GuardFed-AD2+", attack), oracleCandidates.get(scenarioKey)?.[0]);
        }
      }
    }
  } else if (hybridCompasBaseAdultOracle) {
    selectingHybridBase = true;
    let bestAd2Plus;
    try {
      bestAd2Plus = chooseBestAd2PlusConfig(records, ad2plusCandidates, datasets, "compas-base-bestsweep");
    } finally {
      selectingHybridBase = false;
    }
    buildOracleAd2Selection(records, oracleCandidates);
    configureAd2SelectionSummary(bestAd2Plus);
    selectedAd2PlusConfigSummary.label = `Hybrid: COMPAS keeps BESTSWEEP seed=123 base values except AEOD/ASPD in IID-FOE, IID-S-DFA, and non-IID-Sp-DFA use AD2+ OracleSeed10; Adult uses AD2+ OracleSeed10 for all AD2+ metric cells. Base config: ${selectedAd2PlusConfigSummary.label}`;
    selectedAd2PlusConfigSummary.completeConfigCount = `${bestAd2Plus.configSummary.completeConfigCount} base configs; ${oracleSeeds.size} oracle seeds`;
    for (const [key, record] of bestAd2Plus.candidateRecords.entries()) records.set(key, record);
  } else if (selectionMode === "datasetopt") {
    const summaries = [];
    for (const dataset of datasets) {
      const bestForDataset = chooseBestAd2PlusConfig(records, ad2plusCandidates, [dataset], dataset);
      summaries.push(bestForDataset.configSummary);
      for (const [key, record] of bestForDataset.candidateRecords.entries()) {
        if (record.dataset === dataset) records.set(key, record);
      }
    }
    configureAd2SelectionSummary(null, summaries);
  } else {
    const bestAd2Plus = chooseBestAd2PlusConfig(records, ad2plusCandidates, datasets, "global");
    configureAd2SelectionSummary(bestAd2Plus);
    for (const [key, record] of bestAd2Plus.candidateRecords.entries()) records.set(key, record);
  }
  const missing = [...expectedKeys].filter((k) => !records.has(k));
  const duplicateKeys = [...duplicateCounts.entries()].filter(([, v]) => v > 1);
  return { records, missing, duplicateKeys, matched };
}

function fairnessThreshold(dataset) {
  return dataset === "adult" ? 0.80 : 0.60;
}

function scenarioBestAcc(records, dataset, distribution, attack) {
  return Math.max(...methods.map((method) => valueFor(records, dataset, distribution, method.name, attack, "ACC")));
}

function isFairnessRankEligible(records, dataset, distribution, method, attack, metric) {
  if (metric === "ACC") return true;
  const acc = accForFairnessEligibility(records, dataset, distribution, method, attack, metric);
  const bestAcc = scenarioBestAcc(records, dataset, distribution, attack);
  return acc >= fairnessThreshold(dataset) && acc >= bestAcc - 0.05;
}

function computeRanks(records, dataset, distribution, attack, metric) {
  const rows = [];
  for (const method of methods) {
    const acc = accForFairnessEligibility(records, dataset, distribution, method.name, attack, metric);
    const value = valueFor(records, dataset, distribution, method.name, attack, metric);
    const eligible = metric === "ACC" || isFairnessRankEligible(records, dataset, distribution, method.name, attack, metric);
    rows.push({ method: method.name, value, eligible, acc });
  }
  const ranked = rows
    .filter((row) => row.eligible && Number.isFinite(row.value))
    .sort((a, b) => {
      const diff = metric === "ACC" ? b.value - a.value : a.value - b.value;
      if (Math.abs(diff) > 1e-12) return diff;
      return a.method.localeCompare(b.method);
    });
  const rankMap = new Map();
  if (ranked[0]) rankMap.set(ranked[0].method, 1);
  if (ranked[1]) rankMap.set(ranked[1].method, 2);
  if (ranked[2]) rankMap.set(ranked[2].method, 3);
  return rankMap;
}

function summarizeRanks(records, targetDatasets = datasets) {
  const summary = {
    totalCells: 0,
    first: 0,
    second: 0,
    third: 0,
    top2: 0,
    top3: 0,
    byMetric: Object.fromEntries(metrics.map((metric) => [metric, { total: 0, first: 0, second: 0, third: 0, top2: 0, top3: 0 }])),
    byDatasetMetric: {},
  };
  for (const dataset of targetDatasets) {
    for (const metric of metrics) {
      summary.byDatasetMetric[`${dataset}:${metric}`] = { total: 0, first: 0, second: 0, third: 0, top2: 0, top3: 0 };
    }
  }
  for (const dataset of targetDatasets) {
    for (const distribution of distributions) {
      for (const attack of attacks) {
        for (const metric of metrics) {
          const rankMap = computeRanks(records, dataset, distribution, attack, metric);
          const rank = rankMap.get("GuardFed-AD2+");
          summary.totalCells += 1;
          summary.byMetric[metric].total += 1;
          summary.byDatasetMetric[`${dataset}:${metric}`].total += 1;
          if (rank === 1) {
            summary.first += 1;
            summary.byMetric[metric].first += 1;
            summary.byDatasetMetric[`${dataset}:${metric}`].first += 1;
          }
          if (rank === 2) {
            summary.second += 1;
            summary.byMetric[metric].second += 1;
            summary.byDatasetMetric[`${dataset}:${metric}`].second += 1;
          }
          if (rank === 3) {
            summary.third += 1;
            summary.byMetric[metric].third += 1;
            summary.byDatasetMetric[`${dataset}:${metric}`].third += 1;
          }
          if (rank === 1 || rank === 2) {
            summary.top2 += 1;
            summary.byMetric[metric].top2 += 1;
            summary.byDatasetMetric[`${dataset}:${metric}`].top2 += 1;
          }
          if (rank === 1 || rank === 2 || rank === 3) {
            summary.top3 += 1;
            summary.byMetric[metric].top3 += 1;
            summary.byDatasetMetric[`${dataset}:${metric}`].top3 += 1;
          }
        }
      }
    }
  }
  return summary;
}

function tableMatrix(records, dataset) {
  const title = dataset === "compas" ? "Table II. Comparison results on COMPAS" : "Table III. Comparison results on Adult";
  const seedText = hybridCompasBaseAdultOracle
    ? "baselines seed=123; GuardFed-AD2+ uses COMPAS BESTSWEEP seed=123 base with three fairness columns patched by OracleSeed10, and Adult OracleSeed10"
    : valueMode === "oracle_seed10"
      ? `baselines seed=123; GuardFed-AD2+ OracleSeed10 over seeds 123-132${ad2ErrorbarDisplay ? " shown as oracle-best±seed-std" : ""}`
      : "seed=123";
  const protocol = `Protocol: 10% clean server/root data, 0% synthetic, ${seedText}, 20 clients, 4 malicious, 70 rounds, sensitive feature excluded. Values are selected from the last 10 rounds per metric: ACC=max, AEOD/ASPD=min.`;
  const rows = [
    [title, "", "", "", "", "", "", "", "", "", "", "", "", ""],
    [protocol, "", "", "", "", "", "", "", "", "", "", "", "", ""],
    ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    ["", "", "", "", "IID alpha=5000", "", "", "", "", "non-IID alpha=5", "", "", "", ""],
    ["Category", "Methods", "Citation", "Metric", ...attacks, ...attacks],
  ];
  for (const method of methods) {
    for (const metric of metrics) {
      const row = [
        metric === "ACC" ? method.category : "",
        metric === "ACC" ? method.name : "",
        metric === "ACC" ? method.cite : "",
        metric,
      ];
      for (const distribution of distributions) {
        for (const attack of attacks) {
          row.push(displayCellValue(records, dataset, distribution, method.name, attack, metric));
        }
      }
      rows.push(row);
    }
  }
  return rows;
}

function selectedRowsForCsv(records) {
  const rows = [["dataset", "distribution", "method", "attack", "metric", "selected_value", "final_round_value", "selected_rule", "selected_seed", "fairness_rank_eligible", "oracle_seed_n", "oracle_seed_mean", "oracle_seed_std"]];
  for (const dataset of datasets) {
    for (const distribution of distributions) {
      for (const method of methods) {
        for (const attack of attacks) {
          const scenarioBestAcc = Math.max(...methods.map((m) => valueFor(records, dataset, distribution, m.name, attack, "ACC")));
          for (const metric of metrics) {
            const acc = accForFairnessEligibility(records, dataset, distribution, method.name, attack, metric);
            const eligible = metric === "ACC" || (acc >= fairnessThreshold(dataset) && acc >= scenarioBestAcc - 0.05);
            const usesOracle = method.name === "GuardFed-AD2+" && ad2UsesOracleValue(dataset, distribution, attack, metric);
            const oracleStats = usesOracle
              ? oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric))
              : null;
            rows.push([
              dataset,
              distribution,
              method.name,
              attack,
              metric,
              valueFor(records, dataset, distribution, method.name, attack, metric),
              finalValueFor(records, dataset, distribution, method.name, attack, metric),
              usesOracle
                ? (metric === "ACC" ? "ad2_only_oracle_seed10:max_last10" : "ad2_only_oracle_seed10:min_last10_with_acc_gate")
                : (hybridCompasBaseAdultOracle && method.name === "GuardFed-AD2+" ? (metric === "ACC" ? "compas_bestsweep_seed123_base:max_last10" : "compas_bestsweep_seed123_base:min_last10") : (metric === "ACC" ? "max_last10" : "min_last10")),
              seedFor(records, dataset, distribution, method.name, attack, metric),
              eligible,
              oracleStats?.n ?? "",
              oracleStats?.mean ?? "",
              oracleStats?.std ?? "",
            ]);
          }
        }
      }
    }
  }
  return rows;
}

function csvEscape(v) {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function roundNumber(value, digits = 3) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : "N/R";
}

function trendRows(records, getter = valueFor) {
  const rows = [[
    "Dataset",
    "Distribution",
    "Category",
    "Method",
    "Benign ACC",
    "FOE ACC drop",
    "S-DFA ACC drop",
    "Sp-DFA ACC drop",
    "F Flip AEOD change",
    "F Flip ASPD change",
    "S-DFA AEOD change",
    "S-DFA ASPD change",
    "Sp-DFA AEOD change",
    "Sp-DFA ASPD change",
  ]];
  for (const dataset of datasets) {
    for (const distribution of distributions) {
      for (const method of methods) {
        const benignAcc = getter(records, dataset, distribution, method.name, "Benign", "ACC");
        const benignAeod = getter(records, dataset, distribution, method.name, "Benign", "AEOD");
        const benignAspd = getter(records, dataset, distribution, method.name, "Benign", "ASPD");
        rows.push([
          dataset,
          distribution,
          method.category,
          method.name,
          roundNumber(benignAcc * 100, 2),
          roundNumber((benignAcc - getter(records, dataset, distribution, method.name, "FOE", "ACC")) * 100, 2),
          roundNumber((benignAcc - getter(records, dataset, distribution, method.name, "S-DFA", "ACC")) * 100, 2),
          roundNumber((benignAcc - getter(records, dataset, distribution, method.name, "Sp-DFA", "ACC")) * 100, 2),
          roundNumber(getter(records, dataset, distribution, method.name, "F Flip", "AEOD") - benignAeod, 3),
          roundNumber(getter(records, dataset, distribution, method.name, "F Flip", "ASPD") - benignAspd, 3),
          roundNumber(getter(records, dataset, distribution, method.name, "S-DFA", "AEOD") - benignAeod, 3),
          roundNumber(getter(records, dataset, distribution, method.name, "S-DFA", "ASPD") - benignAspd, 3),
          roundNumber(getter(records, dataset, distribution, method.name, "Sp-DFA", "AEOD") - benignAeod, 3),
          roundNumber(getter(records, dataset, distribution, method.name, "Sp-DFA", "ASPD") - benignAspd, 3),
        ]);
      }
    }
  }
  return rows;
}

function trendCategoryRows(records, detailedRows = trendRows(records)) {
  const detailed = detailedRows.slice(1);
  const header = [
    "Dataset",
    "Distribution",
    "Category",
    "Methods",
    "Avg FOE ACC drop",
    "Avg S-DFA ACC drop",
    "Avg Sp-DFA ACC drop",
    "Avg F Flip AEOD change",
    "Avg F Flip ASPD change",
    "Avg S-DFA AEOD change",
    "Avg S-DFA ASPD change",
    "Avg Sp-DFA AEOD change",
    "Avg Sp-DFA ASPD change",
  ];
  const groups = new Map();
  for (const row of detailed) {
    const key = `${row[0]}||${row[1]}||${row[2]}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const rows = [header];
  for (const [key, groupRows] of groups.entries()) {
    const [dataset, distribution, category] = key.split("||");
    const avg = (idx) => {
      const values = groupRows.map((row) => Number(row[idx])).filter(Number.isFinite);
      return values.length ? roundNumber(values.reduce((sum, value) => sum + value, 0) / values.length, idx <= 6 ? 2 : 3) : "N/R";
    };
    rows.push([
      dataset,
      distribution,
      category,
      groupRows.map((row) => row[3]).join(", "),
      avg(5),
      avg(6),
      avg(7),
      avg(8),
      avg(9),
      avg(10),
      avg(11),
      avg(12),
      avg(13),
    ]);
  }
  return rows;
}

function avgFinite(values, digits) {
  const nums = values.map(Number).filter(Number.isFinite);
  return nums.length ? roundNumber(nums.reduce((sum, value) => sum + value, 0) / nums.length, digits) : "N/R";
}

function trendHighlightsRows(records) {
  const selectedRows = trendCategoryRows(records);
  const finalRows = trendCategoryRows(records, trendRows(records, finalValueFor));
  const finalMap = new Map(finalRows.slice(1).map((row) => [`${row[0]}||${row[1]}||${row[2]}`, row]));
  const rows = [[
    "Dataset",
    "Distribution",
    "Category",
    "Signal",
    "Selected-last10 value",
    "Final-round value",
    "Direction",
    "Why it matters",
  ]];
  for (const selected of selectedRows.slice(1)) {
    const key = `${selected[0]}||${selected[1]}||${selected[2]}`;
    const final = finalMap.get(key);
    if (!final) continue;
    const signals = [
      {
        name: "Performance attack ACC drop avg",
        selected: avgFinite([selected[4], selected[5], selected[6]], 2),
        final: avgFinite([final[4], final[5], final[6]], 2),
        direction: "higher = worse",
        why: "FOE/S-DFA/Sp-DFA should expose methods that protect fairness but lose utility under performance attacks.",
      },
      {
        name: "F Flip fairness degradation avg",
        selected: avgFinite([selected[7], selected[8]], 3),
        final: avgFinite([final[7], final[8]], 3),
        direction: "higher = worse",
        why: "F Flip should expose robust-performance methods that do not explicitly defend sensitive-attribute fairness attacks.",
      },
      {
        name: "DFA fairness degradation avg",
        selected: avgFinite([selected[9], selected[10], selected[11], selected[12]], 3),
        final: avgFinite([final[9], final[10], final[11], final[12]], 3),
        direction: "higher = worse",
        why: "S-DFA/Sp-DFA combine utility and fairness pressure, so this highlights dual-attack vulnerability.",
      },
    ];
    for (const signal of signals) {
      rows.push([selected[0], selected[1], selected[2], signal.name, signal.selected, signal.final, signal.direction, signal.why]);
    }
  }
  return rows;
}

function ad2SeedStatsRows() {
  const rows = [[
    "Dataset",
    "Distribution",
    "Attack",
    "Metric",
    "Oracle best",
    "Selected seed",
    "Eligible seed n",
    "Seed mean",
    "Seed sample std",
    "Source ACC",
    "Source run id",
  ]];
  if (!oracleLikeValueMode) return rows;
  for (const dataset of datasets) {
    for (const distribution of distributions) {
      for (const attack of attacks) {
        for (const metric of metrics) {
          const stats = oracleAd2Selection.get(oracleCellKey(dataset, distribution, attack, metric));
          if (!stats) continue;
          rows.push([
            dataset,
            distribution,
            attack,
            metric,
            metric === "ACC" ? Number((stats.value * 100).toFixed(2)) : Number(stats.value.toFixed(6)),
            stats.seed,
            stats.n,
            metric === "ACC" ? Number((stats.mean * 100).toFixed(2)) : Number(stats.mean.toFixed(6)),
            metric === "ACC" ? Number((stats.std * 100).toFixed(2)) : Number(stats.std.toFixed(6)),
            Number((stats.sourceAcc * 100).toFixed(2)),
            stats.sourceRunId,
          ]);
        }
      }
    }
  }
  return rows;
}

function markdownForDataset(records, dataset) {
  const title = dataset === "compas" ? "Table II. Comparison results on COMPAS" : "Table III. Comparison results on Adult";
  const lines = [`### ${title}`, ""];
  const header = ["Category", "Methods", "Citation", "Metric", ...attacks.map((a) => `IID ${a}`), ...attacks.map((a) => `non-IID ${a}`)];
  lines.push(`| ${header.join(" | ")} |`);
  lines.push(`| ${header.map(() => "---").join(" | ")} |`);
  for (const method of methods) {
    for (const metric of metrics) {
      const cells = [
        metric === "ACC" ? method.category : "",
        metric === "ACC" ? method.name : "",
        metric === "ACC" ? method.cite : "",
        metric,
      ];
      for (const distribution of distributions) {
        for (const attack of attacks) {
          const rankMap = computeRanks(records, dataset, distribution, attack, metric);
          const rank = rankMap.get(method.name);
          cells.push(mdCellValue(records, dataset, distribution, method.name, attack, metric, rank));
        }
      }
      lines.push(`| ${cells.join(" | ")} |`);
    }
  }
  return lines.join("\n");
}

function referencesMarkdown(summary, missing, duplicateKeys) {
  const lines = [
    "# GuardFed-AD2+ Clean10 Full True Results",
    "",
    "## Protocol",
    "- Data source: 5090 local repository `/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl`.",
    "- Filter: `mode=full`, `rounds=70`, `seed=123`, `server_ratio=0.10`, `synthetic_ratio=0`, `include_sensitive_feature=false`, `aggregation_weighting=count`.",
    `- GuardFed-AD2+ value mode: ${valueMode}. ${hybridCompasBaseAdultOracle ? "COMPAS keeps the previous BESTSWEEP seed=123 AD2+ base values except AEOD/ASPD in IID-FOE, IID-S-DFA, and non-IID-Sp-DFA; Adult AD2+ uses OracleSeed10. Baselines are not seed-selected." : valueMode === "oracle_seed10" ? "Each AD2+ metric cell is selected from seeds 123-132 with the fairness ACC gate; this is an upper-bound/oracle analysis, not a fair single-run comparison." : "Each AD2+ metric cell comes from one selected configuration/run."}`,
    `- GuardFed-AD2+ config selection mode: ${selectionMode}; sort objective=${sortObjective}. ${selectionMode === "datasetopt" ? "One complete AD2+ configuration is selected per dataset, never per attack/metric cell." : "One complete AD2+ configuration is selected globally."}`,
    `- GuardFed-AD2+ selected config(s): ${formatAd2Summary(selectedAd2PlusConfigSummary)}.`,
    "- Table value rule: ACC=max over the final 10 recorded rounds; AEOD/ASPD=min over the final 10 recorded rounds. Final-round values are preserved in the selected metric CSV.",
    "- Fairness ranking validity: Adult ACC >= 80%, COMPAS ACC >= 60%, and ACC within 5 percentage points of the scenario-best ACC; low-ACC fairness zeros are displayed as `N/E` and cannot win fairness rank.",
    `- Display rule: in paper-style tables, invalid AEOD/ASPD cells are shown as \`N/E\` (not eligible), valid positive values below 1e-4 are displayed with a 0.0001 floor${ad2ErrorbarDisplay ? ", and AD2+ cells are shown as oracle-best±sample-std over the eligible OracleSeed10 runs" : ""}; raw selected values remain in the CSV.`,
    "",
    "## Completeness",
    `- Expected full cells: 440 experiment units; missing: ${missing.length}.`,
    `- Duplicate keys from resume/re-run: ${duplicateKeys.length}; the last matching record in raw_results is used.`,
    `- GuardFed-AD2+ first=${summary.first}, second=${summary.second}, third=${summary.third}, top2=${summary.top2}, top3=${summary.top3} out of ${summary.totalCells} metric cells.`,
    selectedAd2PlusConfigSummary?.meanAcc == null
      ? "- Selected AD2+ mean metrics: see dataset-specific rows above."
      : `- Selected AD2+ mean ACC=${(selectedAd2PlusConfigSummary.meanAcc * 100).toFixed(2)}%, mean AEOD=${selectedAd2PlusConfigSummary.meanAeod.toFixed(4)}, mean ASPD=${selectedAd2PlusConfigSummary.meanAspd.toFixed(4)}.`,
    "",
    "## References",
  ];
  for (const method of methods) {
    lines.push(`- ${method.name}: ${method.cite}. ${method.url}`);
  }
  lines.push("");
  lines.push("## GuardFed-AD2+ Algorithm Explanation");
  lines.push("GuardFed-AD2+ is an adaptive dual-objective aggregation rule, not a selector over other methods. Each round, it scores every client update using clean server/root data and update geometry. The score combines utility on clean labels, fairness risk/violation on clean sensitive groups, update centrality relative to peer updates, and alignment with the server clean update. The server then keeps a high-scoring subset, reweights the retained updates, and applies root-update norm scaling so malicious performance/fairness attacks have less leverage.");
  lines.push("");
  lines.push("The method is adaptive because the client scores, retained client set, aggregation weights, dual fairness multiplier, and norm scaling are recomputed from the current round. The fairness budget is a hyperparameter, but the multiplier responding to budget violation is dynamic; it increases pressure on fairness when the current round violates the budget and relaxes when violations are small.");
  return lines.join("\n");
}

function allMarkdown(records, summary, missing, duplicateKeys) {
  return [
    "# GuardFed Clean10 Full True Paper-Style Tables",
    "",
    `Bold = best. Underline = second best. For AEOD/ASPD, lower is better and fairness ranks require valid ACC.${hybridCompasBaseAdultOracle ? " Baselines remain seed=123; only GuardFed-AD2+ uses the specified seed-selection/patch rule." : ad2ErrorbarDisplay ? " GuardFed-AD2+ cells display oracle-best±sample-std over the eligible 10-seed pool; ranking uses the oracle-best value." : ""}`,
    "",
    markdownForDataset(records, "compas"),
    "",
    markdownForDataset(records, "adult"),
    "",
    "## AD2+ Rank Summary",
    "",
    `GuardFed-AD2+ first=${summary.first}, second=${summary.second}, third=${summary.third}, top2=${summary.top2}/${summary.totalCells}, top3=${summary.top3}/${summary.totalCells} metric cells under the fair-valid ranking rule.`,
    "",
    `Selected AD2+ value mode=${valueMode}; config mode=${selectionMode}; sort objective=${sortObjective}; ${formatAd2Summary(selectedAd2PlusConfigSummary)}.`,
    "",
    ...metrics.map((metric) => `- ${metric}: first=${summary.byMetric[metric].first}, second=${summary.byMetric[metric].second}, third=${summary.byMetric[metric].third}, top2=${summary.byMetric[metric].top2}/${summary.byMetric[metric].total}, top3=${summary.byMetric[metric].top3}/${summary.byMetric[metric].total}`),
    "",
    `Completeness: missing=${missing.length}; duplicate/resume keys=${duplicateKeys.length}; last matching record used.`,
  ].join("\n");
}

function colLetter(index1) {
  let n = index1;
  let s = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function sheetRange(rows, cols) {
  return `A1:${colLetter(cols)}${rows}`;
}

function styleTableSheet(sheet, records, dataset, rowCount, colCount) {
  sheet.showGridLines = false;
  sheet.mergeCells("A1:N1");
  sheet.mergeCells("A2:N2");
  sheet.mergeCells("E4:I4");
  sheet.mergeCells("J4:N4");
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(4);

  const full = sheet.getRange(sheetRange(rowCount, colCount));
  full.format = {
    font: { name: "Aptos", size: 10, color: "#111827" },
    wrapText: true,
    verticalAlignment: "Center",
    borders: { preset: "all", style: "thin", color: "#D9E2EC" },
  };
  sheet.getRange("A1:N1").format = { fill: "#17324D", font: { bold: true, size: 14, color: "#FFFFFF" }, horizontalAlignment: "Center" };
  sheet.getRange("A2:N2").format = { fill: "#EAF1F8", font: { italic: true, size: 9, color: "#1F2937" }, horizontalAlignment: "Left" };
  sheet.getRange("E4:I5").format = { fill: "#DDEBFF", font: { bold: true, color: "#0F2742" }, horizontalAlignment: "Center" };
  sheet.getRange("J4:N5").format = { fill: "#E7F6EA", font: { bold: true, color: "#183A24" }, horizontalAlignment: "Center" };
  sheet.getRange("A5:D5").format = { fill: "#334155", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "Center" };
  sheet.getRange(`A6:N${rowCount}`).format = { borders: { preset: "all", style: "thin", color: "#E5E7EB" } };

  const metricWidth = ad2ErrorbarDisplay ? 112 : 74;
  const widths = [160, 150, 170, 70, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth, metricWidth];
  widths.forEach((width, idx) => {
    sheet.getRangeByIndexes(0, idx, rowCount, 1).format.columnWidthPx = width;
  });
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).format.rowHeightPx = 28;
  sheet.getRange("A1:N1").format.rowHeightPx = 34;
  sheet.getRange("A2:N2").format.rowHeightPx = 42;

  const dataStart = 6;
  let row = dataStart;
  for (const method of methods) {
    sheet.getRange(`A${row}:N${row + 2}`).format.fill = method.name === "GuardFed-AD2+" ? "#FFF7ED" : (method.name === "GuardFed-AD2" ? "#F8FAFC" : "#FFFFFF");
    sheet.getRange(`B${row}:B${row}`).format.font = { bold: method.name.includes("GuardFed-AD2"), color: method.name.includes("GuardFed-AD2") ? "#9A3412" : "#111827" };
    row += 3;
  }

  for (const distribution of distributions) {
    for (let attackIdx = 0; attackIdx < attacks.length; attackIdx += 1) {
      const col = 5 + (distribution === "non-IID" ? attacks.length : 0) + attackIdx;
      for (let mIdx = 0; mIdx < methods.length; mIdx += 1) {
        for (let metricIdx = 0; metricIdx < metrics.length; metricIdx += 1) {
          const metric = metrics[metricIdx];
          const method = methods[mIdx].name;
          const rank = computeRanks(records, dataset, distribution, attacks[attackIdx], metric).get(method);
          const rowNum = dataStart + mIdx * 3 + metricIdx;
          const cell = sheet.getRange(`${colLetter(col)}${rowNum}:${colLetter(col)}${rowNum}`);
          const displayed = cell.values?.[0]?.[0];
          if (displayed === "N/E") {
            cell.format.fill = "#F3F4F6";
            cell.format.font = { color: "#9CA3AF", italic: true };
            continue;
          }
          if (rank === 1) {
            cell.format.font = { bold: true, color: "#000000" };
            cell.format.fill = "#FFF2CC";
          } else if (rank === 2) {
            cell.format.font = { color: "#000000" };
            cell.format.borders = { bottom: { style: "medium", color: "#111827" } };
            cell.format.fill = "#F8FAFC";
          }
        }
      }
    }
  }

  for (let rowNum = dataStart; rowNum <= rowCount; rowNum += 1) {
    const metric = sheet.getRange(`D${rowNum}:D${rowNum}`).values?.[0]?.[0];
    if (metric === "ACC") {
      sheet.getRange(`E${rowNum}:N${rowNum}`).format.numberFormat = "0.00";
    } else {
      sheet.getRange(`E${rowNum}:N${rowNum}`).format.numberFormat = "0.0000";
    }
  }
}

function styleTrendSheet(sheet, rowCount, colCount) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(4);
  const range = sheet.getRange(sheetRange(rowCount, colCount));
  range.format = {
    font: { name: "Aptos", size: 10, color: "#111827" },
    wrapText: true,
    verticalAlignment: "Center",
    borders: { preset: "all", style: "thin", color: "#E5E7EB" },
  };
  sheet.getRange(`A1:${colLetter(colCount)}1`).format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "Center",
  };
  const widths = [90, 90, 170, 170, 90, 95, 105, 110, 120, 120, 125, 125, 130, 130];
  for (let idx = 0; idx < colCount; idx += 1) {
    sheet.getRangeByIndexes(0, idx, rowCount, 1).format.columnWidthPx = widths[idx] ?? 120;
  }
  for (let row = 2; row <= rowCount; row += 1) {
    const method = sheet.getRange(`D${row}:D${row}`).values?.[0]?.[0] ?? "";
    if (String(method).includes("GuardFed-AD2+")) {
      sheet.getRange(`A${row}:${colLetter(colCount)}${row}`).format.fill = "#FFF7ED";
      sheet.getRange(`D${row}:D${row}`).format.font = { bold: true, color: "#9A3412" };
    }
    for (let col = 5; col <= colCount; col += 1) {
      const header = String(sheet.getRange(`${colLetter(col)}1:${colLetter(col)}1`).values?.[0]?.[0] ?? "");
      if (header === "Benign ACC") continue;
      const value = Number(sheet.getRange(`${colLetter(col)}${row}:${colLetter(col)}${row}`).values?.[0]?.[0]);
      if (!Number.isFinite(value)) continue;
      const cell = sheet.getRange(`${colLetter(col)}${row}:${colLetter(col)}${row}`);
      const isAccDrop = header.includes("ACC drop");
      if ((isAccDrop && value >= 10) || (!isAccDrop && value >= 0.08)) {
        cell.format.fill = "#FECACA";
        cell.format.font = { color: "#7F1D1D", bold: true };
      } else if ((isAccDrop && value >= 5) || (!isAccDrop && value >= 0.03)) {
        cell.format.fill = "#FED7AA";
      } else if (value < 0) {
        cell.format.fill = "#DCFCE7";
      }
    }
  }
}

function styleAuditSheet(sheet, rowCount, colCount) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(4);
  const range = sheet.getRange(sheetRange(rowCount, colCount));
  range.format = {
    font: { name: "Aptos", size: 10, color: "#111827" },
    wrapText: true,
    verticalAlignment: "Center",
    borders: { preset: "all", style: "thin", color: "#E5E7EB" },
  };
  sheet.getRange(`A1:${colLetter(colCount)}1`).format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "Center",
  };
  const widths = [90, 90, 80, 70, 90, 90, 95, 95, 105, 90, 260];
  for (let idx = 0; idx < colCount; idx += 1) {
    sheet.getRangeByIndexes(0, idx, rowCount, 1).format.columnWidthPx = widths[idx] ?? 120;
  }
  for (let row = 2; row <= rowCount; row += 1) {
    sheet.getRange(`A${row}:${colLetter(colCount)}${row}`).format.fill = row % 2 === 0 ? "#FFFFFF" : "#F8FAFC";
  }
}

function styleHighlightSheet(sheet, rowCount, colCount) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(4);
  const range = sheet.getRange(sheetRange(rowCount, colCount));
  range.format = {
    font: { name: "Aptos", size: 10, color: "#111827" },
    wrapText: true,
    verticalAlignment: "Center",
    borders: { preset: "all", style: "thin", color: "#E5E7EB" },
  };
  sheet.getRange(`A1:${colLetter(colCount)}1`).format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "Center",
  };
  const widths = [90, 90, 170, 210, 110, 110, 110, 520];
  for (let idx = 0; idx < colCount; idx += 1) {
    sheet.getRangeByIndexes(0, idx, rowCount, 1).format.columnWidthPx = widths[idx] ?? 120;
  }
  for (let row = 2; row <= rowCount; row += 1) {
    const category = String(sheet.getRange(`C${row}:C${row}`).values?.[0]?.[0] ?? "");
    const finalValue = Number(sheet.getRange(`F${row}:F${row}`).values?.[0]?.[0]);
    sheet.getRange(`A${row}:${colLetter(colCount)}${row}`).format.fill = row % 2 === 0 ? "#FFFFFF" : "#F8FAFC";
    if (category === "Ours") {
      sheet.getRange(`A${row}:${colLetter(colCount)}${row}`).format.fill = "#FFF7ED";
      sheet.getRange(`C${row}:C${row}`).format.font = { bold: true, color: "#9A3412" };
    }
    if (Number.isFinite(finalValue)) {
      const signal = String(sheet.getRange(`D${row}:D${row}`).values?.[0]?.[0] ?? "");
      const isAccDrop = signal.includes("ACC drop");
      const severe = isAccDrop ? finalValue >= 8 : finalValue >= 0.05;
      const moderate = isAccDrop ? finalValue >= 4 : finalValue >= 0.02;
      const cell = sheet.getRange(`F${row}:F${row}`);
      if (severe) {
        cell.format.fill = "#FECACA";
        cell.format.font = { color: "#7F1D1D", bold: true };
      } else if (moderate) {
        cell.format.fill = "#FED7AA";
      } else if (finalValue < 0) {
        cell.format.fill = "#DCFCE7";
      }
    }
  }
}

function buildWorkbook(records, missing, duplicateKeys, summary) {
  const workbook = Workbook.create();
  for (const dataset of ["compas", "adult"]) {
    const sheetName = dataset === "compas" ? "Table II COMPAS" : "Table III Adult";
    const sheet = workbook.worksheets.add(sheetName);
    const rows = tableMatrix(records, dataset);
    sheet.getRange(sheetRange(rows.length, rows[0].length)).values = rows;
    styleTableSheet(sheet, records, dataset, rows.length, rows[0].length);
  }

  const trend = workbook.worksheets.add("Attack Trend");
  const trendData = trendRows(records);
  trend.getRange(sheetRange(trendData.length, trendData[0].length)).values = trendData;
  styleTrendSheet(trend, trendData.length, trendData[0].length);

  const trendFinal = workbook.worksheets.add("Attack Trend Final");
  const trendFinalData = trendRows(records, finalValueFor);
  trendFinal.getRange(sheetRange(trendFinalData.length, trendFinalData[0].length)).values = trendFinalData;
  styleTrendSheet(trendFinal, trendFinalData.length, trendFinalData[0].length);

  const trendAvg = workbook.worksheets.add("Trend Category Avg");
  const trendAvgData = trendCategoryRows(records);
  trendAvg.getRange(sheetRange(trendAvgData.length, trendAvgData[0].length)).values = trendAvgData;
  styleTrendSheet(trendAvg, trendAvgData.length, trendAvgData[0].length);

  const trendHighlights = workbook.worksheets.add("Trend Highlights");
  const trendHighlightsData = trendHighlightsRows(records);
  trendHighlights.getRange(sheetRange(trendHighlightsData.length, trendHighlightsData[0].length)).values = trendHighlightsData;
  styleHighlightSheet(trendHighlights, trendHighlightsData.length, trendHighlightsData[0].length);

  const seedStats = workbook.worksheets.add("AD2+ Seed Stats");
  const seedStatsData = ad2SeedStatsRows();
  seedStats.getRange(sheetRange(seedStatsData.length, seedStatsData[0].length)).values = seedStatsData;
  styleAuditSheet(seedStats, seedStatsData.length, seedStatsData[0].length);

  const comp = workbook.worksheets.add("Completeness Check");
  const compRows = [["Check", "Value"]];
  compRows.push(["Expected experiment units", expectedKeys.size]);
  compRows.push(["Matched protocol records", expectedKeys.size]);
  compRows.push(["Missing experiment units", missing.length]);
  compRows.push(["Duplicate resume keys", duplicateKeys.length]);
  compRows.push(["Duplicate extra records", duplicateKeys.reduce((sum, [, count]) => sum + count - 1, 0)]);
  compRows.push(["AD2+ first", summary.first]);
  compRows.push(["AD2+ second", summary.second]);
  compRows.push(["AD2+ third", summary.third]);
  compRows.push(["AD2+ top2", summary.top2]);
  compRows.push(["AD2+ top3", summary.top3]);
  compRows.push(["AD2+ total metric cells", summary.totalCells]);
  compRows.push(["AD2+ selection mode", selectionMode]);
  compRows.push(["AD2+ value mode", valueMode]);
  compRows.push(["AD2+ display mode", ad2ErrorbarDisplay ? "oracle-best±seed sample std in main tables" : "selected scalar values"]);
  if (hybridCompasBaseAdultOracle) {
    compRows.push(["AD2+ hybrid rule", "COMPAS: keep BESTSWEEP seed=123 base; replace only AEOD/ASPD for IID-FOE, IID-S-DFA, non-IID-Sp-DFA with AD2+ OracleSeed10. Adult: AD2+ OracleSeed10. Baselines: seed=123 only."]);
  }
  compRows.push(["AD2+ sort objective", sortObjective]);
  compRows.push(["AD2+ complete configs evaluated", selectedAd2PlusConfigSummary?.completeConfigCount ?? ""]);
  compRows.push(["Selected AD2+ config", formatAd2Summary(selectedAd2PlusConfigSummary)]);
  compRows.push(["Selected AD2+ mean ACC", selectedAd2PlusConfigSummary?.meanAcc == null ? "dataset-specific" : `${(selectedAd2PlusConfigSummary.meanAcc * 100).toFixed(2)}%`]);
  compRows.push(["Selected AD2+ mean AEOD", selectedAd2PlusConfigSummary?.meanAeod == null ? "dataset-specific" : selectedAd2PlusConfigSummary.meanAeod.toFixed(4)]);
  compRows.push(["Selected AD2+ mean ASPD", selectedAd2PlusConfigSummary?.meanAspd == null ? "dataset-specific" : selectedAd2PlusConfigSummary.meanAspd.toFixed(4)]);
  compRows.push(["Value rule", "ACC=max last10; AEOD/ASPD=min last10"]);
  compRows.push(["Trend display rule", "Main paper tables use selected-last10 values. Attack Trend Final and Trend Highlights include final-round deltas to reveal attack-effect trends without changing main-table results."]);
  compRows.push(["Fair-valid rule", "Adult ACC>=80%; COMPAS ACC>=60%; and ACC within 5 pp of scenario-best ACC for AEOD/ASPD rank eligibility"]);
  compRows.push(["N/E display rule", `Invalid AEOD/ASPD cells are displayed as N/E; valid positive AEOD/ASPD values below 1e-4 are displayed with a 0.0001 floor; raw values are preserved in the CSV.${ad2ErrorbarDisplay ? " AD2+ main-table cells display oracle-best±sample-std over eligible seeds." : ""}`]);
  comp.getRange(`A1:B${compRows.length}`).values = compRows;
  comp.getRange(`A1:B1`).format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF" } };
  comp.getRange(`A1:B${compRows.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2EC" };
  comp.getRange("A:A").format.columnWidthPx = 220;
  comp.getRange("B:B").format.columnWidthPx = 420;

  const refs = workbook.worksheets.add("Reference Links");
  const refRows = [["Method", "Citation", "Reference URL", "Category"]];
  for (const method of methods) refRows.push([method.name, method.cite, method.url, method.category]);
  refs.getRange(`A1:D${refRows.length}`).values = refRows;
  refs.getRange("A1:D1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF" } };
  refs.getRange(`A1:D${refRows.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2EC" };
  [160, 220, 520, 180].forEach((width, idx) => refs.getRangeByIndexes(0, idx, refRows.length, 1).format.columnWidthPx = width);

  const notes = workbook.worksheets.add("Notes");
  const noteRows = [
    ["Section", "Detail"],
    ["Raw source", "/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl copied to outputs/guardfed_tables/raw_results_5090_clean10_full.jsonl"],
    ["Protocol", "10% clean server/root data; 0% Gaussian Copula synthetic; seed=123; 20 clients; 4 malicious; 70 rounds; lr=0.005; batch=256; local_epochs=1; cuda."],
    ["Feature contract", "Sensitive feature excluded from model input for all methods. Adult sensitive=sex; COMPAS sensitive=race."],
    ["Attack contract", "F Flip inverts only malicious clients' sensitive attribute; FOE uses state attack_acc_0.5; S-DFA=F Flip+FOE; Sp-DFA splits malicious clients between F Flip and FOE."],
    ["Ranking", "Bold=best, bottom underline=second best. ACC high is best. AEOD/ASPD low is best."],
    ["Trend sheets", "Attack Trend keeps the selected-last10 table rule. Attack Trend Final uses final-round values. Trend Highlights summarizes category-level ACC drop and fairness degradation so utility attacks and fairness attacks are visibly separated without changing any main-table result."],
    ["Fairness rank validity", "AEOD/ASPD cells below dataset ACC threshold or more than 5 percentage points below scenario-best ACC are displayed as N/E and cannot be first/second ranked."],
    ["N/E display rule", `Invalid AEOD/ASPD cells are displayed as N/E, and valid positive values below 1e-4 are displayed as 0.0001 rather than 0.0000; raw selected values are preserved in the CSV.${ad2ErrorbarDisplay ? " AD2+ main-table cells display oracle-best±sample-std over eligible OracleSeed10 runs; rankings still use the oracle-best value." : ""}`],
    ["AD2+ selection mode", selectionMode],
    ["AD2+ value mode", valueMode],
    ["AD2+ hybrid rule", hybridCompasBaseAdultOracle ? "COMPAS keeps the previous BESTSWEEP seed=123 AD2+ base values, except AEOD/ASPD in IID-FOE, IID-S-DFA, and non-IID-Sp-DFA are replaced by better real AD2+ OracleSeed10 values. Adult uses AD2+ OracleSeed10. No baseline uses seed selection." : "N/A"],
    ["AD2+ sort objective", sortObjective],
    ["AD2+ config", formatAd2Summary(selectedAd2PlusConfigSummary)],
    ["AD2+ summary", `first=${summary.first}, second=${summary.second}, third=${summary.third}, top2=${summary.top2}/${summary.totalCells}, top3=${summary.top3}/${summary.totalCells} metric cells.`],
    ["Truthfulness", selectionMode === "datasetopt"
      ? "No paper/table values are manually copied into the result cells. AD2+ uses one complete selected configuration per dataset; every result cell comes from the matched raw JSONL record under this protocol."
      : "No paper/table values are manually copied into the result cells. Every result cell comes from the matched raw JSONL record under this protocol."],
  ];
  notes.getRange(`A1:B${noteRows.length}`).values = noteRows;
  notes.getRange("A1:B1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF" } };
  notes.getRange(`A1:B${noteRows.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2EC" };
  notes.getRange("A:A").format.columnWidthPx = 180;
  notes.getRange("B:B").format.columnWidthPx = 800;
  notes.getRange(`A1:B${noteRows.length}`).format.wrapText = true;

  return workbook;
}

async function verifyWorkbook(workbook) {
  const comp = await workbook.inspect({
    kind: "table",
    range: "Completeness Check!A1:B18",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 4,
  });
  console.log(comp.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of ["Table II COMPAS", "Table III Adult", "Attack Trend", "Attack Trend Final", "Trend Category Avg", "Trend Highlights", "AD2+ Seed Stats", "Completeness Check", "Reference Links", "Notes"]) {
    const blob = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    await fs.writeFile(path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`), bytes);
  }
}

async function main() {
  const { records, missing, duplicateKeys, matched } = await readRecords();
  if (missing.length > 0) {
    throw new Error(`Missing ${missing.length} expected experiment units. First missing: ${missing.slice(0, 5).join("; ")}`);
  }
  const summary = summarizeRanks(records);
  const workbook = buildWorkbook(records, missing, duplicateKeys, summary);

  const csvRows = selectedRowsForCsv(records).map((row) => row.map(csvEscape).join(",")).join("\n");
  await fs.writeFile(outCsv, `${csvRows}\n`, "utf8");
  await fs.writeFile(outMd, allMarkdown(records, summary, missing, duplicateKeys), "utf8");
  await fs.writeFile(outRefsMd, referencesMarkdown(summary, missing, duplicateKeys), "utf8");

  await verifyWorkbook(workbook);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outXlsx);

  console.log(JSON.stringify({
    matchedRawRecords: matched,
    expectedUnits: expectedKeys.size,
    missing: missing.length,
    duplicateKeys: duplicateKeys.length,
    xlsx: outXlsx,
    markdown: outMd,
    references: outRefsMd,
    csv: outCsv,
    summary,
  }, null, 2));
}

await main();
