import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve("..");
const outputPath = path.join(root, "GuardFed_AD2plus_advisor_experiments_true_results.xlsx");

const csvSheets = [
  ["Adult Ablation Raw", "adult_ad2plus_ablation.csv"],
  ["Adult Ablation Summary", "adult_ablation_by_tag_summary.csv"],
  ["Server Dist Raw", "server_distribution_ablation.csv"],
  ["Server Dist Trend", "server_distribution_trend_by_alpha.csv"],
  ["Synthetic Raw", "server_generation_ablation.csv"],
  ["Synthetic Overall", "server_generation_overall_summary.csv"],
  ["Synthetic By Dataset", "server_generation_by_tag_dataset_summary.csv"],
];

const workbook = Workbook.create();

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(coerceCell(cell));
      cell = "";
    } else if (ch === "\n") {
      row.push(coerceCell(cell));
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(coerceCell(cell));
    rows.push(row);
  }
  const width = Math.max(...rows.map((r) => r.length));
  return rows.map((r) => {
    const out = r.slice();
    while (out.length < width) out.push(null);
    return out;
  });
}

function coerceCell(value) {
  if (value === "") return null;
  if (/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(value)) return Number(value);
  return value;
}

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1:H1").merge();
readme.getRange("A1").values = [["GuardFed-AD2+ Advisor Experiments - True Results"]];
readme.getRange("A1").format = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};
readme.getRange("A3:B13").values = [
  ["Generated from", "5090 /home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl"],
  ["Metric selection", "Last 10 rounds: ACC=max, AEOD=min, ASPD=min"],
  ["Adult ablation rows", 160],
  ["Server distribution rows", 200],
  ["Synthetic generation rows", 220],
  ["New performance attack label", "FedSA in raw files; bounded clean-direction performance attack used as FOE replacement"],
  ["Server root size", "10% for distribution ablation; synthetic ablation uses real10, real1+synth9, real5+synth5"],
  ["Methods held fixed", "GuardFed-AD2+ only in advisor ablations"],
  ["Warning", "No values are manually edited; all table values are derived from raw JSONL summaries"],
  ["Created", new Date().toISOString()],
  ["Workspace", "E:/OneDrive/文档/GuardFed"],
];
readme.getRange("A3:A13").format = { font: { bold: true }, fill: "#EEF4FF" };
readme.getRange("A3:B13").format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
readme.getRange("A:B").format.autofitColumns();

const refs = workbook.worksheets.add("References");
refs.showGridLines = false;
refs.getRange("A1:D1").values = [["Item", "Used for", "Citation", "URL"]];
refs.getRange("A1:D1").format = {
  fill: "#2E74B5",
  font: { bold: true, color: "#FFFFFF" },
};
refs.getRange("A2:D9").values = [
  ["CTGAN", "Synthetic server data generator", "Xu et al., NeurIPS 2019", "https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan"],
  ["TVAE", "Synthetic server data generator", "Xu et al., NeurIPS 2019 CTGAN paper includes TVAE baseline", "https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf"],
  ["SMOTE", "Interpolation-based tabular oversampling baseline", "Chawla et al., JAIR 2002", "https://www.jair.org/index.php/jair/article/view/10302"],
  ["Gaussian Copula", "Classical copula-based tabular generator", "Gaussian copula family / SDV-style empirical implementation", "https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/gaussiancopulasynthesizer"],
  ["ForestDiffusion", "Diffusion/flow-matching tabular generator", "Jolicoeur-Martineau et al., AISTATS 2024", "https://github.com/SamsungSAILMontreal/ForestDiffusion"],
  ["PCA-Gaussian", "Classical low-rank Gaussian control baseline", "Implemented as a reproducible parametric control baseline", "N/A"],
  ["PoisonedFL", "Recent model-poisoning reference for FOE replacement discussion", "Xie et al., CVPR 2025 / arXiv 2024", "https://arxiv.org/abs/2404.15611"],
  ["EAB-FL", "Recent fairness attack context", "IJCAI 2024", "https://www.ijcai.org/proceedings/2024/51"],
];
refs.getRange("A1:D9").format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
refs.getRange("A:D").format.autofitColumns();

for (const [sheetName, fileName] of csvSheets) {
  const csvText = await fs.readFile(path.join(root, fileName), "utf8");
  const rows = parseCsv(csvText);
  const sheet = workbook.worksheets.add(sheetName);
  if (rows.length > 0 && rows[0].length > 0) {
    sheet.getRangeByIndexes(0, 0, rows.length, rows[0].length).values = rows;
  }
  sheet.showGridLines = false;
  const used = sheet.getUsedRange(true);
  used.format.borders = { preset: "all", style: "thin", color: "#E5E7EB" };
  const header = sheet.getRange("A1:Z1");
  header.format = {
    fill: "#2E74B5",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.freezePanes.freezeRows(1);
  used.format.autofitColumns();
  used.format.autofitRows();
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Adult Ablation Summary",
  range: "A1:M18",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(root, "GuardFed_AD2plus_advisor_experiments_preview.png"), new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
