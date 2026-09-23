import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const outputXlsx = path.join(here, "GuardFed_AD2plus_Expanded_FedSA_TrueResults_FINAL.xlsx");

const csvSheets = [
  ["fedsa_paper_style_selected_ad2plus_display.csv", "FedSA_Table"],
  ["fedsa_paper_style_selected_ad2plus_raw.csv", "FedSA_Raw"],
  ["fedsa_ad2plus_candidate_compact_ranking.csv", "FedSA_Ranking"],
  ["synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv", "Synthetic_v4_Dataset"],
  ["synthetic_stratified_clean_cap10_dense_v4_by_slice.csv", "Synthetic_v4_Slices"],
  ["server_dist30_strongfloor_v4_by_alpha_dataset.csv", "ServerDist30_Strong"],
  ["server_dist30_strongfloor_v4_curve_diagnostics.csv", "CurveDiag_Strong"],
];

function stripMarkup(value) {
  if (typeof value !== "string") return { text: value, bold: false, underline: false };
  let text = value;
  let bold = false;
  let underline = false;
  if (text.startsWith("**") && text.endsWith("**")) {
    bold = true;
    text = text.slice(2, -2);
  }
  if (text.startsWith("<u>") && text.endsWith("</u>")) {
    underline = true;
    text = text.slice(3, -4);
  }
  return { text, bold, underline };
}

function colLetter(index) {
  let n = index + 1;
  let s = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

async function addCsvSheet(workbook, fileName, sheetName) {
  const csvText = await fs.readFile(path.join(here, fileName), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
  return workbook.worksheets.getItem(sheetName);
}

function styleUsedSheet(sheet, options = {}) {
  const used = sheet.getUsedRange(true);
  if (!used) return;
  used.format.font = { name: "Arial", size: 10, color: "#111827" };
  used.format.borders = { preset: "all", style: "thin", color: "#D9DEE8" };
  const values = used.values;
  const rows = values.length;
  const cols = values[0]?.length || 0;
  if (rows === 0 || cols === 0) return;
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF", size: 10 },
    wrapText: true,
  };
  sheet.freezePanes.freezeRows(1);
  if (options.freezeCols) sheet.freezePanes.freezeColumns(options.freezeCols);
  for (let c = 0; c < cols; c++) {
    const width = c < 3 ? [18, 30, 10][c] || 14 : 16;
    sheet.getRangeByIndexes(0, c, rows, 1).format.columnWidth = width;
  }
  used.format.wrapText = true;
  used.format.rowHeight = 21;
}

function cleanFedSATable(sheet) {
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const rows = values.length;
  const cols = values[0].length;
  const cleaned = values.map((row) => row.map((cell) => stripMarkup(cell).text));
  used.values = cleaned;

  const lastCol = colLetter(cols - 1);
  sheet.getRange(`A1:${lastCol}1`).format.rowHeight = 30;
  sheet.getRange(`A1:${lastCol}${rows}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9DEE8" },
    insideVertical: { style: "thin", color: "#D9DEE8" },
    top: { style: "medium", color: "#111827" },
    bottom: { style: "medium", color: "#111827" },
    left: { style: "thin", color: "#D9DEE8" },
    right: { style: "thin", color: "#D9DEE8" },
  };

  for (let r = 1; r < rows; r++) {
    for (let c = 3; c < cols; c++) {
      const parsed = stripMarkup(values[r][c]);
      if (parsed.bold || parsed.underline) {
        const cell = sheet.getRangeByIndexes(r, c, 1, 1);
        cell.format.font = {
          bold: parsed.bold,
          underline: parsed.underline,
          color: parsed.bold ? "#000000" : "#1D4ED8",
        };
        if (parsed.bold) {
          cell.format.fill = "#D9EAD3";
        } else if (parsed.underline) {
          cell.format.fill = "#EAF2FF";
          cell.format.borders = {
            bottom: { style: "medium", color: "#1D4ED8" },
          };
        }
      }
    }
  }
}

function addFigures(workbook) {
  const sheet = workbook.worksheets.add("Figures");
  sheet.showGridLines = false;
  sheet.getRange("A1:H1").merge();
  sheet.getRange("A1").values = [["Server/root distribution curves"]];
  sheet.getRange("A1").format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };

  const src = workbook.worksheets.getItem("ServerDist30_Strong");
  const values = src.getUsedRange(true).values;
  const header = values[0];
  const idx = Object.fromEntries(header.map((name, i) => [String(name), i]));
  const byAlpha = new Map();
  for (const row of values.slice(1)) {
    const dataset = row[idx.dataset];
    const alpha = Number(row[idx.server_alpha]);
    if (!byAlpha.has(alpha)) byAlpha.set(alpha, { alpha });
    const item = byAlpha.get(alpha);
    item[`${dataset}_acc`] = Number(row[idx.acc_mean]);
    item[`${dataset}_score`] = Number(row[idx.score_mean]);
    item[`${dataset}_aeod`] = Number(row[idx.aeod_mean]);
    item[`${dataset}_aspd`] = Number(row[idx.aspd_mean]);
  }
  const rows = Array.from(byAlpha.values()).sort((a, b) => a.alpha - b.alpha);
  const matrix = [
    ["alpha", "Adult ACC", "COMPAS ACC", "Adult score", "COMPAS score", "Adult AEOD", "Adult ASPD", "COMPAS AEOD", "COMPAS ASPD"],
    ...rows.map((r) => [
      r.alpha,
      r.adult_acc,
      r.compas_acc,
      r.adult_score,
      r.compas_score,
      r.adult_aeod,
      r.adult_aspd,
      r.compas_aeod,
      r.compas_aspd,
    ]),
  ];
  sheet.getRangeByIndexes(2, 0, matrix.length, matrix[0].length).values = matrix;
  const dataRange = sheet.getRangeByIndexes(2, 0, matrix.length, matrix[0].length);
  dataRange.format.borders = { preset: "all", style: "thin", color: "#D9DEE8" };
  sheet.getRangeByIndexes(2, 0, 1, matrix[0].length).format = {
    fill: "#EAF2FF",
    font: { bold: true, color: "#17324D" },
  };
  sheet.getRange("A3:I33").format.numberFormat = "0.000";
  sheet.getRange("A3:I33").format.columnWidth = 13;

  const scoreMatrix = [
    ["alpha", "Adult score", "COMPAS score"],
    ...rows.map((r) => [r.alpha, r.adult_score, r.compas_score]),
  ];
  sheet.getRangeByIndexes(35, 0, scoreMatrix.length, scoreMatrix[0].length).values = scoreMatrix;
  sheet.getRange("A36:C66").format.numberFormat = "0.000";

  const fairMatrix = [
    ["alpha", "Adult AEOD", "Adult ASPD", "COMPAS AEOD", "COMPAS ASPD"],
    ...rows.map((r) => [r.alpha, r.adult_aeod, r.adult_aspd, r.compas_aeod, r.compas_aspd]),
  ];
  sheet.getRangeByIndexes(68, 0, fairMatrix.length, fairMatrix[0].length).values = fairMatrix;
  sheet.getRange("A69:E99").format.numberFormat = "0.000";

  const accChart = sheet.charts.add("line", sheet.getRange("A3:C33"));
  accChart.title = "Server/root distribution: ACC";
  accChart.hasLegend = true;
  accChart.xAxis = { axisType: "textAxis", tickLabelInterval: 4, textStyle: { fontSize: 8 } };
  accChart.yAxis = { numberFormatCode: "0.00" };
  accChart.setPosition("K3", "Z20");

  const scoreChart = sheet.charts.add("line", sheet.getRange("A36:C66"));
  scoreChart.title = "Server/root distribution: joint score";
  scoreChart.hasLegend = true;
  scoreChart.xAxis = { axisType: "textAxis", tickLabelInterval: 4, textStyle: { fontSize: 8 } };
  scoreChart.yAxis = { numberFormatCode: "0.00" };
  scoreChart.setPosition("K22", "Z39");

  const fairChart = sheet.charts.add("line", sheet.getRange("A69:E99"));
  fairChart.title = "Server/root distribution: fairness gaps";
  fairChart.hasLegend = true;
  fairChart.xAxis = { axisType: "textAxis", tickLabelInterval: 4, textStyle: { fontSize: 8 } };
  fairChart.yAxis = { numberFormatCode: "0.00" };
  fairChart.setPosition("K41", "Z58");
}

function addNotes(workbook) {
  const sheet = workbook.worksheets.add("Notes");
  sheet.showGridLines = false;
  const rows = [
    ["GuardFed-AD2+ expanded experiment workbook"],
    [""],
    ["FedSA_Table", "Paper-style FedSA-only table. Bold = best; underline/blue fill = second."],
    ["FedSA_Raw", "Unformatted numeric values for the displayed FedSA table."],
    ["FedSA_Ranking", "All completed baseline/candidate rows with score, ACC, AEOD, ASPD ranks."],
    ["Synthetic_v4_Dataset", "Stratified clean-root synthetic/ratio ablation by dataset; 10% clean is the target cap setting."],
    ["Synthetic_v4_Slices", "Synthetic/root-data ablation by dataset, distribution, and attack."],
    ["ServerDist30_Strong", "Strong-floor 30 Dirichlet-alpha server/root distribution ablation, 3 seeds."],
    ["CurveDiag_Strong", "Curve ranges and adjacent jump diagnostics for strong-floor sampling."],
    ["Figures", "Rendered ACC, joint-score, and fairness curves."],
    [""],
    ["Important", "All values are copied from completed raw_results.jsonl experiment rows. No table cell is hand-edited."],
    ["AD2+ FedSA profile sources", "Adult IID/non-IID: b007_cal002_q81; COMPAS IID: b006_cal002_q81; COMPAS non-IID: b008_cal003_q81."],
    ["Fairness validity", "Adult AEOD/ASPD ranks require ACC >= 80%; COMPAS ranks require ACC >= 60%."],
  ];
  sheet.getRangeByIndexes(0, 0, rows.length, 2).values = rows;
  sheet.getRange("A1:B1").merge();
  sheet.getRange("A1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF", size: 14 } };
  sheet.getRange("A3:A14").format.font = { bold: true, color: "#17324D" };
  sheet.getRange("A1:B14").format.wrapText = true;
  sheet.getRange("A1:B14").format.borders = { preset: "all", style: "thin", color: "#D9DEE8" };
  sheet.getRange("A1:A14").format.columnWidth = 28;
  sheet.getRange("B1:B14").format.columnWidth = 110;
}

const workbook = Workbook.create();

for (const [fileName, sheetName] of csvSheets) {
  const sheet = await addCsvSheet(workbook, fileName, sheetName);
  styleUsedSheet(sheet, { freezeCols: sheetName === "FedSA_Table" ? 3 : 0 });
  if (sheetName === "FedSA_Table") cleanFedSATable(sheet);
}

addNotes(workbook);
addFigures(workbook);

const inspect = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 3000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of [
  "FedSA_Table",
  "FedSA_Raw",
  "FedSA_Ranking",
  "Synthetic_v4_Dataset",
  "Synthetic_v4_Slices",
  "ServerDist30_Strong",
  "CurveDiag_Strong",
  "Notes",
  "Figures",
]) {
  const previewConfig = sheetName === "Figures"
    ? { sheetName, range: "A1:Z100", scale: 1, format: "png" }
    : { sheetName, autoCrop: "all", scale: 1, format: "png" };
  const preview = await workbook.render(previewConfig);
  await fs.writeFile(
    path.join(here, `preview_${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputXlsx);
console.log(`saved=${outputXlsx}`);
