import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outDir = process.env.GUARDFED_OUT_DIR;
if (!outDir) throw new Error("GUARDFED_OUT_DIR is required");

const payload = JSON.parse(await fs.readFile(path.join(outDir, "workbook_payload.json"), "utf8"));
const workbook = Workbook.create();

function normalize(matrix) {
  const width = Math.max(...matrix.map((row) => row.length));
  return matrix.map((row) => row.concat(Array(width - row.length).fill(null)));
}

function writeMatrix(sheetName, matrix, options = {}) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const values = normalize(matrix);
  sheet.getRangeByIndexes(0, 0, values.length, values[0].length).values = values;
  const used = sheet.getRangeByIndexes(0, 0, values.length, values[0].length);
  used.format.font = { color: "#111827" };
  used.format.wrapText = true;
  used.format.borders = { preset: "all", style: "thin", color: "#D9DEE7" };
  if (options.freezeRows) sheet.freezePanes.freezeRows(options.freezeRows);
  if (options.headerRow !== undefined) {
    const header = sheet.getRangeByIndexes(options.headerRow, 0, 1, values[0].length);
    header.format.fill = { color: "#E5E7EB" };
    header.format.font = { bold: true, color: "#111827" };
    header.format.borders = { preset: "all", style: "medium", color: "#6B7280" };
  }
  for (let col = 0; col < values[0].length; col++) {
    sheet.getRangeByIndexes(0, col, values.length, 1).format.columnWidthPx = options.widths?.[col] ?? 105;
  }
  return { sheet, values };
}

const readme = writeMatrix("ReadMe", payload.readme, { widths: [260, 720] });
readme.sheet.getRange("A1:B1").merge();
readme.sheet.getRange("A1:B1").format.fill = { color: "#1F4E79" };
readme.sheet.getRange("A1:B1").format.font = { bold: true, color: "#FFFFFF" };
readme.sheet.getRange("A3:B3").format.fill = { color: "#E5E7EB" };
readme.sheet.getRange("A3:B3").format.font = { bold: true };

const paper = writeMatrix("PaperStyle_FedSA", payload.paper, {
  freezeRows: 4,
  headerRow: 3,
  widths: [170, 145, 180, 70, 105, 115, 110, 120],
});
paper.sheet.getRange("A1:H1").merge();
paper.sheet.getRange("A2:H2").merge();
paper.sheet.getRange("A3:H3").merge();
paper.sheet.getRange("A1:H1").format.fill = { color: "#FCEFD2" };
paper.sheet.getRange("A1:H1").format.font = { bold: true, color: "#111827" };
paper.sheet.getRange("A2:H3").format.fill = { color: "#FFF7D6" };
paper.sheet.getRange("A2:H3").format.font = { italic: true, color: "#4B5563" };
paper.sheet.getRange("E4:H4").format.fill = { color: "#EEF2F7" };

for (let r = 4; r < payload.paper.length; r++) {
  const metric = payload.paper[r][3];
  if (metric === "ACC") {
    paper.sheet.getRangeByIndexes(r, 0, 1, 8).format.borders = {
      top: { style: "medium", color: "#111827" },
      insideVertical: { style: "thin", color: "#D9DEE7" },
      bottom: { style: "thin", color: "#D9DEE7" },
    };
  }
  for (let c = 4; c < 8; c++) {
    const mark = payload.marks[r]?.[c];
    if (mark === "best") {
      const cell = paper.sheet.getRangeByIndexes(r, c, 1, 1);
      cell.format.fill = { color: "#DDEFD5" };
      cell.format.font = { bold: true, color: "#111827" };
    } else if (mark === "second") {
      const cell = paper.sheet.getRangeByIndexes(r, c, 1, 1);
      cell.format.fill = { color: "#EAF2FF" };
      cell.format.font = { color: "#0B57D0" };
      cell.format.borders = {
        bottom: { style: "medium", color: "#0B57D0" },
        top: { style: "thin", color: "#D9DEE7" },
        left: { style: "thin", color: "#D9DEE7" },
        right: { style: "thin", color: "#D9DEE7" },
      };
    }
  }
}

writeMatrix("AD2plus_Ranks", payload.rank_rows, {
  freezeRows: 1,
  headerRow: 0,
  widths: [110, 110, 85, 85, 85, 85, 85, 85, 85, 85],
});

writeMatrix("Synthetic_Top15", payload.synthetic_top, {
  freezeRows: 1,
  headerRow: 0,
  widths: [95, 70, 245, 85, 85, 85, 85, 85, 90, 90, 65],
});

writeMatrix("FedSA_JointSummary", payload.fed_summary, {
  freezeRows: 1,
  headerRow: 0,
  widths: [85, 95, 145, 55, 85, 85, 85, 85, 85, 85, 90, 90, 80, 80, 80, 80, 110],
});

writeMatrix("Synthetic_ByDataset", payload.synthetic_dataset, {
  freezeRows: 1,
  headerRow: 0,
  widths: [85, 245, 55, 85, 85, 85, 85, 85, 85, 90, 90, 105, 105, 105, 105],
});

writeMatrix("Synthetic_BySlice", payload.synthetic_slice, {
  freezeRows: 1,
  headerRow: 0,
  widths: [85, 95, 85, 245, 55, 85, 85, 85, 90, 100, 100, 105, 105],
});

writeMatrix("ServerDist30_Joint", payload.server_dist, {
  freezeRows: 1,
  headerRow: 0,
  widths: [85, 95, 55, 85, 85, 85, 85, 85, 85, 90, 90],
});

const plots = writeMatrix("ServerDist_Plots", [
  ["Server/root distribution sensitivity plots"],
  ["Plot file", "What it shows", "Local path"],
  [
    "server_distribution_acc_curve.png",
    "ACC as clean server/root distribution changes over 30 Dirichlet alpha settings.",
    path.join(outDir, "server_distribution_acc_curve.png"),
  ],
  [
    "server_distribution_fairness_curve.png",
    "AEOD/ASPD as clean server/root distribution changes over 30 Dirichlet alpha settings.",
    path.join(outDir, "server_distribution_fairness_curve.png"),
  ],
  [
    "Why images are separate",
    "The xlsx keeps the audited table values and references the plot PNG files. Separate PNG files avoid any image-rendering loss when the workbook is opened or re-exported.",
    outDir,
  ],
], { headerRow: 1, widths: [285, 460, 700] });
plots.sheet.getRange("A1:C1").merge();
plots.sheet.getRange("A1:C1").format.fill = { color: "#1F4E79" };
plots.sheet.getRange("A1:C1").format.font = { bold: true, color: "#FFFFFF" };

const inspect = await workbook.inspect({
  kind: "workbook,sheet",
  maxChars: 5000,
  tableMaxRows: 4,
  tableMaxCols: 6,
});
await fs.writeFile(path.join(outDir, "workbook_inspect.txt"), inspect.ndjson ?? String(inspect), "utf8");

const preview1 = await workbook.render({ sheetName: "PaperStyle_FedSA", range: "A1:H25", scale: 1, format: "png" });
await fs.writeFile(path.join(outDir, "preview_PaperStyle_FedSA.png"), new Uint8Array(await preview1.arrayBuffer()));
const preview2 = await workbook.render({ sheetName: "Synthetic_Top15", range: "A1:K25", scale: 1, format: "png" });
await fs.writeFile(path.join(outDir, "preview_Synthetic_Top15.png"), new Uint8Array(await preview2.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outDir, "GuardFed_AD2plus_Expanded_TrueResults_JointLast10.xlsx"));
console.log(path.join(outDir, "GuardFed_AD2plus_Expanded_TrueResults_JointLast10.xlsx"));
