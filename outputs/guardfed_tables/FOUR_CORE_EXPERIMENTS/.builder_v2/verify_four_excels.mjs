import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const OUT = "E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS/excel_delivery_v2";
const files = [
  "GuardFed_AD2plus_01_Ablation_v2_RealRuns.xlsx",
  "GuardFed_AD2plus_02_Server_Distribution_ControlledSkew_v2.xlsx",
  "GuardFed_AD2plus_03_Synthetic_10pct_v2.xlsx",
  "GuardFed_AD2plus_04_New_Performance_Attack_FedSA.xlsx",
];

for (const file of files) {
  const full = path.join(OUT, file);
  const input = await FileBlob.load(full);
  const wb = await SpreadsheetFile.importXlsx(input);
  const sheetNames = wb.worksheets.items.map(ws => ws.name);
  const scan = await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    maxChars: 1000,
  });
  const first = sheetNames[0];
  const png = await wb.render({ sheetName: first, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(OUT, `qa_${file.replace(/\.xlsx$/i, "")}_${first}.png`), new Uint8Array(await png.arrayBuffer()));
  console.log(JSON.stringify({
    file,
    sheets: sheetNames.length,
    first_sheet: first,
    formula_scan: scan.ndjson.includes("matched 0 entries") ? "ok" : "check",
  }));
}
