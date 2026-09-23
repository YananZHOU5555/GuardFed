import fs from "node:fs/promises";
const text = await fs.readFile("E:/OneDrive/文档/GuardFed/results/attack_strength/raw_results.jsonl", "utf8");
const lines = text.trim().split(/\r?\n/).filter(Boolean);
let ok = 0, ratio = 0, bad = 0, sample = null, badSample = null;
for (const line of lines) {
  try {
    const x = JSON.parse(line);
    ok++;
    if (x.mode === "ratio" && ["S-DFA", "Sp-DFA"].includes(x.attack)) { ratio++; sample ??= { mode: x.mode, study: x.study, attack: x.attack }; }
  } catch (e) { bad++; badSample ??= { start: line.slice(0, 120), error: String(e) }; }
}
console.log(JSON.stringify({ chars: text.length, lines: lines.length, ok, ratio, bad, sample, badSample }));
