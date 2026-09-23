# GuardFed-AD2 Clean-10% Main Results

Protocol: 70 rounds, 20 clients, 4 malicious clients, local epoch 1, lr 0.005, batch 256, root/server data = 10% clean, no synthetic. This table uses the fairness-first GuardFed-AD2 mode: root-norm scaling + original adaptive calibration objective.

GuardFed-AD2 fairness rank: first/second in 13/16 AEOD/ASPD cells, first in 7/16 cells.

Ranking: ACC higher is better; AEOD/ASPD lower is better. Best is **bold**, second best is <u>underlined</u>. [nr] means the fairness value is not ranked because ACC is more than 10 percentage points below the best ACC in that scenario.

## ADULT
| Method | Source | IID S-DFA<br>ACC / AEOD / ASPD | IID Sp-DFA<br>ACC / AEOD / ASPD | non-IID S-DFA<br>ACC / AEOD / ASPD | non-IID Sp-DFA<br>ACC / AEOD / ASPD |
| --- | --- | --- | --- | --- | --- |
| Fed-NGA | [NeurIPS'25](https://neurips.cc/virtual/2025/poster/118753) | 82.02 / 0.150 / <u>0.052</u> | 82.75 / 0.093 / 0.070 | 81.25 / 0.060 / 0.067 | 81.85 / 0.091 / 0.067 |
| Huber-BRFL | [AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | **83.54** / 0.060 / 0.110 | <u>83.19</u> / <u>0.040</u> / 0.134 | **83.98** / 0.032 / 0.125 | **83.98** / 0.012 / 0.142 |
| LoGoFair | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559) | 75.53 / <u>0.023</u> / **0.002** | 76.52 / 0.080 / **0.002** | 75.54 / 0.020 / **0.002** | 77.31 / 0.095 / **0.012** |
| AdaAggRL | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888) | 82.84 / 0.076 / 0.113 | 83.00 / 0.058 / 0.124 | 83.49 / <u>0.018</u> / 0.128 | <u>83.75</u> / <u>0.009</u> / 0.139 |
| FedAMM | [TIFS'25](https://doi.org/10.1109/TIFS.2025.3607273) | 83.11 / 0.067 / 0.119 | 83.01 / 0.053 / 0.127 | 83.62 / 0.028 / 0.122 | 83.70 / **0.003** / 0.139 |
| FedAA | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | <u>83.31</u> / 0.036 / 0.129 | **83.37** / 0.069 / 0.113 | <u>83.98</u> / **0.002** / 0.120 | 83.75 / 0.055 / 0.122 |
| GuardFed-AD2 | Ours | 82.08 / **0.009** / 0.070 | 82.22 / **0.028** / <u>0.066</u> | 82.11 / 0.042 / <u>0.066</u> | 81.99 / 0.031 / <u>0.067</u> |

## COMPAS
| Method | Source | IID S-DFA<br>ACC / AEOD / ASPD | IID Sp-DFA<br>ACC / AEOD / ASPD | non-IID S-DFA<br>ACC / AEOD / ASPD | non-IID Sp-DFA<br>ACC / AEOD / ASPD |
| --- | --- | --- | --- | --- | --- |
| Fed-NGA | [NeurIPS'25](https://neurips.cc/virtual/2025/poster/118753) | **67.55** / 0.214 / 0.238 | 65.39 / 0.193 / 0.213 | **67.28** / 0.197 / 0.234 | 66.41 / 0.188 / 0.221 |
| Huber-BRFL | [AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | 65.93 / 0.206 / <u>0.228</u> | 66.14 / 0.197 / 0.225 | <u>66.95</u> / 0.221 / 0.237 | 66.63 / 0.221 / 0.232 |
| LoGoFair | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559) | 55.40 / 0.002[nr] / 0.001[nr] | 63.55 / **0.060** / <u>0.045</u> | 55.67 / 0.002[nr] / 0.001[nr] | 62.42 / **0.061** / **0.001** |
| AdaAggRL | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888) | <u>66.31</u> / 0.204 / 0.231 | **66.36** / 0.196 / 0.225 | 65.77 / 0.205 / <u>0.223</u> | 66.63 / 0.220 / 0.235 |
| FedAMM | [TIFS'25](https://doi.org/10.1109/TIFS.2025.3607273) | 66.04 / <u>0.198</u> / 0.234 | 66.20 / 0.191 / 0.224 | 66.41 / 0.209 / 0.225 | <u>66.79</u> / 0.230 / 0.241 |
| FedAA | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | 66.25 / 0.212 / 0.242 | <u>66.31</u> / 0.203 / 0.246 | 66.95 / <u>0.192</u> / 0.238 | **67.22** / 0.183 / 0.239 |
| GuardFed-AD2 | Ours | 63.88 / **0.101** / **0.018** | 64.58 / <u>0.106</u> / **0.024** | 64.04 / **0.149** / **0.047** | 64.09 / <u>0.134</u> / <u>0.039</u> |

## Reference Links

- Fed-NGA: https://neurips.cc/virtual/2025/poster/118753
- Huber-BRFL: https://ojs.aaai.org/index.php/AAAI/article/view/30181
- LoGoFair: https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559
- AdaAggRL: https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888 ; code: https://github.com/TAP-LLM/AdaAggRL
- FedAMM: https://doi.org/10.1109/TIFS.2025.3607273
- FedAA: https://ojs.aaai.org/index.php/AAAI/article/view/33878 ; code: https://github.com/Gp1g/FedAA
