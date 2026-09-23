# GuardFed-AD2 Clean-5% Dual-Attack Results

Protocol: 70 rounds, 20 clients, 4 malicious clients, local epoch 1, lr 0.005, batch 256, root/server data = 5% clean, no synthetic. Class-B FL and GuardFed-ACT are excluded.

Ranking: ACC higher is better; AEOD/ASPD lower is better. Best is **bold**, second best is <u>underlined</u>. [nr] means the fairness value is not ranked because ACC is more than 10 percentage points below the best ACC in that scenario.

## ADULT
| Method | Source | IID S-DFA<br>ACC / AEOD / ASPD | IID Sp-DFA<br>ACC / AEOD / ASPD | non-IID S-DFA<br>ACC / AEOD / ASPD | non-IID Sp-DFA<br>ACC / AEOD / ASPD |
| --- | --- | --- | --- | --- | --- |
| Fed-NGA | [NeurIPS'25](https://neurips.cc/virtual/2025/poster/118753) | 80.85 / 0.160 / <u>0.035</u> | 81.52 / 0.103 / <u>0.064</u> | 83.03 / 0.028 / 0.115 | 82.20 / 0.064 / 0.103 |
| Huber-BRFL | [AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | <u>83.51</u> / 0.036 / 0.119 | <u>83.40</u> / 0.064 / 0.126 | 82.95 / <u>0.010</u> / 0.112 | <u>83.45</u> / <u>0.019</u> / 0.136 |
| LoGoFair | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559) | 75.63 / <u>0.022</u> / **0.002** | 77.94 / <u>0.025</u> / **0.028** | 75.59 / 0.018 / **0.001** | 78.02 / **0.001** / **0.022** |
| AdaAggRL | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888) | 83.37 / 0.075 / 0.121 | 83.30 / 0.056 / 0.134 | 83.30 / 0.059 / 0.121 | 83.15 / 0.037 / 0.138 |
| FedAMM | [TIFS'25](https://doi.org/10.1109/TIFS.2025.3607273) | 83.25 / 0.073 / 0.118 | 83.40 / 0.060 / 0.128 | <u>83.40</u> / 0.058 / 0.122 | 83.37 / 0.034 / 0.136 |
| FedAA | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | **83.74** / 0.056 / 0.111 | **83.93** / **0.024** / 0.127 | **83.70** / 0.017 / 0.138 | **83.63** / 0.032 / 0.131 |
| GuardFed-AD2 | Ours | 81.99 / **0.007** / 0.074 | 81.94 / 0.032 / 0.067 | 83.37 / **0.006** / <u>0.100</u> | 81.72 / 0.058 / <u>0.084</u> |

## COMPAS
| Method | Source | IID S-DFA<br>ACC / AEOD / ASPD | IID Sp-DFA<br>ACC / AEOD / ASPD | non-IID S-DFA<br>ACC / AEOD / ASPD | non-IID Sp-DFA<br>ACC / AEOD / ASPD |
| --- | --- | --- | --- | --- | --- |
| Fed-NGA | [NeurIPS'25](https://neurips.cc/virtual/2025/poster/118753) | 66.20 / 0.212 / 0.227 | **68.03** / <u>0.203</u> / 0.250 | **66.90** / 0.229 / 0.242 | <u>67.06</u> / 0.215 / 0.222 |
| Huber-BRFL | [AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | 67.06 / 0.233 / 0.248 | 67.44 / 0.236 / 0.261 | 66.31 / 0.210 / 0.221 | 66.58 / 0.230 / 0.236 |
| LoGoFair | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559) | 57.24 / 0.009[nr] / 0.002[nr] | 62.20 / **0.067** / **0.002** | 61.72 / **0.096** / **0.029** | 61.99 / **0.075** / **0.013** |
| AdaAggRL | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888) | 66.79 / 0.207 / <u>0.226</u> | 67.44 / 0.233 / 0.253 | 66.20 / 0.201 / 0.219 | 66.79 / 0.231 / 0.240 |
| FedAMM | [TIFS'25](https://doi.org/10.1109/TIFS.2025.3607273) | **67.76** / 0.229 / 0.257 | <u>67.60</u> / 0.230 / 0.254 | <u>66.68</u> / 0.227 / 0.240 | **67.12** / 0.233 / 0.244 |
| FedAA | [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | <u>67.12</u> / <u>0.176</u> / 0.237 | 67.22 / 0.211 / 0.241 | 66.47 / 0.187 / 0.218 | 66.36 / 0.192 / 0.226 |
| GuardFed-AD2 | Ours | 64.15 / **0.157** / **0.084** | 63.55 / 0.233 / <u>0.141</u> | 65.06 / <u>0.172</u> / <u>0.087</u> | 64.36 / <u>0.164</u> / <u>0.052</u> |

## Reference Links

- Fed-NGA: https://neurips.cc/virtual/2025/poster/118753
- Huber-BRFL: https://ojs.aaai.org/index.php/AAAI/article/view/30181
- LoGoFair: https://ojs.aaai.org/index.php/AAAI/article/download/34404/36559
- AdaAggRL: https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888 ; code: https://github.com/TAP-LLM/AdaAggRL
- FedAMM: https://doi.org/10.1109/TIFS.2025.3607273
- FedAA: https://ojs.aaai.org/index.php/AAAI/article/view/33878 ; code: https://github.com/Gp1g/FedAA
