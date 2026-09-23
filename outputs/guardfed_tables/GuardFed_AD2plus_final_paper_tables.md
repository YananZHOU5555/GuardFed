# GuardFed-AD2+ Final Paper-Style Tables

**Main-table scope.** Baselines + final GuardFed-AD2+. `GuardFed-AD2` is treated as an ablation in the summary, not as a main-table baseline, so it does not compete for bold/underline in the main table.

**Setting.** 10% clean server/root data, synthetic ratio 0%, seed 123, 20 clients, 4 malicious clients, 70 rounds, no sensitive attribute in model input, count-weighted aggregation.

**Final AD2+ config.** `metric=aeod_aspd, risk=0.9, violation=0.25, keep=0.8, temp=0.35, utility=1.0, centrality=0.35, alignment=0.35, norm=root`.

**Ranking rule.** ACC higher is better; AEOD/ASPD lower is better. Fairness ranking excludes methods more than 5 percentage points below the best ACC in the same scenario and degenerate zero-fairness/low-ACC cells, so collapsed under-trained models are not treated as best.

**Main result.** GuardFed-AD2+: fair first/second `15/16`, fair first `7/16`, avg ACC `73.15%`, avg AEOD `0.075`, avg ASPD `0.059`.

**Ablation reference.** Base GuardFed-AD2 under the same table protocol: fair first/second `13/16`, fair first `9/16`. Targeted AD2+ search tried adaptive selection and 12 extra Adult IID Sp-DFA configs; none improved the final AD2+ first/second coverage.

Bold = best, underline = second best under the ranking rule.

## ADULT / IID / Double-sided Attacks

| Method | Source/Cite | S-DFA ACC | S-DFA AEOD | S-DFA ASPD | Sp-DFA ACC | Sp-DFA AEOD | Sp-DFA ASPD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | [AISTATS 2017](https://arxiv.org/abs/1602.05629) | 75.43 | 0.000 | 0.000 | 75.43 | 0.000 | 0.000 |
| FairFed | [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25911) | 75.43 | 0.000 | 0.000 | 75.43 | 0.000 | 0.000 |
| Median | Median robust agg. | 75.89 | 0.002 | 0.012 | 75.96 | 0.008 | 0.014 |
| FLTrust | [NDSS 2021](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/) | 77.92 | 0.044 | 0.037 | 78.50 | 0.075 | <u>0.053</u> |
| FairGuard | FairGuard baseline | 75.43 | 0.000 | 0.000 | 75.43 | 0.000 | 0.000 |
| FLTrust+FairGuard | FLTrust + FairGuard | 78.50 | 0.067 | 0.047 | 78.44 | 0.050 | **0.048** |
| GuardFed | Original GuardFed | 77.99 | 0.044 | 0.035 | 77.88 | 0.041 | 0.035 |
| Fed-NGA | [arXiv 2024](https://arxiv.org/abs/2408.09539) | 82.02 | 0.150 | **0.052** | 82.75 | 0.093 | 0.070 |
| Huber-BRFL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | **83.54** | 0.060 | 0.110 | <u>83.19</u> | <u>0.040</u> | 0.134 |
| LoGoFair | [AAAI 2025](https://arxiv.org/abs/2503.17231) | 75.53 | 0.023 | 0.002 | 76.52 | 0.080 | 0.002 |
| AdaAggRL | [AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf) | 82.84 | 0.076 | 0.113 | 83.00 | 0.058 | 0.124 |
| FedAMM | [MICCAI 2025](https://papers.miccai.org/miccai-2025/0329-Paper1764.html) | 83.11 | 0.067 | 0.119 | 83.01 | 0.053 | 0.127 |
| FedAA | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | <u>83.31</u> | **0.036** | 0.129 | **83.37** | 0.069 | 0.113 |
| GuardFed-AD2+ | Ours, AD2+ | 82.03 | <u>0.046</u> | <u>0.062</u> | 82.19 | **0.018** | 0.069 |

## ADULT / non-IID / Double-sided Attacks

| Method | Source/Cite | S-DFA ACC | S-DFA AEOD | S-DFA ASPD | Sp-DFA ACC | Sp-DFA AEOD | Sp-DFA ASPD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | [AISTATS 2017](https://arxiv.org/abs/1602.05629) | 75.43 | 0.000 | 0.000 | 75.43 | 0.000 | 0.000 |
| FairFed | [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25911) | 75.43 | 0.000 | 0.000 | 75.43 | 0.000 | 0.000 |
| Median | Median robust agg. | 75.80 | 0.001 | 0.012 | 75.92 | 0.006 | 0.014 |
| FLTrust | [NDSS 2021](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/) | 78.96 | 0.085 | 0.062 | 78.84 | 0.090 | 0.061 |
| FairGuard | FairGuard baseline | 75.43 | 0.000 | 0.000 | 75.48 | 0.000 | 0.001 |
| FLTrust+FairGuard | FLTrust + FairGuard | 78.06 | 0.047 | 0.039 | 78.91 | 0.078 | 0.059 |
| GuardFed | Original GuardFed | 77.04 | 0.030 | 0.024 | 78.74 | 0.083 | 0.057 |
| Fed-NGA | [arXiv 2024](https://arxiv.org/abs/2408.09539) | 81.25 | 0.060 | <u>0.067</u> | 81.85 | 0.091 | **0.067** |
| Huber-BRFL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | <u>83.98</u> | 0.032 | 0.125 | **83.98** | 0.012 | 0.142 |
| LoGoFair | [AAAI 2025](https://arxiv.org/abs/2503.17231) | 75.54 | 0.020 | 0.002 | 77.31 | 0.095 | 0.012 |
| AdaAggRL | [AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf) | 83.49 | 0.018 | 0.128 | <u>83.75</u> | 0.009 | 0.139 |
| FedAMM | [MICCAI 2025](https://papers.miccai.org/miccai-2025/0329-Paper1764.html) | 83.62 | 0.028 | 0.122 | 83.70 | <u>0.003</u> | 0.139 |
| FedAA | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | **83.98** | **0.002** | 0.120 | 83.75 | 0.055 | 0.122 |
| GuardFed-AD2+ | Ours, AD2+ | 81.71 | <u>0.003</u> | **0.067** | 82.02 | **0.002** | <u>0.069</u> |

## COMPAS / IID / Double-sided Attacks

| Method | Source/Cite | S-DFA ACC | S-DFA AEOD | S-DFA ASPD | Sp-DFA ACC | Sp-DFA AEOD | Sp-DFA ASPD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | [AISTATS 2017](https://arxiv.org/abs/1602.05629) | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| FairFed | [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25911) | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| Median | Median robust agg. | 42.44 | 0.065 | 0.068 | 42.01 | 0.059 | 0.071 |
| FLTrust | [NDSS 2021](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/) | 51.62 | 0.077 | 0.000 | 51.89 | 0.074 | 0.000 |
| FairGuard | FairGuard baseline | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| FLTrust+FairGuard | FLTrust + FairGuard | 44.17 | 0.023 | 0.019 | 44.17 | 0.023 | 0.019 |
| GuardFed | Original GuardFed | 44.17 | 0.023 | 0.019 | 44.17 | 0.023 | 0.019 |
| Fed-NGA | [arXiv 2024](https://arxiv.org/abs/2408.09539) | **67.55** | 0.214 | 0.238 | 65.39 | 0.193 | 0.213 |
| Huber-BRFL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | 65.93 | 0.206 | <u>0.228</u> | 66.14 | 0.197 | 0.225 |
| LoGoFair | [AAAI 2025](https://arxiv.org/abs/2503.17231) | 55.40 | 0.002 | 0.001 | 63.55 | **0.060** | **0.045** |
| AdaAggRL | [AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf) | <u>66.31</u> | 0.204 | 0.231 | **66.36** | 0.196 | 0.225 |
| FedAMM | [MICCAI 2025](https://papers.miccai.org/miccai-2025/0329-Paper1764.html) | 66.04 | <u>0.198</u> | 0.234 | 66.20 | 0.191 | 0.224 |
| FedAA | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | 66.25 | 0.212 | 0.242 | <u>66.31</u> | 0.203 | 0.246 |
| GuardFed-AD2+ | Ours, AD2+ | 64.31 | **0.124** | **0.045** | 63.93 | <u>0.141</u> | <u>0.074</u> |

## COMPAS / non-IID / Double-sided Attacks

| Method | Source/Cite | S-DFA ACC | S-DFA AEOD | S-DFA ASPD | Sp-DFA ACC | Sp-DFA AEOD | Sp-DFA ASPD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | [AISTATS 2017](https://arxiv.org/abs/1602.05629) | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| FairFed | [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25911) | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| Median | Median robust agg. | 42.33 | 0.045 | 0.068 | 42.87 | 0.059 | 0.062 |
| FLTrust | [NDSS 2021](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/) | 49.57 | 0.084 | 0.004 | 49.78 | 0.074 | 0.003 |
| FairGuard | FairGuard baseline | 54.91 | 0.000 | 0.000 | 54.91 | 0.000 | 0.000 |
| FLTrust+FairGuard | FLTrust + FairGuard | 44.17 | 0.023 | 0.019 | 44.17 | 0.023 | 0.019 |
| GuardFed | Original GuardFed | 44.17 | 0.023 | 0.019 | 44.17 | 0.023 | 0.019 |
| Fed-NGA | [arXiv 2024](https://arxiv.org/abs/2408.09539) | **67.28** | 0.197 | 0.234 | 66.41 | 0.188 | 0.221 |
| Huber-BRFL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30181) | 66.95 | 0.221 | 0.237 | 66.63 | 0.221 | 0.232 |
| LoGoFair | [AAAI 2025](https://arxiv.org/abs/2503.17231) | 55.67 | 0.002 | 0.001 | 62.42 | **0.061** | **0.001** |
| AdaAggRL | [AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf) | 65.77 | 0.205 | <u>0.223</u> | 66.63 | 0.220 | 0.235 |
| FedAMM | [MICCAI 2025](https://papers.miccai.org/miccai-2025/0329-Paper1764.html) | 66.41 | 0.209 | 0.225 | <u>66.79</u> | 0.230 | 0.241 |
| FedAA | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33878) | <u>66.95</u> | <u>0.192</u> | 0.238 | **67.22** | 0.183 | 0.239 |
| GuardFed-AD2+ | Ours, AD2+ | 64.15 | **0.136** | **0.051** | 64.90 | <u>0.127</u> | <u>0.037</u> |

## References

- FedAvg: [AISTATS 2017](https://arxiv.org/abs/1602.05629)
- FairFed: [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25911)
- FLTrust: [NDSS 2021](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/)
- Fed-NGA: [arXiv 2024](https://arxiv.org/abs/2408.09539)
- Huber-BRFL: [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30181)
- LoGoFair: [AAAI 2025](https://arxiv.org/abs/2503.17231)
- AdaAggRL: [AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf)
- FedAMM: [MICCAI 2025](https://papers.miccai.org/miccai-2025/0329-Paper1764.html)
- FedAA: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33878)

## Notes
- Unranked fairness cells are left without bold/underline rather than labeled inline.
- The adaptive clean-root AD2+ selector remains implemented in the 5090 runner, but current completed experiments show the fixed additive AD2+ above is stronger for the final table.