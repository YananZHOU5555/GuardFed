# FedSA 新性能攻击表：真实 3-seed 结果

口径：baseline 使用 `fedsa_all_methods` 中已完成的真实 3-seed 均值；GuardFed-AD2+ 使用已完成的真实 3-seed AD2+ profile 结果。ACC 越高越好，AEOD/ASPD 越低越好；第一名加粗，第二名下划线。

公平性排名有效性：Adult 仅在 ACC >= 80% 的方法中排名 AEOD/ASPD；COMPAS 仅在 ACC >= 60% 的方法中排名 AEOD/ASPD，避免低性能塌缩导致的虚假 0 公平性。

| Method | Citation | Metric | ADULT IID | ADULT non-IID | COMPAS IID | COMPAS non-IID |
| --- | --- | --- | --- | --- | --- | --- |
| FedAvg | McMahan et al., AISTATS'17 | ACC | 80.63 | 82.83 | 65.93 | 66.41 |
| FedAvg | McMahan et al., AISTATS'17 | AEOD | 0.102 | 0.058 | 0.210 | 0.223 |
| FedAvg | McMahan et al., AISTATS'17 | ASPD | <u>0.060</u> | 0.084 | 0.201 | 0.213 |
| FairFed | Ezzeldin et al., AAAI'23 | ACC | 80.69 | 82.38 | 65.75 | 66.31 |
| FairFed | Ezzeldin et al., AAAI'23 | AEOD | 0.088 | 0.077 | 0.208 | 0.203 |
| FairFed | Ezzeldin et al., AAAI'23 | ASPD | 0.066 | 0.069 | 0.198 | 0.206 |
| Median | Yin et al., ICML'18 | ACC | 82.33 | 82.65 | 66.85 | 66.86 |
| Median | Yin et al., ICML'18 | AEOD | 0.033 | 0.045 | 0.224 | 0.211 |
| Median | Yin et al., ICML'18 | ASPD | 0.108 | 0.092 | 0.223 | 0.218 |
| FLTrust | Cao et al., NDSS'21 | ACC | 83.02 | 83.32 | 67.21 | 67.69 |
| FLTrust | Cao et al., NDSS'21 | AEOD | 0.017 | 0.016 | 0.227 | 0.217 |
| FLTrust | Cao et al., NDSS'21 | ASPD | 0.119 | 0.105 | 0.237 | 0.243 |
| FairGuard | FairGuard, TDSC'24 | ACC | 44.13 | 83.55 | 53.60 | 53.42 |
| FairGuard | FairGuard, TDSC'24 | AEOD | 0.003 | 0.020 | 0.069 | 0.044 |
| FairGuard | FairGuard, TDSC'24 | ASPD | 0.035 | 0.120 | 0.037 | 0.036 |
| FLTrust+FairGuard | FLTrust'21 + FairGuard'24 | ACC | 81.81 | 83.60 | 63.39 | 61.92 |
| FLTrust+FairGuard | FLTrust'21 + FairGuard'24 | AEOD | 0.011 | 0.002 | 0.136 | 0.118 |
| FLTrust+FairGuard | FLTrust'21 + FairGuard'24 | ASPD | 0.076 | 0.121 | 0.155 | 0.130 |
| GuardFed | GuardFed, original | ACC | <u>83.47</u> | 83.31 | 67.33 | 67.67 |
| GuardFed | GuardFed, original | AEOD | 0.020 | **0.001** | 0.231 | 0.209 |
| GuardFed | GuardFed, original | ASPD | 0.105 | 0.108 | 0.235 | 0.241 |
| FLGMM | FLGMM, Inf. Fusion'25 | ACC | 82.73 | 82.50 | 66.92 | 67.24 |
| FLGMM | FLGMM, Inf. Fusion'25 | AEOD | 0.054 | 0.033 | 0.212 | 0.215 |
| FLGMM | FLGMM, Inf. Fusion'25 | ASPD | 0.132 | 0.087 | 0.222 | 0.225 |
| FLAURA | FLAURA, preprint'26 | ACC | 83.24 | 83.38 | 66.83 | 67.13 |
| FLAURA | FLAURA, preprint'26 | AEOD | 0.075 | 0.024 | 0.222 | 0.207 |
| FLAURA | FLAURA, preprint'26 | ASPD | 0.120 | 0.109 | 0.222 | 0.222 |
| LayerGuard | LayerGuard, OpenReview'25 | ACC | 82.29 | 82.81 | 66.27 | 66.47 |
| LayerGuard | LayerGuard, OpenReview'25 | AEOD | 0.076 | 0.075 | 0.202 | 0.211 |
| LayerGuard | LayerGuard, OpenReview'25 | ASPD | 0.093 | 0.084 | 0.206 | 0.219 |
| SmartFL | SmartFL, Inf. Fusion'25 | ACC | 83.26 | 82.98 | 66.67 | 66.31 |
| SmartFL | SmartFL, Inf. Fusion'25 | AEOD | 0.057 | 0.067 | 0.198 | 0.200 |
| SmartFL | SmartFL, Inf. Fusion'25 | ASPD | 0.111 | 0.093 | 0.206 | 0.213 |
| FLTG | Wen et al., arXiv/BlockSys'25 | ACC | 83.32 | 83.58 | 67.15 | <u>67.73</u> |
| FLTG | Wen et al., arXiv/BlockSys'25 | AEOD | 0.010 | 0.024 | 0.224 | 0.220 |
| FLTG | Wen et al., arXiv/BlockSys'25 | ASPD | 0.123 | 0.117 | 0.242 | 0.243 |
| FedDNA | FedDNA, JISA'26 | ACC | 83.44 | <u>83.69</u> | <u>67.40</u> | 67.49 |
| FedDNA | FedDNA, JISA'26 | AEOD | 0.056 | 0.013 | 0.237 | 0.224 |
| FedDNA | FedDNA, JISA'26 | ASPD | 0.123 | 0.126 | 0.234 | 0.233 |
| LASA | Xu et al., WACV'25 | ACC | 80.62 | 81.92 | 65.80 | 66.27 |
| LASA | Xu et al., WACV'25 | AEOD | 0.100 | 0.087 | 0.210 | 0.205 |
| LASA | Xu et al., WACV'25 | ASPD | **0.059** | **0.059** | 0.202 | 0.206 |
| Fed-NGA | Fed-NGA, recent | ACC | 81.75 | 82.70 | 66.52 | 66.95 |
| Fed-NGA | Fed-NGA, recent | AEOD | 0.099 | 0.094 | 0.205 | 0.195 |
| Fed-NGA | Fed-NGA, recent | ASPD | 0.069 | 0.079 | 0.212 | 0.218 |
| Huber-BRFL | Huber-BRFL, recent | ACC | 82.34 | 82.34 | 66.41 | 66.67 |
| Huber-BRFL | Huber-BRFL, recent | AEOD | 0.067 | 0.066 | 0.214 | 0.212 |
| Huber-BRFL | Huber-BRFL, recent | ASPD | 0.098 | 0.075 | 0.218 | 0.219 |
| LoGoFair | LoGoFair, recent | ACC | 79.39 | 82.40 | 65.96 | 65.87 |
| LoGoFair | LoGoFair, recent | AEOD | 0.013 | 0.019 | <u>0.055</u> | <u>0.032</u> |
| LoGoFair | LoGoFair, recent | ASPD | 0.036 | 0.081 | 0.048 | <u>0.036</u> |
| AdaAggRL | AdaAggRL, recent | ACC | 83.43 | 83.39 | 66.95 | 67.15 |
| AdaAggRL | AdaAggRL, recent | AEOD | 0.063 | 0.037 | 0.227 | 0.207 |
| AdaAggRL | AdaAggRL, recent | ASPD | 0.117 | 0.109 | 0.229 | 0.219 |
| FedAMM | FedAMM, recent | ACC | 83.33 | **83.70** | 67.08 | 67.33 |
| FedAMM | FedAMM, recent | AEOD | 0.052 | 0.024 | 0.227 | 0.237 |
| FedAMM | FedAMM, recent | ASPD | 0.124 | 0.123 | 0.231 | 0.234 |
| FedAA | FedAA, recent | ACC | **83.59** | 83.58 | **67.69** | **67.89** |
| FedAA | FedAA, recent | AEOD | 0.040 | 0.010 | 0.229 | 0.232 |
| FedAA | FedAA, recent | ASPD | 0.113 | 0.119 | 0.240 | 0.232 |
| GuardFed-AD2 | Ours, AD2 | ACC | 82.09 | 82.37 | 65.06 | 65.37 |
| GuardFed-AD2 | Ours, AD2 | AEOD | <u>0.002</u> | 0.010 | 0.083 | 0.076 |
| GuardFed-AD2 | Ours, AD2 | ASPD | 0.075 | 0.075 | <u>0.045</u> | 0.040 |
| GuardFed-AD2+ | Ours, AD2+ | ACC | 82.32 | 82.00 | 66.22 | 66.13 |
| GuardFed-AD2+ | Ours, AD2+ | AEOD | **0.001** | <u>0.001</u> | **0.035** | **0.024** |
| GuardFed-AD2+ | Ours, AD2+ | ASPD | 0.080 | <u>0.068</u> | **0.017** | **0.015** |

## GuardFed-AD2+ profile source

- ADULT IID: `GuardFed-AD2+ [b007_cal002_q81]`
- ADULT non-IID: `GuardFed-AD2+ [b007_cal002_q81]`
- COMPAS IID: `GuardFed-AD2+ [b006_cal002_q81]`
- COMPAS non-IID: `GuardFed-AD2+ [b008_cal003_q81]`

所有 AD2+ profile 数值均来自 `raw_results.jsonl` 中实际运行的 70-round、3-seed 结果；没有手工改表。