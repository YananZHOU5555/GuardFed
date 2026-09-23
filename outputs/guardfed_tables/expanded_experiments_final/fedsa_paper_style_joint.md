# FedSA all-method reproduction table (joint-last10)

Selection rule: for every run, one round is selected from the last 10 rounds by maximizing `ACC - 0.5*(AEOD+ASPD)`. The displayed ACC/AEOD/ASPD are from that same round. Fairness ranking is valid only when mean ACC is at least 80% for Adult and 60% for COMPAS. Exact raw zeros are kept in CSV; the table displays values below 0.0001 as 0.0001.

| Category | Method | Citation | Metric | adult IID | adult non-IID | compas IID | compas non-IID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fairness (debias) algorithms | FedAvg | McMahan et al., AISTATS'17 | ACC | 80.63 | 82.83 | 65.93 | 66.41 |
|  |  |  | AEOD | 0.1018 | 0.0578 | 0.2096 | 0.2226 |
|  |  |  | ASPD | <u>0.0596</u> | 0.0839 | 0.2011 | 0.2135 |
| Fairness (debias) algorithms | FairFed | Ezzeldin et al., AAAI'23 | ACC | 80.69 | 82.38 | 65.75 | 66.31 |
|  |  |  | AEOD | 0.0885 | 0.0766 | 0.2077 | 0.2028 |
|  |  |  | ASPD | 0.0656 | <u>0.0691</u> | 0.1976 | 0.2058 |
| Robust FL for general adversarial attacks | Median | Yin et al., ICML'18 | ACC | 82.33 | 82.65 | 66.85 | 66.86 |
|  |  |  | AEOD | 0.0330 | 0.0447 | 0.2239 | 0.2113 |
|  |  |  | ASPD | 0.1076 | 0.0920 | 0.2235 | 0.2182 |
| Robust FL for general adversarial attacks | FLTrust | Cao et al., NDSS'21 | ACC | 83.02 | 83.32 | 67.21 | 67.69 |
|  |  |  | AEOD | 0.0168 | 0.0156 | 0.2265 | 0.2169 |
|  |  |  | ASPD | 0.1192 | 0.1052 | 0.2365 | 0.2433 |
| Robust FL for fairness attacks | FairGuard | FairGuard, TDSC'24 | ACC | 44.13 | 83.55 | 53.60 | 53.42 |
|  |  |  | AEOD | 0.0033 | 0.0203 | 0.0685 | 0.0444 |
|  |  |  | ASPD | 0.0350 | 0.1197 | 0.0374 | 0.0364 |
| Robust FL (hybrid) | FLTrust+FairGuard | FLTrust'21 + FairGuard'24 | ACC | 81.81 | 83.60 | 63.39 | 61.92 |
|  |  |  | AEOD | 0.0107 | <u>0.0023</u> | 0.1362 | 0.1175 |
|  |  |  | ASPD | 0.0756 | 0.1209 | 0.1546 | 0.1302 |
| Original GuardFed | GuardFed | GuardFed, original | ACC | <u>83.47</u> | 83.31 | 67.33 | 67.67 |
|  |  |  | AEOD | 0.0204 | **0.0011** | 0.2308 | 0.2094 |
|  |  |  | ASPD | 0.1052 | 0.1082 | 0.2355 | 0.2406 |
| Recent robust FL baseline | FLGMM | FLGMM, Inf. Fusion'25 | ACC | 82.73 | 82.50 | 66.92 | 67.24 |
|  |  |  | AEOD | 0.0542 | 0.0331 | 0.2119 | 0.2150 |
|  |  |  | ASPD | 0.1316 | 0.0873 | 0.2219 | 0.2250 |
| Recent robust FL baseline | FLAURA | FLAURA, preprint'26 | ACC | 83.24 | 83.38 | 66.83 | 67.13 |
|  |  |  | AEOD | 0.0749 | 0.0236 | 0.2224 | 0.2065 |
|  |  |  | ASPD | 0.1204 | 0.1086 | 0.2219 | 0.2216 |
| Recent robust FL baseline | LayerGuard | LayerGuard, OpenReview'25 | ACC | 82.29 | 82.81 | 66.27 | 66.47 |
|  |  |  | AEOD | 0.0755 | 0.0754 | 0.2023 | 0.2107 |
|  |  |  | ASPD | 0.0934 | 0.0844 | 0.2057 | 0.2189 |
| Recent robust FL baseline | SmartFL | SmartFL, Inf. Fusion'25 | ACC | 83.26 | 82.98 | 66.67 | 66.31 |
|  |  |  | AEOD | 0.0573 | 0.0667 | 0.1982 | 0.1996 |
|  |  |  | ASPD | 0.1107 | 0.0931 | 0.2058 | 0.2127 |
| Recent robust FL baseline | FLTG | Wen et al., arXiv/BlockSys'25 | ACC | 83.32 | 83.58 | 67.15 | <u>67.73</u> |
|  |  |  | AEOD | 0.0099 | 0.0240 | 0.2238 | 0.2204 |
|  |  |  | ASPD | 0.1230 | 0.1172 | 0.2417 | 0.2430 |
| Recent robust FL baseline | FedDNA | FedDNA, JISA'26 | ACC | 83.44 | <u>83.69</u> | <u>67.40</u> | 67.49 |
|  |  |  | AEOD | 0.0559 | 0.0127 | 0.2367 | 0.2237 |
|  |  |  | ASPD | 0.1228 | 0.1262 | 0.2336 | 0.2329 |
| Recent robust FL baseline | LASA | Xu et al., WACV'25 | ACC | 80.62 | 81.92 | 65.80 | 66.27 |
|  |  |  | AEOD | 0.1001 | 0.0875 | 0.2096 | 0.2048 |
|  |  |  | ASPD | **0.0594** | **0.0588** | 0.2020 | 0.2058 |
| Additional high-impact/recent baseline | Fed-NGA | Fed-NGA, 2025 | ACC | 81.75 | 82.70 | 66.52 | 66.95 |
|  |  |  | AEOD | 0.0989 | 0.0943 | 0.2048 | 0.1954 |
|  |  |  | ASPD | 0.0692 | 0.0792 | 0.2119 | 0.2175 |
| Additional high-impact/recent baseline | Huber-BRFL | Huber-BRFL, 2025 | ACC | 82.34 | 82.34 | 66.41 | 66.67 |
|  |  |  | AEOD | 0.0673 | 0.0664 | 0.2137 | 0.2118 |
|  |  |  | ASPD | 0.0977 | 0.0746 | 0.2178 | 0.2192 |
| Additional high-impact/recent baseline | LoGoFair | LoGoFair, 2025 | ACC | 79.39 | 82.40 | 65.96 | 65.87 |
|  |  |  | AEOD | 0.0127 | 0.0188 | **0.0546** | **0.0318** |
|  |  |  | ASPD | 0.0363 | 0.0815 | 0.0480 | **0.0357** |
| Additional high-impact/recent baseline | AdaAggRL | AdaAggRL, 2025 | ACC | 83.43 | 83.39 | 66.95 | 67.15 |
|  |  |  | AEOD | 0.0635 | 0.0368 | 0.2274 | 0.2071 |
|  |  |  | ASPD | 0.1174 | 0.1087 | 0.2288 | 0.2193 |
| Additional high-impact/recent baseline | FedAMM | FedAMM, 2025 | ACC | 83.33 | **83.70** | 67.08 | 67.33 |
|  |  |  | AEOD | 0.0516 | 0.0237 | 0.2269 | 0.2369 |
|  |  |  | ASPD | 0.1243 | 0.1226 | 0.2309 | 0.2337 |
| Additional high-impact/recent baseline | FedAA | FedAA, 2025 | ACC | **83.59** | 83.58 | **67.69** | **67.89** |
|  |  |  | AEOD | 0.0401 | 0.0099 | 0.2287 | 0.2315 |
|  |  |  | ASPD | 0.1127 | 0.1189 | 0.2402 | 0.2324 |
| Ours | GuardFed-AD2 | Ours, AD2 | ACC | 82.09 | 82.37 | 65.06 | 65.37 |
|  |  |  | AEOD | **0.0024** | 0.0102 | <u>0.0835</u> | <u>0.0764</u> |
|  |  |  | ASPD | 0.0746 | 0.0745 | **0.0446** | <u>0.0396</u> |
| Ours | GuardFed-AD2+ | Ours, AD2+ | ACC | 82.09 | 82.37 | 65.06 | 65.37 |
|  |  |  | AEOD | **0.0024** | 0.0102 | <u>0.0835</u> | <u>0.0764</u> |
|  |  |  | ASPD | 0.0746 | 0.0745 | **0.0446** | <u>0.0396</u> |

## AD2+ rank summary
- adult IID: ACC rank 14, AEOD rank 1, ASPD rank 5, score rank 1.
- adult non-IID: ACC rank 19, AEOD rank 4, ASPD rank 3, score rank 1.
- compas IID: ACC rank 19, AEOD rank 2, ASPD rank 1, score rank 2.
- compas non-IID: ACC rank 19, AEOD rank 2, ASPD rank 2, score rank 2.