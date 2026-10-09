# CelebA Full–minus_U: interim three-view paired validation tables

Six coverage-complete scenes; all three views are reported in parallel. This snapshot does not select a primary endpoint.

## native — All 10 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID | Benign | minus_U | 10 | 87.152 ± 1.796 | 0.01138 ± 0.00831 | 0.03982 ± 0.01737 |
| IID | Benign | minus_U minus Full | 10 | -1.107 ± 2.044 | 0.00164 ± 0.00571 | -0.02272 ± 0.01963 |
| IID | F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID | F Flip | minus_U | 10 | 88.082 ± 1.551 | 0.01276 ± 0.00997 | 0.05153 ± 0.02025 |
| IID | F Flip | minus_U minus Full | 10 | -0.309 ± 1.337 | 0.00208 ± 0.01164 | -0.00916 ± 0.02366 |
| IID | FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID | FedSA | minus_U | 10 | 88.143 ± 1.359 | 0.01061 ± 0.00888 | 0.05184 ± 0.01671 |
| IID | FedSA | minus_U minus Full | 10 | -0.327 ± 1.294 | 0.00453 ± 0.01009 | -0.01217 ± 0.01722 |
| IID | S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID | S-DFA | minus_U | 10 | 87.759 ± 1.228 | 0.01191 ± 0.01163 | 0.05051 ± 0.02281 |
| IID | S-DFA | minus_U minus Full | 10 | -0.931 ± 1.140 | 0.00421 ± 0.01321 | -0.01325 ± 0.02639 |
| IID | Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID | Sp-DFA | minus_U | 10 | 86.998 ± 4.345 | 0.01311 ± 0.01074 | 0.05267 ± 0.02474 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.384 ± 3.938 | -0.00337 ± 0.01161 | -0.00321 ± 0.03629 |
| non-IID | Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID | Benign | minus_U | 10 | 87.290 ± 1.870 | 0.01300 ± 0.00732 | 0.05140 ± 0.01598 |
| non-IID | Benign | minus_U minus Full | 10 | -1.303 ± 0.892 | 0.00604 ± 0.00899 | -0.01371 ± 0.01219 |

## native — Exclude selection seed: 9 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID | Benign | minus_U | 9 | 87.107 ± 1.899 | 0.01125 ± 0.00881 | 0.03948 ± 0.01838 |
| IID | Benign | minus_U minus Full | 9 | -1.245 ± 2.117 | 0.00150 ± 0.00603 | -0.02284 ± 0.02082 |
| IID | F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID | F Flip | minus_U | 9 | 88.073 ± 1.645 | 0.01266 ± 0.01056 | 0.04963 ± 0.02052 |
| IID | F Flip | minus_U minus Full | 9 | -0.339 ± 1.415 | 0.00119 ± 0.01198 | -0.01080 ± 0.02448 |
| IID | FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID | FedSA | minus_U | 9 | 88.230 ± 1.412 | 0.01108 ± 0.00929 | 0.05251 ± 0.01758 |
| IID | FedSA | minus_U minus Full | 9 | -0.392 ± 1.355 | 0.00563 ± 0.01005 | -0.01424 ± 0.01691 |
| IID | S-DFA | Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID | S-DFA | minus_U | 9 | 87.925 ± 1.177 | 0.01199 ± 0.01234 | 0.05347 ± 0.02207 |
| IID | S-DFA | minus_U minus Full | 9 | -0.920 ± 1.208 | 0.00524 ± 0.01358 | -0.00982 ± 0.02552 |
| IID | Sp-DFA | Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID | Sp-DFA | minus_U | 9 | 86.979 ± 4.608 | 0.01314 ± 0.01140 | 0.05060 ± 0.02530 |
| IID | Sp-DFA | minus_U minus Full | 9 | -1.567 ± 4.131 | -0.00392 ± 0.01217 | -0.00489 ± 0.03808 |
| non-IID | Benign | Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID | Benign | minus_U | 9 | 87.245 ± 1.978 | 0.01230 ± 0.00740 | 0.04929 ± 0.01540 |
| non-IID | Benign | minus_U minus Full | 9 | -1.347 ± 0.935 | 0.00599 ± 0.00954 | -0.01456 ± 0.01261 |

## native — Seeds 91005–91010: 6 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID | Benign | minus_U | 6 | 87.467 ± 1.093 | 0.01129 ± 0.00914 | 0.04017 ± 0.01468 |
| IID | Benign | minus_U minus Full | 6 | -1.091 ± 1.163 | -0.00009 ± 0.00619 | -0.02632 ± 0.01120 |
| IID | F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID | F Flip | minus_U | 6 | 87.676 ± 1.877 | 0.01099 ± 0.01064 | 0.04632 ± 0.02276 |
| IID | F Flip | minus_U minus Full | 6 | -0.612 ± 1.685 | -0.00385 ± 0.00832 | -0.01176 ± 0.02947 |
| IID | FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID | FedSA | minus_U | 6 | 88.576 ± 0.521 | 0.00728 ± 0.00729 | 0.05668 ± 0.00947 |
| IID | FedSA | minus_U minus Full | 6 | 0.007 ± 0.631 | 0.00178 ± 0.00926 | -0.01297 ± 0.01100 |
| IID | S-DFA | Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID | S-DFA | minus_U | 6 | 87.961 ± 0.921 | 0.01546 ± 0.01384 | 0.04828 ± 0.02248 |
| IID | S-DFA | minus_U minus Full | 6 | -0.784 ± 1.074 | 0.00843 ± 0.01564 | -0.01342 ± 0.02896 |
| IID | Sp-DFA | Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID | Sp-DFA | minus_U | 6 | 86.259 ± 5.627 | 0.01084 ± 0.00958 | 0.04996 ± 0.02617 |
| IID | Sp-DFA | minus_U minus Full | 6 | -2.095 ± 5.107 | -0.00626 ± 0.00504 | -0.00599 ± 0.04233 |
| non-IID | Benign | Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID | Benign | minus_U | 6 | 87.497 ± 2.201 | 0.01282 ± 0.00747 | 0.04494 ± 0.01300 |
| non-IID | Benign | minus_U minus Full | 6 | -1.088 ± 0.958 | 0.00659 ± 0.01056 | -0.01917 ± 0.01271 |

## raw — All 10 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID | Benign | minus_U | 10 | 87.391 ± 2.237 | 0.03930 ± 0.01011 | 0.09930 ± 0.01160 |
| IID | Benign | minus_U minus Full | 10 | -1.158 ± 2.504 | -0.00097 ± 0.01606 | -0.00378 ± 0.01447 |
| IID | F Flip | Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID | F Flip | minus_U | 10 | 88.401 ± 1.511 | 0.03210 ± 0.01597 | 0.09698 ± 0.01028 |
| IID | F Flip | minus_U minus Full | 10 | -0.302 ± 1.330 | -0.00474 ± 0.01405 | -0.00382 ± 0.00806 |
| IID | FedSA | Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID | FedSA | minus_U | 10 | 88.456 ± 1.346 | 0.02835 ± 0.00976 | 0.09138 ± 0.01511 |
| IID | FedSA | minus_U minus Full | 10 | -0.376 ± 1.336 | -0.00880 ± 0.01183 | -0.01120 ± 0.01949 |
| IID | S-DFA | Full | 10 | 89.059 ± 0.840 | 0.03671 ± 0.00377 | 0.10194 ± 0.00664 |
| IID | S-DFA | minus_U | 10 | 88.005 ± 1.297 | 0.03623 ± 0.01343 | 0.09576 ± 0.01605 |
| IID | S-DFA | minus_U minus Full | 10 | -1.054 ± 1.111 | -0.00048 ± 0.01566 | -0.00618 ± 0.01346 |
| IID | Sp-DFA | Full | 10 | 88.766 ± 0.924 | 0.03465 ± 0.01394 | 0.09978 ± 0.01059 |
| IID | Sp-DFA | minus_U | 10 | 87.264 ± 4.711 | 0.03612 ± 0.01022 | 0.09364 ± 0.02496 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.502 ± 4.227 | 0.00147 ± 0.02085 | -0.00614 ± 0.03121 |
| non-IID | Benign | Full | 10 | 88.875 ± 1.209 | 0.03157 ± 0.00643 | 0.09978 ± 0.00770 |
| non-IID | Benign | minus_U | 10 | 87.723 ± 1.779 | 0.03036 ± 0.01038 | 0.09083 ± 0.01732 |
| non-IID | Benign | minus_U minus Full | 10 | -1.152 ± 0.775 | -0.00122 ± 0.01165 | -0.00896 ± 0.01156 |

## raw — Exclude selection seed: 9 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID | Benign | minus_U | 9 | 87.329 ± 2.363 | 0.03973 ± 0.01063 | 0.10018 ± 0.01194 |
| IID | Benign | minus_U minus Full | 9 | -1.317 ± 2.602 | -0.00030 ± 0.01689 | -0.00320 ± 0.01523 |
| IID | F Flip | Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID | F Flip | minus_U | 9 | 88.406 ± 1.603 | 0.03250 ± 0.01688 | 0.09827 ± 0.01001 |
| IID | F Flip | minus_U minus Full | 9 | -0.325 ± 1.408 | -0.00520 ± 0.01482 | -0.00373 ± 0.00854 |
| IID | FedSA | Full | 9 | 89.001 ± 0.767 | 0.03797 ± 0.00689 | 0.10454 ± 0.01032 |
| IID | FedSA | minus_U | 9 | 88.546 ± 1.395 | 0.02858 ± 0.01032 | 0.09226 ± 0.01575 |
| IID | FedSA | minus_U minus Full | 9 | -0.455 ± 1.392 | -0.00939 ± 0.01239 | -0.01228 ± 0.02035 |
| IID | S-DFA | Full | 9 | 89.233 ± 0.673 | 0.03652 ± 0.00395 | 0.10342 ± 0.00500 |
| IID | S-DFA | minus_U | 9 | 88.179 ± 1.246 | 0.03762 ± 0.01346 | 0.09929 ± 0.01226 |
| IID | S-DFA | minus_U minus Full | 9 | -1.054 ± 1.178 | 0.00110 ± 0.01574 | -0.00413 ± 0.01252 |
| IID | Sp-DFA | Full | 9 | 88.925 ± 0.824 | 0.03222 ± 0.01233 | 0.09968 ± 0.01123 |
| IID | Sp-DFA | minus_U | 9 | 87.257 ± 4.997 | 0.03527 ± 0.01046 | 0.09271 ± 0.02628 |
| IID | Sp-DFA | minus_U minus Full | 9 | -1.667 ± 4.449 | 0.00305 ± 0.02147 | -0.00697 ± 0.03298 |
| non-IID | Benign | Full | 9 | 88.877 ± 1.282 | 0.03006 ± 0.00454 | 0.09914 ± 0.00788 |
| non-IID | Benign | minus_U | 9 | 87.686 ± 1.882 | 0.02940 ± 0.01053 | 0.09039 ± 0.01831 |
| non-IID | Benign | minus_U minus Full | 9 | -1.191 ± 0.811 | -0.00066 ± 0.01221 | -0.00875 ± 0.01224 |

## raw — Seeds 91005–91010: 6 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID | Benign | minus_U | 6 | 87.866 ± 1.214 | 0.04029 ± 0.01244 | 0.10016 ± 0.01509 |
| IID | Benign | minus_U minus Full | 6 | -0.947 ± 1.254 | -0.00196 ± 0.02091 | -0.00694 ± 0.01640 |
| IID | F Flip | Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID | F Flip | minus_U | 6 | 87.972 ± 1.795 | 0.03745 ± 0.01712 | 0.10004 ± 0.00716 |
| IID | F Flip | minus_U minus Full | 6 | -0.680 ± 1.611 | 0.00098 ± 0.01109 | -0.00109 ± 0.00616 |
| IID | FedSA | Full | 6 | 88.894 ± 0.796 | 0.03453 ± 0.00524 | 0.10012 ± 0.00832 |
| IID | FedSA | minus_U | 6 | 88.859 ± 0.454 | 0.03224 ± 0.01016 | 0.09741 ± 0.00753 |
| IID | FedSA | minus_U minus Full | 6 | -0.034 ± 0.494 | -0.00230 ± 0.00674 | -0.00271 ± 0.00846 |
| IID | S-DFA | Full | 6 | 89.175 ± 0.783 | 0.03681 ± 0.00432 | 0.10370 ± 0.00558 |
| IID | S-DFA | minus_U | 6 | 88.251 ± 1.005 | 0.03181 ± 0.00828 | 0.09446 ± 0.01195 |
| IID | S-DFA | minus_U minus Full | 6 | -0.924 ± 0.926 | -0.00501 ± 0.01046 | -0.00924 ± 0.01226 |
| IID | Sp-DFA | Full | 6 | 88.713 ± 0.934 | 0.03446 ± 0.01385 | 0.09848 ± 0.01283 |
| IID | Sp-DFA | minus_U | 6 | 86.410 ± 6.083 | 0.03548 ± 0.01298 | 0.08910 ± 0.03224 |
| IID | Sp-DFA | minus_U minus Full | 6 | -2.303 ± 5.477 | 0.00102 ± 0.02599 | -0.00938 ± 0.04144 |
| non-IID | Benign | Full | 6 | 88.879 ± 1.416 | 0.02856 ± 0.00486 | 0.09666 ± 0.00846 |
| non-IID | Benign | minus_U | 6 | 87.966 ± 2.076 | 0.02638 ± 0.01182 | 0.09011 ± 0.02266 |
| non-IID | Benign | minus_U minus Full | 6 | -0.914 ± 0.823 | -0.00218 ± 0.01496 | -0.00655 ± 0.01471 |

## shared_calibration — All 10 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID | Benign | minus_U | 10 | 87.152 ± 1.796 | 0.01138 ± 0.00831 | 0.03982 ± 0.01737 |
| IID | Benign | minus_U minus Full | 10 | -1.107 ± 2.044 | 0.00164 ± 0.00571 | -0.02272 ± 0.01963 |
| IID | F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID | F Flip | minus_U | 10 | 88.082 ± 1.551 | 0.01276 ± 0.00997 | 0.05153 ± 0.02025 |
| IID | F Flip | minus_U minus Full | 10 | -0.309 ± 1.337 | 0.00208 ± 0.01164 | -0.00916 ± 0.02366 |
| IID | FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID | FedSA | minus_U | 10 | 88.143 ± 1.359 | 0.01061 ± 0.00888 | 0.05184 ± 0.01671 |
| IID | FedSA | minus_U minus Full | 10 | -0.327 ± 1.294 | 0.00453 ± 0.01009 | -0.01217 ± 0.01722 |
| IID | S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID | S-DFA | minus_U | 10 | 87.759 ± 1.228 | 0.01191 ± 0.01163 | 0.05051 ± 0.02281 |
| IID | S-DFA | minus_U minus Full | 10 | -0.931 ± 1.140 | 0.00421 ± 0.01321 | -0.01325 ± 0.02639 |
| IID | Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID | Sp-DFA | minus_U | 10 | 86.998 ± 4.345 | 0.01311 ± 0.01074 | 0.05267 ± 0.02474 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.384 ± 3.938 | -0.00337 ± 0.01161 | -0.00321 ± 0.03629 |
| non-IID | Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID | Benign | minus_U | 10 | 87.290 ± 1.870 | 0.01300 ± 0.00732 | 0.05140 ± 0.01598 |
| non-IID | Benign | minus_U minus Full | 10 | -1.303 ± 0.892 | 0.00604 ± 0.00899 | -0.01371 ± 0.01219 |

## shared_calibration — Exclude selection seed: 9 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID | Benign | minus_U | 9 | 87.107 ± 1.899 | 0.01125 ± 0.00881 | 0.03948 ± 0.01838 |
| IID | Benign | minus_U minus Full | 9 | -1.245 ± 2.117 | 0.00150 ± 0.00603 | -0.02284 ± 0.02082 |
| IID | F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID | F Flip | minus_U | 9 | 88.073 ± 1.645 | 0.01266 ± 0.01056 | 0.04963 ± 0.02052 |
| IID | F Flip | minus_U minus Full | 9 | -0.339 ± 1.415 | 0.00119 ± 0.01198 | -0.01080 ± 0.02448 |
| IID | FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID | FedSA | minus_U | 9 | 88.230 ± 1.412 | 0.01108 ± 0.00929 | 0.05251 ± 0.01758 |
| IID | FedSA | minus_U minus Full | 9 | -0.392 ± 1.355 | 0.00563 ± 0.01005 | -0.01424 ± 0.01691 |
| IID | S-DFA | Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID | S-DFA | minus_U | 9 | 87.925 ± 1.177 | 0.01199 ± 0.01234 | 0.05347 ± 0.02207 |
| IID | S-DFA | minus_U minus Full | 9 | -0.920 ± 1.208 | 0.00524 ± 0.01358 | -0.00982 ± 0.02552 |
| IID | Sp-DFA | Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID | Sp-DFA | minus_U | 9 | 86.979 ± 4.608 | 0.01314 ± 0.01140 | 0.05060 ± 0.02530 |
| IID | Sp-DFA | minus_U minus Full | 9 | -1.567 ± 4.131 | -0.00392 ± 0.01217 | -0.00489 ± 0.03808 |
| non-IID | Benign | Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID | Benign | minus_U | 9 | 87.245 ± 1.978 | 0.01230 ± 0.00740 | 0.04929 ± 0.01540 |
| non-IID | Benign | minus_U minus Full | 9 | -1.347 ± 0.935 | 0.00599 ± 0.00954 | -0.01456 ± 0.01261 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID | Benign | minus_U | 6 | 87.467 ± 1.093 | 0.01129 ± 0.00914 | 0.04017 ± 0.01468 |
| IID | Benign | minus_U minus Full | 6 | -1.091 ± 1.163 | -0.00009 ± 0.00619 | -0.02632 ± 0.01120 |
| IID | F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID | F Flip | minus_U | 6 | 87.676 ± 1.877 | 0.01099 ± 0.01064 | 0.04632 ± 0.02276 |
| IID | F Flip | minus_U minus Full | 6 | -0.612 ± 1.685 | -0.00385 ± 0.00832 | -0.01176 ± 0.02947 |
| IID | FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID | FedSA | minus_U | 6 | 88.576 ± 0.521 | 0.00728 ± 0.00729 | 0.05668 ± 0.00947 |
| IID | FedSA | minus_U minus Full | 6 | 0.007 ± 0.631 | 0.00178 ± 0.00926 | -0.01297 ± 0.01100 |
| IID | S-DFA | Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID | S-DFA | minus_U | 6 | 87.961 ± 0.921 | 0.01546 ± 0.01384 | 0.04828 ± 0.02248 |
| IID | S-DFA | minus_U minus Full | 6 | -0.784 ± 1.074 | 0.00843 ± 0.01564 | -0.01342 ± 0.02896 |
| IID | Sp-DFA | Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID | Sp-DFA | minus_U | 6 | 86.259 ± 5.627 | 0.01084 ± 0.00958 | 0.04996 ± 0.02617 |
| IID | Sp-DFA | minus_U minus Full | 6 | -2.095 ± 5.107 | -0.00626 ± 0.00504 | -0.00599 ± 0.04233 |
| non-IID | Benign | Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID | Benign | minus_U | 6 | 87.497 ± 2.201 | 0.01282 ± 0.00747 | 0.04494 ± 0.01300 |
| non-IID | Benign | minus_U minus Full | 6 | -1.088 ± 0.958 | 0.00659 ± 0.01056 | -0.01917 ± 0.01271 |

## Evidence and limits

Validation-only terminal checkpoints. This snapshot covers six complete Full–minus_U scenes, not all900 mechanism records or final test.

ACC is percent; AEOD is absolute TPR gap, not full equalized odds; ASPD is absolute positive-rate gap. Mean ± sample SD uses ddof=1. Differences are minus_U minus Full; ACC differences are percentage points.

Native includes each procedure’s original root-fitted calibration. Raw uses margin>0 (ties predict0); shared calibration uses the unchanged common root-only fit and >= group thresholds. These are three views of each same checkpoint, not three independent experiments.

For all120 displayed checkpoint records, native and shared-calibration saved metrics and confusion counts are identical. Their equality in this subset supplies no independent calibration-gain evidence; it does not select the final reporting endpoint.

Historical Full100 training used98 cu128 and2 cu130 records; current minus_U training used cu128/driver595. The displayed60 Full records include59 cu128 and1 cu130 (non-IID Benign91001). PyTorch-build and driver differences limit causal attribution.

Full replay mixes CPU/GPU; minus_U replay uses CPU. Per-ID runtime/source provenance is retained. This is an implementation/numerical validation snapshot, not a uniform-device final fairness comparison.

Seed91001 participated in recipe selection. Removing91001 or retaining91005–91010 applies equally to both procedures. All these validation seeds were previously exposed; neither panel is a prospectively untouched confirmation set.

The unchanged loader materialized full-split Smiling/Male metadata during original replays. No test image inference, test fitting or test selection is performed here; this local extraction reads no label arrays.

Completed scenes are included by coverage, independent of which variant wins. Improvements and regressions after deletion are retained. No significance test, Pareto claim, seed selection or claim that every component is necessary is made.

Native/shared main endpoint remains pending user choice. Other mechanism variants and four remaining non-IID scenes are incomplete in this frozen60-control snapshot; later native71 records are excluded.

Full identity join SHA256: `8c67ee29de18ef136d4bdba58c3b82318b371aeb461e93c389961c6d6abc5701`.
Original statistics source SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`.
