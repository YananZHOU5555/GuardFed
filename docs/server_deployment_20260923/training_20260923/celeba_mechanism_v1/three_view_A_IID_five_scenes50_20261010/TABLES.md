# CelebA Full–minus_A: five complete IID scenes, three views

IID Benign, F Flip, FedSA, S-DFA and Sp-DFA only. Validation (19,867 images), round70; ten paired model seeds per scene, with the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1). Paired difference = minus_A − Full; ACC is percent and ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| IID / Benign / minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_A | 10 | 88.392 ± 1.190 | 0.01069 ± 0.00706 | 0.05980 ± 0.01789 |
| IID / F Flip / minus_A minus Full | 10 | 0.001 ± 1.030 | 0.00001 ± 0.01228 | -0.00088 ± 0.02417 |
| IID / FedSA / Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID / FedSA / minus_A | 10 | 88.604 ± 1.348 | 0.00543 ± 0.00508 | 0.06543 ± 0.01689 |
| IID / FedSA / minus_A minus Full | 10 | 0.134 ± 0.985 | -0.00065 ± 0.00662 | 0.00141 ± 0.01497 |
| IID / S-DFA / Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID / S-DFA / minus_A | 10 | 88.328 ± 0.734 | 0.01311 ± 0.00684 | 0.05573 ± 0.01404 |
| IID / S-DFA / minus_A minus Full | 10 | -0.361 ± 0.901 | 0.00542 ± 0.00901 | -0.00804 ± 0.01754 |
| IID / Sp-DFA / Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID / Sp-DFA / minus_A | 10 | 88.152 ± 0.807 | 0.00697 ± 0.00296 | 0.05705 ± 0.00990 |
| IID / Sp-DFA / minus_A minus Full | 10 | -0.230 ± 0.746 | -0.00950 ± 0.00897 | 0.00117 ± 0.01870 |

## native — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| IID / Benign / minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_A | 9 | 88.382 ± 1.262 | 0.01122 ± 0.00728 | 0.05889 ± 0.01872 |
| IID / F Flip / minus_A minus Full | 9 | -0.030 ± 1.088 | -0.00024 ± 0.01300 | -0.00154 ± 0.02554 |
| IID / FedSA / Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID / FedSA / minus_A | 9 | 88.895 ± 1.044 | 0.00432 ± 0.00392 | 0.06988 ± 0.00993 |
| IID / FedSA / minus_A minus Full | 9 | 0.273 ± 0.933 | -0.00112 ± 0.00684 | 0.00313 ± 0.01480 |
| IID / S-DFA / Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID / S-DFA / minus_A | 9 | 88.367 ± 0.768 | 0.01362 ± 0.00705 | 0.05476 ± 0.01453 |
| IID / S-DFA / minus_A minus Full | 9 | -0.478 ± 0.872 | 0.00687 ± 0.00822 | -0.00853 ± 0.01853 |
| IID / Sp-DFA / Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID / Sp-DFA / minus_A | 9 | 88.272 ± 0.756 | 0.00705 ± 0.00313 | 0.05775 ± 0.01024 |
| IID / Sp-DFA / minus_A minus Full | 9 | -0.274 ± 0.777 | -0.01001 ± 0.00936 | 0.00226 ± 0.01949 |

## native — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| IID / Benign / minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_A | 6 | 88.090 ± 1.453 | 0.00958 ± 0.00537 | 0.05617 ± 0.01602 |
| IID / F Flip / minus_A minus Full | 6 | -0.198 ± 1.213 | -0.00526 ± 0.01077 | -0.00190 ± 0.02730 |
| IID / FedSA / Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID / FedSA / minus_A | 6 | 88.564 ± 1.127 | 0.00371 ± 0.00249 | 0.06690 ± 0.00935 |
| IID / FedSA / minus_A minus Full | 6 | -0.005 ± 0.983 | -0.00179 ± 0.00515 | -0.00276 ± 0.01435 |
| IID / S-DFA / Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID / S-DFA / minus_A | 6 | 88.491 ± 0.715 | 0.01312 ± 0.00636 | 0.05497 ± 0.00805 |
| IID / S-DFA / minus_A minus Full | 6 | -0.254 ± 0.535 | 0.00609 ± 0.00817 | -0.00674 ± 0.01486 |
| IID / Sp-DFA / Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID / Sp-DFA / minus_A | 6 | 88.185 ± 0.731 | 0.00694 ± 0.00230 | 0.05833 ± 0.01110 |
| IID / Sp-DFA / minus_A minus Full | 6 | -0.169 ± 0.749 | -0.01016 ± 0.01011 | 0.00239 ± 0.02216 |

## raw — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID / Benign / minus_A | 10 | 88.214 ± 1.695 | 0.04092 ± 0.00933 | 0.10124 ± 0.00931 |
| IID / Benign / minus_A minus Full | 10 | -0.335 ± 1.319 | 0.00064 ± 0.01538 | -0.00184 ± 0.01319 |
| IID / F Flip / Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID / F Flip / minus_A | 10 | 88.774 ± 1.169 | 0.03945 ± 0.01694 | 0.10525 ± 0.00936 |
| IID / F Flip / minus_A minus Full | 10 | 0.070 ± 1.194 | 0.00261 ± 0.01613 | 0.00445 ± 0.00985 |
| IID / FedSA / Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID / FedSA / minus_A | 10 | 88.996 ± 1.217 | 0.04106 ± 0.00989 | 0.10687 ± 0.00661 |
| IID / FedSA / minus_A minus Full | 10 | 0.164 ± 0.771 | 0.00390 ± 0.01159 | 0.00430 ± 0.01102 |
| IID / S-DFA / Full | 10 | 89.059 ± 0.840 | 0.03671 ± 0.00377 | 0.10194 ± 0.00664 |
| IID / S-DFA / minus_A | 10 | 88.757 ± 0.821 | 0.03773 ± 0.00712 | 0.10191 ± 0.00517 |
| IID / S-DFA / minus_A minus Full | 10 | -0.302 ± 0.939 | 0.00102 ± 0.00865 | -0.00003 ± 0.00644 |
| IID / Sp-DFA / Full | 10 | 88.766 ± 0.924 | 0.03465 ± 0.01394 | 0.09978 ± 0.01059 |
| IID / Sp-DFA / minus_A | 10 | 88.500 ± 0.783 | 0.04196 ± 0.01721 | 0.10123 ± 0.01657 |
| IID / Sp-DFA / minus_A minus Full | 10 | -0.266 ± 0.612 | 0.00731 ± 0.02104 | 0.00145 ± 0.01756 |

## raw — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID / Benign / minus_A | 9 | 88.676 ± 0.910 | 0.04071 ± 0.00987 | 0.10366 ± 0.00560 |
| IID / Benign / minus_A minus Full | 9 | 0.030 ± 0.675 | 0.00067 ± 0.01631 | 0.00028 ± 0.01204 |
| IID / F Flip / Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID / F Flip / minus_A | 9 | 88.763 ± 1.240 | 0.03930 ± 0.01796 | 0.10519 ± 0.00992 |
| IID / F Flip / minus_A minus Full | 9 | 0.032 ± 1.260 | 0.00161 ± 0.01677 | 0.00318 ± 0.00954 |
| IID / FedSA / Full | 9 | 89.001 ± 0.767 | 0.03797 ± 0.00689 | 0.10454 ± 0.01032 |
| IID / FedSA / minus_A | 9 | 89.269 ± 0.911 | 0.03894 ± 0.00773 | 0.10614 ± 0.00657 |
| IID / FedSA / minus_A minus Full | 9 | 0.267 ± 0.740 | 0.00097 ± 0.00740 | 0.00160 ± 0.00741 |
| IID / S-DFA / Full | 9 | 89.233 ± 0.673 | 0.03652 ± 0.00395 | 0.10342 ± 0.00500 |
| IID / S-DFA / minus_A | 9 | 88.801 ± 0.859 | 0.03781 ± 0.00754 | 0.10248 ± 0.00513 |
| IID / S-DFA / minus_A minus Full | 9 | -0.432 ± 0.894 | 0.00130 ± 0.00913 | -0.00094 ± 0.00612 |
| IID / Sp-DFA / Full | 9 | 88.925 ± 0.824 | 0.03222 ± 0.01233 | 0.09968 ± 0.01123 |
| IID / Sp-DFA / minus_A | 9 | 88.619 ± 0.728 | 0.04238 ± 0.01820 | 0.10220 ± 0.01727 |
| IID / Sp-DFA / minus_A minus Full | 9 | -0.306 ± 0.635 | 0.01016 ± 0.02016 | 0.00252 ± 0.01827 |

## raw — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID / Benign / minus_A | 6 | 88.683 ± 0.631 | 0.04186 ± 0.00928 | 0.10352 ± 0.00430 |
| IID / Benign / minus_A minus Full | 6 | -0.130 ± 0.778 | -0.00040 ± 0.01774 | -0.00358 ± 0.01009 |
| IID / F Flip / Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID / F Flip / minus_A | 6 | 88.493 ± 1.430 | 0.04089 ± 0.02240 | 0.10482 ± 0.01182 |
| IID / F Flip / minus_A minus Full | 6 | -0.159 ± 1.343 | 0.00443 ± 0.01417 | 0.00369 ± 0.00911 |
| IID / FedSA / Full | 6 | 88.894 ± 0.796 | 0.03453 ± 0.00524 | 0.10012 ± 0.00832 |
| IID / FedSA / minus_A | 6 | 89.005 ± 1.000 | 0.03717 ± 0.00593 | 0.10292 ± 0.00230 |
| IID / FedSA / minus_A minus Full | 6 | 0.112 ± 0.812 | 0.00264 ± 0.00726 | 0.00280 ± 0.00806 |
| IID / S-DFA / Full | 6 | 89.175 ± 0.783 | 0.03681 ± 0.00432 | 0.10370 ± 0.00558 |
| IID / S-DFA / minus_A | 6 | 88.964 ± 0.813 | 0.03741 ± 0.00650 | 0.10101 ± 0.00517 |
| IID / S-DFA / minus_A minus Full | 6 | -0.211 ± 0.505 | 0.00060 ± 0.00661 | -0.00269 ± 0.00552 |
| IID / Sp-DFA / Full | 6 | 88.713 ± 0.934 | 0.03446 ± 0.01385 | 0.09848 ± 0.01283 |
| IID / Sp-DFA / minus_A | 6 | 88.520 ± 0.771 | 0.03705 ± 0.00732 | 0.09719 ± 0.00875 |
| IID / Sp-DFA / minus_A minus Full | 6 | -0.193 ± 0.711 | 0.00260 ± 0.01612 | -0.00129 ± 0.01838 |

## shared_calibration — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| IID / Benign / minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_A | 10 | 88.392 ± 1.190 | 0.01069 ± 0.00706 | 0.05980 ± 0.01789 |
| IID / F Flip / minus_A minus Full | 10 | 0.001 ± 1.030 | 0.00001 ± 0.01228 | -0.00088 ± 0.02417 |
| IID / FedSA / Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID / FedSA / minus_A | 10 | 88.604 ± 1.348 | 0.00543 ± 0.00508 | 0.06543 ± 0.01689 |
| IID / FedSA / minus_A minus Full | 10 | 0.134 ± 0.985 | -0.00065 ± 0.00662 | 0.00141 ± 0.01497 |
| IID / S-DFA / Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID / S-DFA / minus_A | 10 | 88.328 ± 0.734 | 0.01311 ± 0.00684 | 0.05573 ± 0.01404 |
| IID / S-DFA / minus_A minus Full | 10 | -0.361 ± 0.901 | 0.00542 ± 0.00901 | -0.00804 ± 0.01754 |
| IID / Sp-DFA / Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID / Sp-DFA / minus_A | 10 | 88.152 ± 0.807 | 0.00697 ± 0.00296 | 0.05705 ± 0.00990 |
| IID / Sp-DFA / minus_A minus Full | 10 | -0.230 ± 0.746 | -0.00950 ± 0.00897 | 0.00117 ± 0.01870 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| IID / Benign / minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_A | 9 | 88.382 ± 1.262 | 0.01122 ± 0.00728 | 0.05889 ± 0.01872 |
| IID / F Flip / minus_A minus Full | 9 | -0.030 ± 1.088 | -0.00024 ± 0.01300 | -0.00154 ± 0.02554 |
| IID / FedSA / Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID / FedSA / minus_A | 9 | 88.895 ± 1.044 | 0.00432 ± 0.00392 | 0.06988 ± 0.00993 |
| IID / FedSA / minus_A minus Full | 9 | 0.273 ± 0.933 | -0.00112 ± 0.00684 | 0.00313 ± 0.01480 |
| IID / S-DFA / Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID / S-DFA / minus_A | 9 | 88.367 ± 0.768 | 0.01362 ± 0.00705 | 0.05476 ± 0.01453 |
| IID / S-DFA / minus_A minus Full | 9 | -0.478 ± 0.872 | 0.00687 ± 0.00822 | -0.00853 ± 0.01853 |
| IID / Sp-DFA / Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID / Sp-DFA / minus_A | 9 | 88.272 ± 0.756 | 0.00705 ± 0.00313 | 0.05775 ± 0.01024 |
| IID / Sp-DFA / minus_A minus Full | 9 | -0.274 ± 0.777 | -0.01001 ± 0.00936 | 0.00226 ± 0.01949 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| IID / Benign / minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_A | 6 | 88.090 ± 1.453 | 0.00958 ± 0.00537 | 0.05617 ± 0.01602 |
| IID / F Flip / minus_A minus Full | 6 | -0.198 ± 1.213 | -0.00526 ± 0.01077 | -0.00190 ± 0.02730 |
| IID / FedSA / Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID / FedSA / minus_A | 6 | 88.564 ± 1.127 | 0.00371 ± 0.00249 | 0.06690 ± 0.00935 |
| IID / FedSA / minus_A minus Full | 6 | -0.005 ± 0.983 | -0.00179 ± 0.00515 | -0.00276 ± 0.01435 |
| IID / S-DFA / Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID / S-DFA / minus_A | 6 | 88.491 ± 0.715 | 0.01312 ± 0.00636 | 0.05497 ± 0.00805 |
| IID / S-DFA / minus_A minus Full | 6 | -0.254 ± 0.535 | 0.00609 ± 0.00817 | -0.00674 ± 0.01486 |
| IID / Sp-DFA / Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID / Sp-DFA / minus_A | 6 | 88.185 ± 0.731 | 0.00694 ± 0.00230 | 0.05833 ± 0.01110 |
| IID / Sp-DFA / minus_A minus Full | 6 | -0.169 ± 0.749 | -0.01016 ± 0.01011 | 0.00239 ± 0.02216 |

AEOD is the absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0). Native retains each original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. The three views are parallel descriptions, with no endpoint selected or threshold refitted by this builder.
Actual replay devices across50 pairs: Full {'cpu': 3, 'cuda:0': 47}; minus_A {'cpu': 50}. Training Torch: Full {'2.11.0+cu128': 50}; minus_A {'2.11.0+cu128': 50}. Per-record configuration, source, checkpoint, environment and driver provenance remain in records.json. Broader Full100 history includes98 cu128 and2 cu130 records; these50 actual source records, not that broader count, define this table.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets. No final test was run for this table.
All negative and constant outcomes are retained. These are all five IID minus_A scenes; all five non-IID scenes remain outside this delivery (one non-IID Benign seed is retained in the accepted source index only). The separate five-IID-scene aggregate first averages within each model seed, and scenes are not treated as independent model seeds. This is not A100 or completion of all mechanism controls. No significance, necessity, causal-isolation or whole-rebuttal-completion claim is made.

# Five IID scenes: seed-first aggregate

AUTHOR-REVIEW CANDIDATE. Each model seed contributes once after its five IID scenes are equally averaged. This is not a balanced ten-scene or non-IID aggregate.

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.438 ± 0.670 | 0.01013 ± 0.00323 | 0.06138 ± 0.00730 |
| minus_A | 10 | 88.261 ± 0.897 | 0.00965 ± 0.00324 | 0.05948 ± 0.00567 |
| minus_A minus Full | 10 | -0.177 ± 0.454 | -0.00048 ± 0.00533 | -0.00190 ± 0.00941 |

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.555 ± 0.592 | 0.01009 ± 0.00342 | 0.06165 ± 0.00768 |
| minus_A | 9 | 88.436 ± 0.748 | 0.00940 ± 0.00333 | 0.06015 ± 0.00557 |
| minus_A minus Full | 9 | -0.119 ± 0.440 | -0.00070 ± 0.00561 | -0.00150 ± 0.00990 |

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.503 ± 0.634 | 0.01117 ± 0.00371 | 0.06237 ± 0.00943 |
| minus_A | 6 | 88.315 ± 0.851 | 0.00886 ± 0.00238 | 0.05965 ± 0.00572 |
| minus_A minus Full | 6 | -0.187 ± 0.441 | -0.00231 ± 0.00594 | -0.00272 ± 0.01100 |

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.782 ± 0.693 | 0.03713 ± 0.00423 | 0.10163 ± 0.00429 |
| minus_A | 10 | 88.648 ± 0.888 | 0.04022 ± 0.00701 | 0.10330 ± 0.00530 |
| minus_A minus Full | 10 | -0.134 ± 0.400 | 0.00310 ± 0.00449 | 0.00167 ± 0.00275 |

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.907 ± 0.604 | 0.03689 ± 0.00442 | 0.10260 ± 0.00318 |
| minus_A | 9 | 88.825 ± 0.730 | 0.03983 ± 0.00732 | 0.10393 ± 0.00520 |
| minus_A minus Full | 9 | -0.082 ± 0.386 | 0.00294 ± 0.00473 | 0.00133 ± 0.00269 |

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.849 ± 0.640 | 0.03690 ± 0.00488 | 0.10211 ± 0.00289 |
| minus_A | 6 | 88.733 ± 0.840 | 0.03887 ± 0.00725 | 0.10189 ± 0.00335 |
| minus_A minus Full | 6 | -0.116 ± 0.383 | 0.00197 ± 0.00466 | -0.00021 ± 0.00088 |

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.438 ± 0.670 | 0.01013 ± 0.00323 | 0.06138 ± 0.00730 |
| minus_A | 10 | 88.261 ± 0.897 | 0.00965 ± 0.00324 | 0.05948 ± 0.00567 |
| minus_A minus Full | 10 | -0.177 ± 0.454 | -0.00048 ± 0.00533 | -0.00190 ± 0.00941 |

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.555 ± 0.592 | 0.01009 ± 0.00342 | 0.06165 ± 0.00768 |
| minus_A | 9 | 88.436 ± 0.748 | 0.00940 ± 0.00333 | 0.06015 ± 0.00557 |
| minus_A minus Full | 9 | -0.119 ± 0.440 | -0.00070 ± 0.00561 | -0.00150 ± 0.00990 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.503 ± 0.634 | 0.01117 ± 0.00371 | 0.06237 ± 0.00943 |
| minus_A | 6 | 88.315 ± 0.851 | 0.00886 ± 0.00238 | 0.05965 ± 0.00572 |
| minus_A minus Full | 6 | -0.187 ± 0.441 | -0.00231 ± 0.00594 | -0.00272 ± 0.01100 |
