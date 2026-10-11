# CelebA Full–minus_F: five IID scenes and non-IID Benign/F Flip/FedSA, three views

Validation only (19,867 images), round70; IID is frozen Dirichlet alpha5000; non-IID scenes use alpha5. Ten paired model seeds per scene; each model uses the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1); paired difference = minus_F − Full. ACC is percent; ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| IID / Benign / minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_F | 10 | 88.325 ± 1.212 | 0.01439 ± 0.00651 | 0.06600 ± 0.01819 |
| IID / F Flip / minus_F minus Full | 10 | -0.066 ± 1.239 | 0.00372 ± 0.00942 | 0.00532 ± 0.02066 |
| IID / FedSA / Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID / FedSA / minus_F | 10 | 88.606 ± 1.323 | 0.00929 ± 0.00785 | 0.06794 ± 0.01401 |
| IID / FedSA / minus_F minus Full | 10 | 0.136 ± 1.125 | 0.00322 ± 0.00911 | 0.00393 ± 0.01621 |
| IID / S-DFA / Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID / S-DFA / minus_F | 10 | 88.412 ± 1.202 | 0.01054 ± 0.00697 | 0.06941 ± 0.01477 |
| IID / S-DFA / minus_F minus Full | 10 | -0.277 ± 1.091 | 0.00284 ± 0.00705 | 0.00564 ± 0.01932 |
| IID / Sp-DFA / Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID / Sp-DFA / minus_F | 10 | 88.499 ± 1.171 | 0.00987 ± 0.00716 | 0.07313 ± 0.01145 |
| IID / Sp-DFA / minus_F minus Full | 10 | 0.117 ± 0.995 | -0.00661 ± 0.01017 | 0.01725 ± 0.01433 |
| non-IID / Benign / Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID / Benign / minus_F | 10 | 88.768 ± 0.936 | 0.00923 ± 0.00704 | 0.06936 ± 0.01266 |
| non-IID / Benign / minus_F minus Full | 10 | 0.175 ± 1.283 | 0.00227 ± 0.00752 | 0.00425 ± 0.01616 |
| non-IID / F Flip / Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID / F Flip / minus_F | 10 | 88.449 ± 1.046 | 0.00889 ± 0.00742 | 0.06411 ± 0.01444 |
| non-IID / F Flip / minus_F minus Full | 10 | 0.006 ± 0.846 | -0.00113 ± 0.01122 | 0.00518 ± 0.01759 |
| non-IID / FedSA / Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID / FedSA / minus_F | 10 | 88.174 ± 1.292 | 0.01138 ± 0.00960 | 0.06204 ± 0.00740 |
| non-IID / FedSA / minus_F minus Full | 10 | -0.239 ± 1.297 | 0.00359 ± 0.01201 | 0.00047 ± 0.01507 |

## native — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| IID / Benign / minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_F | 9 | 88.657 ± 0.642 | 0.01473 ± 0.00681 | 0.07064 ± 0.01141 |
| IID / F Flip / minus_F minus Full | 9 | 0.244 ± 0.802 | 0.00327 ± 0.00987 | 0.01021 ± 0.01452 |
| IID / FedSA / Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID / FedSA / minus_F | 9 | 88.979 ± 0.630 | 0.00927 ± 0.00832 | 0.06996 ± 0.01323 |
| IID / FedSA / minus_F minus Full | 9 | 0.358 ± 0.933 | 0.00382 ± 0.00944 | 0.00322 ± 0.01703 |
| IID / S-DFA / Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID / S-DFA / minus_F | 9 | 88.709 ± 0.799 | 0.01100 ± 0.00722 | 0.07284 ± 0.01063 |
| IID / S-DFA / minus_F minus Full | 9 | -0.136 ± 1.057 | 0.00425 ± 0.00578 | 0.00955 ± 0.01575 |
| IID / Sp-DFA / Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID / Sp-DFA / minus_F | 9 | 88.785 ± 0.790 | 0.00862 ± 0.00634 | 0.07407 ± 0.01174 |
| IID / Sp-DFA / minus_F minus Full | 9 | 0.239 ± 0.973 | -0.00844 ± 0.00888 | 0.01858 ± 0.01453 |
| non-IID / Benign / Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID / Benign / minus_F | 9 | 88.927 ± 0.838 | 0.00892 ± 0.00739 | 0.06966 ± 0.01339 |
| non-IID / Benign / minus_F minus Full | 9 | 0.336 ± 1.248 | 0.00260 ± 0.00790 | 0.00582 ± 0.01631 |
| non-IID / F Flip / Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID / F Flip / minus_F | 9 | 88.634 ± 0.919 | 0.00929 ± 0.00776 | 0.06614 ± 0.01372 |
| non-IID / F Flip / minus_F minus Full | 9 | 0.130 ± 0.795 | -0.00110 ± 0.01190 | 0.00827 ± 0.01551 |
| non-IID / FedSA / Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID / FedSA / minus_F | 9 | 88.269 ± 1.332 | 0.01038 ± 0.00961 | 0.06090 ± 0.00686 |
| non-IID / FedSA / minus_F minus Full | 9 | -0.125 ± 1.321 | 0.00212 ± 0.01175 | 0.00002 ± 0.01592 |

## native — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| IID / Benign / minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_F | 6 | 88.333 ± 0.500 | 0.01745 ± 0.00682 | 0.07303 ± 0.01359 |
| IID / F Flip / minus_F minus Full | 6 | 0.045 ± 0.928 | 0.00261 ± 0.01214 | 0.01496 ± 0.01590 |
| IID / FedSA / Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID / FedSA / minus_F | 6 | 88.721 ± 0.622 | 0.00677 ± 0.00740 | 0.06405 ± 0.00902 |
| IID / FedSA / minus_F minus Full | 6 | 0.152 ± 0.995 | 0.00127 ± 0.00805 | -0.00560 ± 0.01074 |
| IID / S-DFA / Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID / S-DFA / minus_F | 6 | 88.349 ± 0.745 | 0.01056 ± 0.00767 | 0.07202 ± 0.00890 |
| IID / S-DFA / minus_F minus Full | 6 | -0.396 ± 1.212 | 0.00353 ± 0.00614 | 0.01031 ± 0.01691 |
| IID / Sp-DFA / Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID / Sp-DFA / minus_F | 6 | 88.483 ± 0.683 | 0.00881 ± 0.00703 | 0.07089 ± 0.01131 |
| IID / Sp-DFA / minus_F minus Full | 6 | 0.129 ± 1.172 | -0.00829 ± 0.01089 | 0.01495 ± 0.01241 |
| non-IID / Benign / Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID / Benign / minus_F | 6 | 88.557 ± 0.762 | 0.00839 ± 0.00825 | 0.06565 ± 0.01246 |
| non-IID / Benign / minus_F minus Full | 6 | -0.028 ± 1.290 | 0.00216 ± 0.00899 | 0.00154 ± 0.01804 |
| non-IID / F Flip / Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID / F Flip / minus_F | 6 | 88.480 ± 1.031 | 0.00888 ± 0.00952 | 0.06483 ± 0.01679 |
| non-IID / F Flip / minus_F minus Full | 6 | 0.049 ± 0.974 | -0.00214 ± 0.01489 | 0.00787 ± 0.01663 |
| non-IID / FedSA / Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID / FedSA / minus_F | 6 | 88.819 ± 0.555 | 0.00726 ± 0.00669 | 0.06242 ± 0.00529 |
| non-IID / FedSA / minus_F minus Full | 6 | 0.331 ± 0.630 | -0.00105 ± 0.00773 | 0.00582 ± 0.00998 |

## raw — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID / Benign / minus_F | 10 | 88.778 ± 0.921 | 0.04605 ± 0.00944 | 0.10965 ± 0.00771 |
| IID / Benign / minus_F minus Full | 10 | 0.229 ± 1.374 | 0.00578 ± 0.01538 | 0.00658 ± 0.01550 |
| IID / F Flip / Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID / F Flip / minus_F | 10 | 88.695 ± 1.220 | 0.04510 ± 0.01187 | 0.10941 ± 0.01526 |
| IID / F Flip / minus_F minus Full | 10 | -0.009 ± 1.303 | 0.00826 ± 0.01806 | 0.00861 ± 0.01499 |
| IID / FedSA / Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID / FedSA / minus_F | 10 | 88.962 ± 1.249 | 0.03793 ± 0.00601 | 0.10419 ± 0.01005 |
| IID / FedSA / minus_F minus Full | 10 | 0.129 ± 0.943 | 0.00077 ± 0.00767 | 0.00162 ± 0.00787 |
| IID / S-DFA / Full | 10 | 89.059 ± 0.840 | 0.03671 ± 0.00377 | 0.10194 ± 0.00664 |
| IID / S-DFA / minus_F | 10 | 88.750 ± 1.152 | 0.04140 ± 0.00906 | 0.10505 ± 0.00950 |
| IID / S-DFA / minus_F minus Full | 10 | -0.309 ± 1.100 | 0.00469 ± 0.01091 | 0.00312 ± 0.00945 |
| IID / Sp-DFA / Full | 10 | 88.766 ± 0.924 | 0.03465 ± 0.01394 | 0.09978 ± 0.01059 |
| IID / Sp-DFA / minus_F | 10 | 88.778 ± 1.151 | 0.04543 ± 0.00839 | 0.11031 ± 0.00795 |
| IID / Sp-DFA / minus_F minus Full | 10 | 0.012 ± 1.003 | 0.01077 ± 0.01216 | 0.01053 ± 0.00748 |
| non-IID / Benign / Full | 10 | 88.875 ± 1.209 | 0.03157 ± 0.00643 | 0.09978 ± 0.00770 |
| non-IID / Benign / minus_F | 10 | 89.005 ± 0.927 | 0.03993 ± 0.01001 | 0.10702 ± 0.01067 |
| non-IID / Benign / minus_F minus Full | 10 | 0.130 ± 1.188 | 0.00835 ± 0.01020 | 0.00724 ± 0.01106 |
| non-IID / F Flip / Full | 10 | 88.728 ± 0.717 | 0.03162 ± 0.00730 | 0.09947 ± 0.00578 |
| non-IID / F Flip / minus_F | 10 | 88.835 ± 1.022 | 0.04530 ± 0.00972 | 0.10929 ± 0.00982 |
| non-IID / F Flip / minus_F minus Full | 10 | 0.108 ± 0.785 | 0.01367 ± 0.01175 | 0.00982 ± 0.01137 |
| non-IID / FedSA / Full | 10 | 88.678 ± 0.894 | 0.03531 ± 0.00824 | 0.09889 ± 0.01124 |
| non-IID / FedSA / minus_F | 10 | 88.509 ± 1.374 | 0.03638 ± 0.01136 | 0.09983 ± 0.00368 |
| non-IID / FedSA / minus_F minus Full | 10 | -0.169 ± 1.337 | 0.00107 ± 0.01448 | 0.00094 ± 0.01116 |

## raw — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID / Benign / minus_F | 9 | 88.978 ± 0.709 | 0.04401 ± 0.00729 | 0.10907 ± 0.00794 |
| IID / Benign / minus_F minus Full | 9 | 0.333 ± 1.415 | 0.00397 ± 0.01514 | 0.00569 ± 0.01616 |
| IID / F Flip / Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID / F Flip / minus_F | 9 | 89.026 ± 0.661 | 0.04664 ± 0.01148 | 0.11306 ± 0.01058 |
| IID / F Flip / minus_F minus Full | 9 | 0.296 ± 0.932 | 0.00894 ± 0.01902 | 0.01106 ± 0.01361 |
| IID / FedSA / Full | 9 | 89.001 ± 0.767 | 0.03797 ± 0.00689 | 0.10454 ± 0.01032 |
| IID / FedSA / minus_F | 9 | 89.327 ± 0.501 | 0.03830 ± 0.00625 | 0.10675 ± 0.00628 |
| IID / FedSA / minus_F minus Full | 9 | 0.326 ± 0.752 | 0.00033 ± 0.00800 | 0.00221 ± 0.00810 |
| IID / S-DFA / Full | 9 | 89.233 ± 0.673 | 0.03652 ± 0.00395 | 0.10342 ± 0.00500 |
| IID / S-DFA / minus_F | 9 | 89.019 ± 0.822 | 0.04206 ± 0.00936 | 0.10735 ± 0.00648 |
| IID / S-DFA / minus_F minus Full | 9 | -0.214 ± 1.122 | 0.00555 ± 0.01121 | 0.00394 ± 0.00964 |
| IID / Sp-DFA / Full | 9 | 88.925 ± 0.824 | 0.03222 ± 0.01233 | 0.09968 ± 0.01123 |
| IID / Sp-DFA / minus_F | 9 | 89.075 ± 0.707 | 0.04399 ± 0.00748 | 0.11131 ± 0.00774 |
| IID / Sp-DFA / minus_F minus Full | 9 | 0.150 ± 0.957 | 0.01177 ± 0.01245 | 0.01163 ± 0.00703 |
| non-IID / Benign / Full | 9 | 88.877 ± 1.282 | 0.03006 ± 0.00454 | 0.09914 ± 0.00788 |
| non-IID / Benign / minus_F | 9 | 89.162 ± 0.830 | 0.03941 ± 0.01048 | 0.10789 ± 0.01094 |
| non-IID / Benign / minus_F minus Full | 9 | 0.285 ± 1.148 | 0.00936 ± 0.01029 | 0.00875 ± 0.01057 |
| non-IID / F Flip / Full | 9 | 88.811 ± 0.708 | 0.03131 ± 0.00768 | 0.09975 ± 0.00605 |
| non-IID / F Flip / minus_F | 9 | 89.020 ± 0.890 | 0.04537 ± 0.01031 | 0.11078 ± 0.00914 |
| non-IID / F Flip / minus_F minus Full | 9 | 0.209 ± 0.759 | 0.01406 ± 0.01240 | 0.01103 ± 0.01136 |
| non-IID / FedSA / Full | 9 | 88.672 ± 0.948 | 0.03560 ± 0.00868 | 0.09942 ± 0.01178 |
| non-IID / FedSA / minus_F | 9 | 88.620 ± 1.408 | 0.03481 ± 0.01083 | 0.09936 ± 0.00356 |
| non-IID / FedSA / minus_F minus Full | 9 | -0.052 ± 1.363 | -0.00079 ± 0.01404 | -0.00006 ± 0.01134 |

## raw — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID / Benign / minus_F | 6 | 88.821 ± 0.706 | 0.04104 ± 0.00709 | 0.10609 ± 0.00591 |
| IID / Benign / minus_F minus Full | 6 | 0.008 ± 1.004 | -0.00121 ± 0.01600 | -0.00101 ± 0.01015 |
| IID / F Flip / Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID / F Flip / minus_F | 6 | 88.728 ± 0.564 | 0.04872 ± 0.01196 | 0.11191 ± 0.01134 |
| IID / F Flip / minus_F minus Full | 6 | 0.076 ± 1.088 | 0.01226 ± 0.02037 | 0.01078 ± 0.01345 |
| IID / FedSA / Full | 6 | 88.894 ± 0.796 | 0.03453 ± 0.00524 | 0.10012 ± 0.00832 |
| IID / FedSA / minus_F | 6 | 89.132 ± 0.498 | 0.03786 ± 0.00509 | 0.10437 ± 0.00397 |
| IID / FedSA / minus_F minus Full | 6 | 0.238 ± 0.829 | 0.00333 ± 0.00757 | 0.00425 ± 0.00942 |
| IID / S-DFA / Full | 6 | 89.175 ± 0.783 | 0.03681 ± 0.00432 | 0.10370 ± 0.00558 |
| IID / S-DFA / minus_F | 6 | 88.634 ± 0.724 | 0.04388 ± 0.00989 | 0.10548 ± 0.00586 |
| IID / S-DFA / minus_F minus Full | 6 | -0.540 ± 1.241 | 0.00707 ± 0.01143 | 0.00178 ± 0.01068 |
| IID / Sp-DFA / Full | 6 | 88.713 ± 0.934 | 0.03446 ± 0.01385 | 0.09848 ± 0.01283 |
| IID / Sp-DFA / minus_F | 6 | 88.832 ± 0.651 | 0.04722 ± 0.00615 | 0.11155 ± 0.00692 |
| IID / Sp-DFA / minus_F minus Full | 6 | 0.118 ± 1.197 | 0.01277 ± 0.01524 | 0.01307 ± 0.00808 |
| non-IID / Benign / Full | 6 | 88.879 ± 1.416 | 0.02856 ± 0.00486 | 0.09666 ± 0.00846 |
| non-IID / Benign / minus_F | 6 | 88.855 ± 0.856 | 0.03480 ± 0.00731 | 0.10199 ± 0.00598 |
| non-IID / Benign / minus_F minus Full | 6 | -0.024 ± 1.140 | 0.00624 ± 0.00882 | 0.00533 ± 0.01116 |
| non-IID / F Flip / Full | 6 | 88.741 ± 0.837 | 0.02861 ± 0.00802 | 0.09658 ± 0.00263 |
| non-IID / F Flip / minus_F | 6 | 88.879 ± 1.046 | 0.04636 ± 0.00835 | 0.11047 ± 0.01129 |
| non-IID / F Flip / minus_F minus Full | 6 | 0.138 ± 0.945 | 0.01775 ± 0.01021 | 0.01390 ± 0.01205 |
| non-IID / FedSA / Full | 6 | 88.800 ± 0.820 | 0.03206 ± 0.00774 | 0.09659 ± 0.00664 |
| non-IID / FedSA / minus_F | 6 | 89.166 ± 0.624 | 0.03086 ± 0.00251 | 0.10007 ± 0.00391 |
| non-IID / FedSA / minus_F minus Full | 6 | 0.367 ± 0.442 | -0.00120 ± 0.00926 | 0.00348 ± 0.00669 |

## shared_calibration — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| IID / Benign / minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_F | 10 | 88.325 ± 1.212 | 0.01439 ± 0.00651 | 0.06600 ± 0.01819 |
| IID / F Flip / minus_F minus Full | 10 | -0.066 ± 1.239 | 0.00372 ± 0.00942 | 0.00532 ± 0.02066 |
| IID / FedSA / Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID / FedSA / minus_F | 10 | 88.606 ± 1.323 | 0.00929 ± 0.00785 | 0.06794 ± 0.01401 |
| IID / FedSA / minus_F minus Full | 10 | 0.136 ± 1.125 | 0.00322 ± 0.00911 | 0.00393 ± 0.01621 |
| IID / S-DFA / Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID / S-DFA / minus_F | 10 | 88.412 ± 1.202 | 0.01054 ± 0.00697 | 0.06941 ± 0.01477 |
| IID / S-DFA / minus_F minus Full | 10 | -0.277 ± 1.091 | 0.00284 ± 0.00705 | 0.00564 ± 0.01932 |
| IID / Sp-DFA / Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID / Sp-DFA / minus_F | 10 | 88.499 ± 1.171 | 0.00987 ± 0.00716 | 0.07313 ± 0.01145 |
| IID / Sp-DFA / minus_F minus Full | 10 | 0.117 ± 0.995 | -0.00661 ± 0.01017 | 0.01725 ± 0.01433 |
| non-IID / Benign / Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID / Benign / minus_F | 10 | 88.768 ± 0.936 | 0.00923 ± 0.00704 | 0.06936 ± 0.01266 |
| non-IID / Benign / minus_F minus Full | 10 | 0.175 ± 1.283 | 0.00227 ± 0.00752 | 0.00425 ± 0.01616 |
| non-IID / F Flip / Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID / F Flip / minus_F | 10 | 88.449 ± 1.046 | 0.00889 ± 0.00742 | 0.06411 ± 0.01444 |
| non-IID / F Flip / minus_F minus Full | 10 | 0.006 ± 0.846 | -0.00113 ± 0.01122 | 0.00518 ± 0.01759 |
| non-IID / FedSA / Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID / FedSA / minus_F | 10 | 88.174 ± 1.292 | 0.01138 ± 0.00960 | 0.06204 ± 0.00740 |
| non-IID / FedSA / minus_F minus Full | 10 | -0.239 ± 1.297 | 0.00359 ± 0.01201 | 0.00047 ± 0.01507 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| IID / Benign / minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_F | 9 | 88.657 ± 0.642 | 0.01473 ± 0.00681 | 0.07064 ± 0.01141 |
| IID / F Flip / minus_F minus Full | 9 | 0.244 ± 0.802 | 0.00327 ± 0.00987 | 0.01021 ± 0.01452 |
| IID / FedSA / Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID / FedSA / minus_F | 9 | 88.979 ± 0.630 | 0.00927 ± 0.00832 | 0.06996 ± 0.01323 |
| IID / FedSA / minus_F minus Full | 9 | 0.358 ± 0.933 | 0.00382 ± 0.00944 | 0.00322 ± 0.01703 |
| IID / S-DFA / Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID / S-DFA / minus_F | 9 | 88.709 ± 0.799 | 0.01100 ± 0.00722 | 0.07284 ± 0.01063 |
| IID / S-DFA / minus_F minus Full | 9 | -0.136 ± 1.057 | 0.00425 ± 0.00578 | 0.00955 ± 0.01575 |
| IID / Sp-DFA / Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID / Sp-DFA / minus_F | 9 | 88.785 ± 0.790 | 0.00862 ± 0.00634 | 0.07407 ± 0.01174 |
| IID / Sp-DFA / minus_F minus Full | 9 | 0.239 ± 0.973 | -0.00844 ± 0.00888 | 0.01858 ± 0.01453 |
| non-IID / Benign / Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID / Benign / minus_F | 9 | 88.927 ± 0.838 | 0.00892 ± 0.00739 | 0.06966 ± 0.01339 |
| non-IID / Benign / minus_F minus Full | 9 | 0.336 ± 1.248 | 0.00260 ± 0.00790 | 0.00582 ± 0.01631 |
| non-IID / F Flip / Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID / F Flip / minus_F | 9 | 88.634 ± 0.919 | 0.00929 ± 0.00776 | 0.06614 ± 0.01372 |
| non-IID / F Flip / minus_F minus Full | 9 | 0.130 ± 0.795 | -0.00110 ± 0.01190 | 0.00827 ± 0.01551 |
| non-IID / FedSA / Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID / FedSA / minus_F | 9 | 88.269 ± 1.332 | 0.01038 ± 0.00961 | 0.06090 ± 0.00686 |
| non-IID / FedSA / minus_F minus Full | 9 | -0.125 ± 1.321 | 0.00212 ± 0.01175 | 0.00002 ± 0.01592 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| IID / Benign / minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_F | 6 | 88.333 ± 0.500 | 0.01745 ± 0.00682 | 0.07303 ± 0.01359 |
| IID / F Flip / minus_F minus Full | 6 | 0.045 ± 0.928 | 0.00261 ± 0.01214 | 0.01496 ± 0.01590 |
| IID / FedSA / Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID / FedSA / minus_F | 6 | 88.721 ± 0.622 | 0.00677 ± 0.00740 | 0.06405 ± 0.00902 |
| IID / FedSA / minus_F minus Full | 6 | 0.152 ± 0.995 | 0.00127 ± 0.00805 | -0.00560 ± 0.01074 |
| IID / S-DFA / Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID / S-DFA / minus_F | 6 | 88.349 ± 0.745 | 0.01056 ± 0.00767 | 0.07202 ± 0.00890 |
| IID / S-DFA / minus_F minus Full | 6 | -0.396 ± 1.212 | 0.00353 ± 0.00614 | 0.01031 ± 0.01691 |
| IID / Sp-DFA / Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID / Sp-DFA / minus_F | 6 | 88.483 ± 0.683 | 0.00881 ± 0.00703 | 0.07089 ± 0.01131 |
| IID / Sp-DFA / minus_F minus Full | 6 | 0.129 ± 1.172 | -0.00829 ± 0.01089 | 0.01495 ± 0.01241 |
| non-IID / Benign / Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID / Benign / minus_F | 6 | 88.557 ± 0.762 | 0.00839 ± 0.00825 | 0.06565 ± 0.01246 |
| non-IID / Benign / minus_F minus Full | 6 | -0.028 ± 1.290 | 0.00216 ± 0.00899 | 0.00154 ± 0.01804 |
| non-IID / F Flip / Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID / F Flip / minus_F | 6 | 88.480 ± 1.031 | 0.00888 ± 0.00952 | 0.06483 ± 0.01679 |
| non-IID / F Flip / minus_F minus Full | 6 | 0.049 ± 0.974 | -0.00214 ± 0.01489 | 0.00787 ± 0.01663 |
| non-IID / FedSA / Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID / FedSA / minus_F | 6 | 88.819 ± 0.555 | 0.00726 ± 0.00669 | 0.06242 ± 0.00529 |
| non-IID / FedSA / minus_F minus Full | 6 | 0.331 ± 0.630 | -0.00105 ± 0.00773 | 0.00582 ± 0.00998 |

AEOD = absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0); native retains the original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. Native/shared metrics and counts are compared below, not treated as independent evidence of calibration gain. No threshold is refitted by this builder.
Complete-scene replay devices: Full {'cpu': 5, 'cuda:0': 75}; minus_F {'cpu': 80}. Training Torch: Full {'2.11.0+cu128': 79, '2.11.0+cu130': 1}; minus_F {'2.11.0+cu128': 80}. Historical/current driver provenance is retained per source record. The broader Full100 history includes98 cu128 and2 cu130 records; that fact does not turn these selected scenes into a unified-device comparison.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets.
Only the five IID scenes and non-IID Benign/F Flip/FedSA seeds91001–91010 enter this eight-scene table. No cross-scene aggregate is computed or published; scenes are not independent seeds. The other two non-IID minus_F scenes and other control coverages remain incomplete. This is not F100 or all mechanism controls. No significance, necessity, causal-isolation, final-test or whole-rebuttal-completion claim is made.
