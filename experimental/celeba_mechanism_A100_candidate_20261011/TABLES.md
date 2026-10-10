# CelebA Full–minus_A: all ten IID/non-IID scenes, three views

IID and non-IID Benign, F Flip, FedSA, S-DFA and Sp-DFA: all ten scenes, 100 matched Full/minus_A pairs. Validation (19,867 images), round70; ten paired model seeds per scene, with the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1). Paired difference = minus_A − Full; ACC is percent and ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Frozen Dirichlet partitions use alpha = 5000 for IID and alpha = 5 for non-IID. Formal primary endpoint remains pending.

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
| non-IID / Benign / Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID / Benign / minus_A | 10 | 88.054 ± 0.979 | 0.01183 ± 0.00727 | 0.05929 ± 0.01867 |
| non-IID / Benign / minus_A minus Full | 10 | -0.540 ± 0.561 | 0.00488 ± 0.00856 | -0.00582 ± 0.01772 |
| non-IID / F Flip / Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID / F Flip / minus_A | 10 | 88.039 ± 1.143 | 0.00661 ± 0.00481 | 0.06383 ± 0.01068 |
| non-IID / F Flip / minus_A minus Full | 10 | -0.403 ± 0.702 | -0.00341 ± 0.00875 | 0.00490 ± 0.01339 |
| non-IID / FedSA / Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID / FedSA / minus_A | 10 | 88.466 ± 0.893 | 0.01151 ± 0.00904 | 0.06364 ± 0.01829 |
| non-IID / FedSA / minus_A minus Full | 10 | 0.053 ± 0.948 | 0.00372 ± 0.01267 | 0.00207 ± 0.02164 |
| non-IID / S-DFA / Full | 10 | 88.215 ± 0.781 | 0.01255 ± 0.00926 | 0.05844 ± 0.01889 |
| non-IID / S-DFA / minus_A | 10 | 88.179 ± 0.769 | 0.00821 ± 0.00536 | 0.05699 ± 0.00714 |
| non-IID / S-DFA / minus_A minus Full | 10 | -0.036 ± 0.727 | -0.00434 ± 0.01082 | -0.00145 ± 0.02006 |
| non-IID / Sp-DFA / Full | 10 | 88.344 ± 0.965 | 0.00869 ± 0.00663 | 0.05956 ± 0.01464 |
| non-IID / Sp-DFA / minus_A | 10 | 88.187 ± 1.343 | 0.01361 ± 0.01735 | 0.05824 ± 0.01674 |
| non-IID / Sp-DFA / minus_A minus Full | 10 | -0.158 ± 1.560 | 0.00492 ± 0.01996 | -0.00131 ± 0.02604 |

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
| non-IID / Benign / Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID / Benign / minus_A | 9 | 88.008 ± 1.027 | 0.01116 ± 0.00738 | 0.05767 ± 0.01905 |
| non-IID / Benign / minus_A minus Full | 9 | -0.583 ± 0.577 | 0.00485 ± 0.00908 | -0.00617 ± 0.01876 |
| non-IID / F Flip / Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID / F Flip / minus_A | 9 | 88.139 ± 1.165 | 0.00572 ± 0.00414 | 0.06351 ± 0.01128 |
| non-IID / F Flip / minus_A minus Full | 9 | -0.365 ± 0.734 | -0.00468 ± 0.00825 | 0.00563 ± 0.01399 |
| non-IID / FedSA / Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID / FedSA / minus_A | 9 | 88.514 ± 0.934 | 0.01183 ± 0.00952 | 0.06507 ± 0.01879 |
| non-IID / FedSA / minus_A minus Full | 9 | 0.120 ± 0.980 | 0.00357 ± 0.01343 | 0.00419 ± 0.02183 |
| non-IID / S-DFA / Full | 9 | 88.241 ± 0.824 | 0.01187 ± 0.00954 | 0.06046 ± 0.01885 |
| non-IID / S-DFA / minus_A | 9 | 88.176 ± 0.815 | 0.00847 ± 0.00562 | 0.05604 ± 0.00687 |
| non-IID / S-DFA / minus_A minus Full | 9 | -0.065 ± 0.765 | -0.00340 ± 0.01103 | -0.00442 ± 0.01880 |
| non-IID / Sp-DFA / Full | 9 | 88.467 ± 0.937 | 0.00793 ± 0.00655 | 0.06127 ± 0.01443 |
| non-IID / Sp-DFA / minus_A | 9 | 88.172 ± 1.423 | 0.01390 ± 0.01838 | 0.05707 ± 0.01731 |
| non-IID / Sp-DFA / minus_A minus Full | 9 | -0.295 ± 1.589 | 0.00597 ± 0.02088 | -0.00420 ± 0.02587 |

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
| non-IID / Benign / Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID / Benign / minus_A | 6 | 88.004 ± 1.241 | 0.00800 ± 0.00500 | 0.06022 ± 0.00618 |
| non-IID / Benign / minus_A minus Full | 6 | -0.581 ± 0.413 | 0.00177 ± 0.00568 | -0.00388 ± 0.00654 |
| non-IID / F Flip / Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID / F Flip / minus_A | 6 | 88.286 ± 1.319 | 0.00594 ± 0.00403 | 0.06731 ± 0.00919 |
| non-IID / F Flip / minus_A minus Full | 6 | -0.145 ± 0.678 | -0.00507 ± 0.00920 | 0.01035 ± 0.01279 |
| non-IID / FedSA / Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID / FedSA / minus_A | 6 | 88.455 ± 1.139 | 0.01216 ± 0.01085 | 0.05894 ± 0.01990 |
| non-IID / FedSA / minus_A minus Full | 6 | -0.034 ± 0.614 | 0.00384 ± 0.01397 | 0.00234 ± 0.01907 |
| non-IID / S-DFA / Full | 6 | 88.423 ± 0.829 | 0.00670 ± 0.00677 | 0.06111 ± 0.01143 |
| non-IID / S-DFA / minus_A | 6 | 88.325 ± 0.951 | 0.00890 ± 0.00606 | 0.05702 ± 0.00723 |
| non-IID / S-DFA / minus_A minus Full | 6 | -0.098 ± 0.650 | 0.00220 ± 0.00816 | -0.00409 ± 0.01515 |
| non-IID / Sp-DFA / Full | 6 | 88.420 ± 1.102 | 0.00892 ± 0.00797 | 0.05835 ± 0.01724 |
| non-IID / Sp-DFA / minus_A | 6 | 88.535 ± 0.748 | 0.00855 ± 0.00603 | 0.06022 ± 0.01023 |
| non-IID / Sp-DFA / minus_A minus Full | 6 | 0.115 ± 1.320 | -0.00037 ± 0.00996 | 0.00187 ± 0.02258 |

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
| non-IID / Benign / Full | 10 | 88.875 ± 1.209 | 0.03157 ± 0.00643 | 0.09978 ± 0.00770 |
| non-IID / Benign / minus_A | 10 | 88.331 ± 0.928 | 0.03210 ± 0.01097 | 0.09587 ± 0.01073 |
| non-IID / Benign / minus_A minus Full | 10 | -0.544 ± 0.488 | 0.00053 ± 0.01126 | -0.00391 ± 0.01100 |
| non-IID / F Flip / Full | 10 | 88.728 ± 0.717 | 0.03162 ± 0.00730 | 0.09947 ± 0.00578 |
| non-IID / F Flip / minus_A | 10 | 88.340 ± 1.229 | 0.04257 ± 0.02819 | 0.10491 ± 0.02358 |
| non-IID / F Flip / minus_A minus Full | 10 | -0.388 ± 0.770 | 0.01095 ± 0.02834 | 0.00544 ± 0.02129 |
| non-IID / FedSA / Full | 10 | 88.678 ± 0.894 | 0.03531 ± 0.00824 | 0.09889 ± 0.01124 |
| non-IID / FedSA / minus_A | 10 | 88.756 ± 0.948 | 0.03023 ± 0.01247 | 0.09718 ± 0.01285 |
| non-IID / FedSA / minus_A minus Full | 10 | 0.078 ± 0.950 | -0.00508 ± 0.01328 | -0.00170 ± 0.01643 |
| non-IID / S-DFA / Full | 10 | 88.501 ± 0.789 | 0.04395 ± 0.01690 | 0.10460 ± 0.01218 |
| non-IID / S-DFA / minus_A | 10 | 88.515 ± 0.854 | 0.03239 ± 0.00985 | 0.09625 ± 0.01175 |
| non-IID / S-DFA / minus_A minus Full | 10 | 0.014 ± 0.828 | -0.01156 ± 0.02246 | -0.00835 ± 0.01716 |
| non-IID / Sp-DFA / Full | 10 | 88.699 ± 0.932 | 0.03332 ± 0.00712 | 0.09938 ± 0.00993 |
| non-IID / Sp-DFA / minus_A | 10 | 88.461 ± 1.428 | 0.03755 ± 0.01537 | 0.10244 ± 0.00883 |
| non-IID / Sp-DFA / minus_A minus Full | 10 | -0.238 ± 1.567 | 0.00423 ± 0.01094 | 0.00306 ± 0.00791 |

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
| non-IID / Benign / Full | 9 | 88.877 ± 1.282 | 0.03006 ± 0.00454 | 0.09914 ± 0.00788 |
| non-IID / Benign / minus_A | 9 | 88.307 ± 0.981 | 0.03109 ± 0.01113 | 0.09542 ± 0.01128 |
| non-IID / Benign / minus_A minus Full | 9 | -0.570 ± 0.510 | 0.00103 ± 0.01183 | -0.00372 ± 0.01165 |
| non-IID / F Flip / Full | 9 | 88.811 ± 0.708 | 0.03131 ± 0.00768 | 0.09975 ± 0.00605 |
| non-IID / F Flip / minus_A | 9 | 88.434 ± 1.265 | 0.04323 ± 0.02982 | 0.10623 ± 0.02462 |
| non-IID / F Flip / minus_A minus Full | 9 | -0.377 ± 0.816 | 0.01192 ± 0.02989 | 0.00648 ± 0.02231 |
| non-IID / FedSA / Full | 9 | 88.672 ± 0.948 | 0.03560 ± 0.00868 | 0.09942 ± 0.01178 |
| non-IID / FedSA / minus_A | 9 | 88.779 ± 1.003 | 0.02921 ± 0.01277 | 0.09674 ± 0.01355 |
| non-IID / FedSA / minus_A minus Full | 9 | 0.106 ± 1.004 | -0.00639 ± 0.01338 | -0.00269 ± 0.01711 |
| non-IID / S-DFA / Full | 9 | 88.510 ± 0.837 | 0.04511 ± 0.01750 | 0.10578 ± 0.01230 |
| non-IID / S-DFA / minus_A | 9 | 88.519 ± 0.905 | 0.03164 ± 0.01015 | 0.09594 ± 0.01242 |
| non-IID / S-DFA / minus_A minus Full | 9 | 0.009 ± 0.878 | -0.01347 ± 0.02294 | -0.00984 ± 0.01751 |
| non-IID / Sp-DFA / Full | 9 | 88.807 ± 0.920 | 0.03292 ± 0.00743 | 0.10017 ± 0.01020 |
| non-IID / Sp-DFA / minus_A | 9 | 88.445 ± 1.514 | 0.03549 ± 0.01476 | 0.10133 ± 0.00859 |
| non-IID / Sp-DFA / minus_A minus Full | 9 | -0.362 ± 1.609 | 0.00257 ± 0.01018 | 0.00116 ± 0.00546 |

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
| non-IID / Benign / Full | 6 | 88.879 ± 1.416 | 0.02856 ± 0.00486 | 0.09666 ± 0.00846 |
| non-IID / Benign / minus_A | 6 | 88.259 ± 1.137 | 0.02965 ± 0.00874 | 0.09408 ± 0.00535 |
| non-IID / Benign / minus_A minus Full | 6 | -0.620 ± 0.454 | 0.00110 ± 0.00929 | -0.00259 ± 0.00870 |
| non-IID / F Flip / Full | 6 | 88.741 ± 0.837 | 0.02861 ± 0.00802 | 0.09658 ± 0.00263 |
| non-IID / F Flip / minus_A | 6 | 88.591 ± 1.428 | 0.03267 ± 0.01205 | 0.09944 ± 0.01511 |
| non-IID / F Flip / minus_A minus Full | 6 | -0.150 ± 0.770 | 0.00406 ± 0.01516 | 0.00286 ± 0.01710 |
| non-IID / FedSA / Full | 6 | 88.800 ± 0.820 | 0.03206 ± 0.00774 | 0.09659 ± 0.00664 |
| non-IID / FedSA / minus_A | 6 | 88.681 ± 1.231 | 0.02492 ± 0.01131 | 0.09247 ± 0.01415 |
| non-IID / FedSA / minus_A minus Full | 6 | -0.118 ± 0.684 | -0.00714 ± 0.01201 | -0.00412 ± 0.01340 |
| non-IID / S-DFA / Full | 6 | 88.728 ± 0.747 | 0.03523 ± 0.00655 | 0.09979 ± 0.00749 |
| non-IID / S-DFA / minus_A | 6 | 88.664 ± 1.059 | 0.03294 ± 0.00842 | 0.09891 ± 0.01181 |
| non-IID / S-DFA / minus_A minus Full | 6 | -0.065 ± 0.689 | -0.00229 ± 0.00653 | -0.00088 ± 0.00879 |
| non-IID / Sp-DFA / Full | 6 | 88.824 ± 1.068 | 0.02940 ± 0.00482 | 0.09711 ± 0.00943 |
| non-IID / Sp-DFA / minus_A | 6 | 88.810 ± 0.873 | 0.02904 ± 0.00896 | 0.09693 ± 0.00609 |
| non-IID / Sp-DFA / minus_A minus Full | 6 | -0.014 ± 1.302 | -0.00036 ± 0.00953 | -0.00018 ± 0.00530 |

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
| non-IID / Benign / Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID / Benign / minus_A | 10 | 88.054 ± 0.979 | 0.01183 ± 0.00727 | 0.05929 ± 0.01867 |
| non-IID / Benign / minus_A minus Full | 10 | -0.540 ± 0.561 | 0.00488 ± 0.00856 | -0.00582 ± 0.01772 |
| non-IID / F Flip / Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID / F Flip / minus_A | 10 | 88.039 ± 1.143 | 0.00661 ± 0.00481 | 0.06383 ± 0.01068 |
| non-IID / F Flip / minus_A minus Full | 10 | -0.403 ± 0.702 | -0.00341 ± 0.00875 | 0.00490 ± 0.01339 |
| non-IID / FedSA / Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID / FedSA / minus_A | 10 | 88.466 ± 0.893 | 0.01151 ± 0.00904 | 0.06364 ± 0.01829 |
| non-IID / FedSA / minus_A minus Full | 10 | 0.053 ± 0.948 | 0.00372 ± 0.01267 | 0.00207 ± 0.02164 |
| non-IID / S-DFA / Full | 10 | 88.215 ± 0.781 | 0.01255 ± 0.00926 | 0.05844 ± 0.01889 |
| non-IID / S-DFA / minus_A | 10 | 88.179 ± 0.769 | 0.00821 ± 0.00536 | 0.05699 ± 0.00714 |
| non-IID / S-DFA / minus_A minus Full | 10 | -0.036 ± 0.727 | -0.00434 ± 0.01082 | -0.00145 ± 0.02006 |
| non-IID / Sp-DFA / Full | 10 | 88.344 ± 0.965 | 0.00869 ± 0.00663 | 0.05956 ± 0.01464 |
| non-IID / Sp-DFA / minus_A | 10 | 88.187 ± 1.343 | 0.01361 ± 0.01735 | 0.05824 ± 0.01674 |
| non-IID / Sp-DFA / minus_A minus Full | 10 | -0.158 ± 1.560 | 0.00492 ± 0.01996 | -0.00131 ± 0.02604 |

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
| non-IID / Benign / Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID / Benign / minus_A | 9 | 88.008 ± 1.027 | 0.01116 ± 0.00738 | 0.05767 ± 0.01905 |
| non-IID / Benign / minus_A minus Full | 9 | -0.583 ± 0.577 | 0.00485 ± 0.00908 | -0.00617 ± 0.01876 |
| non-IID / F Flip / Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID / F Flip / minus_A | 9 | 88.139 ± 1.165 | 0.00572 ± 0.00414 | 0.06351 ± 0.01128 |
| non-IID / F Flip / minus_A minus Full | 9 | -0.365 ± 0.734 | -0.00468 ± 0.00825 | 0.00563 ± 0.01399 |
| non-IID / FedSA / Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID / FedSA / minus_A | 9 | 88.514 ± 0.934 | 0.01183 ± 0.00952 | 0.06507 ± 0.01879 |
| non-IID / FedSA / minus_A minus Full | 9 | 0.120 ± 0.980 | 0.00357 ± 0.01343 | 0.00419 ± 0.02183 |
| non-IID / S-DFA / Full | 9 | 88.241 ± 0.824 | 0.01187 ± 0.00954 | 0.06046 ± 0.01885 |
| non-IID / S-DFA / minus_A | 9 | 88.176 ± 0.815 | 0.00847 ± 0.00562 | 0.05604 ± 0.00687 |
| non-IID / S-DFA / minus_A minus Full | 9 | -0.065 ± 0.765 | -0.00340 ± 0.01103 | -0.00442 ± 0.01880 |
| non-IID / Sp-DFA / Full | 9 | 88.467 ± 0.937 | 0.00793 ± 0.00655 | 0.06127 ± 0.01443 |
| non-IID / Sp-DFA / minus_A | 9 | 88.172 ± 1.423 | 0.01390 ± 0.01838 | 0.05707 ± 0.01731 |
| non-IID / Sp-DFA / minus_A minus Full | 9 | -0.295 ± 1.589 | 0.00597 ± 0.02088 | -0.00420 ± 0.02587 |

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
| non-IID / Benign / Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID / Benign / minus_A | 6 | 88.004 ± 1.241 | 0.00800 ± 0.00500 | 0.06022 ± 0.00618 |
| non-IID / Benign / minus_A minus Full | 6 | -0.581 ± 0.413 | 0.00177 ± 0.00568 | -0.00388 ± 0.00654 |
| non-IID / F Flip / Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID / F Flip / minus_A | 6 | 88.286 ± 1.319 | 0.00594 ± 0.00403 | 0.06731 ± 0.00919 |
| non-IID / F Flip / minus_A minus Full | 6 | -0.145 ± 0.678 | -0.00507 ± 0.00920 | 0.01035 ± 0.01279 |
| non-IID / FedSA / Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID / FedSA / minus_A | 6 | 88.455 ± 1.139 | 0.01216 ± 0.01085 | 0.05894 ± 0.01990 |
| non-IID / FedSA / minus_A minus Full | 6 | -0.034 ± 0.614 | 0.00384 ± 0.01397 | 0.00234 ± 0.01907 |
| non-IID / S-DFA / Full | 6 | 88.423 ± 0.829 | 0.00670 ± 0.00677 | 0.06111 ± 0.01143 |
| non-IID / S-DFA / minus_A | 6 | 88.325 ± 0.951 | 0.00890 ± 0.00606 | 0.05702 ± 0.00723 |
| non-IID / S-DFA / minus_A minus Full | 6 | -0.098 ± 0.650 | 0.00220 ± 0.00816 | -0.00409 ± 0.01515 |
| non-IID / Sp-DFA / Full | 6 | 88.420 ± 1.102 | 0.00892 ± 0.00797 | 0.05835 ± 0.01724 |
| non-IID / Sp-DFA / minus_A | 6 | 88.535 ± 0.748 | 0.00855 ± 0.00603 | 0.06022 ± 0.01023 |
| non-IID / Sp-DFA / minus_A minus Full | 6 | 0.115 ± 1.320 | -0.00037 ± 0.00996 | 0.00187 ± 0.02258 |

AEOD is the absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0). Native retains each original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. The three views are parallel descriptions, with no endpoint selected or threshold refitted by this builder.
Actual replay devices across100 pairs: Full {'cpu': 5, 'cuda:0': 95}; minus_A {'cpu': 100}. Training Torch: Full {'2.11.0+cu128': 98, '2.11.0+cu130': 2}; minus_A {'2.11.0+cu128': 100}. Per-record configuration, source, checkpoint, environment and driver provenance remain in records.json. Broader Full100 history includes98 cu128 and2 cu130 records; these100 actual source records, not that broader count, define this table.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets. No final test was run for this table. Preserved Windows exact-refit/whole-check failures in the broader baseline evaluation remain disclosed; they are not relabelled as passes by this mechanism table.
All negative and constant outcomes are retained. These are all ten minus_A scenes. Separate five-IID, five-non-IID and balanced-ten-scene summaries first equally average scenes within each model seed, then summarize seeds; scenes are never treated as independent model seeds. This completes A100 coverage only, not the other five image-control variants or all800 controls. No significance, necessity, causal-isolation or whole-rebuttal-completion claim is made.

# Across-scene summaries (seed-first)

AUTHOR-REVIEW CANDIDATE. Each model seed contributes once after its five IID scenes are equally averaged. The separate five-non-IID and balanced-ten-scene summaries below use the same seed-first rule.

### IID / Five-scene equal mean within seed

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.438 ± 0.670 | 0.01013 ± 0.00323 | 0.06138 ± 0.00730 |
| minus_A | 10 | 88.261 ± 0.897 | 0.00965 ± 0.00324 | 0.05948 ± 0.00567 |
| minus_A minus Full | 10 | -0.177 ± 0.454 | -0.00048 ± 0.00533 | -0.00190 ± 0.00941 |

### IID / Five-scene equal mean within seed

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.555 ± 0.592 | 0.01009 ± 0.00342 | 0.06165 ± 0.00768 |
| minus_A | 9 | 88.436 ± 0.748 | 0.00940 ± 0.00333 | 0.06015 ± 0.00557 |
| minus_A minus Full | 9 | -0.119 ± 0.440 | -0.00070 ± 0.00561 | -0.00150 ± 0.00990 |

### IID / Five-scene equal mean within seed

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.503 ± 0.634 | 0.01117 ± 0.00371 | 0.06237 ± 0.00943 |
| minus_A | 6 | 88.315 ± 0.851 | 0.00886 ± 0.00238 | 0.05965 ± 0.00572 |
| minus_A minus Full | 6 | -0.187 ± 0.441 | -0.00231 ± 0.00594 | -0.00272 ± 0.01100 |

### IID / Five-scene equal mean within seed

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.782 ± 0.693 | 0.03713 ± 0.00423 | 0.10163 ± 0.00429 |
| minus_A | 10 | 88.648 ± 0.888 | 0.04022 ± 0.00701 | 0.10330 ± 0.00530 |
| minus_A minus Full | 10 | -0.134 ± 0.400 | 0.00310 ± 0.00449 | 0.00167 ± 0.00275 |

### IID / Five-scene equal mean within seed

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.907 ± 0.604 | 0.03689 ± 0.00442 | 0.10260 ± 0.00318 |
| minus_A | 9 | 88.825 ± 0.730 | 0.03983 ± 0.00732 | 0.10393 ± 0.00520 |
| minus_A minus Full | 9 | -0.082 ± 0.386 | 0.00294 ± 0.00473 | 0.00133 ± 0.00269 |

### IID / Five-scene equal mean within seed

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.849 ± 0.640 | 0.03690 ± 0.00488 | 0.10211 ± 0.00289 |
| minus_A | 6 | 88.733 ± 0.840 | 0.03887 ± 0.00725 | 0.10189 ± 0.00335 |
| minus_A minus Full | 6 | -0.116 ± 0.383 | 0.00197 ± 0.00466 | -0.00021 ± 0.00088 |

### IID / Five-scene equal mean within seed

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.438 ± 0.670 | 0.01013 ± 0.00323 | 0.06138 ± 0.00730 |
| minus_A | 10 | 88.261 ± 0.897 | 0.00965 ± 0.00324 | 0.05948 ± 0.00567 |
| minus_A minus Full | 10 | -0.177 ± 0.454 | -0.00048 ± 0.00533 | -0.00190 ± 0.00941 |

### IID / Five-scene equal mean within seed

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.555 ± 0.592 | 0.01009 ± 0.00342 | 0.06165 ± 0.00768 |
| minus_A | 9 | 88.436 ± 0.748 | 0.00940 ± 0.00333 | 0.06015 ± 0.00557 |
| minus_A minus Full | 9 | -0.119 ± 0.440 | -0.00070 ± 0.00561 | -0.00150 ± 0.00990 |

### IID / Five-scene equal mean within seed

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.503 ± 0.634 | 0.01117 ± 0.00371 | 0.06237 ± 0.00943 |
| minus_A | 6 | 88.315 ± 0.851 | 0.00886 ± 0.00238 | 0.05965 ± 0.00572 |
| minus_A minus Full | 6 | -0.187 ± 0.441 | -0.00231 ± 0.00594 | -0.00272 ± 0.01100 |

### non-IID / Five-scene equal mean within seed

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.402 ± 0.709 | 0.00920 ± 0.00366 | 0.06072 ± 0.00561 |
| minus_A | 10 | 88.185 ± 0.832 | 0.01035 ± 0.00412 | 0.06040 ± 0.00870 |
| minus_A minus Full | 10 | -0.217 ± 0.399 | 0.00115 ± 0.00364 | -0.00032 ± 0.00692 |

### non-IID / Five-scene equal mean within seed

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.440 ± 0.742 | 0.00895 ± 0.00379 | 0.06087 ± 0.00593 |
| minus_A | 9 | 88.202 ± 0.881 | 0.01022 ± 0.00435 | 0.05987 ± 0.00906 |
| minus_A minus Full | 9 | -0.238 ± 0.417 | 0.00126 ± 0.00384 | -0.00099 ± 0.00699 |

### non-IID / Five-scene equal mean within seed

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.469 ± 0.903 | 0.00824 ± 0.00429 | 0.05943 ± 0.00516 |
| minus_A | 6 | 88.321 ± 1.021 | 0.00871 ± 0.00296 | 0.06074 ± 0.00779 |
| minus_A minus Full | 6 | -0.149 ± 0.430 | 0.00047 ± 0.00414 | 0.00132 ± 0.00634 |

### non-IID / Five-scene equal mean within seed

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.696 ± 0.706 | 0.03516 ± 0.00714 | 0.10043 ± 0.00761 |
| minus_A | 10 | 88.481 ± 0.884 | 0.03497 ± 0.00935 | 0.09933 ± 0.00929 |
| minus_A minus Full | 10 | -0.215 ± 0.397 | -0.00019 ± 0.00625 | -0.00109 ± 0.00553 |

### non-IID / Five-scene equal mean within seed

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.735 ± 0.737 | 0.03500 ± 0.00755 | 0.10085 ± 0.00795 |
| minus_A | 9 | 88.497 ± 0.936 | 0.03413 ± 0.00950 | 0.09913 ± 0.00983 |
| minus_A minus Full | 9 | -0.239 ± 0.414 | -0.00087 ± 0.00623 | -0.00172 ± 0.00547 |

### non-IID / Five-scene equal mean within seed

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.794 ± 0.879 | 0.03077 ± 0.00391 | 0.09734 ± 0.00517 |
| minus_A | 6 | 88.601 ± 1.076 | 0.02985 ± 0.00753 | 0.09637 ± 0.00936 |
| minus_A minus Full | 6 | -0.193 ± 0.452 | -0.00093 ± 0.00686 | -0.00098 ± 0.00553 |

### non-IID / Five-scene equal mean within seed

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.402 ± 0.709 | 0.00920 ± 0.00366 | 0.06072 ± 0.00561 |
| minus_A | 10 | 88.185 ± 0.832 | 0.01035 ± 0.00412 | 0.06040 ± 0.00870 |
| minus_A minus Full | 10 | -0.217 ± 0.399 | 0.00115 ± 0.00364 | -0.00032 ± 0.00692 |

### non-IID / Five-scene equal mean within seed

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.440 ± 0.742 | 0.00895 ± 0.00379 | 0.06087 ± 0.00593 |
| minus_A | 9 | 88.202 ± 0.881 | 0.01022 ± 0.00435 | 0.05987 ± 0.00906 |
| minus_A minus Full | 9 | -0.238 ± 0.417 | 0.00126 ± 0.00384 | -0.00099 ± 0.00699 |

### non-IID / Five-scene equal mean within seed

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.469 ± 0.903 | 0.00824 ± 0.00429 | 0.05943 ± 0.00516 |
| minus_A | 6 | 88.321 ± 1.021 | 0.00871 ± 0.00296 | 0.06074 ± 0.00779 |
| minus_A minus Full | 6 | -0.149 ± 0.430 | 0.00047 ± 0.00414 | 0.00132 ± 0.00634 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.420 ± 0.623 | 0.00967 ± 0.00297 | 0.06105 ± 0.00466 |
| minus_A | 10 | 88.223 ± 0.756 | 0.01000 ± 0.00319 | 0.05994 ± 0.00655 |
| minus_A minus Full | 10 | -0.197 ± 0.373 | 0.00033 ± 0.00378 | -0.00111 ± 0.00707 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.498 ± 0.607 | 0.00952 ± 0.00311 | 0.06126 ± 0.00489 |
| minus_A | 9 | 88.319 ± 0.734 | 0.00981 ± 0.00332 | 0.06001 ± 0.00694 |
| minus_A minus Full | 9 | -0.178 ± 0.391 | 0.00028 ± 0.00400 | -0.00125 ± 0.00748 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.486 ± 0.754 | 0.00970 ± 0.00370 | 0.06090 ± 0.00550 |
| minus_A | 6 | 88.318 ± 0.918 | 0.00879 ± 0.00138 | 0.06020 ± 0.00649 |
| minus_A minus Full | 6 | -0.168 ± 0.409 | -0.00092 ± 0.00396 | -0.00070 ± 0.00807 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.739 ± 0.620 | 0.03614 ± 0.00440 | 0.10103 ± 0.00505 |
| minus_A | 10 | 88.564 ± 0.775 | 0.03760 ± 0.00702 | 0.10131 ± 0.00633 |
| minus_A minus Full | 10 | -0.175 ± 0.362 | 0.00145 ± 0.00454 | 0.00029 ± 0.00308 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.821 ± 0.597 | 0.03594 ± 0.00462 | 0.10173 ± 0.00482 |
| minus_A | 9 | 88.661 ± 0.755 | 0.03698 ± 0.00715 | 0.10153 ± 0.00667 |
| minus_A minus Full | 9 | -0.160 ± 0.381 | 0.00104 ± 0.00460 | -0.00020 ± 0.00284 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.822 ± 0.748 | 0.03384 ± 0.00380 | 0.09973 ± 0.00252 |
| minus_A | 6 | 88.667 ± 0.940 | 0.03436 ± 0.00568 | 0.09913 ± 0.00523 |
| minus_A minus Full | 6 | -0.155 ± 0.399 | 0.00052 ± 0.00472 | -0.00060 ± 0.00297 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.420 ± 0.623 | 0.00967 ± 0.00297 | 0.06105 ± 0.00466 |
| minus_A | 10 | 88.223 ± 0.756 | 0.01000 ± 0.00319 | 0.05994 ± 0.00655 |
| minus_A minus Full | 10 | -0.197 ± 0.373 | 0.00033 ± 0.00378 | -0.00111 ± 0.00707 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.498 ± 0.607 | 0.00952 ± 0.00311 | 0.06126 ± 0.00489 |
| minus_A | 9 | 88.319 ± 0.734 | 0.00981 ± 0.00332 | 0.06001 ± 0.00694 |
| minus_A minus Full | 9 | -0.178 ± 0.391 | 0.00028 ± 0.00400 | -0.00125 ± 0.00748 |

### Balanced IID/non-IID / Ten-scene equal mean within seed

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.486 ± 0.754 | 0.00970 ± 0.00370 | 0.06090 ± 0.00550 |
| minus_A | 6 | 88.318 ± 0.918 | 0.00879 ± 0.00138 | 0.06020 ± 0.00649 |
| minus_A minus Full | 6 | -0.168 ± 0.409 | -0.00092 ± 0.00396 | -0.00070 ± 0.00807 |
