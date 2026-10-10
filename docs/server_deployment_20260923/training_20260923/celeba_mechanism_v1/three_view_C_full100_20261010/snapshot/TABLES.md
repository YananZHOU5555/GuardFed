# CelebA Full–minus_C: complete IID/non-IID coverage, three views

Ten complete scenes,100 matched pairs; valid19867,round70. Mean ± sampleSD(ddof1); differences minus_C−Full. ACC percent; ΔACC pp. Higher ACC and lower gaps are favorable; a positive deletion-minus-Full gap favors Full.

## native — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID Benign | minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| IID Benign | minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |
| IID F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID F Flip | minus_C | 10 | 88.909 ± 0.783 | 0.01207 ± 0.00962 | 0.07167 ± 0.01580 |
| IID F Flip | minus_C minus Full | 10 | 0.518 ± 0.814 | 0.00139 ± 0.01487 | 0.01098 ± 0.02176 |
| IID FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID FedSA | minus_C | 10 | 88.749 ± 0.805 | 0.00584 ± 0.00414 | 0.06500 ± 0.00669 |
| IID FedSA | minus_C minus Full | 10 | 0.279 ± 0.498 | -0.00024 ± 0.00472 | 0.00099 ± 0.01311 |
| IID S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID S-DFA | minus_C | 10 | 88.612 ± 0.699 | 0.00819 ± 0.00512 | 0.06399 ± 0.01224 |
| IID S-DFA | minus_C minus Full | 10 | -0.077 ± 0.715 | 0.00050 ± 0.00760 | 0.00022 ± 0.01411 |
| IID Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID Sp-DFA | minus_C | 10 | 88.424 ± 1.253 | 0.00980 ± 0.00717 | 0.06398 ± 0.01417 |
| IID Sp-DFA | minus_C minus Full | 10 | 0.042 ± 0.796 | -0.00668 ± 0.00848 | 0.00810 ± 0.02432 |
| non-IID Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID Benign | minus_C | 10 | 89.188 ± 0.789 | 0.00615 ± 0.00668 | 0.06778 ± 0.00453 |
| non-IID Benign | minus_C minus Full | 10 | 0.594 ± 0.795 | -0.00081 ± 0.00644 | 0.00267 ± 0.00947 |
| non-IID F Flip | Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID F Flip | minus_C | 10 | 89.385 ± 0.716 | 0.00694 ± 0.00536 | 0.07070 ± 0.01076 |
| non-IID F Flip | minus_C minus Full | 10 | 0.943 ± 0.551 | -0.00308 ± 0.00892 | 0.01177 ± 0.01233 |
| non-IID FedSA | Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID FedSA | minus_C | 10 | 88.823 ± 1.207 | 0.00969 ± 0.00839 | 0.06156 ± 0.01068 |
| non-IID FedSA | minus_C minus Full | 10 | 0.410 ± 1.124 | 0.00190 ± 0.01023 | -0.00001 ± 0.01504 |
| non-IID S-DFA | Full | 10 | 88.215 ± 0.781 | 0.01255 ± 0.00926 | 0.05844 ± 0.01889 |
| non-IID S-DFA | minus_C | 10 | 88.747 ± 1.234 | 0.01144 ± 0.00845 | 0.05761 ± 0.01807 |
| non-IID S-DFA | minus_C minus Full | 10 | 0.532 ± 1.120 | -0.00111 ± 0.01011 | -0.00083 ± 0.01396 |
| non-IID Sp-DFA | Full | 10 | 88.344 ± 0.965 | 0.00869 ± 0.00663 | 0.05956 ± 0.01464 |
| non-IID Sp-DFA | minus_C | 10 | 89.025 ± 0.639 | 0.00810 ± 0.00348 | 0.06434 ± 0.00800 |
| non-IID Sp-DFA | minus_C minus Full | 10 | 0.681 ± 1.132 | -0.00058 ± 0.00718 | 0.00479 ± 0.01739 |

## native — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID Benign | minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| IID Benign | minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |
| IID F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID F Flip | minus_C | 9 | 89.029 ± 0.726 | 0.01172 ± 0.01014 | 0.07218 ± 0.01667 |
| IID F Flip | minus_C minus Full | 9 | 0.616 ± 0.797 | 0.00025 ± 0.01530 | 0.01175 ± 0.02294 |
| IID FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID FedSA | minus_C | 9 | 88.841 ± 0.797 | 0.00571 ± 0.00437 | 0.06441 ± 0.00681 |
| IID FedSA | minus_C minus Full | 9 | 0.219 ± 0.489 | 0.00026 ± 0.00472 | -0.00233 ± 0.00832 |
| IID S-DFA | Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID S-DFA | minus_C | 9 | 88.691 ± 0.693 | 0.00821 ± 0.00543 | 0.06355 ± 0.01291 |
| IID S-DFA | minus_C minus Full | 9 | -0.154 ± 0.714 | 0.00146 ± 0.00739 | 0.00026 ± 0.01496 |
| IID Sp-DFA | Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID Sp-DFA | minus_C | 9 | 88.680 ± 1.012 | 0.01053 ± 0.00721 | 0.06654 ± 0.01234 |
| IID Sp-DFA | minus_C minus Full | 9 | 0.134 ± 0.786 | -0.00653 ± 0.00898 | 0.01106 ± 0.02382 |
| non-IID Benign | Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID Benign | minus_C | 9 | 89.251 ± 0.810 | 0.00558 ± 0.00682 | 0.06716 ± 0.00433 |
| non-IID Benign | minus_C minus Full | 9 | 0.659 ± 0.814 | -0.00074 ± 0.00683 | 0.00332 ± 0.00980 |
| non-IID F Flip | Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID F Flip | minus_C | 9 | 89.506 ± 0.643 | 0.00669 ± 0.00563 | 0.07105 ± 0.01135 |
| non-IID F Flip | minus_C minus Full | 9 | 1.002 ± 0.550 | -0.00370 ± 0.00923 | 0.01318 ± 0.01219 |
| non-IID FedSA | Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID FedSA | minus_C | 9 | 89.140 ± 0.713 | 0.00905 ± 0.00864 | 0.06241 ± 0.01096 |
| non-IID FedSA | minus_C minus Full | 9 | 0.746 ± 0.396 | 0.00079 ± 0.01020 | 0.00152 ± 0.01510 |
| non-IID S-DFA | Full | 9 | 88.241 ± 0.824 | 0.01187 ± 0.00954 | 0.06046 ± 0.01885 |
| non-IID S-DFA | minus_C | 9 | 89.039 ± 0.867 | 0.01187 ± 0.00884 | 0.05944 ± 0.01814 |
| non-IID S-DFA | minus_C minus Full | 9 | 0.798 ± 0.784 | 0.00000 ± 0.01005 | -0.00102 ± 0.01479 |
| non-IID Sp-DFA | Full | 9 | 88.467 ± 0.937 | 0.00793 ± 0.00655 | 0.06127 ± 0.01443 |
| non-IID Sp-DFA | minus_C | 9 | 89.097 ± 0.634 | 0.00872 ± 0.00306 | 0.06437 ± 0.00848 |
| non-IID Sp-DFA | minus_C minus Full | 9 | 0.630 ± 1.188 | 0.00079 ± 0.00606 | 0.00310 ± 0.01756 |

## native — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID Benign | minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| IID Benign | minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |
| IID F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID F Flip | minus_C | 6 | 88.953 ± 0.876 | 0.01193 ± 0.01091 | 0.07928 ± 0.01311 |
| IID F Flip | minus_C minus Full | 6 | 0.665 ± 0.997 | -0.00291 ± 0.01773 | 0.02120 ± 0.02060 |
| IID FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID FedSA | minus_C | 6 | 88.717 ± 0.911 | 0.00566 ± 0.00427 | 0.06372 ± 0.00750 |
| IID FedSA | minus_C minus Full | 6 | 0.148 ± 0.569 | 0.00016 ± 0.00415 | -0.00593 ± 0.00634 |
| IID S-DFA | Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID S-DFA | minus_C | 6 | 88.767 ± 0.847 | 0.00754 ± 0.00652 | 0.06817 ± 0.01271 |
| IID S-DFA | minus_C minus Full | 6 | 0.022 ± 0.770 | 0.00051 ± 0.00843 | 0.00646 ± 0.01382 |
| IID Sp-DFA | Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID Sp-DFA | minus_C | 6 | 88.288 ± 0.998 | 0.01089 ± 0.00891 | 0.06373 ± 0.01194 |
| IID Sp-DFA | minus_C minus Full | 6 | -0.066 ± 0.881 | -0.00622 ± 0.01103 | 0.00779 ± 0.02924 |
| non-IID Benign | Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID Benign | minus_C | 6 | 89.103 ± 0.962 | 0.00623 ± 0.00807 | 0.06674 ± 0.00477 |
| non-IID Benign | minus_C minus Full | 6 | 0.518 ± 0.545 | -0.00000 ± 0.00681 | 0.00264 ± 0.00655 |
| non-IID F Flip | Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID F Flip | minus_C | 6 | 89.475 ± 0.720 | 0.00677 ± 0.00581 | 0.07106 ± 0.00560 |
| non-IID F Flip | minus_C minus Full | 6 | 1.044 ± 0.690 | -0.00424 ± 0.01154 | 0.01409 ± 0.01382 |
| non-IID FedSA | Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID FedSA | minus_C | 6 | 89.191 ± 0.663 | 0.01008 ± 0.01054 | 0.06268 ± 0.01173 |
| non-IID FedSA | minus_C minus Full | 6 | 0.702 ± 0.419 | 0.00176 ± 0.01210 | 0.00608 ± 0.01423 |
| non-IID S-DFA | Full | 6 | 88.423 ± 0.829 | 0.00670 ± 0.00677 | 0.06111 ± 0.01143 |
| non-IID S-DFA | minus_C | 6 | 89.232 ± 0.827 | 0.00985 ± 0.00542 | 0.06108 ± 0.00967 |
| non-IID S-DFA | minus_C minus Full | 6 | 0.809 ± 0.723 | 0.00315 ± 0.00815 | -0.00003 ± 0.01694 |
| non-IID Sp-DFA | Full | 6 | 88.420 ± 1.102 | 0.00892 ± 0.00797 | 0.05835 ± 0.01724 |
| non-IID Sp-DFA | minus_C | 6 | 88.977 ± 0.757 | 0.00926 ± 0.00352 | 0.06227 ± 0.00923 |
| non-IID Sp-DFA | minus_C minus Full | 6 | 0.557 ± 1.467 | 0.00034 ± 0.00719 | 0.00392 ± 0.02136 |

## raw — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID Benign | minus_C | 10 | 88.535 ± 1.226 | 0.03544 ± 0.00804 | 0.09925 ± 0.01001 |
| IID Benign | minus_C minus Full | 10 | -0.014 ± 1.385 | -0.00483 ± 0.01119 | -0.00383 ± 0.01524 |
| IID F Flip | Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID F Flip | minus_C | 10 | 89.233 ± 0.770 | 0.04149 ± 0.00883 | 0.10783 ± 0.00966 |
| IID F Flip | minus_C minus Full | 10 | 0.530 ± 0.875 | 0.00465 ± 0.01053 | 0.00703 ± 0.01157 |
| IID FedSA | Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID FedSA | minus_C | 10 | 89.067 ± 0.783 | 0.03584 ± 0.00831 | 0.10399 ± 0.00893 |
| IID FedSA | minus_C minus Full | 10 | 0.235 ± 0.535 | -0.00132 ± 0.01034 | 0.00142 ± 0.01245 |
| IID S-DFA | Full | 10 | 89.059 ± 0.840 | 0.03671 ± 0.00377 | 0.10194 ± 0.00664 |
| IID S-DFA | minus_C | 10 | 89.024 ± 0.694 | 0.03871 ± 0.00764 | 0.10338 ± 0.01001 |
| IID S-DFA | minus_C minus Full | 10 | -0.035 ± 0.715 | 0.00200 ± 0.00936 | 0.00144 ± 0.01139 |
| IID Sp-DFA | Full | 10 | 88.766 ± 0.924 | 0.03465 ± 0.01394 | 0.09978 ± 0.01059 |
| IID Sp-DFA | minus_C | 10 | 88.863 ± 1.290 | 0.03988 ± 0.00743 | 0.10370 ± 0.01027 |
| IID Sp-DFA | minus_C minus Full | 10 | 0.097 ± 0.909 | 0.00523 ± 0.01289 | 0.00392 ± 0.01422 |
| non-IID Benign | Full | 10 | 88.875 ± 1.209 | 0.03157 ± 0.00643 | 0.09978 ± 0.00770 |
| non-IID Benign | minus_C | 10 | 89.516 ± 0.776 | 0.03262 ± 0.00766 | 0.10177 ± 0.00674 |
| non-IID Benign | minus_C minus Full | 10 | 0.641 ± 0.802 | 0.00105 ± 0.00915 | 0.00198 ± 0.00656 |
| non-IID F Flip | Full | 10 | 88.728 ± 0.717 | 0.03162 ± 0.00730 | 0.09947 ± 0.00578 |
| non-IID F Flip | minus_C | 10 | 89.649 ± 0.665 | 0.03116 ± 0.00618 | 0.10403 ± 0.00963 |
| non-IID F Flip | minus_C minus Full | 10 | 0.921 ± 0.501 | -0.00046 ± 0.01166 | 0.00457 ± 0.01143 |
| non-IID FedSA | Full | 10 | 88.678 ± 0.894 | 0.03531 ± 0.00824 | 0.09889 ± 0.01124 |
| non-IID FedSA | minus_C | 10 | 89.171 ± 1.209 | 0.03181 ± 0.01093 | 0.10017 ± 0.00800 |
| non-IID FedSA | minus_C minus Full | 10 | 0.493 ± 1.138 | -0.00351 ± 0.00999 | 0.00128 ± 0.00778 |
| non-IID S-DFA | Full | 10 | 88.501 ± 0.789 | 0.04395 ± 0.01690 | 0.10460 ± 0.01218 |
| non-IID S-DFA | minus_C | 10 | 89.152 ± 1.259 | 0.02906 ± 0.01060 | 0.09744 ± 0.01365 |
| non-IID S-DFA | minus_C minus Full | 10 | 0.651 ± 1.276 | -0.01489 ± 0.02257 | -0.00717 ± 0.01965 |
| non-IID Sp-DFA | Full | 10 | 88.699 ± 0.932 | 0.03332 ± 0.00712 | 0.09938 ± 0.00993 |
| non-IID Sp-DFA | minus_C | 10 | 89.407 ± 0.534 | 0.02663 ± 0.01204 | 0.09742 ± 0.00797 |
| non-IID Sp-DFA | minus_C minus Full | 10 | 0.708 ± 1.077 | -0.00669 ± 0.01496 | -0.00196 ± 0.01530 |

## raw — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID Benign | minus_C | 9 | 88.827 ± 0.855 | 0.03710 ± 0.00648 | 0.10213 ± 0.00439 |
| IID Benign | minus_C minus Full | 9 | 0.181 ± 1.315 | -0.00294 ± 0.01004 | -0.00125 ± 0.01366 |
| IID F Flip | Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID F Flip | minus_C | 9 | 89.348 ± 0.720 | 0.04256 ± 0.00865 | 0.11002 ± 0.00713 |
| IID F Flip | minus_C minus Full | 9 | 0.617 ± 0.880 | 0.00486 ± 0.01115 | 0.00802 ± 0.01181 |
| IID FedSA | Full | 9 | 89.001 ± 0.767 | 0.03797 ± 0.00689 | 0.10454 ± 0.01032 |
| IID FedSA | minus_C | 9 | 89.160 ± 0.770 | 0.03520 ± 0.00855 | 0.10382 ± 0.00945 |
| IID FedSA | minus_C minus Full | 9 | 0.158 ± 0.505 | -0.00277 ± 0.00983 | -0.00072 ± 0.01109 |
| IID S-DFA | Full | 9 | 89.233 ± 0.673 | 0.03652 ± 0.00395 | 0.10342 ± 0.00500 |
| IID S-DFA | minus_C | 9 | 89.096 ± 0.695 | 0.03967 ± 0.00744 | 0.10409 ± 0.01035 |
| IID S-DFA | minus_C minus Full | 9 | -0.137 ± 0.677 | 0.00315 ± 0.00915 | 0.00067 ± 0.01180 |
| IID Sp-DFA | Full | 9 | 88.925 ± 0.824 | 0.03222 ± 0.01233 | 0.09968 ± 0.01123 |
| IID Sp-DFA | minus_C | 9 | 89.123 ± 1.056 | 0.03945 ± 0.00775 | 0.10542 ± 0.00924 |
| IID Sp-DFA | minus_C minus Full | 9 | 0.198 ± 0.902 | 0.00723 ± 0.01190 | 0.00574 ± 0.01378 |
| non-IID Benign | Full | 9 | 88.877 ± 1.282 | 0.03006 ± 0.00454 | 0.09914 ± 0.00788 |
| non-IID Benign | minus_C | 9 | 89.574 ± 0.800 | 0.03145 ± 0.00711 | 0.10131 ± 0.00698 |
| non-IID Benign | minus_C minus Full | 9 | 0.697 ± 0.830 | 0.00139 ± 0.00963 | 0.00217 ± 0.00692 |
| non-IID F Flip | Full | 9 | 88.811 ± 0.708 | 0.03131 ± 0.00768 | 0.09975 ± 0.00605 |
| non-IID F Flip | minus_C | 9 | 89.766 ± 0.586 | 0.03046 ± 0.00611 | 0.10464 ± 0.01000 |
| non-IID F Flip | minus_C minus Full | 9 | 0.955 ± 0.519 | -0.00086 ± 0.01229 | 0.00489 ± 0.01207 |
| non-IID FedSA | Full | 9 | 88.672 ± 0.948 | 0.03560 ± 0.00868 | 0.09942 ± 0.01178 |
| non-IID FedSA | minus_C | 9 | 89.503 ± 0.637 | 0.02989 ± 0.00965 | 0.10116 ± 0.00781 |
| non-IID FedSA | minus_C minus Full | 9 | 0.831 ± 0.421 | -0.00571 ± 0.00760 | 0.00173 ± 0.00811 |
| non-IID S-DFA | Full | 9 | 88.510 ± 0.837 | 0.04511 ± 0.01750 | 0.10578 ± 0.01230 |
| non-IID S-DFA | minus_C | 9 | 89.452 ± 0.877 | 0.02870 ± 0.01118 | 0.09927 ± 0.01312 |
| non-IID S-DFA | minus_C minus Full | 9 | 0.942 ± 0.937 | -0.01641 ± 0.02339 | -0.00651 ± 0.02073 |
| non-IID Sp-DFA | Full | 9 | 88.807 ± 0.920 | 0.03292 ± 0.00743 | 0.10017 ± 0.01020 |
| non-IID Sp-DFA | minus_C | 9 | 89.469 ± 0.526 | 0.02599 ± 0.01258 | 0.09751 ± 0.00845 |
| non-IID Sp-DFA | minus_C minus Full | 9 | 0.663 ± 1.132 | -0.00694 ± 0.01585 | -0.00266 ± 0.01606 |

## raw — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID Benign | minus_C | 6 | 88.494 ± 0.812 | 0.03809 ± 0.00776 | 0.10094 ± 0.00496 |
| IID Benign | minus_C minus Full | 6 | -0.319 ± 0.678 | -0.00416 ± 0.01235 | -0.00617 ± 0.01176 |
| IID F Flip | Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID F Flip | minus_C | 6 | 89.287 ± 0.872 | 0.04425 ± 0.00831 | 0.11101 ± 0.00666 |
| IID F Flip | minus_C minus Full | 6 | 0.636 ± 1.099 | 0.00779 ± 0.00863 | 0.00988 ± 0.01213 |
| IID FedSA | Full | 6 | 88.894 ± 0.796 | 0.03453 ± 0.00524 | 0.10012 ± 0.00832 |
| IID FedSA | minus_C | 6 | 89.099 ± 0.942 | 0.03486 ± 0.00821 | 0.10253 ± 0.00705 |
| IID FedSA | minus_C minus Full | 6 | 0.206 ± 0.556 | 0.00033 ± 0.00842 | 0.00241 ± 0.01063 |
| IID S-DFA | Full | 6 | 89.175 ± 0.783 | 0.03681 ± 0.00432 | 0.10370 ± 0.00558 |
| IID S-DFA | minus_C | 6 | 89.105 ± 0.816 | 0.03940 ± 0.00809 | 0.10465 ± 0.01099 |
| IID S-DFA | minus_C minus Full | 6 | -0.070 ± 0.660 | 0.00259 ± 0.00937 | 0.00095 ± 0.01126 |
| IID Sp-DFA | Full | 6 | 88.713 ± 0.934 | 0.03446 ± 0.01385 | 0.09848 ± 0.01283 |
| IID Sp-DFA | minus_C | 6 | 88.733 ± 1.063 | 0.04167 ± 0.00863 | 0.10295 ± 0.01065 |
| IID Sp-DFA | minus_C minus Full | 6 | 0.019 ± 1.012 | 0.00722 ± 0.01293 | 0.00447 ± 0.01603 |
| non-IID Benign | Full | 6 | 88.879 ± 1.416 | 0.02856 ± 0.00486 | 0.09666 ± 0.00846 |
| non-IID Benign | minus_C | 6 | 89.429 ± 0.967 | 0.03332 ± 0.00755 | 0.10105 ± 0.00850 |
| non-IID Benign | minus_C minus Full | 6 | 0.549 ± 0.542 | 0.00476 ± 0.00977 | 0.00439 ± 0.00764 |
| non-IID F Flip | Full | 6 | 88.741 ± 0.837 | 0.02861 ± 0.00802 | 0.09658 ± 0.00263 |
| non-IID F Flip | minus_C | 6 | 89.783 ± 0.649 | 0.03198 ± 0.00713 | 0.10664 ± 0.00967 |
| non-IID F Flip | minus_C minus Full | 6 | 1.042 ± 0.628 | 0.00337 ± 0.01307 | 0.01006 ± 0.01122 |
| non-IID FedSA | Full | 6 | 88.800 ± 0.820 | 0.03206 ± 0.00774 | 0.09659 ± 0.00664 |
| non-IID FedSA | minus_C | 6 | 89.557 ± 0.548 | 0.02589 ± 0.00909 | 0.09977 ± 0.00717 |
| non-IID FedSA | minus_C minus Full | 6 | 0.758 ± 0.439 | -0.00617 ± 0.00887 | 0.00319 ± 0.00758 |
| non-IID S-DFA | Full | 6 | 88.728 ± 0.747 | 0.03523 ± 0.00655 | 0.09979 ± 0.00749 |
| non-IID S-DFA | minus_C | 6 | 89.644 ± 0.849 | 0.02750 ± 0.00503 | 0.10081 ± 0.00926 |
| non-IID S-DFA | minus_C minus Full | 6 | 0.915 ± 0.781 | -0.00773 ± 0.00933 | 0.00103 ± 0.01127 |
| non-IID Sp-DFA | Full | 6 | 88.824 ± 1.068 | 0.02940 ± 0.00482 | 0.09711 ± 0.00943 |
| non-IID Sp-DFA | minus_C | 6 | 89.433 ± 0.656 | 0.02760 ± 0.00895 | 0.09853 ± 0.00355 |
| non-IID Sp-DFA | minus_C minus Full | 6 | 0.609 ± 1.378 | -0.00180 ± 0.00721 | 0.00142 ± 0.00902 |

## shared_calibration — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID Benign | minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| IID Benign | minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |
| IID F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID F Flip | minus_C | 10 | 88.909 ± 0.783 | 0.01207 ± 0.00962 | 0.07167 ± 0.01580 |
| IID F Flip | minus_C minus Full | 10 | 0.518 ± 0.814 | 0.00139 ± 0.01487 | 0.01098 ± 0.02176 |
| IID FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID FedSA | minus_C | 10 | 88.749 ± 0.805 | 0.00584 ± 0.00414 | 0.06500 ± 0.00669 |
| IID FedSA | minus_C minus Full | 10 | 0.279 ± 0.498 | -0.00024 ± 0.00472 | 0.00099 ± 0.01311 |
| IID S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID S-DFA | minus_C | 10 | 88.612 ± 0.699 | 0.00819 ± 0.00512 | 0.06399 ± 0.01224 |
| IID S-DFA | minus_C minus Full | 10 | -0.077 ± 0.715 | 0.00050 ± 0.00760 | 0.00022 ± 0.01411 |
| IID Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID Sp-DFA | minus_C | 10 | 88.424 ± 1.253 | 0.00980 ± 0.00717 | 0.06398 ± 0.01417 |
| IID Sp-DFA | minus_C minus Full | 10 | 0.042 ± 0.796 | -0.00668 ± 0.00848 | 0.00810 ± 0.02432 |
| non-IID Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID Benign | minus_C | 10 | 89.188 ± 0.789 | 0.00615 ± 0.00668 | 0.06778 ± 0.00453 |
| non-IID Benign | minus_C minus Full | 10 | 0.594 ± 0.795 | -0.00081 ± 0.00644 | 0.00267 ± 0.00947 |
| non-IID F Flip | Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID F Flip | minus_C | 10 | 89.385 ± 0.716 | 0.00694 ± 0.00536 | 0.07070 ± 0.01076 |
| non-IID F Flip | minus_C minus Full | 10 | 0.943 ± 0.551 | -0.00308 ± 0.00892 | 0.01177 ± 0.01233 |
| non-IID FedSA | Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID FedSA | minus_C | 10 | 88.823 ± 1.207 | 0.00969 ± 0.00839 | 0.06156 ± 0.01068 |
| non-IID FedSA | minus_C minus Full | 10 | 0.410 ± 1.124 | 0.00190 ± 0.01023 | -0.00001 ± 0.01504 |
| non-IID S-DFA | Full | 10 | 88.215 ± 0.781 | 0.01255 ± 0.00926 | 0.05844 ± 0.01889 |
| non-IID S-DFA | minus_C | 10 | 88.747 ± 1.234 | 0.01144 ± 0.00845 | 0.05761 ± 0.01807 |
| non-IID S-DFA | minus_C minus Full | 10 | 0.532 ± 1.120 | -0.00111 ± 0.01011 | -0.00083 ± 0.01396 |
| non-IID Sp-DFA | Full | 10 | 88.344 ± 0.965 | 0.00869 ± 0.00663 | 0.05956 ± 0.01464 |
| non-IID Sp-DFA | minus_C | 10 | 89.025 ± 0.639 | 0.00810 ± 0.00348 | 0.06434 ± 0.00800 |
| non-IID Sp-DFA | minus_C minus Full | 10 | 0.681 ± 1.132 | -0.00058 ± 0.00718 | 0.00479 ± 0.01739 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID Benign | minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| IID Benign | minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |
| IID F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID F Flip | minus_C | 9 | 89.029 ± 0.726 | 0.01172 ± 0.01014 | 0.07218 ± 0.01667 |
| IID F Flip | minus_C minus Full | 9 | 0.616 ± 0.797 | 0.00025 ± 0.01530 | 0.01175 ± 0.02294 |
| IID FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID FedSA | minus_C | 9 | 88.841 ± 0.797 | 0.00571 ± 0.00437 | 0.06441 ± 0.00681 |
| IID FedSA | minus_C minus Full | 9 | 0.219 ± 0.489 | 0.00026 ± 0.00472 | -0.00233 ± 0.00832 |
| IID S-DFA | Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID S-DFA | minus_C | 9 | 88.691 ± 0.693 | 0.00821 ± 0.00543 | 0.06355 ± 0.01291 |
| IID S-DFA | minus_C minus Full | 9 | -0.154 ± 0.714 | 0.00146 ± 0.00739 | 0.00026 ± 0.01496 |
| IID Sp-DFA | Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID Sp-DFA | minus_C | 9 | 88.680 ± 1.012 | 0.01053 ± 0.00721 | 0.06654 ± 0.01234 |
| IID Sp-DFA | minus_C minus Full | 9 | 0.134 ± 0.786 | -0.00653 ± 0.00898 | 0.01106 ± 0.02382 |
| non-IID Benign | Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID Benign | minus_C | 9 | 89.251 ± 0.810 | 0.00558 ± 0.00682 | 0.06716 ± 0.00433 |
| non-IID Benign | minus_C minus Full | 9 | 0.659 ± 0.814 | -0.00074 ± 0.00683 | 0.00332 ± 0.00980 |
| non-IID F Flip | Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID F Flip | minus_C | 9 | 89.506 ± 0.643 | 0.00669 ± 0.00563 | 0.07105 ± 0.01135 |
| non-IID F Flip | minus_C minus Full | 9 | 1.002 ± 0.550 | -0.00370 ± 0.00923 | 0.01318 ± 0.01219 |
| non-IID FedSA | Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID FedSA | minus_C | 9 | 89.140 ± 0.713 | 0.00905 ± 0.00864 | 0.06241 ± 0.01096 |
| non-IID FedSA | minus_C minus Full | 9 | 0.746 ± 0.396 | 0.00079 ± 0.01020 | 0.00152 ± 0.01510 |
| non-IID S-DFA | Full | 9 | 88.241 ± 0.824 | 0.01187 ± 0.00954 | 0.06046 ± 0.01885 |
| non-IID S-DFA | minus_C | 9 | 89.039 ± 0.867 | 0.01187 ± 0.00884 | 0.05944 ± 0.01814 |
| non-IID S-DFA | minus_C minus Full | 9 | 0.798 ± 0.784 | 0.00000 ± 0.01005 | -0.00102 ± 0.01479 |
| non-IID Sp-DFA | Full | 9 | 88.467 ± 0.937 | 0.00793 ± 0.00655 | 0.06127 ± 0.01443 |
| non-IID Sp-DFA | minus_C | 9 | 89.097 ± 0.634 | 0.00872 ± 0.00306 | 0.06437 ± 0.00848 |
| non-IID Sp-DFA | minus_C minus Full | 9 | 0.630 ± 1.188 | 0.00079 ± 0.00606 | 0.00310 ± 0.01756 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID Benign | minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| IID Benign | minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |
| IID F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID F Flip | minus_C | 6 | 88.953 ± 0.876 | 0.01193 ± 0.01091 | 0.07928 ± 0.01311 |
| IID F Flip | minus_C minus Full | 6 | 0.665 ± 0.997 | -0.00291 ± 0.01773 | 0.02120 ± 0.02060 |
| IID FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID FedSA | minus_C | 6 | 88.717 ± 0.911 | 0.00566 ± 0.00427 | 0.06372 ± 0.00750 |
| IID FedSA | minus_C minus Full | 6 | 0.148 ± 0.569 | 0.00016 ± 0.00415 | -0.00593 ± 0.00634 |
| IID S-DFA | Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID S-DFA | minus_C | 6 | 88.767 ± 0.847 | 0.00754 ± 0.00652 | 0.06817 ± 0.01271 |
| IID S-DFA | minus_C minus Full | 6 | 0.022 ± 0.770 | 0.00051 ± 0.00843 | 0.00646 ± 0.01382 |
| IID Sp-DFA | Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID Sp-DFA | minus_C | 6 | 88.288 ± 0.998 | 0.01089 ± 0.00891 | 0.06373 ± 0.01194 |
| IID Sp-DFA | minus_C minus Full | 6 | -0.066 ± 0.881 | -0.00622 ± 0.01103 | 0.00779 ± 0.02924 |
| non-IID Benign | Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID Benign | minus_C | 6 | 89.103 ± 0.962 | 0.00623 ± 0.00807 | 0.06674 ± 0.00477 |
| non-IID Benign | minus_C minus Full | 6 | 0.518 ± 0.545 | -0.00000 ± 0.00681 | 0.00264 ± 0.00655 |
| non-IID F Flip | Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID F Flip | minus_C | 6 | 89.475 ± 0.720 | 0.00677 ± 0.00581 | 0.07106 ± 0.00560 |
| non-IID F Flip | minus_C minus Full | 6 | 1.044 ± 0.690 | -0.00424 ± 0.01154 | 0.01409 ± 0.01382 |
| non-IID FedSA | Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID FedSA | minus_C | 6 | 89.191 ± 0.663 | 0.01008 ± 0.01054 | 0.06268 ± 0.01173 |
| non-IID FedSA | minus_C minus Full | 6 | 0.702 ± 0.419 | 0.00176 ± 0.01210 | 0.00608 ± 0.01423 |
| non-IID S-DFA | Full | 6 | 88.423 ± 0.829 | 0.00670 ± 0.00677 | 0.06111 ± 0.01143 |
| non-IID S-DFA | minus_C | 6 | 89.232 ± 0.827 | 0.00985 ± 0.00542 | 0.06108 ± 0.00967 |
| non-IID S-DFA | minus_C minus Full | 6 | 0.809 ± 0.723 | 0.00315 ± 0.00815 | -0.00003 ± 0.01694 |
| non-IID Sp-DFA | Full | 6 | 88.420 ± 1.102 | 0.00892 ± 0.00797 | 0.05835 ± 0.01724 |
| non-IID Sp-DFA | minus_C | 6 | 88.977 ± 0.757 | 0.00926 ± 0.00352 | 0.06227 ± 0.00923 |
| non-IID Sp-DFA | minus_C minus Full | 6 | 0.557 ± 1.467 | 0.00034 ± 0.00719 | 0.00392 ± 0.02136 |

AEOD here is absolute TPR gap, not full equalized odds. Raw/native/shared are parallel outputs of the same checkpoint; native includes each method’s original root-only calibration and shared uses the frozen common fitting rule. No inference or refitting occurs in this table build.
Full replay comprises5 CPU/95 GPU records; Full training comprises98 cu128/2 cu130 records. The100 minus_C replays are CPU. Exact runtime and training fields remain in records.json. Recipe-selection seed91001, validation exposure and historical official-test exposure remain disclosed; these are validation tables, not a newly untouched test.
All10/9/6 panels and unfavorable/zero results are retained. Paired deletion contrasts do not establish necessity, pure aggregation causality or statistical significance. No primary endpoint is selected. This completes minus_C coverage only; it does not complete the other six control variants or the overall revision.
cross_scene_seed_first.json preserves the original five-IID summary byte-for-byte. cross_scene_additional.json adds five-non-IID and balanced-ten-scene summaries: first equally average scenes within each seed, then summarize independent seed units.
