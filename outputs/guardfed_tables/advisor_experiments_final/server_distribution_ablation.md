# AD2+ Server Distribution Ablation

| tag | dataset | distribution | attack | ACC_pct | AEOD | ASPD | fair_avg | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| server_alpha_0.03 | adult | IID | Benign | 81.64 | 0.0812 | 0.0330 | 0.0571 | 0.7593 |
| server_alpha_0.03 | adult | IID | F Flip | 81.49 | 0.0617 | 0.0287 | 0.0452 | 0.7697 |
| server_alpha_0.03 | adult | IID | FedSA | 81.11 | 0.0263 | 0.0209 | 0.0236 | 0.7876 |
| server_alpha_0.03 | adult | IID | S-DFA | 81.27 | 0.0305 | 0.0225 | 0.0265 | 0.7862 |
| server_alpha_0.03 | adult | IID | Sp-DFA | 81.77 | 0.0762 | 0.0330 | 0.0546 | 0.7631 |
| server_alpha_0.03 | adult | non-IID | Benign | 81.49 | 0.0472 | 0.0231 | 0.0351 | 0.7797 |
| server_alpha_0.03 | adult | non-IID | F Flip | 81.54 | 0.0377 | 0.0227 | 0.0302 | 0.7852 |
| server_alpha_0.03 | adult | non-IID | FedSA | 81.30 | 0.0653 | 0.0287 | 0.0470 | 0.7660 |
| server_alpha_0.03 | adult | non-IID | S-DFA | 81.33 | 0.0367 | 0.0245 | 0.0306 | 0.7827 |
| server_alpha_0.03 | adult | non-IID | Sp-DFA | 81.93 | 0.0828 | 0.0356 | 0.0592 | 0.7601 |
| server_alpha_0.03 | compas | IID | Benign | 63.34 | 0.5112 | 0.4108 | 0.4610 | 0.1724 |
| server_alpha_0.03 | compas | IID | F Flip | 63.66 | 0.5098 | 0.3948 | 0.4523 | 0.1843 |
| server_alpha_0.03 | compas | IID | FedSA | 63.93 | 0.5142 | 0.4042 | 0.4592 | 0.1801 |
| server_alpha_0.03 | compas | IID | S-DFA | 63.01 | 0.5168 | 0.4044 | 0.4606 | 0.1695 |
| server_alpha_0.03 | compas | IID | Sp-DFA | 63.34 | 0.5380 | 0.4170 | 0.4775 | 0.1559 |
| server_alpha_0.03 | compas | non-IID | Benign | 62.26 | 0.4263 | 0.3281 | 0.3772 | 0.2454 |
| server_alpha_0.03 | compas | non-IID | F Flip | 61.29 | 0.4178 | 0.3199 | 0.3688 | 0.2440 |
| server_alpha_0.03 | compas | non-IID | FedSA | 61.72 | 0.4251 | 0.3239 | 0.3745 | 0.2426 |
| server_alpha_0.03 | compas | non-IID | S-DFA | 61.88 | 0.4333 | 0.3281 | 0.3807 | 0.2381 |
| server_alpha_0.03 | compas | non-IID | Sp-DFA | 61.45 | 0.4228 | 0.3178 | 0.3703 | 0.2442 |
| server_alpha_0.05 | adult | IID | Benign | 81.64 | 0.0812 | 0.0330 | 0.0571 | 0.7593 |
| server_alpha_0.05 | adult | IID | F Flip | 81.49 | 0.0617 | 0.0287 | 0.0452 | 0.7697 |
| server_alpha_0.05 | adult | IID | FedSA | 81.11 | 0.0263 | 0.0209 | 0.0236 | 0.7876 |
| server_alpha_0.05 | adult | IID | S-DFA | 81.27 | 0.0305 | 0.0225 | 0.0265 | 0.7862 |
| server_alpha_0.05 | adult | IID | Sp-DFA | 81.77 | 0.0762 | 0.0330 | 0.0546 | 0.7631 |
| server_alpha_0.05 | adult | non-IID | Benign | 81.49 | 0.0472 | 0.0231 | 0.0351 | 0.7797 |
| server_alpha_0.05 | adult | non-IID | F Flip | 81.54 | 0.0377 | 0.0227 | 0.0302 | 0.7852 |
| server_alpha_0.05 | adult | non-IID | FedSA | 81.30 | 0.0653 | 0.0287 | 0.0470 | 0.7660 |
| server_alpha_0.05 | adult | non-IID | S-DFA | 81.33 | 0.0367 | 0.0245 | 0.0306 | 0.7827 |
| server_alpha_0.05 | adult | non-IID | Sp-DFA | 81.93 | 0.0828 | 0.0356 | 0.0592 | 0.7601 |
| server_alpha_0.05 | compas | IID | Benign | 63.34 | 0.5112 | 0.4108 | 0.4610 | 0.1724 |
| server_alpha_0.05 | compas | IID | F Flip | 63.66 | 0.5098 | 0.3948 | 0.4523 | 0.1843 |
| server_alpha_0.05 | compas | IID | FedSA | 63.93 | 0.5142 | 0.4042 | 0.4592 | 0.1801 |
| server_alpha_0.05 | compas | IID | S-DFA | 63.01 | 0.5168 | 0.4044 | 0.4606 | 0.1695 |
| server_alpha_0.05 | compas | IID | Sp-DFA | 63.34 | 0.5380 | 0.4170 | 0.4775 | 0.1559 |
| server_alpha_0.05 | compas | non-IID | Benign | 62.26 | 0.4263 | 0.3281 | 0.3772 | 0.2454 |
| server_alpha_0.05 | compas | non-IID | F Flip | 61.29 | 0.4178 | 0.3199 | 0.3688 | 0.2440 |
| server_alpha_0.05 | compas | non-IID | FedSA | 61.72 | 0.4251 | 0.3239 | 0.3745 | 0.2426 |
| server_alpha_0.05 | compas | non-IID | S-DFA | 61.88 | 0.4333 | 0.3281 | 0.3807 | 0.2381 |
| server_alpha_0.05 | compas | non-IID | Sp-DFA | 61.45 | 0.4228 | 0.3178 | 0.3703 | 0.2442 |
| server_alpha_0.1 | adult | IID | Benign | 28.48 | 0.0003 | 0.0477 | 0.0240 | 0.2608 |
| server_alpha_0.1 | adult | IID | F Flip | 25.33 | 0.0003 | 0.0001 | 0.0002 | 0.2531 |
| server_alpha_0.1 | adult | IID | FedSA | 27.82 | 0.0000 | 0.0174 | 0.0087 | 0.2695 |
| server_alpha_0.1 | adult | IID | S-DFA | 27.07 | 0.0000 | 0.0033 | 0.0017 | 0.2691 |
| server_alpha_0.1 | adult | IID | Sp-DFA | 30.07 | 0.0003 | 0.0162 | 0.0082 | 0.2924 |
| server_alpha_0.1 | adult | non-IID | Benign | 28.03 | 0.0006 | 0.0306 | 0.0156 | 0.2647 |
| server_alpha_0.1 | adult | non-IID | F Flip | 29.40 | 0.0000 | 0.0137 | 0.0068 | 0.2872 |
| server_alpha_0.1 | adult | non-IID | FedSA | 29.81 | 0.0006 | 0.0497 | 0.0252 | 0.2729 |
| server_alpha_0.1 | adult | non-IID | S-DFA | 28.26 | 0.0006 | 0.0262 | 0.0134 | 0.2692 |
| server_alpha_0.1 | adult | non-IID | Sp-DFA | 31.20 | 0.0006 | 0.0422 | 0.0214 | 0.2906 |
| server_alpha_0.1 | compas | IID | Benign | 45.84 | 0.0082 | 0.0189 | 0.0135 | 0.4449 |
| server_alpha_0.1 | compas | IID | F Flip | 45.90 | 0.0082 | 0.0168 | 0.0125 | 0.4465 |
| server_alpha_0.1 | compas | IID | FedSA | 45.41 | 0.0082 | 0.0136 | 0.0109 | 0.4432 |
| server_alpha_0.1 | compas | IID | S-DFA | 45.79 | 0.0082 | 0.0189 | 0.0135 | 0.4444 |
| server_alpha_0.1 | compas | IID | Sp-DFA | 45.46 | 0.0102 | 0.0168 | 0.0135 | 0.4411 |
| server_alpha_0.1 | compas | non-IID | Benign | 45.90 | 0.0000 | 0.0016 | 0.0008 | 0.4582 |
| server_alpha_0.1 | compas | non-IID | F Flip | 46.33 | 0.0029 | 0.0015 | 0.0022 | 0.4611 |
| server_alpha_0.1 | compas | non-IID | FedSA | 46.17 | 0.0000 | 0.0026 | 0.0013 | 0.4604 |
| server_alpha_0.1 | compas | non-IID | S-DFA | 46.06 | 0.0000 | 0.0124 | 0.0062 | 0.4544 |
| server_alpha_0.1 | compas | non-IID | Sp-DFA | 46.22 | 0.0032 | 0.0196 | 0.0114 | 0.4508 |
| server_alpha_0.2 | adult | IID | Benign | 74.10 | 0.5691 | 0.1698 | 0.3694 | 0.3716 |
| server_alpha_0.2 | adult | IID | F Flip | 72.71 | 0.6858 | 0.2388 | 0.4623 | 0.2649 |
| server_alpha_0.2 | adult | IID | FedSA | 74.61 | 0.6643 | 0.1812 | 0.4227 | 0.3234 |
| server_alpha_0.2 | adult | IID | S-DFA | 74.49 | 0.5404 | 0.1529 | 0.3466 | 0.3983 |
| server_alpha_0.2 | adult | IID | Sp-DFA | 72.30 | 0.6880 | 0.2569 | 0.4724 | 0.2505 |
| server_alpha_0.2 | adult | non-IID | Benign | 71.39 | 0.7211 | 0.2946 | 0.5079 | 0.2060 |
| server_alpha_0.2 | adult | non-IID | F Flip | 72.24 | 0.5778 | 0.2301 | 0.4039 | 0.3185 |
| server_alpha_0.2 | adult | non-IID | FedSA | 71.17 | 0.7477 | 0.3007 | 0.5242 | 0.1874 |
| server_alpha_0.2 | adult | non-IID | S-DFA | 70.26 | 0.7854 | 0.3370 | 0.5612 | 0.1414 |
| server_alpha_0.2 | adult | non-IID | Sp-DFA | 70.53 | 0.7501 | 0.3205 | 0.5353 | 0.1700 |
| server_alpha_0.2 | compas | IID | Benign | 58.96 | 0.4829 | 0.2971 | 0.3900 | 0.1996 |
| server_alpha_0.2 | compas | IID | F Flip | 59.02 | 0.4677 | 0.2875 | 0.3776 | 0.2126 |
| server_alpha_0.2 | compas | IID | FedSA | 59.07 | 0.5107 | 0.3129 | 0.4118 | 0.1789 |
| server_alpha_0.2 | compas | IID | S-DFA | 58.96 | 0.4788 | 0.2918 | 0.3853 | 0.2044 |
| server_alpha_0.2 | compas | IID | Sp-DFA | 58.96 | 0.4982 | 0.3039 | 0.4010 | 0.1886 |
| server_alpha_0.2 | compas | non-IID | Benign | 59.18 | 0.4858 | 0.2883 | 0.3870 | 0.2048 |
| server_alpha_0.2 | compas | non-IID | F Flip | 59.02 | 0.4742 | 0.2827 | 0.3784 | 0.2118 |
| server_alpha_0.2 | compas | non-IID | FedSA | 58.96 | 0.4336 | 0.2571 | 0.3453 | 0.2443 |
| server_alpha_0.2 | compas | non-IID | S-DFA | 58.86 | 0.4858 | 0.2982 | 0.3920 | 0.1965 |
| server_alpha_0.2 | compas | non-IID | Sp-DFA | 58.96 | 0.4655 | 0.2793 | 0.3724 | 0.2172 |
| server_alpha_0.5 | adult | IID | Benign | 45.06 | 0.0267 | 0.1485 | 0.0876 | 0.3630 |
| server_alpha_0.5 | adult | IID | F Flip | 51.19 | 0.0121 | 0.1957 | 0.1039 | 0.4079 |
| server_alpha_0.5 | adult | IID | FedSA | 24.70 | 0.0000 | 0.0000 | 0.0000 | 0.2469 |
| server_alpha_0.5 | adult | IID | S-DFA | 24.96 | 0.0000 | 0.0013 | 0.0006 | 0.2490 |
| server_alpha_0.5 | adult | IID | Sp-DFA | 49.27 | 0.0152 | 0.1891 | 0.1021 | 0.3905 |
| server_alpha_0.5 | adult | non-IID | Benign | 42.29 | 0.0000 | 0.0000 | 0.0000 | 0.4229 |
| server_alpha_0.5 | adult | non-IID | F Flip | 50.69 | 0.0162 | 0.1821 | 0.0992 | 0.4077 |
| server_alpha_0.5 | adult | non-IID | FedSA | 44.13 | 0.0367 | 0.1669 | 0.1018 | 0.3395 |
| server_alpha_0.5 | adult | non-IID | S-DFA | 40.35 | 0.0000 | 0.0001 | 0.0000 | 0.4035 |
| server_alpha_0.5 | adult | non-IID | Sp-DFA | 47.45 | 0.0194 | 0.2001 | 0.1098 | 0.3648 |
| server_alpha_0.5 | compas | IID | Benign | 45.46 | 0.0020 | 0.0073 | 0.0047 | 0.4500 |
| server_alpha_0.5 | compas | IID | F Flip | 45.63 | 0.0012 | 0.0104 | 0.0058 | 0.4505 |
| server_alpha_0.5 | compas | IID | FedSA | 45.41 | 0.0000 | 0.0042 | 0.0021 | 0.4520 |
| server_alpha_0.5 | compas | IID | S-DFA | 45.30 | 0.0020 | 0.0063 | 0.0042 | 0.4489 |
| server_alpha_0.5 | compas | IID | Sp-DFA | 45.84 | 0.0012 | 0.0057 | 0.0034 | 0.4550 |
| server_alpha_0.5 | compas | non-IID | Benign | 46.06 | 0.0102 | 0.0036 | 0.0069 | 0.4537 |
| server_alpha_0.5 | compas | non-IID | F Flip | 46.22 | 0.0082 | 0.0004 | 0.0043 | 0.4579 |
| server_alpha_0.5 | compas | non-IID | FedSA | 46.00 | 0.0082 | 0.0004 | 0.0043 | 0.4558 |
| server_alpha_0.5 | compas | non-IID | S-DFA | 46.06 | 0.0041 | 0.0068 | 0.0054 | 0.4552 |
| server_alpha_0.5 | compas | non-IID | Sp-DFA | 45.52 | 0.0000 | 0.0001 | 0.0000 | 0.4552 |
| server_alpha_1 | adult | IID | Benign | 74.02 | 0.2072 | 0.0784 | 0.1428 | 0.5974 |
| server_alpha_1 | adult | IID | F Flip | 74.40 | 0.0557 | 0.0719 | 0.0638 | 0.6802 |
| server_alpha_1 | adult | IID | FedSA | 75.53 | 0.1965 | 0.0627 | 0.1296 | 0.6257 |
| server_alpha_1 | adult | IID | S-DFA | 78.51 | 0.1738 | 0.0628 | 0.1183 | 0.6668 |
| server_alpha_1 | adult | IID | Sp-DFA | 75.77 | 0.1812 | 0.0681 | 0.1247 | 0.6330 |
| server_alpha_1 | adult | non-IID | Benign | 73.31 | 0.1652 | 0.0491 | 0.1072 | 0.6259 |
| server_alpha_1 | adult | non-IID | F Flip | 77.77 | 0.1441 | 0.0729 | 0.1085 | 0.6693 |
| server_alpha_1 | adult | non-IID | FedSA | 76.05 | 0.1678 | 0.0566 | 0.1122 | 0.6483 |
| server_alpha_1 | adult | non-IID | S-DFA | 75.58 | 0.1355 | 0.0745 | 0.1050 | 0.6508 |
| server_alpha_1 | adult | non-IID | Sp-DFA | 73.40 | 0.1233 | 0.1118 | 0.1175 | 0.6165 |
| server_alpha_1 | compas | IID | Benign | 64.90 | 0.0063 | 0.0027 | 0.0045 | 0.6445 |
| server_alpha_1 | compas | IID | F Flip | 63.50 | 0.0010 | 0.0003 | 0.0006 | 0.6344 |
| server_alpha_1 | compas | IID | FedSA | 64.63 | 0.0069 | 0.0243 | 0.0156 | 0.6307 |
| server_alpha_1 | compas | IID | S-DFA | 64.58 | 0.0003 | 0.0024 | 0.0013 | 0.6444 |
| server_alpha_1 | compas | IID | Sp-DFA | 62.31 | 0.0015 | 0.0092 | 0.0054 | 0.6177 |
| server_alpha_1 | compas | non-IID | Benign | 63.34 | 0.0124 | 0.0157 | 0.0140 | 0.6193 |
| server_alpha_1 | compas | non-IID | F Flip | 62.47 | 0.0536 | 0.0167 | 0.0351 | 0.5896 |
| server_alpha_1 | compas | non-IID | FedSA | 62.85 | 0.0992 | 0.0690 | 0.0841 | 0.5444 |
| server_alpha_1 | compas | non-IID | S-DFA | 61.12 | 0.0639 | 0.0886 | 0.0762 | 0.5350 |
| server_alpha_1 | compas | non-IID | Sp-DFA | 61.77 | 0.1535 | 0.1107 | 0.1321 | 0.4856 |
| server_alpha_2 | adult | IID | Benign | 76.05 | 0.1272 | 0.0152 | 0.0712 | 0.6893 |
| server_alpha_2 | adult | IID | F Flip | 76.07 | 0.1466 | 0.0169 | 0.0817 | 0.6789 |
| server_alpha_2 | adult | IID | FedSA | 76.09 | 0.1520 | 0.0181 | 0.0850 | 0.6759 |
| server_alpha_2 | adult | IID | S-DFA | 76.05 | 0.1307 | 0.0154 | 0.0731 | 0.6874 |
| server_alpha_2 | adult | IID | Sp-DFA | 76.10 | 0.1286 | 0.0149 | 0.0717 | 0.6893 |
| server_alpha_2 | adult | non-IID | Benign | 76.09 | 0.1257 | 0.0147 | 0.0702 | 0.6908 |
| server_alpha_2 | adult | non-IID | F Flip | 76.11 | 0.1358 | 0.0159 | 0.0758 | 0.6853 |
| server_alpha_2 | adult | non-IID | FedSA | 76.13 | 0.1699 | 0.0202 | 0.0950 | 0.6663 |
| server_alpha_2 | adult | non-IID | S-DFA | 76.11 | 0.1538 | 0.0190 | 0.0864 | 0.6747 |
| server_alpha_2 | adult | non-IID | Sp-DFA | 76.13 | 0.1329 | 0.0165 | 0.0747 | 0.6866 |
| server_alpha_2 | compas | IID | Benign | 61.02 | 0.0081 | 0.0010 | 0.0045 | 0.6056 |
| server_alpha_2 | compas | IID | F Flip | 62.80 | 0.0302 | 0.0589 | 0.0445 | 0.5834 |
| server_alpha_2 | compas | IID | FedSA | 62.53 | 0.0169 | 0.0599 | 0.0384 | 0.5869 |
| server_alpha_2 | compas | IID | S-DFA | 59.50 | 0.0369 | 0.0803 | 0.0586 | 0.5364 |
| server_alpha_2 | compas | IID | Sp-DFA | 61.56 | 0.0883 | 0.1422 | 0.1152 | 0.5003 |
| server_alpha_2 | compas | non-IID | Benign | 62.80 | 0.0057 | 0.0200 | 0.0128 | 0.6151 |
| server_alpha_2 | compas | non-IID | F Flip | 62.58 | 0.0105 | 0.0229 | 0.0167 | 0.6091 |
| server_alpha_2 | compas | non-IID | FedSA | 64.36 | 0.0168 | 0.0098 | 0.0133 | 0.6303 |
| server_alpha_2 | compas | non-IID | S-DFA | 63.66 | 0.0025 | 0.0141 | 0.0083 | 0.6283 |
| server_alpha_2 | compas | non-IID | Sp-DFA | 63.07 | 0.0122 | 0.0587 | 0.0354 | 0.5952 |
| server_alpha_5 | adult | IID | Benign | 81.07 | 0.0085 | 0.0005 | 0.0045 | 0.8062 |
| server_alpha_5 | adult | IID | F Flip | 80.86 | 0.0018 | 0.0007 | 0.0013 | 0.8073 |
| server_alpha_5 | adult | IID | FedSA | 82.50 | 0.0092 | 0.0007 | 0.0050 | 0.8201 |
| server_alpha_5 | adult | IID | S-DFA | 82.48 | 0.0088 | 0.0006 | 0.0047 | 0.8200 |
| server_alpha_5 | adult | IID | Sp-DFA | 82.38 | 0.0046 | 0.0000 | 0.0023 | 0.8214 |
| server_alpha_5 | adult | non-IID | Benign | 82.35 | 0.0134 | 0.0013 | 0.0074 | 0.8161 |
| server_alpha_5 | adult | non-IID | F Flip | 81.35 | 0.0071 | 0.0002 | 0.0037 | 0.8098 |
| server_alpha_5 | adult | non-IID | FedSA | 80.88 | 0.0116 | 0.0011 | 0.0064 | 0.8024 |
| server_alpha_5 | adult | non-IID | S-DFA | 80.96 | 0.0125 | 0.0010 | 0.0067 | 0.8029 |
| server_alpha_5 | adult | non-IID | Sp-DFA | 81.03 | 0.0095 | 0.0008 | 0.0052 | 0.8051 |
| server_alpha_5 | compas | IID | Benign | 66.52 | 0.0249 | 0.0516 | 0.0383 | 0.6269 |
| server_alpha_5 | compas | IID | F Flip | 64.42 | 0.0003 | 0.0007 | 0.0005 | 0.6437 |
| server_alpha_5 | compas | IID | FedSA | 65.17 | 0.0391 | 0.0122 | 0.0257 | 0.6261 |
| server_alpha_5 | compas | IID | S-DFA | 65.93 | 0.0091 | 0.0467 | 0.0279 | 0.6314 |
| server_alpha_5 | compas | IID | Sp-DFA | 61.77 | 0.0023 | 0.0171 | 0.0097 | 0.6080 |
| server_alpha_5 | compas | non-IID | Benign | 64.96 | 0.0605 | 0.0002 | 0.0304 | 0.6192 |
| server_alpha_5 | compas | non-IID | F Flip | 65.01 | 0.0096 | 0.0026 | 0.0061 | 0.6440 |
| server_alpha_5 | compas | non-IID | FedSA | 65.50 | 0.0071 | 0.0379 | 0.0225 | 0.6325 |
| server_alpha_5 | compas | non-IID | S-DFA | 65.82 | 0.0463 | 0.0099 | 0.0281 | 0.6301 |
| server_alpha_5 | compas | non-IID | Sp-DFA | 65.28 | 0.0013 | 0.0086 | 0.0050 | 0.6478 |
| server_alpha_50 | adult | IID | Benign | 84.23 | 0.0006 | 0.0903 | 0.0454 | 0.7968 |
| server_alpha_50 | adult | IID | F Flip | 83.84 | 0.0045 | 0.0846 | 0.0446 | 0.7938 |
| server_alpha_50 | adult | IID | FedSA | 83.54 | 0.0057 | 0.0842 | 0.0450 | 0.7904 |
| server_alpha_50 | adult | IID | S-DFA | 83.89 | 0.0114 | 0.0850 | 0.0482 | 0.7907 |
| server_alpha_50 | adult | IID | Sp-DFA | 83.88 | 0.0016 | 0.0904 | 0.0460 | 0.7928 |
| server_alpha_50 | adult | non-IID | Benign | 83.92 | 0.0106 | 0.0863 | 0.0484 | 0.7907 |
| server_alpha_50 | adult | non-IID | F Flip | 83.94 | 0.0273 | 0.0841 | 0.0557 | 0.7837 |
| server_alpha_50 | adult | non-IID | FedSA | 83.61 | 0.0204 | 0.0812 | 0.0508 | 0.7853 |
| server_alpha_50 | adult | non-IID | S-DFA | 83.74 | 0.0009 | 0.0844 | 0.0427 | 0.7948 |
| server_alpha_50 | adult | non-IID | Sp-DFA | 83.57 | 0.0236 | 0.0898 | 0.0567 | 0.7790 |
| server_alpha_50 | compas | IID | Benign | 66.20 | 0.0252 | 0.0037 | 0.0144 | 0.6475 |
| server_alpha_50 | compas | IID | F Flip | 66.14 | 0.0264 | 0.0033 | 0.0149 | 0.6466 |
| server_alpha_50 | compas | IID | FedSA | 66.04 | 0.0179 | 0.0009 | 0.0094 | 0.6510 |
| server_alpha_50 | compas | IID | S-DFA | 65.82 | 0.0313 | 0.0032 | 0.0172 | 0.6410 |
| server_alpha_50 | compas | IID | Sp-DFA | 65.60 | 0.0453 | 0.0030 | 0.0242 | 0.6319 |
| server_alpha_50 | compas | non-IID | Benign | 65.66 | 0.0407 | 0.0077 | 0.0242 | 0.6324 |
| server_alpha_50 | compas | non-IID | F Flip | 65.33 | 0.0511 | 0.0028 | 0.0269 | 0.6264 |
| server_alpha_50 | compas | non-IID | FedSA | 65.55 | 0.0285 | 0.0036 | 0.0160 | 0.6395 |
| server_alpha_50 | compas | non-IID | S-DFA | 65.28 | 0.0218 | 0.0223 | 0.0220 | 0.6308 |
| server_alpha_50 | compas | non-IID | Sp-DFA | 65.77 | 0.0178 | 0.0219 | 0.0199 | 0.6378 |
| server_alpha_5000 | adult | IID | Benign | 83.43 | 0.0041 | 0.0649 | 0.0345 | 0.7998 |
| server_alpha_5000 | adult | IID | F Flip | 83.62 | 0.0235 | 0.0740 | 0.0487 | 0.7875 |
| server_alpha_5000 | adult | IID | FedSA | 83.56 | 0.0033 | 0.0680 | 0.0357 | 0.8000 |
| server_alpha_5000 | adult | IID | S-DFA | 83.66 | 0.0069 | 0.0661 | 0.0365 | 0.8002 |
| server_alpha_5000 | adult | IID | Sp-DFA | 83.47 | 0.0136 | 0.0702 | 0.0419 | 0.7928 |
| server_alpha_5000 | adult | non-IID | Benign | 83.52 | 0.0067 | 0.0687 | 0.0377 | 0.7974 |
| server_alpha_5000 | adult | non-IID | F Flip | 82.76 | 0.0012 | 0.0600 | 0.0306 | 0.7970 |
| server_alpha_5000 | adult | non-IID | FedSA | 83.21 | 0.0066 | 0.0679 | 0.0373 | 0.7949 |
| server_alpha_5000 | adult | non-IID | S-DFA | 83.39 | 0.0019 | 0.0506 | 0.0263 | 0.8077 |
| server_alpha_5000 | adult | non-IID | Sp-DFA | 83.21 | 0.0122 | 0.0731 | 0.0426 | 0.7894 |
| server_alpha_5000 | compas | IID | Benign | 64.63 | 0.1041 | 0.0147 | 0.0594 | 0.5869 |
| server_alpha_5000 | compas | IID | F Flip | 65.06 | 0.0829 | 0.0043 | 0.0436 | 0.6071 |
| server_alpha_5000 | compas | IID | FedSA | 64.15 | 0.0925 | 0.0156 | 0.0541 | 0.5874 |
| server_alpha_5000 | compas | IID | S-DFA | 65.71 | 0.0588 | 0.0041 | 0.0314 | 0.6257 |
| server_alpha_5000 | compas | IID | Sp-DFA | 65.87 | 0.0852 | 0.0171 | 0.0512 | 0.6076 |
| server_alpha_5000 | compas | non-IID | Benign | 64.47 | 0.0935 | 0.0383 | 0.0659 | 0.5788 |
| server_alpha_5000 | compas | non-IID | F Flip | 65.06 | 0.0472 | 0.0128 | 0.0300 | 0.6207 |
| server_alpha_5000 | compas | non-IID | FedSA | 64.96 | 0.0876 | 0.0053 | 0.0465 | 0.6031 |
| server_alpha_5000 | compas | non-IID | S-DFA | 65.44 | 0.0784 | 0.0046 | 0.0415 | 0.6129 |
| server_alpha_5000 | compas | non-IID | Sp-DFA | 64.36 | 0.1371 | 0.0506 | 0.0939 | 0.5498 |
