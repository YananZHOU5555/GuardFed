# AD2+ Server Generation Ablation

| tag | dataset | distribution | attack | ACC_pct | AEOD | ASPD | fair_avg | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real10_none | adult | IID | Benign | 83.64 | 0.0054 | 0.0663 | 0.0358 | 0.8006 |
| real10_none | adult | IID | F Flip | 82.21 | 0.0005 | 0.0625 | 0.0315 | 0.7906 |
| real10_none | adult | IID | FedSA | 83.24 | 0.0005 | 0.0625 | 0.0315 | 0.8009 |
| real10_none | adult | IID | S-DFA | 83.69 | 0.0022 | 0.0658 | 0.0340 | 0.8029 |
| real10_none | adult | IID | Sp-DFA | 83.67 | 0.0072 | 0.0615 | 0.0343 | 0.8024 |
| real10_none | adult | non-IID | Benign | 82.18 | 0.0141 | 0.0329 | 0.0235 | 0.7983 |
| real10_none | adult | non-IID | F Flip | 83.09 | 0.0207 | 0.0474 | 0.0341 | 0.7969 |
| real10_none | adult | non-IID | FedSA | 82.62 | 0.0024 | 0.0484 | 0.0254 | 0.8007 |
| real10_none | adult | non-IID | S-DFA | 82.89 | 0.0095 | 0.0538 | 0.0317 | 0.7973 |
| real10_none | adult | non-IID | Sp-DFA | 83.25 | 0.0167 | 0.0614 | 0.0390 | 0.7935 |
| real10_none | compas | IID | Benign | 64.85 | 0.0583 | 0.0162 | 0.0373 | 0.6112 |
| real10_none | compas | IID | F Flip | 65.12 | 0.0928 | 0.0060 | 0.0494 | 0.6018 |
| real10_none | compas | IID | FedSA | 65.12 | 0.1117 | 0.0238 | 0.0678 | 0.5834 |
| real10_none | compas | IID | S-DFA | 65.06 | 0.0533 | 0.0034 | 0.0283 | 0.6223 |
| real10_none | compas | IID | Sp-DFA | 64.58 | 0.0779 | 0.0004 | 0.0391 | 0.6066 |
| real10_none | compas | non-IID | Benign | 63.55 | 0.0877 | 0.0460 | 0.0668 | 0.5687 |
| real10_none | compas | non-IID | F Flip | 64.69 | 0.1080 | 0.0219 | 0.0649 | 0.5819 |
| real10_none | compas | non-IID | FedSA | 65.44 | 0.0788 | 0.0013 | 0.0400 | 0.6144 |
| real10_none | compas | non-IID | S-DFA | 64.90 | 0.1033 | 0.0062 | 0.0548 | 0.5943 |
| real10_none | compas | non-IID | Sp-DFA | 64.52 | 0.0977 | 0.0241 | 0.0609 | 0.5844 |
| real1_ctgan_synth9 | adult | IID | Benign | 76.23 | 0.0002 | 0.0043 | 0.0023 | 0.7600 |
| real1_ctgan_synth9 | adult | IID | F Flip | 79.54 | 0.0004 | 0.0054 | 0.0029 | 0.7925 |
| real1_ctgan_synth9 | adult | IID | FedSA | 76.57 | 0.0004 | 0.0047 | 0.0026 | 0.7631 |
| real1_ctgan_synth9 | adult | IID | S-DFA | 76.03 | 0.0002 | 0.0047 | 0.0025 | 0.7579 |
| real1_ctgan_synth9 | adult | IID | Sp-DFA | 76.02 | 0.0001 | 0.0045 | 0.0023 | 0.7579 |
| real1_ctgan_synth9 | adult | non-IID | Benign | 76.17 | 0.0017 | 0.0050 | 0.0034 | 0.7584 |
| real1_ctgan_synth9 | adult | non-IID | F Flip | 76.16 | 0.0008 | 0.0051 | 0.0029 | 0.7587 |
| real1_ctgan_synth9 | adult | non-IID | FedSA | 76.18 | 0.0040 | 0.0059 | 0.0050 | 0.7568 |
| real1_ctgan_synth9 | adult | non-IID | S-DFA | 76.19 | 0.0010 | 0.0045 | 0.0027 | 0.7592 |
| real1_ctgan_synth9 | adult | non-IID | Sp-DFA | 76.13 | 0.0031 | 0.0056 | 0.0044 | 0.7570 |
| real1_ctgan_synth9 | compas | IID | Benign | 66.47 | 0.2648 | 0.3525 | 0.3086 | 0.3561 |
| real1_ctgan_synth9 | compas | IID | F Flip | 67.66 | 0.2891 | 0.3775 | 0.3333 | 0.3432 |
| real1_ctgan_synth9 | compas | IID | FedSA | 65.82 | 0.0430 | 0.0967 | 0.0698 | 0.5884 |
| real1_ctgan_synth9 | compas | IID | S-DFA | 66.41 | 0.0270 | 0.0590 | 0.0430 | 0.6212 |
| real1_ctgan_synth9 | compas | IID | Sp-DFA | 66.20 | 0.1472 | 0.2122 | 0.1797 | 0.4823 |
| real1_ctgan_synth9 | compas | non-IID | Benign | 67.44 | 0.2487 | 0.3278 | 0.2882 | 0.3862 |
| real1_ctgan_synth9 | compas | non-IID | F Flip | 67.93 | 0.2741 | 0.3304 | 0.3022 | 0.3770 |
| real1_ctgan_synth9 | compas | non-IID | FedSA | 66.95 | 0.1878 | 0.2772 | 0.2325 | 0.4371 |
| real1_ctgan_synth9 | compas | non-IID | S-DFA | 66.85 | 0.3203 | 0.3947 | 0.3575 | 0.3109 |
| real1_ctgan_synth9 | compas | non-IID | Sp-DFA | 67.44 | 0.2406 | 0.3272 | 0.2839 | 0.3905 |
| real1_forest_diffusion_synth9 | adult | IID | Benign | 80.20 | 0.0165 | 0.0085 | 0.0125 | 0.7895 |
| real1_forest_diffusion_synth9 | adult | IID | F Flip | 79.95 | 0.0036 | 0.0096 | 0.0066 | 0.7929 |
| real1_forest_diffusion_synth9 | adult | IID | FedSA | 77.68 | 0.0033 | 0.0044 | 0.0039 | 0.7730 |
| real1_forest_diffusion_synth9 | adult | IID | S-DFA | 78.08 | 0.0011 | 0.0042 | 0.0026 | 0.7782 |
| real1_forest_diffusion_synth9 | adult | IID | Sp-DFA | 78.29 | 0.0011 | 0.0042 | 0.0027 | 0.7802 |
| real1_forest_diffusion_synth9 | adult | non-IID | Benign | 80.50 | 0.0115 | 0.0114 | 0.0114 | 0.7935 |
| real1_forest_diffusion_synth9 | adult | non-IID | F Flip | 80.77 | 0.0018 | 0.0138 | 0.0078 | 0.7999 |
| real1_forest_diffusion_synth9 | adult | non-IID | FedSA | 80.87 | 0.0108 | 0.0086 | 0.0097 | 0.7990 |
| real1_forest_diffusion_synth9 | adult | non-IID | S-DFA | 81.73 | 0.0040 | 0.0134 | 0.0087 | 0.8086 |
| real1_forest_diffusion_synth9 | adult | non-IID | Sp-DFA | 80.97 | 0.0120 | 0.0099 | 0.0110 | 0.7988 |
| real1_forest_diffusion_synth9 | compas | IID | Benign | 65.60 | 0.0714 | 0.1475 | 0.1095 | 0.5466 |
| real1_forest_diffusion_synth9 | compas | IID | F Flip | 65.77 | 0.0543 | 0.1232 | 0.0887 | 0.5689 |
| real1_forest_diffusion_synth9 | compas | IID | FedSA | 65.87 | 0.0629 | 0.1275 | 0.0952 | 0.5635 |
| real1_forest_diffusion_synth9 | compas | IID | S-DFA | 65.87 | 0.0511 | 0.1237 | 0.0874 | 0.5714 |
| real1_forest_diffusion_synth9 | compas | IID | Sp-DFA | 65.33 | 0.0229 | 0.1125 | 0.0677 | 0.5857 |
| real1_forest_diffusion_synth9 | compas | non-IID | Benign | 66.52 | 0.0706 | 0.1349 | 0.1028 | 0.5625 |
| real1_forest_diffusion_synth9 | compas | non-IID | F Flip | 67.93 | 0.0773 | 0.1256 | 0.1014 | 0.5778 |
| real1_forest_diffusion_synth9 | compas | non-IID | FedSA | 66.25 | 0.0456 | 0.1023 | 0.0739 | 0.5886 |
| real1_forest_diffusion_synth9 | compas | non-IID | S-DFA | 64.85 | 0.0101 | 0.0960 | 0.0531 | 0.5954 |
| real1_forest_diffusion_synth9 | compas | non-IID | Sp-DFA | 67.22 | 0.0382 | 0.1022 | 0.0702 | 0.6020 |
| real1_gaussian_copula_synth9 | adult | IID | Benign | 80.71 | 0.0000 | 0.0000 | 0.0000 | 0.8071 |
| real1_gaussian_copula_synth9 | adult | IID | F Flip | 82.48 | 0.0002 | 0.0001 | 0.0002 | 0.8246 |
| real1_gaussian_copula_synth9 | adult | IID | FedSA | 81.31 | 0.0000 | 0.0000 | 0.0000 | 0.8131 |
| real1_gaussian_copula_synth9 | adult | IID | S-DFA | 78.24 | 0.0000 | 0.0000 | 0.0000 | 0.7824 |
| real1_gaussian_copula_synth9 | adult | IID | Sp-DFA | 81.30 | 0.0015 | 0.0001 | 0.0008 | 0.8122 |
| real1_gaussian_copula_synth9 | adult | non-IID | Benign | 80.82 | 0.0003 | 0.0001 | 0.0002 | 0.8079 |
| real1_gaussian_copula_synth9 | adult | non-IID | F Flip | 82.02 | 0.0033 | 0.0003 | 0.0018 | 0.8184 |
| real1_gaussian_copula_synth9 | adult | non-IID | FedSA | 81.89 | 0.0018 | 0.0002 | 0.0010 | 0.8179 |
| real1_gaussian_copula_synth9 | adult | non-IID | S-DFA | 82.12 | 0.0141 | 0.0053 | 0.0097 | 0.8114 |
| real1_gaussian_copula_synth9 | adult | non-IID | Sp-DFA | 82.03 | 0.0147 | 0.0035 | 0.0091 | 0.8112 |
| real1_gaussian_copula_synth9 | compas | IID | Benign | 65.06 | 0.0051 | 0.0414 | 0.0232 | 0.6274 |
| real1_gaussian_copula_synth9 | compas | IID | F Flip | 66.25 | 0.0033 | 0.0519 | 0.0276 | 0.6349 |
| real1_gaussian_copula_synth9 | compas | IID | FedSA | 66.25 | 0.0008 | 0.0544 | 0.0276 | 0.6350 |
| real1_gaussian_copula_synth9 | compas | IID | S-DFA | 66.09 | 0.0127 | 0.0615 | 0.0371 | 0.6238 |
| real1_gaussian_copula_synth9 | compas | IID | Sp-DFA | 66.36 | 0.0250 | 0.0688 | 0.0469 | 0.6167 |
| real1_gaussian_copula_synth9 | compas | non-IID | Benign | 66.25 | 0.0060 | 0.0819 | 0.0439 | 0.6186 |
| real1_gaussian_copula_synth9 | compas | non-IID | F Flip | 66.04 | 0.0300 | 0.0985 | 0.0643 | 0.5961 |
| real1_gaussian_copula_synth9 | compas | non-IID | FedSA | 66.31 | 0.0102 | 0.0520 | 0.0311 | 0.6320 |
| real1_gaussian_copula_synth9 | compas | non-IID | S-DFA | 64.69 | 0.0047 | 0.0531 | 0.0289 | 0.6180 |
| real1_gaussian_copula_synth9 | compas | non-IID | Sp-DFA | 65.44 | 0.0238 | 0.0910 | 0.0574 | 0.5970 |
| real1_pca_gaussian_synth9 | adult | IID | Benign | 82.73 | 0.0082 | 0.0072 | 0.0077 | 0.8197 |
| real1_pca_gaussian_synth9 | adult | IID | F Flip | 81.14 | 0.0026 | 0.0069 | 0.0047 | 0.8067 |
| real1_pca_gaussian_synth9 | adult | IID | FedSA | 82.04 | 0.0054 | 0.0067 | 0.0061 | 0.8143 |
| real1_pca_gaussian_synth9 | adult | IID | S-DFA | 82.19 | 0.0040 | 0.0136 | 0.0088 | 0.8131 |
| real1_pca_gaussian_synth9 | adult | IID | Sp-DFA | 82.28 | 0.0023 | 0.0068 | 0.0045 | 0.8183 |
| real1_pca_gaussian_synth9 | adult | non-IID | Benign | 81.86 | 0.0120 | 0.0064 | 0.0092 | 0.8094 |
| real1_pca_gaussian_synth9 | adult | non-IID | F Flip | 82.16 | 0.0001 | 0.0057 | 0.0029 | 0.8187 |
| real1_pca_gaussian_synth9 | adult | non-IID | FedSA | 81.61 | 0.0019 | 0.0055 | 0.0037 | 0.8124 |
| real1_pca_gaussian_synth9 | adult | non-IID | S-DFA | 81.39 | 0.0110 | 0.0275 | 0.0192 | 0.7947 |
| real1_pca_gaussian_synth9 | adult | non-IID | Sp-DFA | 81.35 | 0.0008 | 0.0056 | 0.0032 | 0.8103 |
| real1_pca_gaussian_synth9 | compas | IID | Benign | 66.04 | 0.0024 | 0.0402 | 0.0213 | 0.6391 |
| real1_pca_gaussian_synth9 | compas | IID | F Flip | 65.06 | 0.0751 | 0.0007 | 0.0379 | 0.6128 |
| real1_pca_gaussian_synth9 | compas | IID | FedSA | 66.47 | 0.0058 | 0.0175 | 0.0116 | 0.6530 |
| real1_pca_gaussian_synth9 | compas | IID | S-DFA | 66.31 | 0.0074 | 0.0193 | 0.0134 | 0.6497 |
| real1_pca_gaussian_synth9 | compas | IID | Sp-DFA | 65.55 | 0.0034 | 0.0473 | 0.0254 | 0.6302 |
| real1_pca_gaussian_synth9 | compas | non-IID | Benign | 66.14 | 0.0138 | 0.0179 | 0.0158 | 0.6456 |
| real1_pca_gaussian_synth9 | compas | non-IID | F Flip | 65.87 | 0.0039 | 0.0521 | 0.0280 | 0.6307 |
| real1_pca_gaussian_synth9 | compas | non-IID | FedSA | 66.36 | 0.0032 | 0.0526 | 0.0279 | 0.6357 |
| real1_pca_gaussian_synth9 | compas | non-IID | S-DFA | 66.04 | 0.0009 | 0.0474 | 0.0242 | 0.6362 |
| real1_pca_gaussian_synth9 | compas | non-IID | Sp-DFA | 66.36 | 0.0043 | 0.0331 | 0.0187 | 0.6449 |
| real1_smote_synth9 | adult | IID | Benign | 81.26 | 0.2305 | 0.2052 | 0.2178 | 0.5948 |
| real1_smote_synth9 | adult | IID | F Flip | 80.78 | 0.0487 | 0.1061 | 0.0774 | 0.7303 |
| real1_smote_synth9 | adult | IID | FedSA | 78.48 | 0.0642 | 0.0402 | 0.0522 | 0.7326 |
| real1_smote_synth9 | adult | IID | S-DFA | 80.66 | 0.0029 | 0.0500 | 0.0264 | 0.7802 |
| real1_smote_synth9 | adult | IID | Sp-DFA | 82.48 | 0.0408 | 0.1108 | 0.0758 | 0.7490 |
| real1_smote_synth9 | adult | non-IID | Benign | 80.82 | 0.2581 | 0.2405 | 0.2493 | 0.5590 |
| real1_smote_synth9 | adult | non-IID | F Flip | 80.85 | 0.2189 | 0.1933 | 0.2061 | 0.6024 |
| real1_smote_synth9 | adult | non-IID | FedSA | 81.04 | 0.2733 | 0.2297 | 0.2515 | 0.5589 |
| real1_smote_synth9 | adult | non-IID | S-DFA | 80.24 | 0.2585 | 0.2652 | 0.2618 | 0.5406 |
| real1_smote_synth9 | adult | non-IID | Sp-DFA | 80.48 | 0.0973 | 0.1312 | 0.1143 | 0.6906 |
| real1_smote_synth9 | compas | IID | Benign | 64.63 | 0.0492 | 0.0033 | 0.0263 | 0.6201 |
| real1_smote_synth9 | compas | IID | F Flip | 65.50 | 0.0121 | 0.0180 | 0.0150 | 0.6399 |
| real1_smote_synth9 | compas | IID | FedSA | 64.69 | 0.0612 | 0.0045 | 0.0328 | 0.6140 |
| real1_smote_synth9 | compas | IID | S-DFA | 64.52 | 0.0564 | 0.0078 | 0.0321 | 0.6132 |
| real1_smote_synth9 | compas | IID | Sp-DFA | 65.23 | 0.0382 | 0.0041 | 0.0211 | 0.6311 |
| real1_smote_synth9 | compas | non-IID | Benign | 64.90 | 0.0071 | 0.0119 | 0.0095 | 0.6395 |
| real1_smote_synth9 | compas | non-IID | F Flip | 66.41 | 0.0529 | 0.0125 | 0.0327 | 0.6315 |
| real1_smote_synth9 | compas | non-IID | FedSA | 66.85 | 0.0007 | 0.0357 | 0.0182 | 0.6502 |
| real1_smote_synth9 | compas | non-IID | S-DFA | 67.28 | 0.0020 | 0.0566 | 0.0293 | 0.6435 |
| real1_smote_synth9 | compas | non-IID | Sp-DFA | 66.04 | 0.0286 | 0.0037 | 0.0161 | 0.6442 |
| real1_tvae_synth9 | adult | IID | Benign | 76.33 | 0.0048 | 0.0074 | 0.0061 | 0.7572 |
| real1_tvae_synth9 | adult | IID | F Flip | 76.36 | 0.0033 | 0.0073 | 0.0053 | 0.7583 |
| real1_tvae_synth9 | adult | IID | FedSA | 76.17 | 0.0057 | 0.0068 | 0.0063 | 0.7555 |
| real1_tvae_synth9 | adult | IID | S-DFA | 76.17 | 0.0060 | 0.0069 | 0.0065 | 0.7553 |
| real1_tvae_synth9 | adult | IID | Sp-DFA | 76.18 | 0.0040 | 0.0059 | 0.0050 | 0.7568 |
| real1_tvae_synth9 | adult | non-IID | Benign | 76.25 | 0.0009 | 0.0051 | 0.0030 | 0.7595 |
| real1_tvae_synth9 | adult | non-IID | F Flip | 76.21 | 0.0072 | 0.0069 | 0.0071 | 0.7551 |
| real1_tvae_synth9 | adult | non-IID | FedSA | 76.23 | 0.0007 | 0.0061 | 0.0034 | 0.7589 |
| real1_tvae_synth9 | adult | non-IID | S-DFA | 76.19 | 0.0072 | 0.0069 | 0.0071 | 0.7549 |
| real1_tvae_synth9 | adult | non-IID | Sp-DFA | 76.15 | 0.0059 | 0.0065 | 0.0062 | 0.7552 |
| real1_tvae_synth9 | compas | IID | Benign | 66.41 | 0.0597 | 0.0284 | 0.0441 | 0.6201 |
| real1_tvae_synth9 | compas | IID | F Flip | 57.99 | 0.0033 | 0.0004 | 0.0018 | 0.5781 |
| real1_tvae_synth9 | compas | IID | FedSA | 66.58 | 0.0886 | 0.0411 | 0.0649 | 0.6009 |
| real1_tvae_synth9 | compas | IID | S-DFA | 57.51 | 0.0210 | 0.0229 | 0.0219 | 0.5531 |
| real1_tvae_synth9 | compas | IID | Sp-DFA | 57.56 | 0.0013 | 0.0101 | 0.0057 | 0.5699 |
| real1_tvae_synth9 | compas | non-IID | Benign | 57.45 | 0.1477 | 0.0807 | 0.1142 | 0.4603 |
| real1_tvae_synth9 | compas | non-IID | F Flip | 59.40 | 0.2309 | 0.1219 | 0.1764 | 0.4176 |
| real1_tvae_synth9 | compas | non-IID | FedSA | 59.56 | 0.2486 | 0.1316 | 0.1901 | 0.4055 |
| real1_tvae_synth9 | compas | non-IID | S-DFA | 59.50 | 0.2149 | 0.1091 | 0.1620 | 0.4331 |
| real1_tvae_synth9 | compas | non-IID | Sp-DFA | 59.72 | 0.2590 | 0.1385 | 0.1988 | 0.3984 |
| real5_ctgan_synth5 | adult | IID | Benign | 82.26 | 0.0107 | 0.0002 | 0.0055 | 0.8171 |
| real5_ctgan_synth5 | adult | IID | F Flip | 82.06 | 0.0416 | 0.0141 | 0.0279 | 0.7927 |
| real5_ctgan_synth5 | adult | IID | FedSA | 82.73 | 0.0201 | 0.0109 | 0.0155 | 0.8118 |
| real5_ctgan_synth5 | adult | IID | S-DFA | 81.96 | 0.0264 | 0.0286 | 0.0275 | 0.7921 |
| real5_ctgan_synth5 | adult | IID | Sp-DFA | 82.73 | 0.0254 | 0.0105 | 0.0179 | 0.8094 |
| real5_ctgan_synth5 | adult | non-IID | Benign | 81.27 | 0.0076 | 0.0092 | 0.0084 | 0.8043 |
| real5_ctgan_synth5 | adult | non-IID | F Flip | 82.79 | 0.0270 | 0.0129 | 0.0199 | 0.8080 |
| real5_ctgan_synth5 | adult | non-IID | FedSA | 82.50 | 0.0414 | 0.0071 | 0.0243 | 0.8007 |
| real5_ctgan_synth5 | adult | non-IID | S-DFA | 82.24 | 0.0229 | 0.0108 | 0.0169 | 0.8056 |
| real5_ctgan_synth5 | adult | non-IID | Sp-DFA | 82.29 | 0.0091 | 0.0102 | 0.0097 | 0.8132 |
| real5_ctgan_synth5 | compas | IID | Benign | 65.55 | 0.0749 | 0.1252 | 0.1000 | 0.5555 |
| real5_ctgan_synth5 | compas | IID | F Flip | 65.71 | 0.0540 | 0.1206 | 0.0873 | 0.5698 |
| real5_ctgan_synth5 | compas | IID | FedSA | 65.71 | 0.0615 | 0.1240 | 0.0927 | 0.5644 |
| real5_ctgan_synth5 | compas | IID | S-DFA | 64.63 | 0.0713 | 0.1146 | 0.0929 | 0.5534 |
| real5_ctgan_synth5 | compas | IID | Sp-DFA | 65.01 | 0.0297 | 0.1013 | 0.0655 | 0.5846 |
| real5_ctgan_synth5 | compas | non-IID | Benign | 65.93 | 0.0248 | 0.0408 | 0.0328 | 0.6265 |
| real5_ctgan_synth5 | compas | non-IID | F Flip | 63.61 | 0.0206 | 0.0690 | 0.0448 | 0.5913 |
| real5_ctgan_synth5 | compas | non-IID | FedSA | 61.39 | 0.0236 | 0.0397 | 0.0316 | 0.5823 |
| real5_ctgan_synth5 | compas | non-IID | S-DFA | 60.53 | 0.0064 | 0.0168 | 0.0116 | 0.5937 |
| real5_ctgan_synth5 | compas | non-IID | Sp-DFA | 66.79 | 0.0641 | 0.1228 | 0.0935 | 0.5745 |
| real5_forest_diffusion_synth5 | adult | IID | Benign | 83.15 | 0.0113 | 0.0508 | 0.0311 | 0.8004 |
| real5_forest_diffusion_synth5 | adult | IID | F Flip | 82.95 | 0.0022 | 0.0353 | 0.0188 | 0.8108 |
| real5_forest_diffusion_synth5 | adult | IID | FedSA | 83.04 | 0.0142 | 0.0534 | 0.0338 | 0.7966 |
| real5_forest_diffusion_synth5 | adult | IID | S-DFA | 83.19 | 0.0080 | 0.0299 | 0.0189 | 0.8130 |
| real5_forest_diffusion_synth5 | adult | IID | Sp-DFA | 83.48 | 0.0096 | 0.0361 | 0.0228 | 0.8119 |
| real5_forest_diffusion_synth5 | adult | non-IID | Benign | 82.63 | 0.0005 | 0.0576 | 0.0291 | 0.7972 |
| real5_forest_diffusion_synth5 | adult | non-IID | F Flip | 83.03 | 0.0001 | 0.0600 | 0.0300 | 0.8002 |
| real5_forest_diffusion_synth5 | adult | non-IID | FedSA | 82.63 | 0.0161 | 0.0563 | 0.0362 | 0.7901 |
| real5_forest_diffusion_synth5 | adult | non-IID | S-DFA | 83.13 | 0.0011 | 0.0617 | 0.0314 | 0.7999 |
| real5_forest_diffusion_synth5 | adult | non-IID | Sp-DFA | 83.42 | 0.0056 | 0.0694 | 0.0375 | 0.7967 |
| real5_forest_diffusion_synth5 | compas | IID | Benign | 64.96 | 0.0739 | 0.0096 | 0.0418 | 0.6078 |
| real5_forest_diffusion_synth5 | compas | IID | F Flip | 65.66 | 0.1018 | 0.0013 | 0.0516 | 0.6050 |
| real5_forest_diffusion_synth5 | compas | IID | FedSA | 65.98 | 0.0307 | 0.0012 | 0.0159 | 0.6439 |
| real5_forest_diffusion_synth5 | compas | IID | S-DFA | 65.55 | 0.0892 | 0.0007 | 0.0449 | 0.6106 |
| real5_forest_diffusion_synth5 | compas | IID | Sp-DFA | 64.47 | 0.0725 | 0.0016 | 0.0370 | 0.6077 |
| real5_forest_diffusion_synth5 | compas | non-IID | Benign | 63.77 | 0.1209 | 0.0175 | 0.0692 | 0.5685 |
| real5_forest_diffusion_synth5 | compas | non-IID | F Flip | 64.42 | 0.0978 | 0.0111 | 0.0544 | 0.5897 |
| real5_forest_diffusion_synth5 | compas | non-IID | FedSA | 64.74 | 0.0489 | 0.0026 | 0.0257 | 0.6217 |
| real5_forest_diffusion_synth5 | compas | non-IID | S-DFA | 65.87 | 0.0419 | 0.0020 | 0.0220 | 0.6368 |
| real5_forest_diffusion_synth5 | compas | non-IID | Sp-DFA | 65.06 | 0.1111 | 0.0167 | 0.0639 | 0.5867 |
| real5_gaussian_copula_synth5 | adult | IID | Benign | 82.36 | 0.0084 | 0.0439 | 0.0261 | 0.7974 |
| real5_gaussian_copula_synth5 | adult | IID | F Flip | 82.18 | 0.0365 | 0.0420 | 0.0392 | 0.7825 |
| real5_gaussian_copula_synth5 | adult | IID | FedSA | 82.16 | 0.0086 | 0.0329 | 0.0208 | 0.8008 |
| real5_gaussian_copula_synth5 | adult | IID | S-DFA | 81.87 | 0.0052 | 0.0413 | 0.0232 | 0.7955 |
| real5_gaussian_copula_synth5 | adult | IID | Sp-DFA | 82.51 | 0.0128 | 0.0450 | 0.0289 | 0.7962 |
| real5_gaussian_copula_synth5 | adult | non-IID | Benign | 83.43 | 0.0146 | 0.0404 | 0.0275 | 0.8068 |
| real5_gaussian_copula_synth5 | adult | non-IID | F Flip | 83.27 | 0.0038 | 0.0436 | 0.0237 | 0.8090 |
| real5_gaussian_copula_synth5 | adult | non-IID | FedSA | 81.92 | 0.0088 | 0.0488 | 0.0288 | 0.7904 |
| real5_gaussian_copula_synth5 | adult | non-IID | S-DFA | 82.85 | 0.0062 | 0.0509 | 0.0285 | 0.7999 |
| real5_gaussian_copula_synth5 | adult | non-IID | Sp-DFA | 83.10 | 0.0117 | 0.0585 | 0.0351 | 0.7959 |
| real5_gaussian_copula_synth5 | compas | IID | Benign | 65.17 | 0.0977 | 0.0191 | 0.0584 | 0.5933 |
| real5_gaussian_copula_synth5 | compas | IID | F Flip | 63.98 | 0.1227 | 0.0188 | 0.0708 | 0.5691 |
| real5_gaussian_copula_synth5 | compas | IID | FedSA | 65.55 | 0.0702 | 0.0016 | 0.0359 | 0.6196 |
| real5_gaussian_copula_synth5 | compas | IID | S-DFA | 64.90 | 0.0768 | 0.0024 | 0.0396 | 0.6094 |
| real5_gaussian_copula_synth5 | compas | IID | Sp-DFA | 64.74 | 0.0965 | 0.0021 | 0.0493 | 0.5981 |
| real5_gaussian_copula_synth5 | compas | non-IID | Benign | 65.06 | 0.0558 | 0.0008 | 0.0283 | 0.6224 |
| real5_gaussian_copula_synth5 | compas | non-IID | F Flip | 65.39 | 0.1046 | 0.0106 | 0.0576 | 0.5963 |
| real5_gaussian_copula_synth5 | compas | non-IID | FedSA | 65.44 | 0.0626 | 0.0003 | 0.0314 | 0.6230 |
| real5_gaussian_copula_synth5 | compas | non-IID | S-DFA | 65.17 | 0.0946 | 0.0182 | 0.0564 | 0.5953 |
| real5_gaussian_copula_synth5 | compas | non-IID | Sp-DFA | 65.60 | 0.0364 | 0.0007 | 0.0185 | 0.6375 |
| real5_pca_gaussian_synth5 | adult | IID | Benign | 80.84 | 0.0039 | 0.0100 | 0.0070 | 0.8014 |
| real5_pca_gaussian_synth5 | adult | IID | F Flip | 80.84 | 0.0133 | 0.0112 | 0.0122 | 0.7962 |
| real5_pca_gaussian_synth5 | adult | IID | FedSA | 81.33 | 0.0048 | 0.0097 | 0.0072 | 0.8060 |
| real5_pca_gaussian_synth5 | adult | IID | S-DFA | 80.00 | 0.0023 | 0.0004 | 0.0014 | 0.7986 |
| real5_pca_gaussian_synth5 | adult | IID | Sp-DFA | 80.61 | 0.0075 | 0.0285 | 0.0180 | 0.7881 |
| real5_pca_gaussian_synth5 | adult | non-IID | Benign | 79.77 | 0.0137 | 0.0014 | 0.0076 | 0.7902 |
| real5_pca_gaussian_synth5 | adult | non-IID | F Flip | 80.52 | 0.0051 | 0.0135 | 0.0093 | 0.7959 |
| real5_pca_gaussian_synth5 | adult | non-IID | FedSA | 80.03 | 0.0031 | 0.0186 | 0.0108 | 0.7894 |
| real5_pca_gaussian_synth5 | adult | non-IID | S-DFA | 78.36 | 0.0117 | 0.0094 | 0.0105 | 0.7730 |
| real5_pca_gaussian_synth5 | adult | non-IID | Sp-DFA | 80.30 | 0.0003 | 0.0072 | 0.0038 | 0.7992 |
| real5_pca_gaussian_synth5 | compas | IID | Benign | 64.63 | 0.1067 | 0.0130 | 0.0598 | 0.5865 |
| real5_pca_gaussian_synth5 | compas | IID | F Flip | 65.01 | 0.0862 | 0.0016 | 0.0439 | 0.6062 |
| real5_pca_gaussian_synth5 | compas | IID | FedSA | 64.74 | 0.0940 | 0.0184 | 0.0562 | 0.5912 |
| real5_pca_gaussian_synth5 | compas | IID | S-DFA | 64.52 | 0.1046 | 0.0108 | 0.0577 | 0.5875 |
| real5_pca_gaussian_synth5 | compas | IID | Sp-DFA | 64.31 | 0.0581 | 0.0065 | 0.0323 | 0.6108 |
| real5_pca_gaussian_synth5 | compas | non-IID | Benign | 64.36 | 0.0780 | 0.0035 | 0.0408 | 0.6029 |
| real5_pca_gaussian_synth5 | compas | non-IID | F Flip | 64.04 | 0.0447 | 0.0009 | 0.0228 | 0.6176 |
| real5_pca_gaussian_synth5 | compas | non-IID | FedSA | 64.90 | 0.0702 | 0.0054 | 0.0378 | 0.6112 |
| real5_pca_gaussian_synth5 | compas | non-IID | S-DFA | 64.47 | 0.0700 | 0.0032 | 0.0366 | 0.6081 |
| real5_pca_gaussian_synth5 | compas | non-IID | Sp-DFA | 63.34 | 0.1044 | 0.0204 | 0.0624 | 0.5710 |
| real5_smote_synth5 | adult | IID | Benign | 80.92 | 0.0168 | 0.1417 | 0.0793 | 0.7299 |
| real5_smote_synth5 | adult | IID | F Flip | 83.68 | 0.1303 | 0.1909 | 0.1606 | 0.6762 |
| real5_smote_synth5 | adult | IID | FedSA | 82.67 | 0.0762 | 0.1775 | 0.1269 | 0.6999 |
| real5_smote_synth5 | adult | IID | S-DFA | 78.51 | 0.0012 | 0.0161 | 0.0087 | 0.7764 |
| real5_smote_synth5 | adult | IID | Sp-DFA | 83.87 | 0.0955 | 0.1567 | 0.1261 | 0.7126 |
| real5_smote_synth5 | adult | non-IID | Benign | 82.71 | 0.0898 | 0.1778 | 0.1338 | 0.6933 |
| real5_smote_synth5 | adult | non-IID | F Flip | 83.93 | 0.1035 | 0.2038 | 0.1536 | 0.6857 |
| real5_smote_synth5 | adult | non-IID | FedSA | 83.25 | 0.0770 | 0.1673 | 0.1221 | 0.7104 |
| real5_smote_synth5 | adult | non-IID | S-DFA | 83.37 | 0.0873 | 0.1746 | 0.1310 | 0.7027 |
| real5_smote_synth5 | adult | non-IID | Sp-DFA | 84.12 | 0.1027 | 0.1822 | 0.1425 | 0.6987 |
| real5_smote_synth5 | compas | IID | Benign | 64.09 | 0.0121 | 0.0149 | 0.0135 | 0.6274 |
| real5_smote_synth5 | compas | IID | F Flip | 63.98 | 0.0239 | 0.0096 | 0.0167 | 0.6231 |
| real5_smote_synth5 | compas | IID | FedSA | 61.88 | 0.0310 | 0.0029 | 0.0170 | 0.6018 |
| real5_smote_synth5 | compas | IID | S-DFA | 64.47 | 0.0226 | 0.0071 | 0.0149 | 0.6298 |
| real5_smote_synth5 | compas | IID | Sp-DFA | 63.66 | 0.0416 | 0.0008 | 0.0212 | 0.6154 |
| real5_smote_synth5 | compas | non-IID | Benign | 64.15 | 0.0602 | 0.0035 | 0.0318 | 0.6097 |
| real5_smote_synth5 | compas | non-IID | F Flip | 63.93 | 0.1045 | 0.0155 | 0.0600 | 0.5793 |
| real5_smote_synth5 | compas | non-IID | FedSA | 62.69 | 0.0349 | 0.0016 | 0.0183 | 0.6086 |
| real5_smote_synth5 | compas | non-IID | S-DFA | 64.79 | 0.0229 | 0.0008 | 0.0118 | 0.6361 |
| real5_smote_synth5 | compas | non-IID | Sp-DFA | 63.34 | 0.0802 | 0.0044 | 0.0423 | 0.5910 |
| real5_tvae_synth5 | adult | IID | Benign | 83.30 | 0.0108 | 0.0912 | 0.0510 | 0.7820 |
| real5_tvae_synth5 | adult | IID | F Flip | 83.60 | 0.0080 | 0.1205 | 0.0642 | 0.7718 |
| real5_tvae_synth5 | adult | IID | FedSA | 83.31 | 0.0037 | 0.0564 | 0.0301 | 0.8030 |
| real5_tvae_synth5 | adult | IID | S-DFA | 83.45 | 0.0017 | 0.1005 | 0.0511 | 0.7835 |
| real5_tvae_synth5 | adult | IID | Sp-DFA | 84.19 | 0.0029 | 0.1089 | 0.0559 | 0.7860 |
| real5_tvae_synth5 | adult | non-IID | Benign | 83.57 | 0.0054 | 0.1123 | 0.0588 | 0.7769 |
| real5_tvae_synth5 | adult | non-IID | F Flip | 83.88 | 0.0145 | 0.0730 | 0.0438 | 0.7950 |
| real5_tvae_synth5 | adult | non-IID | FedSA | 83.50 | 0.0183 | 0.1056 | 0.0619 | 0.7731 |
| real5_tvae_synth5 | adult | non-IID | S-DFA | 83.78 | 0.0022 | 0.0949 | 0.0486 | 0.7893 |
| real5_tvae_synth5 | adult | non-IID | Sp-DFA | 83.65 | 0.0193 | 0.1357 | 0.0775 | 0.7590 |
| real5_tvae_synth5 | compas | IID | Benign | 62.31 | 0.0400 | 0.0000 | 0.0200 | 0.6031 |
| real5_tvae_synth5 | compas | IID | F Flip | 60.48 | 0.0698 | 0.0177 | 0.0437 | 0.5610 |
| real5_tvae_synth5 | compas | IID | FedSA | 58.15 | 0.0256 | 0.0076 | 0.0166 | 0.5649 |
| real5_tvae_synth5 | compas | IID | S-DFA | 61.34 | 0.0593 | 0.0127 | 0.0360 | 0.5774 |
| real5_tvae_synth5 | compas | IID | Sp-DFA | 60.96 | 0.1109 | 0.0423 | 0.0766 | 0.5330 |
| real5_tvae_synth5 | compas | non-IID | Benign | 61.12 | 0.0287 | 0.0048 | 0.0167 | 0.5945 |
| real5_tvae_synth5 | compas | non-IID | F Flip | 60.48 | 0.0303 | 0.0020 | 0.0161 | 0.5886 |
| real5_tvae_synth5 | compas | non-IID | FedSA | 60.42 | 0.0197 | 0.0081 | 0.0139 | 0.5903 |
| real5_tvae_synth5 | compas | non-IID | S-DFA | 58.42 | 0.0011 | 0.0016 | 0.0014 | 0.5829 |
| real5_tvae_synth5 | compas | non-IID | Sp-DFA | 62.96 | 0.0196 | 0.0011 | 0.0103 | 0.6192 |
