# GuardFed-AD2+ Advisor Experiments Summary

All values are generated from results/paper_tables/raw_results.jsonl. Metrics use the configured last-10-round selection rule: ACC=max, AEOD=min, ASPD=min.

## Adult AD2+ Ablation: tag-level summary

| score_rank | tag | acc_mean | aeod_mean | aspd_mean | fair_mean | score_mean | acc_rank | fair_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_score | 0.8305 | 0.0079 | 0.0563 | 0.0321 | 0.7984 | 2 | 4 |
| 2 | geo_only_CA | 0.8290 | 0.0081 | 0.0543 | 0.0312 | 0.7978 | 6 | 2 |
| 3 | no_centrality_C | 0.8329 | 0.0089 | 0.0624 | 0.0357 | 0.7973 | 1 | 15 |
| 4 | macro_R0.50_P0.50 | 0.8293 | 0.0026 | 0.0632 | 0.0329 | 0.7964 | 5 | 5 |
| 5 | macro_R0.75_P0.25 | 0.8305 | 0.0068 | 0.0619 | 0.0343 | 0.7961 | 3 | 11 |
| 6 | no_fairness_risk_F | 0.8271 | 0.0049 | 0.0572 | 0.0310 | 0.7960 | 12 | 1 |
| 7 | no_utility_U | 0.8271 | 0.0047 | 0.0591 | 0.0319 | 0.7952 | 11 | 3 |
| 8 | no_alignment_A | 0.8279 | 0.0055 | 0.0624 | 0.0339 | 0.7940 | 7 | 9 |
| 9 | no_violation_V | 0.8277 | 0.0080 | 0.0602 | 0.0341 | 0.7936 | 8 | 10 |
| 10 | macro_R0.25_P0.75 | 0.8297 | 0.0082 | 0.0645 | 0.0364 | 0.7933 | 4 | 16 |
| 11 | macro_R1.00_P0.00 | 0.8276 | 0.0097 | 0.0598 | 0.0347 | 0.7929 | 9 | 12 |
| 11 | reward_only_R | 0.8276 | 0.0097 | 0.0598 | 0.0347 | 0.7929 | 9 | 12 |
| 13 | fair_only_FV | 0.8256 | 0.0071 | 0.0597 | 0.0334 | 0.7922 | 14 | 6 |
| 13 | macro_R0.00_P1.00 | 0.8256 | 0.0071 | 0.0597 | 0.0334 | 0.7922 | 14 | 6 |
| 13 | penalty_only_P | 0.8256 | 0.0071 | 0.0597 | 0.0334 | 0.7922 | 14 | 6 |
| 16 | utility_only_U | 0.8262 | 0.0075 | 0.0631 | 0.0353 | 0.7908 | 13 | 14 |

## Server/root distribution trend

| dataset | server_alpha | acc_mean | aeod_mean | aspd_mean | fair_mean | score_mean | delta_score_vs_5000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 0.0300 | 0.8149 | 0.0546 | 0.0273 | 0.0409 | 0.7740 | -0.0227 |
| adult | 0.0500 | 0.8149 | 0.0546 | 0.0273 | 0.0409 | 0.7740 | -0.0227 |
| adult | 0.1000 | 0.2855 | 0.0003 | 0.0247 | 0.0125 | 0.2729 | -0.5237 |
| adult | 0.2000 | 0.7238 | 0.6730 | 0.2482 | 0.4606 | 0.2632 | -0.5335 |
| adult | 0.5000 | 0.4201 | 0.0126 | 0.1084 | 0.0605 | 0.3596 | -0.4371 |
| adult | 1.0000 | 0.7543 | 0.1550 | 0.0709 | 0.1129 | 0.6414 | -0.1553 |
| adult | 2.0000 | 0.7609 | 0.1403 | 0.0167 | 0.0785 | 0.6825 | -0.1142 |
| adult | 5.0000 | 0.8158 | 0.0087 | 0.0007 | 0.0047 | 0.8111 | 0.0145 |
| adult | 50.0000 | 0.8382 | 0.0107 | 0.0860 | 0.0484 | 0.7898 | -0.0068 |
| adult | 5000.0000 | 0.8338 | 0.0080 | 0.0663 | 0.0372 | 0.7967 | 0.0000 |
| compas | 0.0300 | 0.6259 | 0.4715 | 0.3649 | 0.4182 | 0.2076 | -0.3903 |
| compas | 0.0500 | 0.6259 | 0.4715 | 0.3649 | 0.4182 | 0.2076 | -0.3903 |
| compas | 0.1000 | 0.4591 | 0.0049 | 0.0123 | 0.0086 | 0.4505 | -0.1475 |
| compas | 0.2000 | 0.5900 | 0.4783 | 0.2899 | 0.3841 | 0.2059 | -0.3921 |
| compas | 0.5000 | 0.4575 | 0.0037 | 0.0045 | 0.0041 | 0.4534 | -0.1446 |
| compas | 1.0000 | 0.6315 | 0.0398 | 0.0340 | 0.0369 | 0.5946 | -0.0034 |
| compas | 2.0000 | 0.6239 | 0.0228 | 0.0468 | 0.0348 | 0.5891 | -0.0089 |
| compas | 5.0000 | 0.6504 | 0.0200 | 0.0187 | 0.0194 | 0.6310 | 0.0330 |
| compas | 50.0000 | 0.6574 | 0.0306 | 0.0072 | 0.0189 | 0.6385 | 0.0405 |
| compas | 5000.0000 | 0.6497 | 0.0867 | 0.0167 | 0.0517 | 0.5980 | 0.0000 |

## Synthetic server data: overall ranking

| score_rank | tag | server_ratio | synthetic_ratio | synthetic_method | acc_mean | aeod_mean | aspd_mean | fair_mean | score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | real1_pca_gaussian_synth9 | 0.0100 | 0.0900 | pca_gaussian | 0.7395 | 0.0084 | 0.0210 | 0.0147 | 0.7248 |
| 2 | real1_gaussian_copula_synth9 | 0.0100 | 0.0900 | gaussian_copula | 0.7358 | 0.0079 | 0.0332 | 0.0205 | 0.7153 |
| 3 | real5_forest_diffusion_synth5 | 0.0500 | 0.0500 | forest_diffusion | 0.7406 | 0.0429 | 0.0287 | 0.0358 | 0.7048 |
| 4 | real5_gaussian_copula_synth5 | 0.0500 | 0.0500 | gaussian_copula | 0.7383 | 0.0467 | 0.0261 | 0.0364 | 0.7019 |
| 5 | real10_none | 0.1000 | 0.0000 | none | 0.7392 | 0.0474 | 0.0356 | 0.0415 | 0.6976 |
| 6 | real5_pca_gaussian_synth5 | 0.0500 | 0.0500 | pca_gaussian | 0.7235 | 0.0441 | 0.0097 | 0.0269 | 0.6966 |
| 7 | real5_ctgan_synth5 | 0.0500 | 0.0500 | ctgan | 0.7339 | 0.0332 | 0.0495 | 0.0413 | 0.6925 |
| 8 | real1_forest_diffusion_synth9 | 0.0100 | 0.0900 | forest_diffusion | 0.7301 | 0.0285 | 0.0642 | 0.0463 | 0.6838 |
| 9 | real5_tvae_synth5 | 0.0500 | 0.0500 | tvae | 0.7214 | 0.0246 | 0.0548 | 0.0397 | 0.6817 |
| 10 | real5_smote_synth5 | 0.0500 | 0.0500 | smote | 0.7320 | 0.0607 | 0.0825 | 0.0716 | 0.6604 |
| 11 | real1_smote_synth9 | 0.0100 | 0.0900 | smote | 0.7316 | 0.0901 | 0.0865 | 0.0883 | 0.6433 |
| 12 | real1_tvae_synth9 | 0.0100 | 0.0900 | tvae | 0.6820 | 0.0660 | 0.0375 | 0.0518 | 0.6302 |
| 13 | real1_ctgan_synth9 | 0.0100 | 0.0900 | ctgan | 0.7172 | 0.1027 | 0.1402 | 0.1215 | 0.5957 |

## Synthetic server data: by dataset

| dataset | score_rank_in_dataset | tag | acc_mean | aeod_mean | aspd_mean | fair_mean | score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 1 | real1_pca_gaussian_synth9 | 0.8188 | 0.0048 | 0.0092 | 0.0070 | 0.8118 |
| adult | 2 | real1_gaussian_copula_synth9 | 0.8129 | 0.0036 | 0.0010 | 0.0023 | 0.8106 |
| adult | 3 | real5_ctgan_synth5 | 0.8228 | 0.0232 | 0.0114 | 0.0173 | 0.8055 |
| adult | 4 | real5_forest_diffusion_synth5 | 0.8306 | 0.0069 | 0.0510 | 0.0290 | 0.8017 |
| adult | 5 | real10_none | 0.8305 | 0.0079 | 0.0563 | 0.0321 | 0.7984 |
| adult | 6 | real5_gaussian_copula_synth5 | 0.8256 | 0.0117 | 0.0447 | 0.0282 | 0.7975 |
| adult | 7 | real5_pca_gaussian_synth5 | 0.8026 | 0.0066 | 0.0110 | 0.0088 | 0.7938 |
| adult | 8 | real1_forest_diffusion_synth9 | 0.7990 | 0.0066 | 0.0088 | 0.0077 | 0.7914 |
| adult | 9 | real5_tvae_synth5 | 0.8362 | 0.0087 | 0.0999 | 0.0543 | 0.7819 |
| adult | 10 | real1_ctgan_synth9 | 0.7652 | 0.0012 | 0.0050 | 0.0031 | 0.7621 |
| adult | 11 | real1_tvae_synth9 | 0.7622 | 0.0046 | 0.0066 | 0.0056 | 0.7567 |
| adult | 12 | real5_smote_synth5 | 0.8270 | 0.0780 | 0.1589 | 0.1184 | 0.7086 |
| adult | 13 | real1_smote_synth9 | 0.8071 | 0.1493 | 0.1572 | 0.1533 | 0.6538 |
| compas | 1 | real1_pca_gaussian_synth9 | 0.6602 | 0.0120 | 0.0328 | 0.0224 | 0.6378 |
| compas | 2 | real1_smote_synth9 | 0.6560 | 0.0308 | 0.0158 | 0.0233 | 0.6327 |
| compas | 3 | real1_gaussian_copula_synth9 | 0.6587 | 0.0122 | 0.0655 | 0.0388 | 0.6199 |
| compas | 4 | real5_smote_synth5 | 0.6370 | 0.0434 | 0.0061 | 0.0248 | 0.6122 |
| compas | 5 | real5_forest_diffusion_synth5 | 0.6505 | 0.0789 | 0.0064 | 0.0427 | 0.6078 |
| compas | 6 | real5_gaussian_copula_synth5 | 0.6510 | 0.0818 | 0.0075 | 0.0446 | 0.6064 |
| compas | 7 | real5_pca_gaussian_synth5 | 0.6443 | 0.0817 | 0.0084 | 0.0450 | 0.5993 |
| compas | 8 | real10_none | 0.6478 | 0.0869 | 0.0149 | 0.0509 | 0.5969 |
| compas | 9 | real5_tvae_synth5 | 0.6066 | 0.0405 | 0.0098 | 0.0252 | 0.5815 |
| compas | 10 | real5_ctgan_synth5 | 0.6449 | 0.0431 | 0.0875 | 0.0653 | 0.5796 |
| compas | 11 | real1_forest_diffusion_synth9 | 0.6612 | 0.0504 | 0.1195 | 0.0850 | 0.5762 |
| compas | 12 | real1_tvae_synth9 | 0.6017 | 0.1275 | 0.0685 | 0.0980 | 0.5037 |
| compas | 13 | real1_ctgan_synth9 | 0.6692 | 0.2043 | 0.2755 | 0.2399 | 0.4293 |
