# Adult AD2+ Score Ablation

| tag | dataset | distribution | attack | ACC_pct | AEOD | ASPD | fair_avg | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fair_only_FV | adult | IID | Benign | 82.85 | 0.0004 | 0.0641 | 0.0323 | 0.7963 |
| fair_only_FV | adult | IID | F Flip | 82.23 | 0.0002 | 0.0666 | 0.0334 | 0.7889 |
| fair_only_FV | adult | IID | FedSA | 82.28 | 0.0001 | 0.0434 | 0.0218 | 0.8010 |
| fair_only_FV | adult | IID | S-DFA | 82.12 | 0.0049 | 0.0416 | 0.0232 | 0.7980 |
| fair_only_FV | adult | IID | Sp-DFA | 82.70 | 0.0035 | 0.0646 | 0.0340 | 0.7930 |
| fair_only_FV | adult | non-IID | Benign | 82.32 | 0.0290 | 0.0654 | 0.0472 | 0.7760 |
| fair_only_FV | adult | non-IID | F Flip | 83.64 | 0.0210 | 0.0692 | 0.0451 | 0.7913 |
| fair_only_FV | adult | non-IID | FedSA | 81.85 | 0.0003 | 0.0692 | 0.0348 | 0.7837 |
| fair_only_FV | adult | non-IID | S-DFA | 83.37 | 0.0001 | 0.0705 | 0.0353 | 0.7983 |
| fair_only_FV | adult | non-IID | Sp-DFA | 82.20 | 0.0120 | 0.0419 | 0.0269 | 0.7951 |
| full_score | adult | IID | Benign | 83.64 | 0.0054 | 0.0663 | 0.0358 | 0.8006 |
| full_score | adult | IID | F Flip | 82.21 | 0.0005 | 0.0625 | 0.0315 | 0.7906 |
| full_score | adult | IID | FedSA | 83.24 | 0.0005 | 0.0625 | 0.0315 | 0.8009 |
| full_score | adult | IID | S-DFA | 83.69 | 0.0022 | 0.0658 | 0.0340 | 0.8029 |
| full_score | adult | IID | Sp-DFA | 83.67 | 0.0072 | 0.0615 | 0.0343 | 0.8024 |
| full_score | adult | non-IID | Benign | 82.18 | 0.0141 | 0.0329 | 0.0235 | 0.7983 |
| full_score | adult | non-IID | F Flip | 83.09 | 0.0207 | 0.0474 | 0.0341 | 0.7969 |
| full_score | adult | non-IID | FedSA | 82.62 | 0.0024 | 0.0484 | 0.0254 | 0.8007 |
| full_score | adult | non-IID | S-DFA | 82.89 | 0.0095 | 0.0538 | 0.0317 | 0.7973 |
| full_score | adult | non-IID | Sp-DFA | 83.25 | 0.0167 | 0.0614 | 0.0390 | 0.7935 |
| geo_only_CA | adult | IID | Benign | 82.26 | 0.0028 | 0.0657 | 0.0343 | 0.7884 |
| geo_only_CA | adult | IID | F Flip | 82.43 | 0.0129 | 0.0628 | 0.0378 | 0.7865 |
| geo_only_CA | adult | IID | FedSA | 82.12 | 0.0051 | 0.0602 | 0.0326 | 0.7886 |
| geo_only_CA | adult | IID | S-DFA | 82.18 | 0.0024 | 0.0628 | 0.0326 | 0.7892 |
| geo_only_CA | adult | IID | Sp-DFA | 82.96 | 0.0041 | 0.0420 | 0.0230 | 0.8066 |
| geo_only_CA | adult | non-IID | Benign | 83.15 | 0.0035 | 0.0475 | 0.0255 | 0.8060 |
| geo_only_CA | adult | non-IID | F Flip | 82.62 | 0.0150 | 0.0492 | 0.0321 | 0.7941 |
| geo_only_CA | adult | non-IID | FedSA | 83.94 | 0.0063 | 0.0609 | 0.0336 | 0.8058 |
| geo_only_CA | adult | non-IID | S-DFA | 83.82 | 0.0018 | 0.0593 | 0.0306 | 0.8077 |
| geo_only_CA | adult | non-IID | Sp-DFA | 83.53 | 0.0271 | 0.0326 | 0.0298 | 0.8055 |
| macro_R0.00_P1.00 | adult | IID | Benign | 82.85 | 0.0004 | 0.0641 | 0.0323 | 0.7963 |
| macro_R0.00_P1.00 | adult | IID | F Flip | 82.23 | 0.0002 | 0.0666 | 0.0334 | 0.7889 |
| macro_R0.00_P1.00 | adult | IID | FedSA | 82.28 | 0.0001 | 0.0434 | 0.0218 | 0.8010 |
| macro_R0.00_P1.00 | adult | IID | S-DFA | 82.12 | 0.0049 | 0.0416 | 0.0232 | 0.7980 |
| macro_R0.00_P1.00 | adult | IID | Sp-DFA | 82.70 | 0.0035 | 0.0646 | 0.0340 | 0.7930 |
| macro_R0.00_P1.00 | adult | non-IID | Benign | 82.32 | 0.0290 | 0.0654 | 0.0472 | 0.7760 |
| macro_R0.00_P1.00 | adult | non-IID | F Flip | 83.64 | 0.0210 | 0.0692 | 0.0451 | 0.7913 |
| macro_R0.00_P1.00 | adult | non-IID | FedSA | 81.85 | 0.0003 | 0.0692 | 0.0348 | 0.7837 |
| macro_R0.00_P1.00 | adult | non-IID | S-DFA | 83.37 | 0.0001 | 0.0705 | 0.0353 | 0.7983 |
| macro_R0.00_P1.00 | adult | non-IID | Sp-DFA | 82.20 | 0.0120 | 0.0419 | 0.0269 | 0.7951 |
| macro_R0.25_P0.75 | adult | IID | Benign | 83.64 | 0.0024 | 0.0635 | 0.0330 | 0.8035 |
| macro_R0.25_P0.75 | adult | IID | F Flip | 82.24 | 0.0021 | 0.0633 | 0.0327 | 0.7897 |
| macro_R0.25_P0.75 | adult | IID | FedSA | 82.20 | 0.0008 | 0.0684 | 0.0346 | 0.7874 |
| macro_R0.25_P0.75 | adult | IID | S-DFA | 83.75 | 0.0022 | 0.0637 | 0.0330 | 0.8046 |
| macro_R0.25_P0.75 | adult | IID | Sp-DFA | 82.26 | 0.0030 | 0.0679 | 0.0355 | 0.7871 |
| macro_R0.25_P0.75 | adult | non-IID | Benign | 83.46 | 0.0046 | 0.0629 | 0.0337 | 0.8009 |
| macro_R0.25_P0.75 | adult | non-IID | F Flip | 82.26 | 0.0315 | 0.0624 | 0.0470 | 0.7756 |
| macro_R0.25_P0.75 | adult | non-IID | FedSA | 83.74 | 0.0073 | 0.0625 | 0.0349 | 0.8025 |
| macro_R0.25_P0.75 | adult | non-IID | S-DFA | 83.82 | 0.0057 | 0.0657 | 0.0357 | 0.8025 |
| macro_R0.25_P0.75 | adult | non-IID | Sp-DFA | 82.30 | 0.0220 | 0.0651 | 0.0435 | 0.7795 |
| macro_R0.50_P0.50 | adult | IID | Benign | 82.22 | 0.0011 | 0.0688 | 0.0349 | 0.7873 |
| macro_R0.50_P0.50 | adult | IID | F Flip | 82.14 | 0.0058 | 0.0660 | 0.0359 | 0.7855 |
| macro_R0.50_P0.50 | adult | IID | FedSA | 82.10 | 0.0011 | 0.0654 | 0.0332 | 0.7878 |
| macro_R0.50_P0.50 | adult | IID | S-DFA | 83.82 | 0.0024 | 0.0660 | 0.0342 | 0.8040 |
| macro_R0.50_P0.50 | adult | IID | Sp-DFA | 82.18 | 0.0005 | 0.0656 | 0.0331 | 0.7888 |
| macro_R0.50_P0.50 | adult | non-IID | Benign | 83.31 | 0.0003 | 0.0635 | 0.0319 | 0.8012 |
| macro_R0.50_P0.50 | adult | non-IID | F Flip | 83.56 | 0.0009 | 0.0603 | 0.0306 | 0.8050 |
| macro_R0.50_P0.50 | adult | non-IID | FedSA | 82.66 | 0.0047 | 0.0604 | 0.0326 | 0.7940 |
| macro_R0.50_P0.50 | adult | non-IID | S-DFA | 83.48 | 0.0056 | 0.0517 | 0.0286 | 0.8062 |
| macro_R0.50_P0.50 | adult | non-IID | Sp-DFA | 83.81 | 0.0035 | 0.0648 | 0.0341 | 0.8040 |
| macro_R0.75_P0.25 | adult | IID | Benign | 82.16 | 0.0189 | 0.0629 | 0.0409 | 0.7807 |
| macro_R0.75_P0.25 | adult | IID | F Flip | 82.42 | 0.0029 | 0.0651 | 0.0340 | 0.7902 |
| macro_R0.75_P0.25 | adult | IID | FedSA | 82.55 | 0.0003 | 0.0656 | 0.0329 | 0.7926 |
| macro_R0.75_P0.25 | adult | IID | S-DFA | 83.69 | 0.0012 | 0.0644 | 0.0328 | 0.8041 |
| macro_R0.75_P0.25 | adult | IID | Sp-DFA | 83.74 | 0.0023 | 0.0606 | 0.0315 | 0.8059 |
| macro_R0.75_P0.25 | adult | non-IID | Benign | 83.04 | 0.0187 | 0.0632 | 0.0409 | 0.7895 |
| macro_R0.75_P0.25 | adult | non-IID | F Flip | 83.70 | 0.0000 | 0.0654 | 0.0327 | 0.8043 |
| macro_R0.75_P0.25 | adult | non-IID | FedSA | 82.35 | 0.0031 | 0.0620 | 0.0326 | 0.7909 |
| macro_R0.75_P0.25 | adult | non-IID | S-DFA | 83.25 | 0.0150 | 0.0447 | 0.0299 | 0.8026 |
| macro_R0.75_P0.25 | adult | non-IID | Sp-DFA | 83.57 | 0.0052 | 0.0647 | 0.0350 | 0.8008 |
| macro_R1.00_P0.00 | adult | IID | Benign | 83.70 | 0.0067 | 0.0652 | 0.0359 | 0.8011 |
| macro_R1.00_P0.00 | adult | IID | F Flip | 82.27 | 0.0133 | 0.0598 | 0.0366 | 0.7861 |
| macro_R1.00_P0.00 | adult | IID | FedSA | 81.97 | 0.0209 | 0.0616 | 0.0413 | 0.7784 |
| macro_R1.00_P0.00 | adult | IID | S-DFA | 81.96 | 0.0024 | 0.0621 | 0.0323 | 0.7873 |
| macro_R1.00_P0.00 | adult | IID | Sp-DFA | 82.32 | 0.0012 | 0.0668 | 0.0340 | 0.7892 |
| macro_R1.00_P0.00 | adult | non-IID | Benign | 82.74 | 0.0099 | 0.0604 | 0.0352 | 0.7923 |
| macro_R1.00_P0.00 | adult | non-IID | F Flip | 83.19 | 0.0235 | 0.0389 | 0.0312 | 0.8007 |
| macro_R1.00_P0.00 | adult | non-IID | FedSA | 82.95 | 0.0029 | 0.0622 | 0.0326 | 0.7969 |
| macro_R1.00_P0.00 | adult | non-IID | S-DFA | 82.81 | 0.0153 | 0.0563 | 0.0358 | 0.7923 |
| macro_R1.00_P0.00 | adult | non-IID | Sp-DFA | 83.68 | 0.0008 | 0.0641 | 0.0325 | 0.8044 |
| no_alignment_A | adult | IID | Benign | 82.20 | 0.0025 | 0.0628 | 0.0326 | 0.7894 |
| no_alignment_A | adult | IID | F Flip | 83.85 | 0.0020 | 0.0684 | 0.0352 | 0.8033 |
| no_alignment_A | adult | IID | FedSA | 82.09 | 0.0166 | 0.0639 | 0.0403 | 0.7806 |
| no_alignment_A | adult | IID | S-DFA | 82.20 | 0.0114 | 0.0642 | 0.0378 | 0.7842 |
| no_alignment_A | adult | IID | Sp-DFA | 82.18 | 0.0014 | 0.0621 | 0.0317 | 0.7901 |
| no_alignment_A | adult | non-IID | Benign | 83.22 | 0.0007 | 0.0673 | 0.0340 | 0.7982 |
| no_alignment_A | adult | non-IID | F Flip | 83.56 | 0.0062 | 0.0428 | 0.0245 | 0.8111 |
| no_alignment_A | adult | non-IID | FedSA | 82.98 | 0.0097 | 0.0612 | 0.0354 | 0.7944 |
| no_alignment_A | adult | non-IID | S-DFA | 83.04 | 0.0035 | 0.0647 | 0.0341 | 0.7963 |
| no_alignment_A | adult | non-IID | Sp-DFA | 82.61 | 0.0013 | 0.0662 | 0.0337 | 0.7924 |
| no_centrality_C | adult | IID | Benign | 83.37 | 0.0013 | 0.0592 | 0.0302 | 0.8035 |
| no_centrality_C | adult | IID | F Flip | 83.43 | 0.0114 | 0.0624 | 0.0369 | 0.7974 |
| no_centrality_C | adult | IID | FedSA | 83.31 | 0.0279 | 0.0620 | 0.0450 | 0.7881 |
| no_centrality_C | adult | IID | S-DFA | 83.37 | 0.0059 | 0.0624 | 0.0341 | 0.7995 |
| no_centrality_C | adult | IID | Sp-DFA | 83.78 | 0.0012 | 0.0673 | 0.0342 | 0.8036 |
| no_centrality_C | adult | non-IID | Benign | 82.11 | 0.0209 | 0.0583 | 0.0396 | 0.7815 |
| no_centrality_C | adult | non-IID | F Flip | 83.83 | 0.0056 | 0.0681 | 0.0368 | 0.8015 |
| no_centrality_C | adult | non-IID | FedSA | 82.32 | 0.0039 | 0.0596 | 0.0318 | 0.7915 |
| no_centrality_C | adult | non-IID | S-DFA | 83.58 | 0.0043 | 0.0613 | 0.0328 | 0.8030 |
| no_centrality_C | adult | non-IID | Sp-DFA | 83.83 | 0.0062 | 0.0639 | 0.0351 | 0.8032 |
| no_fairness_risk_F | adult | IID | Benign | 82.39 | 0.0006 | 0.0653 | 0.0329 | 0.7909 |
| no_fairness_risk_F | adult | IID | F Flip | 82.39 | 0.0052 | 0.0625 | 0.0338 | 0.7901 |
| no_fairness_risk_F | adult | IID | FedSA | 82.04 | 0.0055 | 0.0624 | 0.0340 | 0.7864 |
| no_fairness_risk_F | adult | IID | S-DFA | 83.27 | 0.0097 | 0.0616 | 0.0357 | 0.7970 |
| no_fairness_risk_F | adult | IID | Sp-DFA | 82.44 | 0.0007 | 0.0665 | 0.0336 | 0.7907 |
| no_fairness_risk_F | adult | non-IID | Benign | 82.58 | 0.0002 | 0.0555 | 0.0279 | 0.7980 |
| no_fairness_risk_F | adult | non-IID | F Flip | 82.94 | 0.0066 | 0.0471 | 0.0268 | 0.8026 |
| no_fairness_risk_F | adult | non-IID | FedSA | 82.30 | 0.0065 | 0.0431 | 0.0248 | 0.7981 |
| no_fairness_risk_F | adult | non-IID | S-DFA | 82.94 | 0.0131 | 0.0429 | 0.0280 | 0.8014 |
| no_fairness_risk_F | adult | non-IID | Sp-DFA | 83.80 | 0.0005 | 0.0653 | 0.0329 | 0.8051 |
| no_utility_U | adult | IID | Benign | 82.32 | 0.0004 | 0.0606 | 0.0305 | 0.7926 |
| no_utility_U | adult | IID | F Flip | 83.47 | 0.0004 | 0.0676 | 0.0340 | 0.8007 |
| no_utility_U | adult | IID | FedSA | 82.72 | 0.0025 | 0.0610 | 0.0317 | 0.7955 |
| no_utility_U | adult | IID | S-DFA | 82.73 | 0.0039 | 0.0598 | 0.0319 | 0.7955 |
| no_utility_U | adult | IID | Sp-DFA | 82.14 | 0.0029 | 0.0594 | 0.0311 | 0.7902 |
| no_utility_U | adult | non-IID | Benign | 82.97 | 0.0052 | 0.0657 | 0.0354 | 0.7943 |
| no_utility_U | adult | non-IID | F Flip | 82.73 | 0.0116 | 0.0490 | 0.0303 | 0.7970 |
| no_utility_U | adult | non-IID | FedSA | 82.88 | 0.0027 | 0.0590 | 0.0308 | 0.7980 |
| no_utility_U | adult | non-IID | S-DFA | 82.26 | 0.0094 | 0.0597 | 0.0346 | 0.7880 |
| no_utility_U | adult | non-IID | Sp-DFA | 82.91 | 0.0078 | 0.0495 | 0.0287 | 0.8005 |
| no_violation_V | adult | IID | Benign | 82.34 | 0.0241 | 0.0620 | 0.0430 | 0.7804 |
| no_violation_V | adult | IID | F Flip | 82.46 | 0.0021 | 0.0620 | 0.0320 | 0.7925 |
| no_violation_V | adult | IID | FedSA | 82.35 | 0.0241 | 0.0628 | 0.0434 | 0.7800 |
| no_violation_V | adult | IID | S-DFA | 82.20 | 0.0011 | 0.0612 | 0.0311 | 0.7909 |
| no_violation_V | adult | IID | Sp-DFA | 83.94 | 0.0015 | 0.0538 | 0.0277 | 0.8117 |
| no_violation_V | adult | non-IID | Benign | 82.32 | 0.0011 | 0.0525 | 0.0268 | 0.7963 |
| no_violation_V | adult | non-IID | F Flip | 82.88 | 0.0047 | 0.0665 | 0.0356 | 0.7932 |
| no_violation_V | adult | non-IID | FedSA | 83.07 | 0.0186 | 0.0596 | 0.0391 | 0.7916 |
| no_violation_V | adult | non-IID | S-DFA | 82.64 | 0.0018 | 0.0595 | 0.0306 | 0.7958 |
| no_violation_V | adult | non-IID | Sp-DFA | 83.53 | 0.0010 | 0.0618 | 0.0314 | 0.8039 |
| penalty_only_P | adult | IID | Benign | 82.85 | 0.0004 | 0.0641 | 0.0323 | 0.7963 |
| penalty_only_P | adult | IID | F Flip | 82.23 | 0.0002 | 0.0666 | 0.0334 | 0.7889 |
| penalty_only_P | adult | IID | FedSA | 82.28 | 0.0001 | 0.0434 | 0.0218 | 0.8010 |
| penalty_only_P | adult | IID | S-DFA | 82.12 | 0.0049 | 0.0416 | 0.0232 | 0.7980 |
| penalty_only_P | adult | IID | Sp-DFA | 82.70 | 0.0035 | 0.0646 | 0.0340 | 0.7930 |
| penalty_only_P | adult | non-IID | Benign | 82.32 | 0.0290 | 0.0654 | 0.0472 | 0.7760 |
| penalty_only_P | adult | non-IID | F Flip | 83.64 | 0.0210 | 0.0692 | 0.0451 | 0.7913 |
| penalty_only_P | adult | non-IID | FedSA | 81.85 | 0.0003 | 0.0692 | 0.0348 | 0.7837 |
| penalty_only_P | adult | non-IID | S-DFA | 83.37 | 0.0001 | 0.0705 | 0.0353 | 0.7983 |
| penalty_only_P | adult | non-IID | Sp-DFA | 82.20 | 0.0120 | 0.0419 | 0.0269 | 0.7951 |
| reward_only_R | adult | IID | Benign | 83.70 | 0.0067 | 0.0652 | 0.0359 | 0.8011 |
| reward_only_R | adult | IID | F Flip | 82.27 | 0.0133 | 0.0598 | 0.0366 | 0.7861 |
| reward_only_R | adult | IID | FedSA | 81.97 | 0.0209 | 0.0616 | 0.0413 | 0.7784 |
| reward_only_R | adult | IID | S-DFA | 81.96 | 0.0024 | 0.0621 | 0.0323 | 0.7873 |
| reward_only_R | adult | IID | Sp-DFA | 82.32 | 0.0012 | 0.0668 | 0.0340 | 0.7892 |
| reward_only_R | adult | non-IID | Benign | 82.74 | 0.0099 | 0.0604 | 0.0352 | 0.7923 |
| reward_only_R | adult | non-IID | F Flip | 83.19 | 0.0235 | 0.0389 | 0.0312 | 0.8007 |
| reward_only_R | adult | non-IID | FedSA | 82.95 | 0.0029 | 0.0622 | 0.0326 | 0.7969 |
| reward_only_R | adult | non-IID | S-DFA | 82.81 | 0.0153 | 0.0563 | 0.0358 | 0.7923 |
| reward_only_R | adult | non-IID | Sp-DFA | 83.68 | 0.0008 | 0.0641 | 0.0325 | 0.8044 |
| utility_only_U | adult | IID | Benign | 83.74 | 0.0057 | 0.0635 | 0.0346 | 0.8028 |
| utility_only_U | adult | IID | F Flip | 82.32 | 0.0085 | 0.0620 | 0.0352 | 0.7879 |
| utility_only_U | adult | IID | FedSA | 83.66 | 0.0052 | 0.0646 | 0.0349 | 0.8017 |
| utility_only_U | adult | IID | S-DFA | 82.21 | 0.0017 | 0.0655 | 0.0336 | 0.7885 |
| utility_only_U | adult | IID | Sp-DFA | 82.22 | 0.0000 | 0.0652 | 0.0326 | 0.7896 |
| utility_only_U | adult | non-IID | Benign | 82.73 | 0.0070 | 0.0640 | 0.0355 | 0.7918 |
| utility_only_U | adult | non-IID | F Flip | 82.29 | 0.0187 | 0.0590 | 0.0388 | 0.7841 |
| utility_only_U | adult | non-IID | FedSA | 82.06 | 0.0155 | 0.0637 | 0.0396 | 0.7810 |
| utility_only_U | adult | non-IID | S-DFA | 82.65 | 0.0067 | 0.0595 | 0.0331 | 0.7934 |
| utility_only_U | adult | non-IID | Sp-DFA | 82.27 | 0.0060 | 0.0643 | 0.0352 | 0.7875 |
