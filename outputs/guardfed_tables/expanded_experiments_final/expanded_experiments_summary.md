# Expanded GuardFed Advisor Experiments Summary

All rows are derived from results/paper_tables/raw_results.jsonl. ACC=max over last 10 rounds; AEOD/ASPD=min over last 10 rounds.

- expanded_server_dist30 rows: 720
- expanded_synthetic_ratios rows: 840
- fedsa_all_methods rows: 264

## Server Distribution Coverage

| dataset   | distribution   | attack   |   num_alphas |
|:----------|:---------------|:---------|-------------:|
| adult     | IID            | Benign   |           30 |
| adult     | IID            | FedSA    |           30 |
| adult     | non-IID        | Benign   |           30 |
| adult     | non-IID        | FedSA    |           30 |
| compas    | IID            | Benign   |           30 |
| compas    | IID            | FedSA    |           30 |
| compas    | non-IID        | Benign   |           30 |
| compas    | non-IID        | FedSA    |           30 |

## Synthetic Dataset-Level Top 12

| dataset   |   score_rank_in_dataset | setting                       |   acc_mean |   aeod_mean |   aspd_mean |   score_mean |   n |
|:----------|------------------------:|:------------------------------|-----------:|------------:|------------:|-------------:|----:|
| adult     |                       1 | real1_gaussian_copula_synth9  |     0.8139 |      0.0037 |      0.0019 |       0.8112 |  12 |
| adult     |                       2 | real3_tvae_synth7             |     0.8128 |      0.0027 |      0.0011 |       0.8109 |  12 |
| adult     |                       3 | real3_gaussian_copula_synth7  |     0.8245 |      0.0089 |      0.0200 |       0.8101 |  12 |
| adult     |                       4 | real2_gaussian_copula_synth8  |     0.8164 |      0.0062 |      0.0120 |       0.8073 |  12 |
| adult     |                       5 | real7_none                    |     0.8379 |      0.0158 |      0.0496 |       0.8052 |  12 |
| adult     |                       6 | real9_forest_diffusion_synth1 |     0.8329 |      0.0059 |      0.0504 |       0.8048 |  12 |
| adult     |                       7 | real9_gaussian_copula_synth1  |     0.8326 |      0.0152 |      0.0416 |       0.8042 |  12 |
| adult     |                       8 | real10_none                   |     0.8329 |      0.0078 |      0.0496 |       0.8042 |  12 |
| adult     |                       9 | real9_tvae_synth1             |     0.8300 |      0.0033 |      0.0501 |       0.8033 |  12 |
| adult     |                      10 | real2_forest_diffusion_synth8 |     0.8284 |      0.0063 |      0.0455 |       0.8025 |  12 |
| adult     |                      11 | real7_forest_diffusion_synth3 |     0.8343 |      0.0068 |      0.0577 |       0.8021 |  12 |
| adult     |                      12 | real3_forest_diffusion_synth7 |     0.8304 |      0.0102 |      0.0485 |       0.8010 |  12 |
| compas    |                       1 | real3_gaussian_copula_synth7  |     0.6636 |      0.0074 |      0.0158 |       0.6520 |  12 |
| compas    |                       2 | real2_forest_diffusion_synth8 |     0.6653 |      0.0107 |      0.0216 |       0.6491 |  12 |
| compas    |                       3 | real7_gaussian_copula_synth3  |     0.6641 |      0.0340 |      0.0051 |       0.6445 |  12 |
| compas    |                       4 | real5_smote_synth5            |     0.6542 |      0.0143 |      0.0052 |       0.6444 |  12 |
| compas    |                       5 | real7_forest_diffusion_synth3 |     0.6631 |      0.0293 |      0.0142 |       0.6414 |  12 |
| compas    |                       6 | real7_smote_synth3            |     0.6607 |      0.0334 |      0.0108 |       0.6386 |  12 |
| compas    |                       7 | real9_tvae_synth1             |     0.6611 |      0.0357 |      0.0096 |       0.6385 |  12 |
| compas    |                       8 | real3_forest_diffusion_synth7 |     0.6662 |      0.0174 |      0.0421 |       0.6364 |  12 |
| compas    |                       9 | real5_forest_diffusion_synth5 |     0.6623 |      0.0351 |      0.0220 |       0.6337 |  12 |
| compas    |                      10 | real1_gaussian_copula_synth9  |     0.6652 |      0.0209 |      0.0486 |       0.6304 |  12 |
| compas    |                      11 | real7_tvae_synth3             |     0.6459 |      0.0234 |      0.0082 |       0.6302 |  12 |
| compas    |                      12 | real5_none                    |     0.6597 |      0.0449 |      0.0158 |       0.6294 |  12 |

## FedSA All-Methods Top 8 Per Dataset/Distribution

| dataset   | distribution   |   score_rank | method            |   acc_mean |   aeod_mean |   aspd_mean |   score_mean |   n |
|:----------|:---------------|-------------:|:------------------|-----------:|------------:|------------:|-------------:|----:|
| adult     | IID            |            1 | GuardFed-AD2      |     0.8335 |      0.0018 |      0.0567 |       0.8043 |   3 |
| adult     | IID            |            1 | GuardFed-AD2+     |     0.8335 |      0.0018 |      0.0567 |       0.8043 |   3 |
| adult     | IID            |            3 | FLTrust+FairGuard |     0.8193 |      0.0081 |      0.0506 |       0.7899 |   3 |
| adult     | IID            |            4 | LoGoFair          |     0.8100 |      0.0123 |      0.0290 |       0.7893 |   3 |
| adult     | IID            |            5 | GuardFed          |     0.8359 |      0.0049 |      0.1009 |       0.7830 |   3 |
| adult     | IID            |            6 | FLTG              |     0.8345 |      0.0075 |      0.0981 |       0.7817 |   3 |
| adult     | IID            |            7 | FLTrust           |     0.8343 |      0.0157 |      0.1007 |       0.7761 |   3 |
| adult     | IID            |            8 | SmartFL           |     0.8326 |      0.0526 |      0.0623 |       0.7752 |   3 |
| adult     | non-IID        |            1 | GuardFed-AD2      |     0.8322 |      0.0074 |      0.0451 |       0.8060 |   3 |
| adult     | non-IID        |            1 | GuardFed-AD2+     |     0.8322 |      0.0074 |      0.0451 |       0.8060 |   3 |
| adult     | non-IID        |            3 | FLTrust+FairGuard |     0.8379 |      0.0018 |      0.0724 |       0.8008 |   3 |
| adult     | non-IID        |            4 | GuardFed          |     0.8357 |      0.0011 |      0.0746 |       0.7979 |   3 |
| adult     | non-IID        |            5 | LoGoFair          |     0.8277 |      0.0184 |      0.0495 |       0.7937 |   3 |
| adult     | non-IID        |            6 | FLTrust           |     0.8395 |      0.0042 |      0.1042 |       0.7853 |   3 |
| adult     | non-IID        |            7 | Fed-NGA           |     0.8289 |      0.0629 |      0.0280 |       0.7834 |   3 |
| adult     | non-IID        |            8 | FedAA             |     0.8366 |      0.0095 |      0.1096 |       0.7770 |   3 |
| compas    | IID            |            1 | LoGoFair          |     0.6602 |      0.0430 |      0.0390 |       0.6192 |   3 |
| compas    | IID            |            2 | GuardFed-AD2      |     0.6550 |      0.0835 |      0.0445 |       0.5910 |   3 |
| compas    | IID            |            2 | GuardFed-AD2+     |     0.6550 |      0.0835 |      0.0445 |       0.5910 |   3 |
| compas    | IID            |            4 | FLTrust+FairGuard |     0.6364 |      0.1362 |      0.1530 |       0.4918 |   3 |
| compas    | IID            |            5 | FairGuard         |     0.5373 |      0.0669 |      0.0306 |       0.4885 |   3 |
| compas    | IID            |            6 | SmartFL           |     0.6679 |      0.1956 |      0.1998 |       0.4702 |   3 |
| compas    | IID            |            7 | Fed-NGA           |     0.6713 |      0.2046 |      0.2119 |       0.4631 |   3 |
| compas    | IID            |            8 | LayerGuard        |     0.6658 |      0.2023 |      0.2042 |       0.4625 |   3 |
| compas    | non-IID        |            1 | LoGoFair          |     0.6604 |      0.0305 |      0.0231 |       0.6336 |   3 |
| compas    | non-IID        |            2 | GuardFed-AD2      |     0.6544 |      0.0724 |      0.0385 |       0.5990 |   3 |
| compas    | non-IID        |            2 | GuardFed-AD2+     |     0.6544 |      0.0724 |      0.0385 |       0.5990 |   3 |
| compas    | non-IID        |            4 | FLTrust+FairGuard |     0.6344 |      0.1175 |      0.1225 |       0.5144 |   3 |
| compas    | non-IID        |            5 | FairGuard         |     0.5391 |      0.0444 |      0.0219 |       0.5060 |   3 |
| compas    | non-IID        |            6 | Fed-NGA           |     0.6762 |      0.1954 |      0.2108 |       0.4731 |   3 |
| compas    | non-IID        |            7 | SmartFL           |     0.6690 |      0.1956 |      0.2098 |       0.4663 |   3 |
| compas    | non-IID        |            8 | FLAURA            |     0.6746 |      0.2064 |      0.2210 |       0.4609 |   3 |
