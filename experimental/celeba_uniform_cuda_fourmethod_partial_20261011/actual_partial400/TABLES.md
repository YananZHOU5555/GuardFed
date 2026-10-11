# partial four-method uniformly-CUDA validation

Partial four-method uniformly-CUDA validation; not the final17-method comparison or a newly selected manuscript endpoint. Native predictions only; ACC in %, AEOD/ASPD on [0,1]. AEOD is the implemented absolute TPR gap, not full equalized odds. Mean ± sample SD(ddof1). Original70-round CNN checkpoints are reused; CNN/root+valid forward evaluation uses CUDA FP32, while each frozen method retains its original postprocessing implementation/device. LoGoFair-DP applies its unchanged fitted recipe07/fitseed1719 state to adopted FedAvg CUDA margins, with netcal1.3.6 CPU postprocessing and20 image-ID virtual cohorts, not real training clients; DP means demographic parity. No refit or new CNN is performed for LoGo. Original cu128/cu130 training histories and adaptation boundaries remain; CUDA evaluation is not a claim of training-runtime equivalence. Seed91001 participated in validation selection; n9/n6 are sensitivity subsets, not untouched test cohorts. All fixed seeds, negative outcomes and constant predictors are retained. No score, best-seed selection, significance or final-test claim. Each native pipeline includes its own postprocessing and does not isolate aggregation causality.

## IID / ten

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 84.72 ± 11.74 | 89.57 ± 0.78 | 68.19 ± 5.77 | 66.01 ± 3.57 | 84.37 ± 2.38 |
| FedAvg | AEOD ↓ | 0.0384 ± 0.0176 | 0.0519 ± 0.0096 | 0.1004 ± 0.0410 | 0.0950 ± 0.0477 | 0.0280 ± 0.0076 |
| FedAvg | ASPD ↓ | 0.0937 ± 0.0366 | 0.1212 ± 0.0064 | 0.0993 ± 0.0380 | 0.0807 ± 0.0413 | 0.0693 ± 0.0161 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.26 ± 1.04 | 88.39 ± 0.63 | 88.47 ± 0.91 | 88.69 ± 0.81 | 88.38 ± 0.89 |
| GuardFed-AD2+ | AEOD ↓ | 0.0097 ± 0.0075 | 0.0107 ± 0.0096 | 0.0061 ± 0.0042 | 0.0077 ± 0.0046 | 0.0165 ± 0.0072 |
| GuardFed-AD2+ | ASPD ↓ | 0.0625 ± 0.0138 | 0.0607 ± 0.0103 | 0.0640 ± 0.0142 | 0.0638 ± 0.0070 | 0.0559 ± 0.0156 |
| FLGMM | ACC (%) ↑ | 89.33 ± 0.99 | 89.72 ± 0.83 | 89.34 ± 1.01 | 89.34 ± 1.01 | 89.49 ± 0.93 |
| FLGMM | AEOD ↓ | 0.0404 ± 0.0065 | 0.0458 ± 0.0078 | 0.0424 ± 0.0070 | 0.0423 ± 0.0075 | 0.0460 ± 0.0061 |
| FLGMM | ASPD ↓ | 0.1074 ± 0.0073 | 0.1171 ± 0.0049 | 0.1113 ± 0.0081 | 0.1113 ± 0.0080 | 0.1143 ± 0.0063 |
| LoGoFair-DP | ACC (%) ↑ | 83.64 ± 11.34 | 88.03 ± 0.79 | 69.18 ± 5.69 | 66.76 ± 3.39 | 83.61 ± 2.26 |
| LoGoFair-DP | AEOD ↓ | 0.0453 ± 0.0189 | 0.0564 ± 0.0077 | 0.0161 ± 0.0115 | 0.0129 ± 0.0106 | 0.0284 ± 0.0115 |
| LoGoFair-DP | ASPD ↓ | 0.0071 ± 0.0059 | 0.0079 ± 0.0040 | 0.0103 ± 0.0071 | 0.0100 ± 0.0055 | 0.0077 ± 0.0061 |

## non-IID / ten

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 86.85 ± 5.45 | 89.51 ± 1.82 | 74.76 ± 13.38 | 74.51 ± 13.37 | 85.48 ± 2.58 |
| FedAvg | AEOD ↓ | 0.0428 ± 0.0137 | 0.0492 ± 0.0045 | 0.0644 ± 0.1097 | 0.0522 ± 0.0913 | 0.0267 ± 0.0157 |
| FedAvg | ASPD ↓ | 0.0958 ± 0.0353 | 0.1161 ± 0.0143 | 0.0614 ± 0.0893 | 0.0591 ± 0.0756 | 0.0741 ± 0.0247 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.59 ± 1.17 | 88.44 ± 0.74 | 88.41 ± 0.92 | 88.22 ± 0.78 | 88.34 ± 0.96 |
| GuardFed-AD2+ | AEOD ↓ | 0.0070 ± 0.0042 | 0.0100 ± 0.0070 | 0.0078 ± 0.0056 | 0.0126 ± 0.0093 | 0.0087 ± 0.0066 |
| GuardFed-AD2+ | ASPD ↓ | 0.0651 ± 0.0090 | 0.0589 ± 0.0092 | 0.0616 ± 0.0141 | 0.0584 ± 0.0189 | 0.0596 ± 0.0146 |
| FLGMM | ACC (%) ↑ | 89.02 ± 0.91 | 89.28 ± 0.98 | 89.15 ± 0.88 | 89.17 ± 1.15 | 89.64 ± 0.72 |
| FLGMM | AEOD ↓ | 0.0396 ± 0.0066 | 0.0525 ± 0.0077 | 0.0428 ± 0.0049 | 0.0423 ± 0.0048 | 0.0446 ± 0.0084 |
| FLGMM | ASPD ↓ | 0.1071 ± 0.0078 | 0.1185 ± 0.0068 | 0.1088 ± 0.0055 | 0.1083 ± 0.0105 | 0.1147 ± 0.0053 |
| LoGoFair-DP | ACC (%) ↑ | 85.64 ± 4.92 | 87.96 ± 1.66 | 74.94 ± 12.90 | 74.53 ± 12.82 | 84.53 ± 2.27 |
| LoGoFair-DP | AEOD ↓ | 0.0445 ± 0.0232 | 0.0535 ± 0.0147 | 0.0192 ± 0.0147 | 0.0196 ± 0.0178 | 0.0358 ± 0.0122 |
| LoGoFair-DP | ASPD ↓ | 0.0055 ± 0.0051 | 0.0060 ± 0.0036 | 0.0084 ± 0.0054 | 0.0090 ± 0.0061 | 0.0051 ± 0.0045 |

## IID / nonselection_nine

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 88.39 ± 1.84 | 89.72 ± 0.65 | 69.08 ± 5.35 | 66.61 ± 3.21 | 84.91 ± 1.77 |
| FedAvg | AEOD ↓ | 0.0426 ± 0.0119 | 0.0522 ± 0.0101 | 0.0939 ± 0.0376 | 0.0904 ± 0.0481 | 0.0289 ± 0.0075 |
| FedAvg | ASPD ↓ | 0.1041 ± 0.0171 | 0.1225 ± 0.0053 | 0.0962 ± 0.0389 | 0.0781 ± 0.0429 | 0.0702 ± 0.0168 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.35 ± 1.06 | 88.41 ± 0.66 | 88.62 ± 0.82 | 88.84 ± 0.68 | 88.55 ± 0.77 |
| GuardFed-AD2+ | AEOD ↓ | 0.0097 ± 0.0080 | 0.0115 ± 0.0098 | 0.0054 ± 0.0039 | 0.0068 ± 0.0038 | 0.0171 ± 0.0074 |
| GuardFed-AD2+ | ASPD ↓ | 0.0623 ± 0.0146 | 0.0604 ± 0.0108 | 0.0667 ± 0.0120 | 0.0633 ± 0.0072 | 0.0555 ± 0.0165 |
| FLGMM | ACC (%) ↑ | 89.40 ± 1.02 | 89.81 ± 0.84 | 89.47 ± 0.98 | 89.48 ± 0.97 | 89.63 ± 0.87 |
| FLGMM | AEOD ↓ | 0.0401 ± 0.0069 | 0.0454 ± 0.0082 | 0.0424 ± 0.0074 | 0.0423 ± 0.0080 | 0.0465 ± 0.0062 |
| FLGMM | ASPD ↓ | 0.1081 ± 0.0074 | 0.1175 ± 0.0050 | 0.1120 ± 0.0083 | 0.1120 ± 0.0082 | 0.1153 ± 0.0058 |
| LoGoFair-DP | ACC (%) ↑ | 87.19 ± 1.67 | 88.20 ± 0.61 | 70.06 ± 5.28 | 67.33 ± 3.06 | 84.12 ± 1.66 |
| LoGoFair-DP | AEOD ↓ | 0.0503 ± 0.0107 | 0.0563 ± 0.0082 | 0.0151 ± 0.0118 | 0.0122 ± 0.0110 | 0.0278 ± 0.0121 |
| LoGoFair-DP | ASPD ↓ | 0.0079 ± 0.0057 | 0.0077 ± 0.0041 | 0.0109 ± 0.0073 | 0.0106 ± 0.0054 | 0.0079 ± 0.0064 |

## non-IID / nonselection_nine

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 88.09 ± 4.01 | 89.91 ± 1.39 | 75.58 ± 13.92 | 75.31 ± 13.93 | 85.69 ± 2.65 |
| FedAvg | AEOD ↓ | 0.0457 ± 0.0107 | 0.0485 ± 0.0043 | 0.0560 ± 0.1128 | 0.0510 ± 0.0968 | 0.0273 ± 0.0165 |
| FedAvg | ASPD ↓ | 0.1050 ± 0.0215 | 0.1190 ± 0.0117 | 0.0552 ± 0.0923 | 0.0603 ± 0.0801 | 0.0748 ± 0.0260 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.59 ± 1.24 | 88.50 ± 0.76 | 88.39 ± 0.97 | 88.24 ± 0.82 | 88.47 ± 0.94 |
| GuardFed-AD2+ | AEOD ↓ | 0.0063 ± 0.0039 | 0.0104 ± 0.0073 | 0.0083 ± 0.0057 | 0.0119 ± 0.0095 | 0.0079 ± 0.0065 |
| GuardFed-AD2+ | ASPD ↓ | 0.0638 ± 0.0086 | 0.0579 ± 0.0090 | 0.0609 ± 0.0148 | 0.0605 ± 0.0188 | 0.0613 ± 0.0144 |
| FLGMM | ACC (%) ↑ | 89.18 ± 0.81 | 89.40 ± 0.95 | 89.38 ± 0.56 | 89.40 ± 0.96 | 89.77 ± 0.62 |
| FLGMM | AEOD ↓ | 0.0392 ± 0.0068 | 0.0514 ± 0.0072 | 0.0422 ± 0.0047 | 0.0417 ± 0.0048 | 0.0434 ± 0.0079 |
| FLGMM | ASPD ↓ | 0.1080 ± 0.0077 | 0.1185 ± 0.0072 | 0.1099 ± 0.0044 | 0.1094 ± 0.0105 | 0.1147 ± 0.0056 |
| LoGoFair-DP | ACC (%) ↑ | 86.76 ± 3.64 | 88.32 ± 1.28 | 75.65 ± 13.47 | 75.26 ± 13.37 | 84.71 ± 2.33 |
| LoGoFair-DP | AEOD ↓ | 0.0488 ± 0.0200 | 0.0567 ± 0.0113 | 0.0202 ± 0.0152 | 0.0218 ± 0.0174 | 0.0356 ± 0.0130 |
| LoGoFair-DP | ASPD ↓ | 0.0053 ± 0.0054 | 0.0060 ± 0.0038 | 0.0077 ± 0.0052 | 0.0082 ± 0.0059 | 0.0044 ± 0.0041 |

## IID / matching_six

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 87.99 ± 2.11 | 89.71 ± 0.80 | 69.36 ± 6.14 | 67.41 ± 3.17 | 85.70 ± 1.38 |
| FedAvg | AEOD ↓ | 0.0406 ± 0.0131 | 0.0535 ± 0.0119 | 0.0852 ± 0.0377 | 0.0823 ± 0.0434 | 0.0280 ± 0.0074 |
| FedAvg | ASPD ↓ | 0.0992 ± 0.0184 | 0.1227 ± 0.0063 | 0.0866 ± 0.0413 | 0.0698 ± 0.0304 | 0.0738 ± 0.0142 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.56 ± 0.68 | 88.29 ± 0.75 | 88.57 ± 0.87 | 88.75 ± 0.79 | 88.35 ± 0.87 |
| GuardFed-AD2+ | AEOD ↓ | 0.0114 ± 0.0073 | 0.0148 ± 0.0104 | 0.0055 ± 0.0036 | 0.0070 ± 0.0040 | 0.0171 ± 0.0083 |
| GuardFed-AD2+ | ASPD ↓ | 0.0665 ± 0.0159 | 0.0581 ± 0.0129 | 0.0697 ± 0.0110 | 0.0617 ± 0.0085 | 0.0559 ± 0.0203 |
| FLGMM | ACC (%) ↑ | 89.12 ± 1.17 | 89.66 ± 1.00 | 89.10 ± 1.01 | 89.12 ± 1.01 | 89.32 ± 0.92 |
| FLGMM | AEOD ↓ | 0.0411 ± 0.0082 | 0.0436 ± 0.0093 | 0.0431 ± 0.0092 | 0.0429 ± 0.0098 | 0.0460 ± 0.0073 |
| FLGMM | ASPD ↓ | 0.1064 ± 0.0083 | 0.1162 ± 0.0044 | 0.1091 ± 0.0089 | 0.1091 ± 0.0088 | 0.1121 ± 0.0044 |
| LoGoFair-DP | ACC (%) ↑ | 86.82 ± 1.92 | 88.22 ± 0.75 | 70.21 ± 6.01 | 68.05 ± 3.01 | 84.88 ± 1.23 |
| LoGoFair-DP | AEOD ↓ | 0.0481 ± 0.0120 | 0.0541 ± 0.0086 | 0.0152 ± 0.0121 | 0.0133 ± 0.0136 | 0.0281 ± 0.0120 |
| LoGoFair-DP | ASPD ↓ | 0.0076 ± 0.0060 | 0.0099 ± 0.0030 | 0.0099 ± 0.0081 | 0.0101 ± 0.0041 | 0.0096 ± 0.0068 |

## non-IID / matching_six

| Method | Metric | Benign | F Flip | FedSA | S-DFA | Sp-DFA |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | ACC (%) ↑ | 86.90 ± 4.54 | 89.58 ± 1.62 | 78.23 ± 12.19 | 78.69 ± 12.03 | 86.13 ± 3.06 |
| FedAvg | AEOD ↓ | 0.0464 ± 0.0125 | 0.0479 ± 0.0036 | 0.0756 ± 0.1376 | 0.0652 ± 0.1170 | 0.0312 ± 0.0166 |
| FedAvg | ASPD ↓ | 0.0977 ± 0.0234 | 0.1151 ± 0.0128 | 0.0788 ± 0.1078 | 0.0804 ± 0.0927 | 0.0817 ± 0.0283 |
| GuardFed-AD2+ | ACC (%) ↑ | 88.58 ± 1.42 | 88.43 ± 0.87 | 88.49 ± 0.88 | 88.42 ± 0.83 | 88.42 ± 1.10 |
| GuardFed-AD2+ | AEOD ↓ | 0.0062 ± 0.0039 | 0.0110 ± 0.0086 | 0.0083 ± 0.0044 | 0.0067 ± 0.0068 | 0.0089 ± 0.0080 |
| GuardFed-AD2+ | ASPD ↓ | 0.0641 ± 0.0067 | 0.0570 ± 0.0090 | 0.0566 ± 0.0081 | 0.0611 ± 0.0114 | 0.0583 ± 0.0172 |
| FLGMM | ACC (%) ↑ | 89.05 ± 0.99 | 89.33 ± 1.13 | 89.34 ± 0.65 | 89.63 ± 0.67 | 89.64 ± 0.73 |
| FLGMM | AEOD ↓ | 0.0398 ± 0.0039 | 0.0508 ± 0.0055 | 0.0403 ± 0.0047 | 0.0418 ± 0.0027 | 0.0444 ± 0.0087 |
| FLGMM | ASPD ↓ | 0.1067 ± 0.0079 | 0.1169 ± 0.0076 | 0.1081 ± 0.0020 | 0.1112 ± 0.0070 | 0.1143 ± 0.0062 |
| LoGoFair-DP | ACC (%) ↑ | 85.68 ± 4.12 | 88.02 ± 1.50 | 78.31 ± 11.31 | 78.54 ± 11.02 | 85.07 ± 2.69 |
| LoGoFair-DP | AEOD ↓ | 0.0411 ± 0.0201 | 0.0547 ± 0.0129 | 0.0263 ± 0.0141 | 0.0306 ± 0.0141 | 0.0344 ± 0.0159 |
| LoGoFair-DP | ASPD ↓ | 0.0061 ± 0.0063 | 0.0066 ± 0.0046 | 0.0077 ± 0.0061 | 0.0085 ± 0.0065 | 0.0056 ± 0.0045 |

# Seed-first descriptive aggregates

Each model seed is averaged across its five scenes per distribution or all ten balanced scenes, before cross-seed mean/SD; scenes are not independent seeds.

| Panel | Scope | Method | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| ten | IID | FedAvg | 10 | 78.57 ± 3.89 | 0.0627 ± 0.0179 | 0.0928 ± 0.0168 |
| ten | non-IID | FedAvg | 10 | 82.22 ± 5.79 | 0.0471 ± 0.0377 | 0.0813 ± 0.0313 |
| ten | balanced_all10 | FedAvg | 10 | 80.40 ± 4.01 | 0.0549 ± 0.0231 | 0.0871 ± 0.0193 |
| ten | IID | GuardFed-AD2+ | 10 | 88.44 ± 0.67 | 0.0101 ± 0.0032 | 0.0614 ± 0.0073 |
| ten | non-IID | GuardFed-AD2+ | 10 | 88.40 ± 0.71 | 0.0092 ± 0.0037 | 0.0607 ± 0.0056 |
| ten | balanced_all10 | GuardFed-AD2+ | 10 | 88.42 ± 0.62 | 0.0097 ± 0.0030 | 0.0610 ± 0.0047 |
| ten | IID | FLGMM | 10 | 89.45 ± 0.88 | 0.0434 ± 0.0052 | 0.1123 ± 0.0058 |
| ten | non-IID | FLGMM | 10 | 89.25 ± 0.82 | 0.0444 ± 0.0053 | 0.1115 ± 0.0060 |
| ten | balanced_all10 | FLGMM | 10 | 89.35 ± 0.78 | 0.0439 ± 0.0046 | 0.1119 ± 0.0047 |
| ten | IID | LoGoFair-DP | 10 | 78.24 ± 3.78 | 0.0318 ± 0.0045 | 0.0086 ± 0.0038 |
| ten | non-IID | LoGoFair-DP | 10 | 81.52 ± 5.50 | 0.0345 ± 0.0115 | 0.0068 ± 0.0040 |
| ten | balanced_all10 | LoGoFair-DP | 10 | 79.88 ± 3.80 | 0.0332 ± 0.0071 | 0.0077 ± 0.0035 |
| nonselection_nine | IID | FedAvg | 9 | 79.74 ± 1.29 | 0.0616 ± 0.0186 | 0.0942 ± 0.0173 |
| nonselection_nine | non-IID | FedAvg | 9 | 82.92 ± 5.69 | 0.0457 ± 0.0397 | 0.0829 ± 0.0328 |
| nonselection_nine | balanced_all10 | FedAvg | 9 | 81.33 ± 2.88 | 0.0537 ± 0.0241 | 0.0885 ± 0.0199 |
| nonselection_nine | IID | GuardFed-AD2+ | 9 | 88.56 ± 0.59 | 0.0101 ± 0.0034 | 0.0617 ± 0.0077 |
| nonselection_nine | non-IID | GuardFed-AD2+ | 9 | 88.44 ± 0.74 | 0.0090 ± 0.0038 | 0.0609 ± 0.0059 |
| nonselection_nine | balanced_all10 | GuardFed-AD2+ | 9 | 88.50 ± 0.61 | 0.0095 ± 0.0031 | 0.0613 ± 0.0049 |
| nonselection_nine | IID | FLGMM | 9 | 89.56 ± 0.85 | 0.0433 ± 0.0055 | 0.1130 ± 0.0057 |
| nonselection_nine | non-IID | FLGMM | 9 | 89.42 ± 0.65 | 0.0436 ± 0.0049 | 0.1121 ± 0.0060 |
| nonselection_nine | balanced_all10 | FLGMM | 9 | 89.49 ± 0.68 | 0.0435 ± 0.0047 | 0.1125 ± 0.0045 |
| nonselection_nine | IID | LoGoFair-DP | 9 | 79.38 ± 1.27 | 0.0324 ± 0.0045 | 0.0090 ± 0.0038 |
| nonselection_nine | non-IID | LoGoFair-DP | 9 | 82.14 ± 5.45 | 0.0366 ± 0.0099 | 0.0063 ± 0.0039 |
| nonselection_nine | balanced_all10 | LoGoFair-DP | 9 | 80.76 ± 2.75 | 0.0345 ± 0.0062 | 0.0077 ± 0.0037 |
| matching_six | IID | FedAvg | 6 | 80.03 ± 1.49 | 0.0579 ± 0.0167 | 0.0904 ± 0.0130 |
| matching_six | non-IID | FedAvg | 6 | 83.91 ± 5.21 | 0.0533 ± 0.0481 | 0.0907 ± 0.0387 |
| matching_six | balanced_all10 | FedAvg | 6 | 81.97 ± 2.42 | 0.0556 ± 0.0290 | 0.0906 ± 0.0234 |
| matching_six | IID | GuardFed-AD2+ | 6 | 88.50 ± 0.63 | 0.0112 ± 0.0037 | 0.0624 ± 0.0094 |
| matching_six | non-IID | GuardFed-AD2+ | 6 | 88.47 ± 0.90 | 0.0082 ± 0.0043 | 0.0594 ± 0.0052 |
| matching_six | balanced_all10 | GuardFed-AD2+ | 6 | 88.49 ± 0.75 | 0.0097 ± 0.0037 | 0.0609 ± 0.0055 |
| matching_six | IID | FLGMM | 6 | 89.26 ± 0.91 | 0.0433 ± 0.0069 | 0.1106 ± 0.0056 |
| matching_six | non-IID | FLGMM | 6 | 89.40 ± 0.79 | 0.0434 ± 0.0038 | 0.1114 ± 0.0052 |
| matching_six | balanced_all10 | FLGMM | 6 | 89.33 ± 0.79 | 0.0434 ± 0.0052 | 0.1110 ± 0.0043 |
| matching_six | IID | LoGoFair-DP | 6 | 79.64 ± 1.50 | 0.0318 ± 0.0039 | 0.0094 ± 0.0043 |
| matching_six | non-IID | LoGoFair-DP | 6 | 83.12 ± 4.74 | 0.0374 ± 0.0120 | 0.0069 ± 0.0047 |
| matching_six | balanced_all10 | LoGoFair-DP | 6 | 81.38 ± 2.14 | 0.0346 ± 0.0070 | 0.0082 ± 0.0044 |

## Constant-prediction outcomes retained

| Panel | Method | Distribution | Scenario | Constant / n |
|---|---|---|---|---:|
| ten | FedAvg | IID | Benign | 1/10 |
| ten | FedAvg | IID | F Flip | 0/10 |
| ten | FedAvg | IID | FedSA | 0/10 |
| ten | FedAvg | IID | S-DFA | 0/10 |
| ten | FedAvg | IID | Sp-DFA | 0/10 |
| ten | FedAvg | non-IID | Benign | 0/10 |
| ten | FedAvg | non-IID | F Flip | 0/10 |
| ten | FedAvg | non-IID | FedSA | 0/10 |
| ten | FedAvg | non-IID | S-DFA | 0/10 |
| ten | FedAvg | non-IID | Sp-DFA | 0/10 |
| ten | GuardFed-AD2+ | IID | Benign | 0/10 |
| ten | GuardFed-AD2+ | IID | F Flip | 0/10 |
| ten | GuardFed-AD2+ | IID | FedSA | 0/10 |
| ten | GuardFed-AD2+ | IID | S-DFA | 0/10 |
| ten | GuardFed-AD2+ | IID | Sp-DFA | 0/10 |
| ten | GuardFed-AD2+ | non-IID | Benign | 0/10 |
| ten | GuardFed-AD2+ | non-IID | F Flip | 0/10 |
| ten | GuardFed-AD2+ | non-IID | FedSA | 0/10 |
| ten | GuardFed-AD2+ | non-IID | S-DFA | 0/10 |
| ten | GuardFed-AD2+ | non-IID | Sp-DFA | 0/10 |
| ten | FLGMM | IID | Benign | 0/10 |
| ten | FLGMM | IID | F Flip | 0/10 |
| ten | FLGMM | IID | FedSA | 0/10 |
| ten | FLGMM | IID | S-DFA | 0/10 |
| ten | FLGMM | IID | Sp-DFA | 0/10 |
| ten | FLGMM | non-IID | Benign | 0/10 |
| ten | FLGMM | non-IID | F Flip | 0/10 |
| ten | FLGMM | non-IID | FedSA | 0/10 |
| ten | FLGMM | non-IID | S-DFA | 0/10 |
| ten | FLGMM | non-IID | Sp-DFA | 0/10 |
| ten | LoGoFair-DP | IID | Benign | 1/10 |
| ten | LoGoFair-DP | IID | F Flip | 0/10 |
| ten | LoGoFair-DP | IID | FedSA | 0/10 |
| ten | LoGoFair-DP | IID | S-DFA | 0/10 |
| ten | LoGoFair-DP | IID | Sp-DFA | 0/10 |
| ten | LoGoFair-DP | non-IID | Benign | 0/10 |
| ten | LoGoFair-DP | non-IID | F Flip | 0/10 |
| ten | LoGoFair-DP | non-IID | FedSA | 0/10 |
| ten | LoGoFair-DP | non-IID | S-DFA | 0/10 |
| ten | LoGoFair-DP | non-IID | Sp-DFA | 0/10 |
| nonselection_nine | FedAvg | IID | Benign | 0/9 |
| nonselection_nine | FedAvg | IID | F Flip | 0/9 |
| nonselection_nine | FedAvg | IID | FedSA | 0/9 |
| nonselection_nine | FedAvg | IID | S-DFA | 0/9 |
| nonselection_nine | FedAvg | IID | Sp-DFA | 0/9 |
| nonselection_nine | FedAvg | non-IID | Benign | 0/9 |
| nonselection_nine | FedAvg | non-IID | F Flip | 0/9 |
| nonselection_nine | FedAvg | non-IID | FedSA | 0/9 |
| nonselection_nine | FedAvg | non-IID | S-DFA | 0/9 |
| nonselection_nine | FedAvg | non-IID | Sp-DFA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | IID | Benign | 0/9 |
| nonselection_nine | GuardFed-AD2+ | IID | F Flip | 0/9 |
| nonselection_nine | GuardFed-AD2+ | IID | FedSA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | IID | S-DFA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | IID | Sp-DFA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | non-IID | Benign | 0/9 |
| nonselection_nine | GuardFed-AD2+ | non-IID | F Flip | 0/9 |
| nonselection_nine | GuardFed-AD2+ | non-IID | FedSA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | non-IID | S-DFA | 0/9 |
| nonselection_nine | GuardFed-AD2+ | non-IID | Sp-DFA | 0/9 |
| nonselection_nine | FLGMM | IID | Benign | 0/9 |
| nonselection_nine | FLGMM | IID | F Flip | 0/9 |
| nonselection_nine | FLGMM | IID | FedSA | 0/9 |
| nonselection_nine | FLGMM | IID | S-DFA | 0/9 |
| nonselection_nine | FLGMM | IID | Sp-DFA | 0/9 |
| nonselection_nine | FLGMM | non-IID | Benign | 0/9 |
| nonselection_nine | FLGMM | non-IID | F Flip | 0/9 |
| nonselection_nine | FLGMM | non-IID | FedSA | 0/9 |
| nonselection_nine | FLGMM | non-IID | S-DFA | 0/9 |
| nonselection_nine | FLGMM | non-IID | Sp-DFA | 0/9 |
| nonselection_nine | LoGoFair-DP | IID | Benign | 0/9 |
| nonselection_nine | LoGoFair-DP | IID | F Flip | 0/9 |
| nonselection_nine | LoGoFair-DP | IID | FedSA | 0/9 |
| nonselection_nine | LoGoFair-DP | IID | S-DFA | 0/9 |
| nonselection_nine | LoGoFair-DP | IID | Sp-DFA | 0/9 |
| nonselection_nine | LoGoFair-DP | non-IID | Benign | 0/9 |
| nonselection_nine | LoGoFair-DP | non-IID | F Flip | 0/9 |
| nonselection_nine | LoGoFair-DP | non-IID | FedSA | 0/9 |
| nonselection_nine | LoGoFair-DP | non-IID | S-DFA | 0/9 |
| nonselection_nine | LoGoFair-DP | non-IID | Sp-DFA | 0/9 |
| matching_six | FedAvg | IID | Benign | 0/6 |
| matching_six | FedAvg | IID | F Flip | 0/6 |
| matching_six | FedAvg | IID | FedSA | 0/6 |
| matching_six | FedAvg | IID | S-DFA | 0/6 |
| matching_six | FedAvg | IID | Sp-DFA | 0/6 |
| matching_six | FedAvg | non-IID | Benign | 0/6 |
| matching_six | FedAvg | non-IID | F Flip | 0/6 |
| matching_six | FedAvg | non-IID | FedSA | 0/6 |
| matching_six | FedAvg | non-IID | S-DFA | 0/6 |
| matching_six | FedAvg | non-IID | Sp-DFA | 0/6 |
| matching_six | GuardFed-AD2+ | IID | Benign | 0/6 |
| matching_six | GuardFed-AD2+ | IID | F Flip | 0/6 |
| matching_six | GuardFed-AD2+ | IID | FedSA | 0/6 |
| matching_six | GuardFed-AD2+ | IID | S-DFA | 0/6 |
| matching_six | GuardFed-AD2+ | IID | Sp-DFA | 0/6 |
| matching_six | GuardFed-AD2+ | non-IID | Benign | 0/6 |
| matching_six | GuardFed-AD2+ | non-IID | F Flip | 0/6 |
| matching_six | GuardFed-AD2+ | non-IID | FedSA | 0/6 |
| matching_six | GuardFed-AD2+ | non-IID | S-DFA | 0/6 |
| matching_six | GuardFed-AD2+ | non-IID | Sp-DFA | 0/6 |
| matching_six | FLGMM | IID | Benign | 0/6 |
| matching_six | FLGMM | IID | F Flip | 0/6 |
| matching_six | FLGMM | IID | FedSA | 0/6 |
| matching_six | FLGMM | IID | S-DFA | 0/6 |
| matching_six | FLGMM | IID | Sp-DFA | 0/6 |
| matching_six | FLGMM | non-IID | Benign | 0/6 |
| matching_six | FLGMM | non-IID | F Flip | 0/6 |
| matching_six | FLGMM | non-IID | FedSA | 0/6 |
| matching_six | FLGMM | non-IID | S-DFA | 0/6 |
| matching_six | FLGMM | non-IID | Sp-DFA | 0/6 |
| matching_six | LoGoFair-DP | IID | Benign | 0/6 |
| matching_six | LoGoFair-DP | IID | F Flip | 0/6 |
| matching_six | LoGoFair-DP | IID | FedSA | 0/6 |
| matching_six | LoGoFair-DP | IID | S-DFA | 0/6 |
| matching_six | LoGoFair-DP | IID | Sp-DFA | 0/6 |
| matching_six | LoGoFair-DP | non-IID | Benign | 0/6 |
| matching_six | LoGoFair-DP | non-IID | F Flip | 0/6 |
| matching_six | LoGoFair-DP | non-IID | FedSA | 0/6 |
| matching_six | LoGoFair-DP | non-IID | S-DFA | 0/6 |
| matching_six | LoGoFair-DP | non-IID | Sp-DFA | 0/6 |
