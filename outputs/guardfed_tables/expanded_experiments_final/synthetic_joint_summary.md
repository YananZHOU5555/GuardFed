# AD2+ server/root data synthetic-ratio summary (joint-last10)

Rows are ranked by joint score within each dataset. Each setting aggregates 12 runs: 2 distributions x 2 attacks x 3 seeds.

## adult
| Rank | Setting | ACC | AEOD | ASPD | Score | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 9% real + 1% tvae | 82.03 | 0.0050 | 0.0737 | 0.7809 | 12 |
| 2 | 10% real clean | 82.29 | 0.0091 | 0.0750 | 0.7809 | 12 |
| 3 | 7% real + 3% forest_diffusion | 82.54 | 0.0102 | 0.0802 | 0.7802 | 12 |
| 4 | 9% real + 1% forest_diffusion | 81.97 | 0.0065 | 0.0737 | 0.7796 | 12 |
| 5 | 2% real + 8% forest_diffusion | 81.89 | 0.0077 | 0.0712 | 0.7795 | 12 |
| 6 | 5% real clean | 82.86 | 0.0068 | 0.0926 | 0.7789 | 12 |
| 7 | 9% real + 1% gaussian_copula | 81.96 | 0.0157 | 0.0691 | 0.7772 | 12 |
| 8 | 5% real + 5% gaussian_copula | 82.05 | 0.0186 | 0.0687 | 0.7768 | 12 |
| 9 | 2% real + 8% gaussian_copula | 80.58 | 0.0097 | 0.0494 | 0.7763 | 12 |
| 10 | 3% real + 7% forest_diffusion | 82.44 | 0.0118 | 0.0850 | 0.7760 | 12 |
| 11 | 7% real clean | 82.62 | 0.0178 | 0.0831 | 0.7757 | 12 |
| 12 | 9% real + 1% smote | 82.89 | 0.0142 | 0.0923 | 0.7756 | 12 |
| 13 | 3% real clean | 82.84 | 0.0113 | 0.0945 | 0.7755 | 12 |
| 14 | 5% real + 5% tvae | 82.67 | 0.0104 | 0.0930 | 0.7750 | 12 |
| 15 | 1% real + 9% gaussian_copula | 80.76 | 0.0153 | 0.0504 | 0.7747 | 12 |

## compas
| Rank | Setting | ACC | AEOD | ASPD | Score | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7% real + 3% gaussian_copula | 66.21 | 0.0374 | 0.0149 | 0.6359 | 12 |
| 2 | 7% real + 3% forest_diffusion | 66.13 | 0.0349 | 0.0227 | 0.6324 | 12 |
| 3 | 3% real + 7% gaussian_copula | 65.50 | 0.0159 | 0.0302 | 0.6320 | 12 |
| 4 | 2% real + 8% forest_diffusion | 65.76 | 0.0140 | 0.0397 | 0.6307 | 12 |
| 5 | 5% real + 5% smote | 64.50 | 0.0169 | 0.0123 | 0.6304 | 12 |
| 6 | 7% real + 3% smote | 65.75 | 0.0359 | 0.0197 | 0.6297 | 12 |
| 7 | 9% real + 1% tvae | 65.88 | 0.0391 | 0.0203 | 0.6291 | 12 |
| 8 | 5% real + 5% forest_diffusion | 65.95 | 0.0383 | 0.0239 | 0.6284 | 12 |
| 9 | 3% real + 7% forest_diffusion | 66.01 | 0.0209 | 0.0494 | 0.6249 | 12 |
| 10 | 5% real clean | 65.47 | 0.0474 | 0.0181 | 0.6220 | 12 |
| 11 | 1% real + 9% gaussian_copula | 66.04 | 0.0243 | 0.0560 | 0.6203 | 12 |
| 12 | 7% real + 3% ctgan | 66.14 | 0.0285 | 0.0563 | 0.6190 | 12 |
| 13 | 7% real clean | 65.69 | 0.0514 | 0.0281 | 0.6172 | 12 |
| 14 | 5% real + 5% gaussian_copula | 65.57 | 0.0458 | 0.0325 | 0.6166 | 12 |
| 15 | 7% real + 3% tvae | 64.16 | 0.0359 | 0.0147 | 0.6163 | 12 |
