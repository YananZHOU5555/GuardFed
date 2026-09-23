# GuardFed-AD2+ Clean10 Best-Sweep Summary

## Main Result

- Protocol: 10% clean server/root data, 0% synthetic, seed=123, 20 clients, 4 malicious, 70 rounds.
- Ranking rule: ACC higher is better; AEOD/ASPD lower is better.
- Fairness ranking validity: Adult ACC >= 80%, COMPAS ACC >= 60%, and ACC within 5 percentage points of the scenario-best ACC.
- Value rule: ACC = max over last 10 rounds; AEOD/ASPD = min over last 10 rounds.
- Full experiment cells: 440/440; missing cells: 0; N/R values: 0.

## AD2+ Improvement

Original clean10 AD2+:

- first = 7
- second = 5
- top2 = 12/60

Best-sweep AD2+:

- first = 11
- second = 7
- top2 = 18/60
- ACC top2 = 3/20
- AEOD top2 = 6/20
- ASPD top2 = 9/20

Selected single global AD2+ configuration:

- mode = fixed
- clean server/root data = 10%
- fairness metric = AEOD+ASPD
- fairness budget = 0.12
- risk weight = 0.10
- violation weight = 0.02
- keep ratio = 1.0
- temperature = 0.8
- utility weight = 3.0
- centrality weight = 0.2
- alignment weight = 1.0
- score clip = 5
- norm mode = root

This is not a per-cell oracle. The same selected AD2+ configuration is used for all datasets, distributions, attacks, and metrics.

## Paper-Value Consistency Note

The original GuardFed paper states that the server/root data is 1% clean original training data plus 4% Gaussian Copula synthetic data, totaling 5% root data. The current table uses the user-requested 10% clean root data and 0% synthetic data. Therefore the clean10 table is a truthful unified experiment table, but it should not be expected to exactly match the published Table II/III values.

For a paper-aligned reproduction table, run a separate protocol with `server_ratio=0.01` and `synthetic_ratio=0.04`, and keep it clearly labeled as paper-protocol reproduction. Do not mix those values into the clean10 table.
