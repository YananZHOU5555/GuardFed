# C ablation update for R3.2/R3.7 — author review

This addendum extends the accepted five-IID-scene C ablation with non-IID Benign. It supplements the frozen 24-comment response; the submitted manuscript and that response have not been edited. This is descriptive validation evidence, not a final-test or component-necessity conclusion.

## Proposed response text

We additionally examine the C component under non-IID benign training, using ten matched seeds and the same round-70 evaluation protocol. Full achieves an accuracy of 88.594 ± 1.168%, an AEOD of 0.00696 ± 0.00423, and an ASPD of 0.06511 ± 0.00902. Removing C yields 89.188 ± 0.789%, 0.00615 ± 0.00668, and 0.06778 ± 0.00453, respectively. Thus, removing C increases accuracy by 0.594 percentage points and slightly reduces the AEOD mean, while increasing the ASPD mean. These results show a tradeoff rather than an improvement in every metric from retaining C. The accuracy increase and ASPD increase also occur in the nine-seed panel excluding the configuration-selection seed and in the six-seed panel using seeds 91005–91010; the AEOD difference in the six-seed panel is close to zero. We therefore do not interpret these observations as evidence that C is universally indispensable.

The raw-threshold evaluation gives a different fairness pattern: removing C increases accuracy by 0.641 percentage points while increasing both AEOD and ASPD by 0.00105 and 0.00198, respectively. Native and shared-calibration evaluations agree for the checkpoint pairs in this snapshot. Reporting all three views makes the effect of the evaluation rule visible and avoids attributing every native-metric difference solely to aggregation. All values are means and sample standard deviations across matched seeds; no significance claim is made. Here AEOD denotes the absolute TPR gap, not a full equalized-odds measure.

## Evidence and remaining scope

- The accepted table contains 60 Full–minus_C pairs: five IID scenarios and non-IID Benign, each with ten seeds. The remaining four non-IID C scenarios and six other mechanism controls are incomplete. No mean over the imbalanced set of six scenarios is presented; the prior five-IID-scenario seed-first aggregate is preserved byte for byte.
- All metrics use the same final checkpoint per run and the validation population of 19,867 images. Full reuses accepted checkpoints and replay records. Selection history, historical test exposure, mixed replay devices and training environments remain disclosed in the source table; these are not untouched-test estimates.
- Independent checking covers 972 scene mean/SD scalars, 486 display cells, 1,080 count-derived metrics and 2,880 confusion-count checks. The old 100 records, 810 scalars, 405 cells and original 162 IID aggregate scalars remain unchanged.

[Accepted three-view table](../training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/snapshot/TABLES.md) · [Machine-readable statistics](../training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/snapshot/tables.json) · [Root adoption](../training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/ROOT_VERIFICATION.json)

Root adoption SHA256: `f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742`.

Next manuscript action: incorporate this supported tradeoff when the full C coverage is complete, while preserving the COMPAS counterexample and the remaining author decisions. Do not replace pending results or declare the full revision complete.
