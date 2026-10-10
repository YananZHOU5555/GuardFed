# F20 detailed response and matching insertion update

Author-review candidates only. No submitted-manuscript application, new statistics, fit, inference or final test. Original comments/order and all text outside the exact changes below are preserved.

Source bindings:
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011/ROOT_REVIEW.json` — SHA256 `195eb98c26cc2a4f0c3576cbb7a6b57a73fbc0d4ccc17a683f9817569e164c54`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011/rebuttal_integrated_20261011.md` — SHA256 `0816847dc412d7338dbf897571175292dbf950a5d6994dc053062b28a52ed30d`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011/manuscript_insertions_integrated_20261011.md` — SHA256 `a6cca62eed7dddd33f9e9fed5023531906e5843f9477e7e3cfeccee1c1f5f9d5`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/ROOT_VERIFICATION.json` — SHA256 `9d222cea9e55a00442ff278eb47fa1b622938611049c44d0565644047a8894ac`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md` — SHA256 `afce33a5ae59f5814c7b2164ce91638813acd82aa8bf8b475351ac41f46e668c`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_clear_F20_20261011/rebuttal_clear_20261011.md` — SHA256 `2f5f07cd26bd8a7eb8f4279d5c0b260924e2ec5fc5c5a32759ea1f3ff3fa7b6a`

## rebuttal_integrated_20261011.md

### 1. AE: include bounded F20 scope

Before:

The A-deletion extension now covers all ten CelebA scenes and 100 matched Full/minus_A pairs, including the completed non-IID Sp-DFA comparison. R3.2 reports its paired differences, fixed 10/9/6-seed panels and seed-first non-IID/balanced summaries; R3.7 retains the counterexamples and bounded interpretation. Five other image-control variants and seven baseline coverages remain pending; this does not establish all-component necessity or complete the full benchmark.

After:

The A-deletion extension now covers all ten CelebA scenes and 100 matched Full/minus_A pairs, including the completed non-IID Sp-DFA comparison. R3.2 reports its paired differences, fixed 10/9/6-seed panels and seed-first non-IID/balanced summaries; R3.7 retains the counterexamples and bounded interpretation. F deletion additionally has two complete IID scenes, Benign and F Flip, with ten matched seeds each. Completion of the other five image-control variants and seven baseline coverages remains pending; this does not establish all-component necessity or complete the full benchmark.

### 2. R3.2: current completed-versus-partial reading guide

Before:

**Image-control reading guide.** The completed U and C comparisons and the partial A comparison are presented separately.

After:

**Image-control reading guide.** The completed U, C and A comparisons and the two-scene F comparison are presented separately.

### 3. R3.2: precise remaining control coverage

Before:

The current A comparison covers all five IID and five non-IID scenes with 100 paired checkpoints; five other control variants and seven remaining baseline coverages are unfinished.

After:

The current A comparison covers all five IID and five non-IID scenes with 100 paired checkpoints. F deletion currently covers IID Benign and F Flip only; the remaining F scenes, four other controls and seven remaining baseline coverages are unfinished.

### 4. R3.2: two F20 evidence/interpretation paragraphs; preserve A100 paragraph

Before:

**A100 provenance and manuscript boundary.** Actual replay-device counts are {"Full": {"cpu": 5, "cuda:0": 95}, "minus_A": {"cpu": 100}}; training-build counts are {"Full": {"2.11.0+cu128": 98, "2.11.0+cu130": 2}, "minus_A": {"2.11.0+cu128": 100}}. Mixed CPU/CUDA, cu128/cu130 and driver histories remain disclosed; no runtime or trajectory equivalence is asserted. Root-only calibration, exposed validation selection and previously inspected official-test results remain explicit: this is not untouched confirmation. U100/C100 and all historical A80/A90 evidence remain unchanged. The other five image-control variants and seven remaining baseline coverages still require completion; Huber's identity-projection CNN adaptation does not inherit its original theorem, and LoGoFair's virtual cohorts are not real-client fairness evidence. P1 and P3–P6, the final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending. These are author-review insertion candidates, not applied manuscript changes or final-test findings; no universal-win, significance or all-component-necessity claim is made.

After:

**A100 provenance and manuscript boundary.** Actual replay-device counts are {"Full": {"cpu": 5, "cuda:0": 95}, "minus_A": {"cpu": 100}}; training-build counts are {"Full": {"2.11.0+cu128": 98, "2.11.0+cu130": 2}, "minus_A": {"2.11.0+cu128": 100}}. Mixed CPU/CUDA, cu128/cu130 and driver histories remain disclosed; no runtime or trajectory equivalence is asserted. Root-only calibration, exposed validation selection and previously inspected official-test results remain explicit: this is not untouched confirmation. U100/C100 and all historical A80/A90 evidence remain unchanged. The other five image-control variants and seven remaining baseline coverages still require completion; Huber's identity-projection CNN adaptation does not inherit its original theorem, and LoGoFair's virtual cohorts are not real-client fairness evidence. P1 and P3–P6, the final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending. These are author-review insertion candidates, not applied manuscript changes or final-test findings; no universal-win, significance or all-component-necessity claim is made.

**F-deletion: two complete IID scenes.** The [F-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md) adds IID Benign and F Flip (frozen Dirichlet α=5000), each with ten matched Full/deletion seeds. Raw, native and shared-calibration ACC/AEOD/ASPD use the same round-70 checkpoint per model and 19,867 validation images. The fixed panels contain all ten seeds, nine excluding selection seed91001, and six retaining seeds91005–91010; each reports mean±sample SD (ddof=1), including paired minus_F−Full differences. Removing F retains prediction calibration; native/shared metrics and group counts coincide, so these views are not independent confirmations. Reused Full evaluations comprise 2 CPU and 18 GPU records, versus 20 CPU deletion records; all selected checkpoints retain cu128 training provenance. This device difference and exposed validation/selection history preclude an isolated causal or untouched-confirmation interpretation.

**F-deletion results and sensitivity.** Under F Flip, Full has higher ten-seed mean ACC and lower AEOD and ASPD in every view. Native/shared paired differences are −0.066±1.239 percentage points, +0.00372±0.00942 and +0.00532±0.02066, respectively. In the fixed nine- and six-seed panels, deletion instead has higher mean ACC in raw and calibrated views, while Full retains lower disparity means. For Benign, native/shared ten-seed deletion improves ACC and AEOD but worsens ASPD; raw deletion improves ACC while worsening both disparities, whereas the six-seed raw panel favors deletion on all three means. We retain these reversals. AEOD means the absolute TPR gap, not full equalized odds. These two scene-specific comparisons neither establish significance or universal component necessity nor complete F100; no cross-scene F aggregate, primary endpoint or final-test result is supplied.

### 5. R3.7: explain F20 dataset/seed dependence without removing prior counterexamples

Before:

The historical A40 comparison in R3.2 adds an image-domain example of the same limitation: its mean directions depend on the prediction rule and fixed seed panel. It preserves accuracy–disparity trade-offs and does not explain them by an isolated causal mechanism. The historical A50 IID Sp-DFA comparison reinforces this boundary: deletion lowers native/shared mean AEOD but raises mean ASPD, while both raw disparity means rise in the ten-seed panel. Raw ASPD reverses sign in the fixed six-seed panel. Its accuracy loss is a paired-mean result, with four positive and six negative seed-level differences in each view. The seed-first summary in R3.2 provides a broader five-IID summary; it averages only the IID scenes and remains separate from the added non-IID Benign comparison. The historical A60 non-IID Benign comparison supplies a further counterexample to uniform component benefit: deleting A lowers mean accuracy and ASPD but raises mean AEOD in every reported prediction view and fixed seed panel. The historical A60 evidence therefore supports bounded utility–disparity trade-offs, without isolating a causal mechanism or completing the other four non-IID A scenes. The added A80 F Flip and FedSA evidence in R3.2 reports each view and fixed-panel direction explicitly. These added comparisons do not convert the earlier counterexamples into evidence of universal component benefit or identify an isolated causal mechanism. The added A90 non-IID S-DFA comparison in R3.2 further limits a uniform-benefit interpretation: native/shared deletion trades lower accuracy for lower ten-seed AEOD and ASPD, whereas raw deletion improves all three ten-seed means. Native/shared AEOD and raw accuracy change direction in the fixed six-seed panel relative to the ten-/nine-seed panels. We retain all panels; the result does not establish significance, necessity or an isolated causal mechanism. The A100 addition closes non-IID Sp-DFA and supplies the saved same-seed differences and seed-first non-IID/balanced summaries in R3.2. All fixed 10/9/6 panels and the earlier counterexamples remain; completing the A scene grid does not turn its conditional tradeoffs into evidence that every score component is necessary. The tabular COMPAS counterexamples above remain unchanged.

After:

The historical A40 comparison in R3.2 adds an image-domain example of the same limitation: its mean directions depend on the prediction rule and fixed seed panel. It preserves accuracy–disparity trade-offs and does not explain them by an isolated causal mechanism. The historical A50 IID Sp-DFA comparison reinforces this boundary: deletion lowers native/shared mean AEOD but raises mean ASPD, while both raw disparity means rise in the ten-seed panel. Raw ASPD reverses sign in the fixed six-seed panel. Its accuracy loss is a paired-mean result, with four positive and six negative seed-level differences in each view. The seed-first summary in R3.2 provides a broader five-IID summary; it averages only the IID scenes and remains separate from the added non-IID Benign comparison. The historical A60 non-IID Benign comparison supplies a further counterexample to uniform component benefit: deleting A lowers mean accuracy and ASPD but raises mean AEOD in every reported prediction view and fixed seed panel. The historical A60 evidence therefore supports bounded utility–disparity trade-offs, without isolating a causal mechanism or completing the other four non-IID A scenes. The added A80 F Flip and FedSA evidence in R3.2 reports each view and fixed-panel direction explicitly. These added comparisons do not convert the earlier counterexamples into evidence of universal component benefit or identify an isolated causal mechanism. The added A90 non-IID S-DFA comparison in R3.2 further limits a uniform-benefit interpretation: native/shared deletion trades lower accuracy for lower ten-seed AEOD and ASPD, whereas raw deletion improves all three ten-seed means. Native/shared AEOD and raw accuracy change direction in the fixed six-seed panel relative to the ten-/nine-seed panels. We retain all panels; the result does not establish significance, necessity or an isolated causal mechanism. The A100 addition closes non-IID Sp-DFA and supplies the saved same-seed differences and seed-first non-IID/balanced summaries in R3.2. All fixed 10/9/6 panels and the earlier counterexamples remain; completing the A scene grid does not turn its conditional tradeoffs into evidence that every score component is necessary. The tabular COMPAS counterexamples above remain unchanged.

The two F-deletion scenes in R3.2 show why this interpretation must also retain fixed-panel sensitivity. F Flip favors Full on all three ten-seed means, but its accuracy direction reverses with nine or six seeds; Benign retains raw/calibrated disparity trade-offs and a six-seed raw counterexample. These descriptive changes do not isolate candidate compensation, establish significance or make every component necessary.

### 6. P2: accepted320/remaining480, without declaring the study complete

Before:

| P2 — CelebA mechanisms; still pending | U100 and C100 each have accepted/offserver same-checkpoint raw/native/shared results for all ten scenes and fixed 10/9/6-seed panels. C100 preserves all earlier C50/C60 values and adds the remaining non-IID attacks. The historical A40 comparison covers IID Benign, F Flip, FedSA and S-DFA with 40 paired checkpoints. The historical A50 extension adds IID Sp-DFA, completing all five IID scenes with 50 paired checkpoints. The historical A60 snapshot adds non-IID Benign, giving six complete scenes and 60 pairs; its then-pending scope was the other four non-IID A scenes and the other five incomplete image controls. The historical A80 extension adds non-IID F Flip and FedSA, giving eight scenes and 80 pairs; its then-pending scope was non-IID S-DFA/Sp-DFA and the other five image controls. At the historical A90 cutoff, non-IID S-DFA gave nine scenes and 90 pairs; non-IID Sp-DFA and the other five image controls were pending, and partial five-seed Sp-DFA records did not enter scene statistics. A100 now completes all ten scenes and 100 pairs, including Sp-DFA and seed-first non-IID/balanced summaries; the other five image controls remain pending. Preserve Full/source/partition identities, all unfavorable effects and calibration/device/runtime/selection boundaries. | R3.2; R3.7; mechanism attribution |

After:

| P2 — CelebA mechanisms; still pending | U100 and C100 each have accepted/offserver same-checkpoint raw/native/shared results for all ten scenes and fixed 10/9/6-seed panels. C100 preserves all earlier C50/C60 values and adds the remaining non-IID attacks. The historical A40 comparison covers IID Benign, F Flip, FedSA and S-DFA with 40 paired checkpoints. The historical A50 extension adds IID Sp-DFA, completing all five IID scenes with 50 paired checkpoints. The historical A60 snapshot adds non-IID Benign, giving six complete scenes and 60 pairs; its then-pending scope was the other four non-IID A scenes and the other five incomplete image controls. The historical A80 extension adds non-IID F Flip and FedSA, giving eight scenes and 80 pairs; its then-pending scope was non-IID S-DFA/Sp-DFA and the other five image controls. At the historical A90 cutoff, non-IID S-DFA gave nine scenes and 90 pairs; non-IID Sp-DFA and the other five image controls were pending, and partial five-seed Sp-DFA records did not enter scene statistics. A100 now completes all ten scenes and 100 pairs, including Sp-DFA and seed-first non-IID/balanced summaries; the other five image controls remain pending. At the current accepted cutoff, F20 adds IID Benign and F Flip with ten paired seeds each, giving 320 new models with native and three-view evidence (U100+C100+A100+F20). The remaining 480 control runs are 80 F runs and four other controls × 100; their accepted evidence and complete matched-seed tables remain required. Preserve Full/source/partition identities, all unfavorable effects and calibration/device/runtime/selection boundaries. | R3.2; R3.7; mechanism attribution |

### 7. Current snapshot: distinguish F20/320 cutoff from retained historical subsets

Before:

The [accepted A100 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) covers all five IID and five non-IID scenes, with 100 matched Full/minus_A pairs and fixed 10/9/6-seed panels. Non-IID Sp-DFA is now complete; the other five image controls and seven remaining baseline coverages remain pending. The adopted non-IID and balanced seed-first summaries average scenes within each seed; U100/C100 and the old IID aggregate are unchanged. No final test or primary endpoint is supplied.

After:

The [accepted A100 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) covers all five IID and five non-IID scenes, with 100 matched Full/minus_A pairs and fixed 10/9/6-seed panels. Non-IID Sp-DFA is now complete; the other five image controls and seven remaining baseline coverages remain pending. The adopted non-IID and balanced seed-first summaries average scenes within each seed; U100/C100 and the old IID aggregate are unchanged. No final test or primary endpoint is supplied.

The [accepted F20 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md) adds 20 matched Full/minus_F pairs for IID Benign and F Flip, ten seeds per scene, with every raw/native/shared and fixed 10/9/6 panel. The accepted mechanism cutoff is 320 new models with native and three-view evidence: U100, C100, A100 and F20. The remaining 480 control runs comprise the other 80 F runs and 100 runs for each of four other controls. Full controls are explicitly reused, not new training runs. Completing two F scenes does not complete all controls, the seven pending baseline coverages, manuscript integration or final evaluation.

### 8. Current-snapshot C paragraph: label its preserved historical cutoff explicitly

Before:

The fixed 10/9/6 raw/native/shared panels are complete for U and C; six other image-control variants remain incomplete. This completes two controls, not the all-component mechanism study.

After:

At that earlier C100 cutoff, the fixed 10/9/6 raw/native/shared panels completed U and C while six other image-control variants remained incomplete. The subsequent A100 and F20 scopes are stated below; none completes the all-component mechanism study.

## manuscript_insertions_integrated_20261011.md

### 1. Section7: two matching compact F20 insertion paragraphs; original A100 body unchanged

Before:

**A100 provenance and manuscript boundary.** Actual replay-device counts are {"Full": {"cpu": 5, "cuda:0": 95}, "minus_A": {"cpu": 100}}; training-build counts are {"Full": {"2.11.0+cu128": 98, "2.11.0+cu130": 2}, "minus_A": {"2.11.0+cu128": 100}}. Mixed CPU/CUDA, cu128/cu130 and driver histories remain disclosed; no runtime or trajectory equivalence is asserted. Root-only calibration, exposed validation selection and previously inspected official-test results remain explicit: this is not untouched confirmation. U100/C100 and all historical A80/A90 evidence remain unchanged. The other five image-control variants and seven remaining baseline coverages still require completion; Huber's identity-projection CNN adaptation does not inherit its original theorem, and LoGoFair's virtual cohorts are not real-client fairness evidence. P1 and P3–P6, the final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending. These are author-review insertion candidates, not applied manuscript changes or final-test findings; no universal-win, significance or all-component-necessity claim is made.

After:

**A100 provenance and manuscript boundary.** Actual replay-device counts are {"Full": {"cpu": 5, "cuda:0": 95}, "minus_A": {"cpu": 100}}; training-build counts are {"Full": {"2.11.0+cu128": 98, "2.11.0+cu130": 2}, "minus_A": {"2.11.0+cu128": 100}}. Mixed CPU/CUDA, cu128/cu130 and driver histories remain disclosed; no runtime or trajectory equivalence is asserted. Root-only calibration, exposed validation selection and previously inspected official-test results remain explicit: this is not untouched confirmation. U100/C100 and all historical A80/A90 evidence remain unchanged. The other five image-control variants and seven remaining baseline coverages still require completion; Huber's identity-projection CNN adaptation does not inherit its original theorem, and LoGoFair's virtual cohorts are not real-client fairness evidence. P1 and P3–P6, the final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending. These are author-review insertion candidates, not applied manuscript changes or final-test findings; no universal-win, significance or all-component-necessity claim is made.

**F-deletion: two complete IID scenes.** The [F-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md) adds IID Benign and F Flip (frozen Dirichlet α=5000), each with ten matched Full/deletion seeds. Raw, native and shared-calibration ACC/AEOD/ASPD use the same round-70 checkpoint per model and 19,867 validation images. The fixed panels contain all ten seeds, nine excluding selection seed91001, and six retaining seeds91005–91010; each reports mean±sample SD (ddof=1), including paired minus_F−Full differences. Removing F retains prediction calibration; native/shared metrics and group counts coincide, so these views are not independent confirmations. Reused Full evaluations comprise 2 CPU and 18 GPU records, versus 20 CPU deletion records; all selected checkpoints retain cu128 training provenance. This device difference and exposed validation/selection history preclude an isolated causal or untouched-confirmation interpretation.

**F-deletion results and sensitivity.** Under F Flip, Full has higher ten-seed mean ACC and lower AEOD and ASPD in every view. Native/shared paired differences are −0.066±1.239 percentage points, +0.00372±0.00942 and +0.00532±0.02066, respectively. In the fixed nine- and six-seed panels, deletion instead has higher mean ACC in raw and calibrated views, while Full retains lower disparity means. For Benign, native/shared ten-seed deletion improves ACC and AEOD but worsens ASPD; raw deletion improves ACC while worsening both disparities, whereas the six-seed raw panel favors deletion on all three means. We retain these reversals. AEOD means the absolute TPR gap, not full equalized odds. These two scene-specific comparisons neither establish significance or universal component necessity nor complete F100; no cross-scene F aggregate, primary endpoint or final-test result is supplied.

### 2. Integration checklist: actual mechanism scope and remaining obligation

Before:

5. Fill the remaining baseline/mechanism/final-evaluation evidence after acceptance; do not convert planned numbers to measured data.

After:

5. Fill the remaining baseline/mechanism/final-evaluation evidence after acceptance; do not convert planned numbers to measured data. For image mechanisms, the accepted native/three-view cutoff is 320 new models (U100+C100+A100+F20), leaving 480 control runs and their accepted evidence/tables; F20 covers only IID Benign and F Flip.

### 3. Current snapshot: distinguish F20/320 cutoff from retained historical subsets

Before:

The [accepted A100 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) covers all five IID and five non-IID scenes, with 100 matched Full/minus_A pairs and fixed 10/9/6-seed panels. Non-IID Sp-DFA is now complete; the other five image controls and seven remaining baseline coverages remain pending. The adopted non-IID and balanced seed-first summaries average scenes within each seed; U100/C100 and the old IID aggregate are unchanged. No final test or primary endpoint is supplied.

After:

The [accepted A100 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) covers all five IID and five non-IID scenes, with 100 matched Full/minus_A pairs and fixed 10/9/6-seed panels. Non-IID Sp-DFA is now complete; the other five image controls and seven remaining baseline coverages remain pending. The adopted non-IID and balanced seed-first summaries average scenes within each seed; U100/C100 and the old IID aggregate are unchanged. No final test or primary endpoint is supplied.

The [accepted F20 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md) adds 20 matched Full/minus_F pairs for IID Benign and F Flip, ten seeds per scene, with every raw/native/shared and fixed 10/9/6 panel. The accepted mechanism cutoff is 320 new models with native and three-view evidence: U100, C100, A100 and F20. The remaining 480 control runs comprise the other 80 F runs and 100 runs for each of four other controls. Full controls are explicitly reused, not new training runs. Completing two F scenes does not complete all controls, the seven pending baseline coverages, manuscript integration or final evaluation.

### 4. Current-snapshot C paragraph: label its preserved historical cutoff explicitly

Before:

The fixed 10/9/6 raw/native/shared panels are complete for U and C; six other image-control variants remain incomplete. This completes two controls, not the all-component mechanism study.

After:

At that earlier C100 cutoff, the fixed 10/9/6 raw/native/shared panels completed U and C while six other image-control variants remained incomplete. The subsequent A100 and F20 scopes are stated below; none completes the all-component mechanism study.
