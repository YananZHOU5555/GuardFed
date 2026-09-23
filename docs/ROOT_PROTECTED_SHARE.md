# Root protected-share protocol v1

This separate experiment preserves the original AD2+ aggregator and uses an explicit fixed train reservoir. For each dataset/seed, 20% of the train set is jointly stratified by sensitive group and label; the other80% is fixed client data. Root size is min(round(10% train), nonprotected reservoir capacity, twice protected reservoir capacity), fixed across proportions. Protected shares are50%,10%,2%,0%, with rounded integer counts; actual counts, index hashes and missing-group support are recorded. Unused reservoir rows never return to clients. Per-group seed permutations make sampling nested. Global reweighting remains based on the same complete training split, as in the selected5090 implementation.

Protected means Female=0 for Adult and African-American=1 for COMPAS, explicitly checked from the loader encoding. This is independent of legacy privileged constant names and F Flip target conventions. New COMPAS uses train_only scaler. The four proportions are new matched controls; no old main runs are repeated or reused as this new design's control.

Matrix:2datasets x4shares x2attacks(Benign/S-DFA) x10seeds=160. Alpha5,70rounds,1local epoch,Adam.005,batch256,20clients,4nominal malicious,root synthesis0,unchanged candidate pool/calibration. Fixed final checkpoint,10-seed mean/sample std; retain weak outcomes. CPU workers use frozen source/config/data hashes.

Zero protected share means that group's root fairness and thresholds are not identifiable. Existing fallbacks remain implementation behavior, not evidence of fairness. Root changes also change the S-DFA attack's FedSA reference update, so this is an end-to-end sensitivity experiment, not isolated defense-only attribution.

Default root_protected_share=None retains the prior behavior. Tests verify invariant clients, RW, test tensors, common root sizes and nested root groups, including zero support; baseline root/ablation tests remain required. Eight2-round endpoint/attack canaries must finish before formal execution.
