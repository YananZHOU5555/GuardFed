# Exact4 metadata rebinding

Original after14 scientific collector body and per-ID loop are byte exact; sole effective collector change is the frozen authorized ID literal. Original after22 transport functions are reused in memory, changing only prior22/counts to prior28, independent output namespace and actual previous/root hashes. The SNAPSHOT command is the one already successful source/guide/CPU observation; redundant INITIAL_GUIDE/INITIAL_OWNER/SNAPSHOT SSH calls are omitted. collect retains its own actual guide/owner/preflight before one strict call. No original verifier, source/data/method/seed/rule/service is modified.

```diff
--- original_after14_collector
+++ effective_exact4_collector
@@ -57,7 +57,7 @@
     prior_count=len(previous['accepted_job_ids'])
     wanted=[row['id'] for row in snapshot['flgmm']['rows'] if row['terminal_acceptance'] and row['progress']['round']==70 and row['id'] not in previous['accepted_job_ids']]
     # Parent-authorized fixed snapshot exact2 only; later terminal IDs remain unaccepted.
-    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91007_fullcoverage']
+    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_FedSA_seed91010_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91004_fullcoverage']
     wanted=[identity for identity in wanted if identity in authorized_ids]
     if not wanted:
         write_json(BATCH/'NO_NEW_TERMINAL.json',dict(status='NO_NEW_TERMINAL_IN_THIS_SNAPSHOT',snapshot_sha256=digest(BATCH/'live_snapshot.json'),prior_accepted_new=prior_count,backup_created=False));return
```
