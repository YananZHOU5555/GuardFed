# Reader-clarity candidates for author review

The two complete English candidates use only the accepted A40/LoGo100 integrated input. They do not incorporate later evidence or replace the canonical documents.

- `rebuttal_reader_candidate.md` leads with the point-by-point responses. The original evidence register, reporting conventions, snapshot chronology and older runtime details follow the pending register. Text before the first response falls from 1,876 to 242 words under the same checker token rule; the evidence is relocated rather than discarded.
- `manuscript_insertions_reader_candidate.md` presents the substantive insertion passages before the evidence history. The passages remain unapplied author-review text.
- C, U and A passages are labelled separately. The misplaced IID Sp-DFA C result and inherited IID C seed-first summary are placed with their respective C evidence. Current A comparability remains distinct from earlier subset runtime counts.

`EDIT_MANIFEST.json` records every exact before/after span, sequential character offset and whole-document SHA. `check_candidate.py` verifies both forward reconstruction and inverse recovery of the original UTF-8 bytes. Run the read-only check from this directory with `python -B check_candidate.py`.

The final `SELF_CHECK.json` confirms all 24 Original-comment blocks and their order, all numeric-string multisets, all 88 Markdown-link occurrences, every table, all math/code blocks, and the unchanged pending table. All original prose is retained except three explicit navigation edits across the two files. Those sentences contain cohort labels only: C comparisons are “reported earlier,” and historical A counts are “retained in the evidence history.” No quantitative sentence or value changes. The full wording of these exceptions is recorded; the check does not claim they are byte-identical.

This is an editorial check against the supplied files. It does not rerun statistics, reopen external evidence links, certify scientific completion, choose an endpoint, apply manuscript changes or authorize submission. Scientific trade-offs, negative results, selection/exposure history, mixed-runtime limits, original evidence numbering and unresolved obligations remain intact. No training, inference, fitting, SSH or bulk writes were performed. Final adoption remains for root and the authors.

During checker development, its first pending-table extractor accidentally selected the closing prose after the table. The actual table comparison already passed. The extractor was corrected and the full final check passed; no candidate table or source evidence was changed for that correction.
