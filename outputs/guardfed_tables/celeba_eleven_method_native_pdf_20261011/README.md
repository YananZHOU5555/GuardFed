# CelebA eleven-method native comparison

`celeba_eleven_method_native.pdf` contains three A3 landscape pages. Page 1 is the 10-seed IID/non-IID main comparison; pages 2 and 3 retain the fixed 9/6-seed sensitivity views.

Updated at the user's request on 2026-10-12: **best displayed mean is bold**, and the second-best distinct displayed mean is underlined. Rank separately within each distribution, seed panel, scenario and metric. Higher ACC and lower AEOD/ASPD are better. Equal displayed means share a rank; SD does not determine rank. All methods use the same rule. These are descriptive marks, not significance tests.

The linked Markdown table uses the same marks, including its separate seed-first aggregate comparisons. Removing only the new marks and legend recovers the accepted original table byte for byte. All scientific JSON sources, means, SDs, seeds, configurations and evaluation choices are unchanged. No statistics, fitting, inference or training was rerun.

Current formatting verification: `HIGHLIGHT_VERIFICATION.json`. It covers 1,287 Markdown cells and 990 PDF cells, the original scientific-source hashes, exact displayed numeric values, page bounds and visual review. Rebuild the formatted table and PDF with `python build_pdf.py` from this directory, or run the script by its repository-relative path.

The unhighlighted PDF is preserved as `previous_unhighlighted.pdf`. `ROOT_REVIEW.json`, `ROOT_CANDIDATE.json`, `FILES_SHA256.json`, `NUMERIC_LAYOUT_VERIFICATION.json`, `SPATIAL_CELL_VERIFICATION.json`, `VISUAL_QA.json`, `marker_receipt.json`, `SOURCE_DIFF.patch` and `attempt1/` describe the earlier unhighlighted edition; their old PDF/table hashes are historical, not checks of the newly formatted edition. The accepted unformatted Markdown remains in `tmp/celeba_eleven_method_native_table_prepare_20261011/actual_native1100/TABLES.md`.

This remains an eleven-method native validation comparison, not the complete 17-method table or final test evaluation. Method adaptations, postprocessing, mixed runtimes and validation-selection boundaries remain disclosed on every PDF page.
