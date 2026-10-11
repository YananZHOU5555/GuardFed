# CelebA eleven-method native paper-style display

Deliverable: `celeba_eleven_method_native.pdf` (3 A3 landscape pages). Page1 shows the complete10-seed IID/non-IID blocks. Pages2-3 show the fixed9/6-seed sensitivity panels; each page repeats explicit interpretation, adaptation, provenance and selection notes. The original ten-method generator is reused with metadata, row-count/layout and disclosure edits only. `SOURCE_DIFF.patch` records the derivation.

The PDF uses only the accepted TABLES.md display strings. All990 mean±sample-SD cells (1980 displayed numeric values) match the source both as page multisets and by distribution/method/attack/metric position. Independent page counting and text bounds pass. Both PyMuPDF and Poppler final renders were inspected on all3 pages: no clipped text, overlap or missing numeric glyphs. Poppler printed two display-font configuration warnings but returned0 and rendered all pages; details in VISUAL_QA.json.

No statistics, models, fits, inference or training were recomputed. Source records/statistics/table bytes remain unchanged. `marker_receipt.json` records the successful required marker immediately before first authoring; it ran once. `attempt1/` retains the superseded first spacing draft; only the top-level PDF is the final candidate.

This is a readable paper-style review artifact, not a compiled IEEE manuscript layout. It shows11 methods, not all17. Native postprocessing, adaptation and validation-selection boundaries remain explicit. No best-value, significance or universal superiority highlighting is used. Root adoption of this PDF is pending; native1100 scientific table adoption is already established separately.

Checks: NUMERIC_LAYOUT_VERIFICATION.json, SPATIAL_CELL_VERIFICATION.json and VISUAL_QA.json. Root can adopt the final PDF and compact checks after review; no STATE or reading-entry writes were made here.
