# CelebA nine-method three-view tables: compiled display PDF

This is a display-only package for the accepted900-ID validation tables. No statistics, threshold fits, model predictions or scientific claims were added. Original inputs are readonly: `outputs/guardfed_tables/celeba_nine_method_three_view_20261009`, source67-member seal SHA1bc00ab3e2f5b69f915730dc1b94fb9da49cc8cc69f9756f8b2ea092b3e33c8e. All67 source files matched that seal before and after compilation. The canonical source directory contained no PDF at the start; its README explicitly described the TeX as uncompiled fragments.

Open `celeba_nine_method_three_view.pdf`:9A3 landscape pages, one combined panel per page. Pages1–3 are raw,4–6 native,7–9 shared calibration. Within each view the panels are10 seeds91001–91010,9 seeds91002–91010,6 seeds91005–91010. The original Category/Method/Metric/IID+non-IID five-scenario structure, captions, values, bold markup and original footnotes are retained. All pages state validation19867/round70; mean±sample SD uses ddof1. AEOD remains the implemented absoluteTPRgap. ACC is in percent; AEOD/ASPD retain their original gap scale. No significance or ranking rule was added.

Raw is margin>0 with ties assigned class0. Native uses GuardFed-AD2+ saved clean-training-root group thresholds and native argmax for the other eight methods. Shared calibration applies the frozen clean-root group-threshold procedure to all nine. No validation-label fitting was introduced. Each page retains the original recipe/seed91001 search disclosure, originalcu128/cu130 composition and mixed CPU434/GPU466 replay provenance. The extra reading note explicitly retains validation exposure and native/shared primary-endpoint pending status. This PDF is not uniform-device final evaluation or test evidence; nine-method coverage does not complete other baselines/mechanism studies.

`fragments_original/` contains9byte-exact original TeX copies. `fragments_unique/` contains9compile copies whose only textual change is a unique label prefix. `wrapper.tex` makes the original table* environment nonfloating, setsA3 landscape margins and adds view/seed/split/round headings plus compact decision-rule/reading-scope notes. It reserves1pt against resizebox floating-point width roundoff; no source table content is changed. `source_bindings.json` binds every copied fragment and original sourceSHA.

`verification.json` records9pages, all methods/labels/nonempty extraction,2,430mean±SD pairs (4,860scalar numbers) matching the original TeX exactly, minimum table numeric font11.5527pt, text inside page bounds, and zero missing-character/undefined/overfull errors. All9pages were rendered to `qa/page_01.png` through `page_09.png` at108dpi and visually inspected. The original67-member source seal remained unchanged. `attempt1_resize_roundoff/` preserves the first rejected layout logs and wrapper (0.13133pt width warnings, with unchanged values); final compilation resolved those warnings through the wrapper only.

The PDF skill marker ran successfully once before first authoring, with exit0 and empty stdout; see `marker_receipt.json`. Compilation used existing local XeLaTeX/MiKTeX and PyMuPDF, with installer and shell escape disabled. No TeX/dependency installation, SSH, network, training, inference, test, Git or canonical write was performed. Existing MiKTeX emits an unchecked-updates advisory, preserved in stdout logs; it is not a build failure.

Rebuild locally with the existing dependencies:

```powershell
python -B tmp/celeba_nine_method_three_view_pdf_20261009/build_pdf.py
```

The script only copies source fragments, compiles, renders and compares serialized display strings; it does not import the experiment/statistics implementations. A rebuild regenerates PDF metadata/hash and marks visual inspection pending again. This delivered package is sealed after actual nine-page visual inspection. Root owns promotion to outputs; this task stops here.
