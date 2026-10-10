"""Assemble a reviewer-facing draft from unchanged comments and reviewed responses."""
from pathlib import Path
import re,json,hashlib
R=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
S=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/rebuttal_integrated_20261011.md'
D=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_20261011'
H=lambda b:hashlib.sha256(b).hexdigest()
assert H(S.read_bytes())=='882dab1aafe77f7a2792c15bfe3322b9e1ac0d0b7e165c95ed06a5baf3b07354'
source=S.read_text(encoding='utf8')
responses={}
for name in ['responses_AE_R1.json','responses_R2.json']:
    responses.update(json.loads((W/name).read_bytes()))
responses.update({h:b.strip() for h,b in re.findall(r'^### ([^\n]+)\n(.*?)(?=^### |\Z)',(W/'responses_R3.md').read_text(encoding='utf8'),re.M|re.S)})
assert len(responses)==24
heading=re.compile(r'^(## Associate Editor|## Reviewer [123]|### [^\n]+)\n',re.M)
matches=list(heading.finditer(source[:source.index('## Pending register')]));parts=[];original_quotes=[];seen=[]
for i,m in enumerate(matches):
    h=m.group(1);name=h.lstrip('# ')
    if h.startswith('## Reviewer '):parts.append(h+'\n');continue
    body=source[m.end():matches[i+1].start() if i+1<len(matches) else source.index('## Pending register')]
    quotes=[line for line in body.splitlines() if line.startswith('>')]
    assert quotes and name in responses,name
    original_quotes.extend(quotes);seen.append(name)
    parts.append(h+'\n\n**Original comment (verbatim).**\n\n'+'\n'.join(quotes)+'\n\n**Response.** '+responses[name]+'\n')
assert len(seen)==len(set(seen))==24
intro='''# Response to the Associate Editor and Reviewers

**Manuscript:** TDSC-2026-07-3058, “To Kill Two Birds with One Stone: Defending Both Utility and Fairness in Federated Learning Systems”

**Author-review draft — revision incomplete.** This version presents the current responses without internal batch histories. All 24 comments are reproduced verbatim. Proposed manuscript changes have not yet been applied to the matching submitted source, and the unfinished items are listed once at the end. CelebA results remain validation evidence; the official test partition was previously exposed. No final evaluation or submission-ready completion is claimed.

Dear Associate Editor and Reviewers,

Thank you for identifying the gaps in contribution scope, theory and experimental evidence. The response below clarifies the implemented method, presents additional datasets and individual-component evidence, and retains unfavorable results. Reviewer 1 and Reviewer 3 keep their original numbering. Reviewer 2's descriptive headings are navigation aids, not original reviewer numbers.

'''
pending='''## Work still required before submission

| Item | Remaining requirement |
|---|---|
| P1 — Complete benchmark | Complete accepted coverage for the seven remaining target methods. FLGMM and the cosine/fairness hybrid have active fixed-recipe queues; Fed-NGA/Huber validation search is ongoing. FedWA, SmartFL and FedDNA still require faithful source/specification resolution. Preserve adaptation labels, all seeds and negative results; do not substitute simplified branches for original methods. |
| P2 — Image mechanisms | Finish the eight-control CelebA study: 800 new runs with 100 explicitly reused Full controls. At this draft's accepted cutoff, 280 new models have native and three-view evidence. U100 and C100 each cover ten cells; A80 covers eight. Finish non-IID A S-DFA/Sp-DFA and the other five controls, then produce complete matched-seed tables. |
| P3 — Frozen final evaluation | Decide the primary prediction/evaluation endpoint and freeze the final protocol before evaluation. Retain raw/native/shared interpretation and common-calibration controls. Official partition2 contains 19,962 images, but prior test exposure prevents an untouched-holdout claim. No final-test performance is supplied here. |
| P4 — Historical corrections | Finalize coherent Table II replacement and actual repeat counts, without inventing unavailable SD. Resolve unsupported method attributions and synthetic/Fig.3 provenance where possible; otherwise explicitly qualify or withdraw unsupported claims. The terminal redraw remains a candidate, not recovery of the original execution. |
| P5 — Submitted manuscript | Obtain the matching submitted LaTeX project, apply the verified method/theory/discussion insertions and accepted tables, and compile/render the revision. Assign final section, equation, table and page references after integration. The located older source is not assumed to match the submission. |
| P6 — Final reproducibility release | Publish the completed accepted scope, faithful adapters, frozen protocols, per-seed results and usable setup instructions. Retain large-artifact hashes and access information, failure/negative evidence, environment differences and unavailable legacy checkpoints. The current Git snapshot is an intermediate evidence release. |

## Supporting material

- [Ten-method native IID/non-IID tables](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md), with the [three-page paper-table PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf).
- [Nine-method raw/native/shared comparison](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md) and [paired calibration attribution](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/REPORT.md).
- [Complete U-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md), [complete C-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md) and [eight-scene A-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011/TABLES.md).
- [Method, root-update pseudocode, theory, notation and manuscript insertion candidates](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/manuscript_insertions_integrated_20261011.md).
- [Detailed evidence and historical audit version](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/rebuttal_integrated_20261011.md), including legacy Table II, synthetic reporting, source histories and reporting conventions.
- [Verified recommended-literature audit](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/recommended_literature_audit.md).

All repeated-model tables use sample SD (ddof=1), matched seed sets and one terminal checkpoint for all metrics. Cross-scenario summaries first average within model seed. Selection history, mixed environments/devices and historical test exposure remain disclosed. Sensitivity subsets and smaller disparities do not imply significance, untouched confirmation or uniform subgroup benefit.
'''
draft=intro+'\n'.join(parts)+'\n'+pending
assert [line for line in draft.splitlines() if line.startswith('>')]==original_quotes
assert draft.count('**Original comment (verbatim).**')==24
local_links=re.findall(r'\]\((E:/[^)]+)\)',draft)
assert all(Path(link).is_file() for link in local_links)
old_words=len(source.split());new_words=len(draft.split());assert new_words<old_words*.65
D.mkdir(exist_ok=True)
path=D/'rebuttal_clear_20261011.md';path.write_text(draft,encoding='utf8')
proof=dict(status='CLEAR_AUTHOR_REVIEW_DRAFT_EDITORIAL_CHECK_PASS',source_sha256=H(S.read_bytes()),draft_sha256=H(path.read_bytes()),original_comments=24,quotation_lines_exact_and_ordered=True,response_sections=24,source_words=old_words,draft_words=new_words,local_link_occurrences_checked=len(local_links),R3_independent_review=(W/'REVIEW_R3.md').is_file(),source_evidence_unchanged=True,science_recomputed=False,final_test=False,submitted_manuscript_applied=False,whole_rebuttal_complete=False)
(D/'EDITORIAL_CHECK.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(proof))
