"""Integrate accepted prose only; no network, training, inference or statistics."""
import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / 'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009'
ADD = REPO / 'tmp/celeba_mechanism_rebuttal_increment_20261009'
SNAP = REPO / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim_20261009T145900Z'
GATE = 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT'
ROOT_SHA = 'd69255d2059ff6a7449b036ff1ef34225dcd2546c1a027699ba6bd8b35dea55c'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def read(path):
    return path.read_text(encoding='utf-8-sig')


def data(path):
    return json.loads(read(path))


def section(text, title):
    marker = next(line for line in text.splitlines() if line.startswith('## ' + title))
    return text.split(marker + '\n', 1)[1].split('\n## ', 1)[0].strip()


def write(name, value):
    target = HERE / name
    if target.exists():
        raise FileExistsError(f'No overwrite: {target}')
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    target.write_text(value, encoding='utf-8', newline='\n')


def rebase(text, source_dir):
    changes = []
    def replace(match):
        target = match.group(1)
        if urlsplit(target).scheme or target.startswith('#'):
            return match.group(0)
        dest = (source_dir / target).resolve()
        if not dest.is_file():
            raise ValueError(f'Missing original local link: {target}')
        changes.append({'original': target, 'resolved': dest.as_posix(), 'target_sha256': sha(dest)})
        return '](' + dest.as_posix() + ')'
    return re.sub(r'\]\(([^)]+)\)', replace, text), changes


def preserved_paragraphs(source, result, source_dir, replacement_lines=()):
    ledger = []
    for i, paragraph in enumerate(re.split(r'\n\s*\n', source.strip()), 1):
        expected, _ = rebase(paragraph, source_dir)
        replaced = any(line in paragraph for line in replacement_lines)
        if not replaced and expected not in result:
            raise ValueError(f'Unapproved original paragraph change: {i}')
        ledger.append({'source_paragraph': i, 'original_LF_utf8_sha256': sha_bytes(paragraph.encode()),
                       'integrated_expected_LF_utf8_sha256': None if replaced else sha_bytes(expected.encode()),
                       'status': 'explicit_exact_anchor_replacement' if replaced else ('local_link_target_only_rebased' if paragraph != expected else 'verbatim_preserved')})
    return ledger


def check_links(documents):
    rows = []
    for name, text in documents.items():
        markdown = re.findall(r'\]\(([^)]+)\)', text)
        bare = re.findall(r'https?://[^\s<>\]\)]+', text)
        for target in sorted(set(markdown + bare)):
            target = target.rstrip('.,;')
            parsed = urlsplit(target)
            if parsed.scheme in ('http', 'https'):
                if not parsed.netloc or ' ' in target:
                    raise ValueError(f'Malformed external URL: {target}')
                rows.append({'document': name, 'target': target, 'status': 'external_syntax_valid_not_fetched_no_network'})
            elif target.startswith('#'):
                raise ValueError(f'Unvalidated fragment link: {target}')
            else:
                dest = Path(target) if re.match(r'^[A-Za-z]:/', target) else HERE / target
                if not dest.is_file():
                    raise ValueError(f'Missing local evidence: {target}')
                rows.append({'document': name, 'target': target, 'status': 'local_file_exists', 'sha256': sha(dest)})
    return rows


def main():
    seal = data(ADD / 'FILES_SHA256.json')
    if sha(ADD / 'FILES_SHA256.json') != '07112ccc2c14a3fdbee0eb5fb1d9a9a5cb817fe99b099614b8726cf33251634f':
        raise ValueError('Increment source seal drift')
    for member in seal['members']:
        if sha(ADD / member['path']) != member['sha256']:
            raise ValueError(f'Increment member drift: {member["path"]}')
    source_map = data(ADD / 'source_map.json')
    for item in source_map['files']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Original source drift: {item["path"]}')
    if sha(SNAP / 'ROOT_REVIEW.json') != ROOT_SHA:
        raise ValueError('Root-accepted six-scene proof drift')
    proof = data(SNAP / 'ROOT_REVIEW.json')
    if (proof['complete_scenes'], proof['paired_checkpoints'], proof['displayed_records_from_original_receipts']) != (6, 60, 120):
        raise ValueError('Wrong frozen scientific scope')
    old_reply, old_manuscript = read(OLD / 'rebuttal_20261009.md'), read(OLD / 'manuscript_insertions_20261009.md')
    increment = read(ADD / 'candidate_replies.en.md')
    ae, r32, r37 = [section(increment, title) for title in ('AE —', 'R3.2 —', 'R3.7 —')]
    disclosure = section(increment, 'Required disclosure')
    p2 = next(line for line in section(increment, 'P2 —').splitlines() if line.startswith('| P2 —'))
    manuscript_addition = section(read(ADD / 'MANUSCRIPT_INSERT.md'), 'English insertion')
    accepted_table = (SNAP / 'TABLES.md').as_posix()
    excerpt_path = (ADD / 'evidence_excerpt.md').as_posix()
    r32 = r32.replace(excerpt_path, accepted_table)
    manuscript_addition = manuscript_addition.replace(excerpt_path, accepted_table)
    scope_note = (f'**{GATE}. Integrated author-review copy; fixed intermediate evidence snapshot.** '
                  'This copy adds only the closed six-scene CelebA Full–minus_U snapshot dated 2026-10-09T14:59:00Z, '
                  f'accepted at {proof["checked_utc"]} in the [independent root review]({(SNAP / "ROOT_REVIEW.json").as_posix()}). '
                  'It contains 60 matched seed pairs (120 checkpoints), with equal-rule 10/9/6-seed panels in the '
                  f'[accepted three-view table]({accepted_table}). The original draft\'s evidence cut-off and completed-cohort descriptions below retain their historical meaning. '
                  'Later mechanism results and tables are not incorporated. Update this fixed snapshot only after explicit acceptance of the replacement evidence. '
                  'The native/shared main endpoint remains pending, as do the remaining full-cohort and manuscript-integration items.')
    lines = old_reply.splitlines()
    expected = data(ADD / 'comment_alignment.json')['targets']
    for key in ('AE', 'R3.2', 'R3.7', 'P2'):
        locator = expected[key]
        if '\n'.join(lines[locator['start_line']-1:locator['end_line']]) != locator['original_excerpt']:
            raise ValueError(f'Exact integration anchor drift: {key}')
    replacement = {255: r32 + '\n\n**Interim image comparability.** ' + disclosure, 352: p2}
    append = {5: scope_note, 43: ae, 303: r37}
    reply = '\n'.join(line if i not in replacement else replacement[i] for i, line in enumerate(lines, 1)
                      for line in [line + ('\n\n' + append[i] if i in append else '')]) + '\n'
    # Manuscript integration keeps every original paragraph and adds the two approved English paragraphs.
    ml = old_manuscript.splitlines()
    if not ml[138].startswith('**Ablation paragraph.**'):
        raise ValueError('Exact manuscript insertion anchor drift')
    manuscript = '\n'.join(line + ('\n\n' + scope_note if i == 3 else '') + ('\n\n' + manuscript_addition if i == 139 else '') for i, line in enumerate(ml, 1)) + '\n'
    reply, reply_links = rebase(reply, OLD)
    manuscript, manuscript_links = rebase(manuscript, OLD)
    comment_pattern = r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*'
    before, after = re.findall(comment_pattern, old_reply, re.S), re.findall(comment_pattern, reply, re.S)
    if len(before) != 24 or before != after:
        raise ValueError('Original comment blocks are not 24/24 verbatim')
    maps = data(OLD / 'comment_source_map.json')['comments']
    comment_rows = []
    for block in before:
        plain = '\n'.join(re.sub(r'^> ?', '', line) for line in block.splitlines())
        keys = [key for key, value in maps.items() if value == plain]
        if len(keys) != 1:
            raise ValueError('Comment block differs from the original 24-comment source map')
        comment_rows.append({'id': keys[0], 'quote_block_LF_utf8_sha256': sha_bytes(block.encode()), 'original_source_map_text_exact': True, 'integrated_quote_block_exact': True})
    for item in ('P1', 'P3', 'P4', 'P5', 'P6'):
        row = next(line for line in lines if line.startswith('| ' + item + ' —'))
        if row not in reply:
            raise ValueError(f'Pending register row changed: {item}')
    para = {'rebuttal': preserved_paragraphs(old_reply, reply, OLD, (lines[254], lines[351])),
            'manuscript': preserved_paragraphs(old_manuscript, manuscript, OLD)}
    documents = {'rebuttal_integrated_20261009.md': reply, 'manuscript_insertions_integrated_20261009.md': manuscript}
    links = check_links(documents)
    for item in source_map['files']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Original source changed during integration')
    for name, text in documents.items():
        write(name, text)
    write('comment_consistency.json', {'status': 'PASS_24_ORIGINAL_COMMENTS_VERBATIM', 'comments': comment_rows})
    write('unchanged_paragraphs.json', para)
    write('link_checks.json', {'status': 'ALL_LOCAL_LINKS_RESOLVE_EXTERNAL_URL_SYNTAX_ONLY_NO_NETWORK', 'links': links, 'rebase_only_changes': reply_links + manuscript_links})
    write('SOURCE_MAP.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'fixed_snapshot': SNAP.as_posix(), 'root_proof_sha256': ROOT_SHA,
          'increment_seal_sha256': sha(ADD / 'FILES_SHA256.json'), 'original_inputs': source_map['files'],
          'integration_inputs': [{'path': (ADD / n).as_posix(), 'sha256': sha(ADD / n)} for n in ['candidate_replies.en.md', 'MANUSCRIPT_INSERT.md', 'comment_alignment.json', 'FILES_SHA256.json']],
          'integration_anchors_original_lines': {'AE_append_after': 43, 'R3.2_replace': 255, 'R3.7_append_after': 303, 'P2_row_replace': 352, 'manuscript_append_after': 139},
          'editorial_changes': ['Insert fixed-snapshot author-review gate in both headers.', 'Rebase existing local evidence links to their original absolute targets.', 'Point two approved new table citations directly to the root-accepted six-scene TABLES.md.'], 'new_scientific_numbers_or_later_results': False})
    write('INTEGRATION_DIFF.patch', ''.join(difflib.unified_diff(old_reply.splitlines(True), reply.splitlines(True), fromfile='sealed/rebuttal_20261009.md', tofile='integrated/rebuttal_integrated_20261009.md')) + ''.join(difflib.unified_diff(old_manuscript.splitlines(True), manuscript.splitlines(True), fromfile='sealed/manuscript_insertions_20261009.md', tofile='integrated/manuscript_insertions_integrated_20261009.md')))
    write('verification.json', {'status': 'PASS_COMPLETE_COPY_COMMENTS_PARAGRAPHS_LINKS_FIXED_SNAPSHOT', 'original_comments_exact': 24, 'original_comment_source_map_exact': 24, 'original_paragraphs': {key: dict((status, sum(row['status'] == status for row in rows)) for status in ('verbatim_preserved', 'local_link_target_only_rebased', 'explicit_exact_anchor_replacement')) for key, rows in para.items()},
          'pending_register_P1_to_P6_retained': True, 'P2_still_pending': True, 'tabular280_and_COMPAS_negative_paragraphs_preserved': True, 'fixed_scenes': 6, 'matched_seed_pairs': 60, 'checkpoint_records': 120,
          'links_n': len(links), 'external_links_network_verified': False, 'no_later_mechanism_output_read': True, 'root_proof_and_source_SHA_before_after': True, 'submission_gate': GATE, 'new_network_CNN_training_test_statistics': False, 'old_files_STATE_RUNNING_Git_modified': False})
    print(json.dumps({'status': 'PASS', 'comments': len(before), 'paragraph_counts': {key: len(rows) for key, rows in para.items()}, 'links': len(links)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
