"""Finite supplementary archive/source search; never executes recovered code."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

from PIL import Image

from audit import OUT, ROOT, ARCHIVE, file_sha, sha, put_json, search_hits

ARCHIVES = [
    (ROOT / 'tmp/git_sync_assets/guardfed-windows-artifacts-20260923.tar.gz',
     'acba809d105eba56382bde6fbd7f47a850a949dc54cdcfa0d138e8911960cfbb'),
    (ROOT / 'tmp/git_sync_assets/guardfed-server-check.tar.gz', None),
]
TEXT_SUFFIXES = {'.py', '.tex', '.ipynb', '.js', '.ts', '.r', '.m', '.sh', '.svg', '.eps'}
DEPENDENCY = re.compile(r'forest[-_]?diffusion|forestflow', re.I)
METADATA_NAME = re.compile(r'requirements|pip.*freeze|environment|pyproject|package|dependency|versions|pip_list', re.I)
EXTRA_SCOPE = ['src', 'compas_code', 'workbook_build']


def archive_scan(path: Path, expected: str | None, target: Image.Image) -> dict:
    observed = file_sha(path)
    if expected:
        assert observed == expected
    sources, images, metadata, bytecode, vector = [], [], [], [], []
    total = 0
    with tarfile.open(path, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            total += 1
            name, suffix = member.name, Path(member.name).suffix.lower()
            if suffix not in TEXT_SUFFIXES | {'.pyc', '.png', '.jpg', '.jpeg', '.md', '.txt', '.yml', '.yaml', '.lock', '.toml'}:
                continue
            ismeta = suffix in {'.md', '.txt', '.yml', '.yaml', '.lock', '.toml'}
            if ismeta and not METADATA_NAME.search(name) and suffix != '.md':
                continue
            data = tar.extractfile(member).read()
            row = {'member': name, 'bytes': len(data), 'sha256': sha(data)}
            if suffix in TEXT_SUFFIXES:
                row['hits'] = search_hits(data)
                sources.append(row)
                if row['hits']:
                    # Flat, content-addressed filenames stay below Windows path limits.
                    archive_key = 'windows' if 'windows-artifacts' in path.name else 'servercheck'
                    dest = OUT / 'additional_sources' / f'{archive_key}__{sha(data)[:16]}{suffix}'
                    assert not Path(name).is_absolute() and '..' not in Path(name).parts
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    row['restored_as'] = str(dest.relative_to(OUT)).replace('\\', '/')
                if suffix in {'.svg', '.eps'}:
                    vector.append(row)
            elif suffix == '.pyc':
                bytecode.append({**row, 'forest_byte_tokens': bool(re.search(b'forest', data, re.I))})
            elif suffix in {'.png', '.jpg', '.jpeg'}:
                with Image.open(io.BytesIO(data)) as im:
                    im = im.convert('RGB')
                    images.append({**row, 'size': list(im.size),
                                   'exact_submission_pixel_match': im.size == target.size and sha(im.tobytes()) == sha(target.tobytes())})
            elif ismeta:
                text = data.decode('utf-8', errors='replace')
                hits = [{'line': n, 'text': s[:600]} for n, s in enumerate(text.splitlines(), 1)
                        if DEPENDENCY.search(s) or 'Fairness score' in s or 'FairScore' in s]
                metadata.append({**row, 'hits': hits})
    return {'archive': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': observed,
            'previous_expected_sha256': expected, 'total_file_members': total,
            'source_files': sources, 'bytecode': bytecode, 'raster_images': images,
            'metadata_files': metadata, 'vector_assets': vector}


def main() -> None:
    target = Image.open(OUT / 'page11_image_35.png').convert('RGB')
    archives = [archive_scan(path, expected, target) for path, expected in ARCHIVES]
    put_json('additional_archive_search.json', archives)
    # Re-open only small environment/dependency members of the already pinned primary archive.
    primary_metadata = []
    with tarfile.open(ARCHIVE, 'r|gz') as tar:
        for member in tar:
            if member.isfile() and METADATA_NAME.search(member.name) and Path(member.name).suffix.lower() in {'.txt', '.toml', '.yml', '.yaml', '.lock', '.json', '.md'}:
                data = tar.extractfile(member).read()
                text = data.decode('utf-8', errors='replace')
                hits = [{'line': n, 'text': s[:600]} for n, s in enumerate(text.splitlines(), 1) if DEPENDENCY.search(s)]
                primary_metadata.append({'member': member.name, 'sha256': sha(data), 'bytes': len(data), 'forest_dependency_hits': hits})
    extras = []
    roots = [x for x in EXTRA_SCOPE if (ROOT / x).exists()]
    paths = subprocess.check_output(['rg', '--files', '-uuu', *roots], cwd=ROOT).decode('utf-8').splitlines()
    for rel in sorted(set(paths)):
        path = ROOT / rel
        if path.suffix.lower() in TEXT_SUFFIXES:
            data = path.read_bytes()
            extras.append({'path': rel.replace('\\', '/'), 'sha256': sha(data), 'bytes': len(data), 'hits': search_hits(data)})
    # Cover historical raster blobs under the originally bounded Git paths and extra code paths.
    listed = subprocess.check_output(['git', 'rev-list', '--objects', '--all', '--',
                                     'scripts', '.codex_remote', '.codex_transfer', 'outputs/guardfed_tables', *roots], cwd=ROOT).decode('utf-8')
    selected = {}
    for line in listed.splitlines():
        if ' ' not in line:
            continue
        oid, path = line.split(' ', 1)
        if '/node_modules/' in path or '/.venv/' in path:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in {'.png', '.jpg', '.jpeg'} or (suffix in TEXT_SUFFIXES and (path.startswith(tuple(roots)) or suffix in {'.js', '.ts', '.r', '.m', '.sh'})):
            selected[oid] = path
    git = []
    object_types = subprocess.check_output(['git', 'cat-file', '--batch-check=%(objectname) %(objecttype)'],
                                         input=('\n'.join(selected) + '\n').encode(), cwd=ROOT).decode().splitlines()
    type_map = dict(row.split(' ', 1) for row in object_types)
    skipped_nonblobs = []
    for oid, path in sorted(selected.items()):
        if type_map[oid] != 'blob':
            skipped_nonblobs.append({'git_object': oid, 'path': path, 'type': type_map[oid]})
            continue
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        row = {'git_blob': oid, 'path': path, 'bytes': len(data), 'sha256': sha(data)}
        if Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert('RGB')
                row.update(size=list(im.size), exact_submission_pixel_match=im.size == target.size and sha(im.tobytes()) == sha(target.tobytes()))
        else:
            row['hits'] = search_hits(data)
        git.append(row)
    result = {'status': 'BOUNDED_SEARCH_COMPLETE', 'additional_archives': archives,
              'primary_archive_dependency_members': primary_metadata,
              'extra_local_source_scope': roots, 'extra_local_sources': extras,
              'additional_git_blobs': git,
              'skipped_nonblob_objects': skipped_nonblobs,
              'limits': 'Only the three named local backup archives and explicit project code/output paths; no global/person-folder scan, no SSH/remote Git fetch, no unreachable Git objects. A missing hit does not establish global absence.'}
    put_json('additional_source_search.json', result)
    summary = {'archives': [{'archive': a['archive'], 'source_files': len(a['source_files']),
                            'bytecode': len(a['bytecode']), 'raster_images': len(a['raster_images']),
                            'forest_sources': [s['member'] for s in a['source_files'] if 'forest' in s['hits']],
                            'plot_label_sources': [s['member'] for s in a['source_files'] if 'plot_label' in s['hits']],
                            'exact_image_matches': [s['member'] for s in a['raster_images'] if s['exact_submission_pixel_match']],
                            'forest_metadata_hits': [s for s in a['metadata_files'] if s['hits']]}
                           for a in archives],
               'primary_dependency_members': len(primary_metadata),
               'primary_forest_dependency_hits': [x for x in primary_metadata if x['forest_dependency_hits']],
               'extra_local_source_files': len(extras),
               'extra_local_hits': [r for r in extras if r['hits']],
               'additional_git_blobs': len(git),
               'additional_git_plot_or_forest_hits': [r for r in git if 'hits' in r and any(k in r['hits'] for k in ('plot_label', 'forest'))],
               'git_exact_image_matches': [r for r in git if r.get('exact_submission_pixel_match')]}
    put_json('additional_search_summary.json', summary)
    print(json.dumps({'status': 'BOUNDED_SEARCH_COMPLETE',
                      'additional_archives': len(archives),
                      'additional_archive_sources': sum(len(a['source_files']) for a in archives),
                      'additional_archive_rasters': sum(len(a['raster_images']) for a in archives),
                      'extra_local_sources': len(extras), 'additional_git_blobs': len(git),
                      'full_receipt': 'additional_source_search.json'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
