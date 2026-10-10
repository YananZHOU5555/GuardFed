"""Typeset accepted scene-display strings only; no science or statistics execution."""
from pathlib import Path
import collections
import hashlib
import json
import re

import fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'outputs/guardfed_tables/celeba_ten_method_native_20261010'
TABLE_SHA = 'c6a38590e39e6ebafa2732498d7cabb479a161d27faee8016c48aecce2eef5a5'
PANELS = [('ten', '10 shared seeds: 91001-91010'),
          ('nonselection_nine', '9 shared seeds: 91002-91010'),
          ('matching_six', '6 shared seeds: 91005-91010')]
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
PAIR = re.compile(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def parse_panels(text):
    panels = {}
    for title, body in re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', text, re.M | re.S):
        rows = [line.split('|')[1:-1] for line in body.splitlines()
                if line.startswith('| ') and '±' in line]
        rows = [[cell.strip() for cell in row] for row in rows]
        # The source also retains a separate seed-first aggregate table. This PDF displays the six scene panels only.
        rows = [row for row in rows if row[1] in ('ACC (%) ↑', 'AEOD ↓', 'ASPD ↓')]
        assert len(rows) == 30 and all(len(row) == 7 for row in rows), title
        methods = list(dict.fromkeys(row[0] for row in rows))
        assert len(methods) == 10
        values = {}
        for method in methods:
            group = [row for row in rows if row[0] == method]
            assert [row[1] for row in group] == ['ACC (%) ↑', 'AEOD ↓', 'ASPD ↓']
            values[method] = [group[metric][attack + 2]
                              for attack in range(5) for metric in range(3)]
            assert all(PAIR.fullmatch(value) for value in values[method])
        panels[title] = (methods, values)
    assert len(panels) == 6
    return panels


def main():
    assert sha(SOURCE / 'TABLES.md') == TABLE_SHA
    proof = json.loads((SOURCE / 'ROOT_REVIEW.json').read_bytes())
    assert proof['status'] == 'ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED'
    assert proof['records'] == 1000 and not proof['final_test'] and not proof['full17_complete']
    before = {name: sha(SOURCE / name) for name in proof['files_sha256']}
    assert before == proof['files_sha256']
    panels = parse_panels((SOURCE / 'TABLES.md').read_text(encoding='utf-8'))
    out = HERE / 'celeba_ten_method_native.pdf'
    assert not out.exists(), 'Do not overwrite an existing delivery'
    width, height = landscape(A3)
    margin = 34
    c = canvas.Canvas(str(out), pagesize=(width, height), pageCompression=1)
    c.setTitle('CelebA: ten-method native validation tables')
    c.setAuthor('GuardFed revision evidence')
    page_checks = []
    all_cells = []
    note_style = ParagraphStyle('notes', fontName='Times-Roman', fontSize=9,
                               leading=12, textColor=colors.HexColor('#333333'))
    for page, (panel, seed_label) in enumerate(PANELS, 1):
        c.setFont('Times-Bold', 17)
        c.drawString(margin, height - 44, 'CelebA - ten-method native validation comparison')
        c.setFont('Times-Roman', 11)
        c.drawString(margin, height - 64, seed_label + ' | mean +/- sample SD (ddof=1)')
        c.setFont('Times-Italic', 9.5)
        c.drawString(margin, height - 80,
                     'Author-review candidate. Terminal checkpoints; validation split. Seven method cohorts remain incomplete.')
        y = height - 107
        page_cells = []
        for distribution in ('IID', 'non-IID'):
            methods, values = panels[distribution + ' / ' + panel]
            c.setFont('Times-Bold', 12)
            c.drawString(margin, y, distribution + (' (alpha=5000)' if distribution == 'IID' else ' (alpha=5)'))
            y -= 10
            top = ['Method'] + [attack if metric == 0 else '' for attack in ATTACKS for metric in range(3)]
            sub = [''] + ['ACC (%)', 'AEOD', 'ASPD'] * 5
            data = [top, sub] + [[method] + values[method] for method in methods]
            styles = [('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
                      ('FONTSIZE', (0, 0), (-1, -1), 8.2),
                      ('FONTNAME', (0, 0), (-1, 1), 'Times-Bold'),
                      ('FONTSIZE', (0, 0), (-1, 0), 10),
                      ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                      ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                      ('LEFTPADDING', (0, 0), (-1, -1), 3),
                      ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                      ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
                      ('LINEBELOW', (0, 1), (-1, 1), .65, colors.black),
                      ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black)]
            for index in range(5):
                left = 1 + index * 3
                styles.append(('SPAN', (left, 0), (left + 2, 0)))
                if index:
                    styles.append(('LINEBEFORE', (left, 0), (left, -1), .3, colors.HexColor('#aaaaaa')))
            guard_row = methods.index('GuardFed-AD2+') + 2
            styles += [('FONTNAME', (0, guard_row), (0, guard_row), 'Times-Bold'),
                       ('BACKGROUND', (0, guard_row), (-1, guard_row), colors.HexColor('#f2f2f2'))]
            table = Table(data, colWidths=[130] + [(width - 2 * margin - 130) / 15] * 15,
                          rowHeights=[20, 18] + [20] * 10)
            table.setStyle(TableStyle(styles))
            tw, th = table.wrap(width - 2 * margin, y)
            table.drawOn(c, margin, y - th)
            y -= th + 30
            page_cells += [value for method in methods for value in values[method]]
        notes = [
            '<b>Reading:</b> Higher ACC and lower AEOD/ASPD are better. ACC is a percentage; gaps are in [0,1]. AEOD is the absolute TPR gap, not full equalized odds. Low gaps can accompany constant predictions or poor accuracy. No best-value or significance highlighting is applied.',
            '<b>Method scope:</b> Native predictions retain each method\'s own reporting/postprocessing. FairFed, FairGuard, FLTrust+FairGuard, FedAA-DDPG and LASA use documented project adaptations. LoGoFair-DP uses 30 frozen postprocessing rounds, fitted native predictions, fit seed 1719 and 20 image-ID virtual cohorts; these are not true training clients.',
            '<b>Evidence boundary:</b> Backbone checkpoints are from round 70. Old nine-method replay uses CPU434/GPU466; training uses cu128886/cu13014. LoGoFair reuses accepted FedAvg checkpoints/caches with CPU netcal1.3.6 fitting. No full runtime equivalence or aggregation-only causal claim is made.',
            '<b>Selection:</b> Seed 91001 participated in validation configuration selection. Other validation seeds and historical test metadata were exposed. The 9/6-seed subsets are sensitivity views, not untouched test cohorts. All negative and constant-prediction outcomes are retained. This is not final-test evidence or a complete 17-method comparison; the primary endpoint remains pending.',
        ]
        y -= 1
        for note in notes:
            p = Paragraph(note, note_style)
            _, ph = p.wrap(width - 2 * margin, height)
            p.drawOn(c, margin, y - ph)
            y -= ph + 7
        assert y > 36, ('Page overflow', page, y)
        c.setFont('Times-Roman', 8)
        c.drawString(margin, 22, 'Source: root-adopted ten-method native1000 display table; no new statistics, fits, inference or training.')
        c.drawRightString(width - margin, 22, f'{page} / 3')
        c.showPage()
        page_checks.append(dict(page=page, panel=panel, display_cells=len(page_cells), content_lower_y=y))
        all_cells += page_cells
    c.save()
    doc = fitz.open(out)
    assert len(doc) == 3
    render = HERE / 'preview'
    render.mkdir(exist_ok=False)
    for page, check in zip(doc, page_checks):
        # Compare every displayed numeric pair as a multiset: no dependence on extraction order.
        rendered = collections.Counter(PAIR.findall(page.get_text()))
        expected = collections.Counter(PAIR.fullmatch(cell).groups() for cell in all_cells[(check['page']-1)*300:check['page']*300])
        assert rendered == expected, (check['page'], rendered - expected, expected - rendered)
        assert all(0 <= span['bbox'][0] < span['bbox'][2] <= width and 0 <= span['bbox'][1] < span['bbox'][3] <= height
                   for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans'])
        page.get_pixmap(matrix=fitz.Matrix(1.15, 1.15), alpha=False).save(render / f'page-{check["page"]}.png')
        check.update(numeric_pairs_extracted_exact=True, text_spans_within_page=True)
    assert len(all_cells) == 900
    assert before == {name: sha(SOURCE / name) for name in before}
    result = dict(status='DISPLAY_PDF_NUMERIC_AND_PAGE_BOUNDS_PASS_VISUAL_REVIEW_PENDING',
                  source_root_sha256=sha(SOURCE / 'ROOT_REVIEW.json'), source_table_sha256=TABLE_SHA,
                  source_member_hashes=before, pdf_sha256=sha(out), pages=page_checks,
                  displayed_mean_SD_pairs=900, displayed_numeric_values=1800,
                  new_statistics=0, new_inference=0, new_fit=0, new_training=0,
                  final_test=False, full17_complete=False, source_bytes_unchanged=True)
    (HERE / 'NUMERIC_LAYOUT_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
