"""Typeset accepted scene-display strings only; no science or statistics execution."""
from pathlib import Path
import collections
from decimal import Decimal
import hashlib
import json
import re
import shutil

import fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'outputs/guardfed_tables/celeba_eleven_method_native_20261011'
TABLE_SHA = 'c40f7909e79a16fcd18b9fb1980decfe541d33195b24caac335555844dbeea76'
LEGACY_LEGEND = ('**Best** and <u>second-best</u> means are marked separately within each distribution, seed panel, '
          'scenario and metric (and within each aggregate scope). ACC: higher is better; AEOD/ASPD: lower is better. '
          'Ranking uses the displayed mean precision; equal displayed means share a rank. SD does not determine rank. '
          'These marks are descriptive, not significance tests.')
LEGEND = (LEGACY_LEGEND + ' Fairness-ranking eligibility: the corresponding full-precision mean ACC must be >80%. '
          'AEOD/ASPD marked † have mean ACC <=80% and are retained but excluded from bold/underline ranking. '
          'The same rule applies to every method and to each aggregate scope, using its corresponding mean ACC. '
          'ACC ranking is unrestricted. All seeds and original statistics are retained; no conditional means are recomputed. '
          'This CelebA display rule was explicitly approved by the author on 2026-10-12 after viewing the results; '
          'it is not a preregistered experiment rule or a claim that the marked disparities are undefined.')
ACC_FLOOR = 0.8
PANELS = [('ten', '10 shared seeds: 91001-91010'),
          ('nonselection_nine', '9 shared seeds: 91002-91010'),
          ('matching_six', '6 shared seeds: 91005-91010')]
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
PAIR = re.compile(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def plain_cell(value):
    return value.removesuffix('†').removeprefix('**').removesuffix('**').removeprefix('<u>').removesuffix('</u>')


def unhighlight_table(marked):
    text = marked.decode('utf-8')
    newline = '\r\n' if b'\r\n' in marked else '\n'
    for legend in (LEGEND, LEGACY_LEGEND):
        text = text.replace(legend + newline + newline, '', 1)
    text = re.sub(r'(?<=\d)†', '', text)
    return re.sub(r'\*\*([\d.]+\s*±\s*[\d.]+)\*\*|<u>([\d.]+\s*±\s*[\d.]+)</u>',
                  lambda match: match[1] or match[2], text).encode('utf-8')


def highlight_table(original):
    # Work only on the accepted display strings; never recompute the statistics.
    statistics = json.loads((SOURCE / 'summary_statistics.json').read_bytes())
    aggregates = json.loads((SOURCE / 'seed_first_aggregates.json').read_bytes())
    scene_acc = {(panel, row['distribution'], row['attack'], row['method']): row['accuracy']['mean']
                 for panel, rows in statistics.items() for row in rows}
    aggregate_acc = {(panel, row['scope'], row['method']): row['accuracy']['mean']
                     for panel, rows in aggregates.items() for row in rows}
    lines = original.decode('utf-8').splitlines(keepends=True)
    groups = collections.defaultdict(list)
    eligible = {}
    section = ''
    for index, line in enumerate(lines):
        if line.startswith('#'):
            section = line.strip()
        if not line.startswith('| ') or '±' not in line:
            continue
        cells = line.split('|')
        if cells[2].strip() in ('ACC (%) ↑', 'AEOD ↓', 'ASPD ↓'):
            distribution, panel = section.removeprefix('## ').split(' / ')
            for column in range(3, 8):
                groups[(section, cells[2].strip(), column)].append((index, column))
                acc = scene_acc[(panel, distribution, ATTACKS[column - 3], cells[1].strip())]
                eligible[index, column] = cells[2].strip() == 'ACC (%) ↑' or acc > ACC_FLOOR
        elif section == '# Seed-first descriptive aggregates':
            for column in range(5, 8):
                groups[(cells[1].strip(), cells[2].strip(), column)].append((index, column))
                acc = aggregate_acc[(cells[1].strip(), cells[2].strip(), cells[3].strip())]
                eligible[index, column] = column == 5 or acc > ACC_FLOOR
    marks = collections.Counter()
    for key, positions in groups.items():
        assert len(positions) == 11, key
        means = {position: Decimal(lines[position[0]].split('|')[position[1]].split('±')[0].strip())
                 for position in positions}
        higher = key[1] == 'ACC (%) ↑' or (key[0] in dict(PANELS) and key[2] == 5)
        levels = sorted({mean for position, mean in means.items() if eligible[position]}, reverse=higher)
        for (index, column), mean in means.items():
            if eligible[index, column] and mean not in levels[:2]:
                continue
            cells = lines[index].split('|')
            value = cells[column].strip()
            assert PAIR.fullmatch(value)
            if not eligible[index, column]:
                marked = value + '†'
                marks['excluded_fairness'] += 1
            else:
                rank = levels.index(mean) + 1
                marked = '**' + value + '**' if rank == 1 else '<u>' + value + '</u>'
                marks[rank] += 1
            cells[column] = ' ' + marked + ' '
            lines[index] = '|'.join(cells)
    assert len(groups) == 117 and len(eligible) == 1287
    assert marks['excluded_fairness'] == sum(not value for value in eligible.values())
    newline = '\r\n' if b'\r\n' in original else '\n'
    highlighted = ''.join(lines).replace(newline + newline, newline + newline + LEGEND + newline + newline, 1)
    assert unhighlight_table(highlighted.encode('utf-8')) == original
    return highlighted.encode('utf-8'), dict(groups=len(groups), numeric_cells=1287,
                                            best_cells=marks[1], second_cells=marks[2],
                                            excluded_fairness_cells=marks['excluded_fairness'],
                                            fairness_eligibility='corresponding full-precision mean accuracy > 0.8',
                                            approved_date='2026-10-12',
                                            posthoc_display_rule=True, seed_filtering=False,
                                            ranking='displayed means among eligible methods; dense ranks; display ties share marks')


def parse_panels(text):
    panels = {}
    for title, body in re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', text, re.M | re.S):
        rows = [line.split('|')[1:-1] for line in body.splitlines()
                if line.startswith('| ') and '±' in line]
        rows = [[cell.strip() for cell in row] for row in rows]
        # The source also retains a separate seed-first aggregate table. This PDF displays the six scene panels only.
        rows = [row for row in rows if row[1] in ('ACC (%) ↑', 'AEOD ↓', 'ASPD ↓')]
        assert len(rows) == 33 and all(len(row) == 7 for row in rows), title
        methods = list(dict.fromkeys(row[0] for row in rows))
        assert len(methods) == 11
        values = {}
        for method in methods:
            group = [row for row in rows if row[0] == method]
            assert [row[1] for row in group] == ['ACC (%) ↑', 'AEOD ↓', 'ASPD ↓']
            values[method] = [group[metric][attack + 2]
                              for attack in range(5) for metric in range(3)]
            assert all(PAIR.fullmatch(plain_cell(value)) for value in values[method])
        panels[title] = (methods, values)
    assert len(panels) == 6
    return panels


def main():
    assert sha(SOURCE / 'ROOT_REVIEW.json') == '69e9c479c061a1aa428157b6f1d121aa5dd72a48e0e6d1d16dd21ac296391d02'
    original = unhighlight_table((SOURCE / 'TABLES.md').read_bytes())
    assert hashlib.sha256(original).hexdigest() == TABLE_SHA
    proof = json.loads((SOURCE / 'ROOT_REVIEW.json').read_bytes())
    assert proof['status'] == 'ROOT_ELEVEN_METHOD_NATIVE1100_DESCRIPTIVE_VALIDATION_TABLE_ADOPTED'
    assert proof['records'] == 1100 and not proof['final_test'] and not proof['full17_complete']
    before = {name: sha(SOURCE / name) for name in proof['files_sha256'] if name != 'TABLES.md'}
    assert before == {name: digest for name, digest in proof['files_sha256'].items() if name != 'TABLES.md'}
    formatted, highlight_checks = highlight_table(original)
    (SOURCE / 'TABLES.md').write_bytes(formatted)
    panels = parse_panels((SOURCE / 'TABLES.md').read_text(encoding='utf-8'))
    out = HERE / 'celeba_eleven_method_native.pdf'
    previous = HERE / 'previous_unhighlighted.pdf'
    if out.exists() and not previous.exists():
        shutil.copy2(out, previous)
    width, height = landscape(A3)
    margin = 34
    c = canvas.Canvas(str(out), pagesize=(width, height), pageCompression=1)
    c.setTitle('CelebA: eleven-method native validation tables')
    c.setAuthor('GuardFed revision evidence')
    page_checks = []
    all_cells = []
    note_style = ParagraphStyle('notes', fontName='Times-Roman', fontSize=8.7,
                               leading=11, textColor=colors.HexColor('#333333'))
    cell_style = ParagraphStyle('numeric', fontName='Times-Roman', fontSize=8.2,
                               leading=9.5, alignment=1)
    for page, (panel, seed_label) in enumerate(PANELS, 1):
        c.setFont('Times-Bold', 17)
        c.drawString(margin, height - 44, 'CelebA - eleven-method native validation comparison')
        c.setFont('Times-Roman', 11)
        section = 'Main table' if panel == 'ten' else 'Appendix sensitivity analysis'
        c.drawString(margin, height - 64, section + ' | ' + seed_label + ' | mean +/- sample SD (ddof=1)')
        c.setFont('Times-Italic', 9.5)
        c.drawString(margin, height - 80,
                     'Best: bold. Second-best: underlined. Fairness ranking requires mean ACC >80%; dagger marks ineligible values. Displayed ties share rank.')
        y = height - 107
        page_cells = []
        for distribution in ('IID', 'non-IID'):
            methods, values = panels[distribution + ' / ' + panel]
            c.setFont('Times-Bold', 12)
            c.drawString(margin, y, distribution + (' (alpha=5000)' if distribution == 'IID' else ' (alpha=5)'))
            y -= 10
            top = ['Method'] + [attack if metric == 0 else '' for attack in ATTACKS for metric in range(3)]
            sub = [''] + ['ACC (%)', 'AEOD', 'ASPD'] * 5
            data = [top, sub] + [[method] + [Paragraph(value.replace('**', '<b>', 1).replace('**', '</b>', 1)
                                                      .replace('†', '<super>†</super>'), cell_style)
                                            for value in values[method]] for method in methods]
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
                          rowHeights=[20, 18] + [18.5] * 11)
            table.setStyle(TableStyle(styles))
            tw, th = table.wrap(width - 2 * margin, y)
            table.drawOn(c, margin, y - th)
            y -= th + 30
            page_cells += [plain_cell(value) for method in methods for value in values[method]]
        notes = [
            '<b>Reading:</b> Higher ACC and lower AEOD/ASPD are better. Best/second-best eligible displayed means are <b>bold</b>/<u>underlined</u>; displayed ties share rank. † retains a disparity whose corresponding mean ACC is <=80%, excluding it only from fairness ranking. ACC ranking and all seeds/means/sample SDs are unchanged. Eligibility uses unrounded mean ACC; SD does not rank. This post-hoc display rule was author-approved on 2026-10-12, not preregistered. Marks do not establish significance or undefined metrics. ACC is percent; gaps are [0,1]; AEOD is the absolute TPR gap, not full equalized odds. Shading identifies GuardFed.',
            '<b>Method scope:</b> Native predictions preserve each method\'s own postprocessing; this does not isolate aggregation effects. FairFed/FairGuard/composition, FedAA and LASA are documented adaptations. LoGoFair-DP uses fitted demographic-parity postprocessing, 20 image-ID virtual cohorts (not training clients), fit seed 1719 and 30 postprocessing rounds. FLGMM uses the author-code full-model/largest-cluster adaptation and native uncalibrated outputs.',
            '<b>Coverage and provenance:</b> Eleven methods, two distributions, five conditions and 10 seeds give 1100 evaluation records, not 1100 independent training models: LoGoFair reuses FedAvg checkpoints. Backbone checkpoints are terminal round 70. FLGMM contributes 96 new and 4 explicitly reused screen records; its selected recipe is warmup 20/control width 2.0/local learning rate 0.001. Original mixed training/replay environments and separate CPU saved-state verification remain disclosed; no cross-runtime equivalence is claimed.',
            '<b>Selection boundary:</b> Seed 91001 participated in validation selection; other validation seeds and historical test metadata were exposed. The 9/6-seed panels are sensitivity views, not untouched test cohorts. No score ranking, significance, final-test or complete 17-method claim is made. This display does not select the manuscript\'s final primary endpoint.',
        ]
        y -= 1
        for note in notes:
            p = Paragraph(note, note_style)
            _, ph = p.wrap(width - 2 * margin, height)
            p.drawOn(c, margin, y - ph)
            y -= ph + 7
        assert y > 36, ('Page overflow', page, y)
        c.setFont('Times-Roman', 8)
        c.drawString(margin, 22, 'Source: root-adopted eleven-method native1100 display table; no new statistics, fits, inference or training.')
        c.drawRightString(width - margin, 22, f'{page} / 3')
        c.showPage()
        page_checks.append(dict(page=page, panel=panel, display_cells=len(page_cells), content_lower_y=y))
        all_cells += page_cells
    c.save()
    doc = fitz.open(out)
    assert len(doc) == 3
    render = HERE / 'preview'
    render.mkdir(exist_ok=True)
    for page, check in zip(doc, page_checks):
        # Compare every displayed numeric pair as a multiset: no dependence on extraction order.
        rendered = collections.Counter(PAIR.findall(page.get_text()))
        expected = collections.Counter(PAIR.fullmatch(cell).groups() for cell in all_cells[(check['page']-1)*330:check['page']*330])
        assert rendered == expected, (check['page'], rendered - expected, expected - rendered)
        assert all(0 <= span['bbox'][0] < span['bbox'][2] <= width and 0 <= span['bbox'][1] < span['bbox'][3] <= height
                   for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans'])
        page.get_pixmap(matrix=fitz.Matrix(1.15, 1.15), alpha=False).save(render / f'page-{check["page"]}.png')
        check.update(numeric_pairs_extracted_exact=True, text_spans_within_page=True)
    assert len(all_cells) == 990
    assert before == {name: sha(SOURCE / name) for name in before}
    result = dict(status='HIGHLIGHT_NUMERIC_AND_PAGE_BOUNDS_PASS_VISUAL_REVIEW_PENDING',
                  source_root_sha256=sha(SOURCE / 'ROOT_REVIEW.json'), original_table_sha256=TABLE_SHA,
                  formatted_table_sha256=sha(SOURCE / 'TABLES.md'), highlight=highlight_checks,
                  source_member_hashes=before, pdf_sha256=sha(out), pages=page_checks,
                  displayed_mean_SD_pairs=990, displayed_numeric_values=1980,
                  new_statistics=0, new_inference=0, new_fit=0, new_training=0,
                  final_test=False, full17_complete=False, scientific_source_bytes_unchanged=True,
                  original_table_recovered_exactly_by_removing_highlights=True,
                  previous_pdf_sha256=sha(previous),
                  historical_checks='ROOT_REVIEW, NUMERIC_LAYOUT_VERIFICATION, SPATIAL_CELL_VERIFICATION and VISUAL_QA predate this formatting edit')
    (HERE / 'HIGHLIGHT_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
