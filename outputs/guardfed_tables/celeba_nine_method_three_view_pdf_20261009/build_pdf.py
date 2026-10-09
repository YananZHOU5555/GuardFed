"""Display-only wrapper for nine accepted TeX panels; no statistics or scientific runtime."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys
import fitz
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/FILES_SHA256.json').is_file())
SOURCE=ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
SOURCE_SEAL='1bc00ab3e2f5b69f915730dc1b94fb9da49cc8cc69f9756f8b2ea092b3e33c8e'
VIEWS=('raw','native','shared_calibration')
PANELS=('ten_seed','exclude_selection','matching_six')
SEEDS={'ten_seed':'10 shared seeds: 91001--91010','exclude_selection':'9 shared seeds: 91002--91010','matching_six':'6 shared seeds: 91005--91010'}
TITLES={'raw':'Raw argmax','native':'Native method rules','shared_calibration':'Shared calibration'}
PAIR=re.compile(rb'(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

def main():
    assert sha(SOURCE/'FILES_SHA256.json')==SOURCE_SEAL
    manifest=json.loads((SOURCE/'FILES_SHA256.json').read_bytes())
    before={name:sha(SOURCE/name) for name in manifest}
    assert len(before)==67 and all(before[name]==row['sha256'] for name,row in manifest.items())
    assert shutil.which('xelatex')
    rows=[]
    wrapper=r'''\documentclass[11pt]{article}
\usepackage[a3paper,landscape,margin=12mm]{geometry}
\usepackage{multirow,graphicx}
\setlength{\parindent}{0pt}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{8pt}
\makeatletter
\renewenvironment{table*}[1][]{\par\begingroup\setlength{\textwidth}{\dimexpr\linewidth-1pt\relax}\def\@captype{table}}{\par\endgroup}
\makeatother
\begin{document}
'''
    for view in VIEWS:
        for panel in PANELS:
            name='celeba_iid_noniid_'+panel+'.tex';source=SOURCE/view/name
            raw=source.read_bytes();assert sha(source)==manifest[view+'/'+name]['sha256']
            copied=HERE/'fragments_original'/view/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes(raw)
            unique=re.sub(rb'(\\label\{)([^}]+)(\})',lambda m:m[1]+b'pdf-'+view.encode()+b'-'+m[2]+m[3],raw)
            assert len(re.findall(rb'\\label\{',raw))==1
            compiled=HERE/'fragments_unique'/view/name;compiled.parent.mkdir(parents=True,exist_ok=True);compiled.write_bytes(unique)
            assert PAIR.findall(raw)==PAIR.findall(unique) and len(PAIR.findall(raw))==270
            assert re.sub(rb'\\label\{[^}]+\}',b'',raw)==re.sub(rb'\\label\{[^}]+\}',b'',unique)
            number=len(rows)+1
            if rows:wrapper+='\n\\newpage\n'
            wrapper+=r'{\small GuardFed / CelebA nine-method validation tables / accepted 2026-10-09 snapshot}\par\medskip'+'\n'
            wrapper+=r'{\LARGE\bfseries '+TITLES[view]+r'}\hfill{\large Panel '+str(number)+r' / 9}\par\smallskip'+'\n'
            wrapper+=r'{\large '+SEEDS[panel]+r'; round 70; validation split (19,867 images)}\par\medskip'+'\n'
            wrapper+='\\input{fragments_unique/'+view+'/'+name+'}\n'
            wrapper+=r'''\par\medskip
{\footnotesize\textbf{Decision rules.} Raw: margin $>0$ (ties class 0). Native: GuardFed-AD2+ uses its saved clean-root group thresholds; the eight baselines use argmax. Shared: all nine use saved clean-root group thresholds ($\geq$); no validation-label fitting.\par
\textbf{Reading scope.} Validation was exposed to recipe selection, including seed 91001. Mixed CPU/GPU replay is not a uniform-device final comparison. The native/shared primary endpoint remains pending.}
'''
            rows.append(dict(page=number,view=view,panel=panel,seeds=SEEDS[panel],source_relative=view+'/'+name,source_sha256=sha(source),original_copy_sha256=sha(copied),unique_label_copy_sha256=sha(compiled),mean_sd_cells=270,original_bold_commands=len(re.findall(rb'\\(?:textbf|mathbf)\{',raw))))
    wrapper+='\n\\end{document}\n';(HERE/'wrapper.tex').write_text(wrapper,encoding='utf-8',newline='\n')
    save(HERE/'source_bindings.json',dict(source_directory=str(SOURCE),source_seal_sha256=SOURCE_SEAL,source_member_count=67,source_before=before,panels=rows,only_fragment_change='Unique label prefix; all original bytes also retained'))
    command=['xelatex','--disable-installer','--interaction=nonstopmode','--halt-on-error','--file-line-error','--no-shell-escape','--synctex=0','--jobname=celeba_nine_method_three_view','wrapper.tex']
    runs=[]
    for number in (1,2):
        result=subprocess.run(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (HERE/('build_pass'+str(number)+'.stdout.log')).write_bytes(result.stdout)
        runs.append(dict(pass_number=number,returncode=result.returncode,command=command))
        result.check_returncode()
    log=(HERE/'celeba_nine_method_three_view.log').read_text(encoding='utf-8',errors='replace')
    errors={term:log.count(term) for term in ('Missing character:','Undefined control sequence','LaTeX Error','Overfull \\hbox','Overfull \\vbox','undefined references','multiply defined')}
    assert not any(errors.values()),errors
    pdf=HERE/'celeba_nine_method_three_view.pdf';doc=fitz.open(pdf);assert len(doc)==9
    qa=HERE/'qa';qa.mkdir(exist_ok=True);pages=[]
    for page,row in zip(doc,rows):
        assert page.rect.width>page.rect.height
        text=page.get_text();assert len(text)>4000
        source_pairs=[(a.decode(),b.decode()) for a,b in PAIR.findall((SOURCE/row['source_relative']).read_bytes())]
        pdf_pairs=re.findall(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)',text)
        assert pdf_pairs==source_pairs,(row['page'],len(pdf_pairs))
        for method in ('FedAvg','FairFed','Median','FLTrust','FedAA-DDPG','LASA','FairGuard','FLTrust+FairGuard','GuardFed-AD2+'):
            assert method in text,(row['page'],method)
        for label in ('Category','Method','Metric','IID: Benign','non-IID: Benign','19867','91010'):
            assert label in text.replace(',',''),(row['page'],label)
        assert str({'ten_seed':10,'exclude_selection':9,'matching_six':6}[row['panel']])+' shared seeds' in text
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        decimal_sizes=[s['size'] for s in spans if s['text']=='.' and s['font'].startswith('CMMI')]
        assert decimal_sizes and min(decimal_sizes)>=8
        assert all(s['bbox'][0]>=0 and s['bbox'][1]>=0 and s['bbox'][2]<=page.rect.width+0.5 and s['bbox'][3]<=page.rect.height+0.5 for s in spans)
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(qa/('page_'+str(row['page']).zfill(2)+'.png'))
        pages.append(dict(page=row['page'],view=row['view'],panel=row['panel'],extracted_characters=len(text),mean_sd_cells_matched=len(pdf_pairs),minimum_numeric_font_pt=min(decimal_sizes),width_pt=page.rect.width,height_pt=page.rect.height,text_within_page=True))
    after={name:sha(SOURCE/name) for name in manifest};assert after==before and sha(SOURCE/'FILES_SHA256.json')==SOURCE_SEAL
    save(HERE/'verification.json',dict(status='DISPLAY_ONLY_BUILD_AND_TEXT_CHECKS_PASS_VISUAL_REVIEW_PENDING',source_67_members_unchanged=True,source_seal_sha256=SOURCE_SEAL,pdf_sha256=sha(pdf),pages=pages,numeric_mean_sd_cells_matched=2430,scalar_numbers_matched=4860,original_fragment_byte_copies=9,unique_label_only_copies=9,compile=runs,log_checks=errors,statistics_recomputed=False,new_inference=False,test_evaluation=False,primary_endpoint_selected=False))
    print(json.dumps(dict(pdf=str(pdf),pages=len(doc),cells_matched=2430,minimum_numeric_font_pt=min(x['minimum_numeric_font_pt'] for x in pages),visual_review='PENDING')))

if __name__=='__main__':main()
