from pathlib import Path
import argparse,ast,hashlib,json,runpy,sys,traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
PRIOR=R/'tmp/celeba_flgmm_fullcoverage_delta_after28_20261010'
ORIGINAL=R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'
S=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
PREVIOUS='b508fd4d0f0d36c05e4ea145ff960551d95d415b79db6c19f4d2da95935da5cf'
ROOTPROOF='5e95872f3216a9842e95c534ba87629b2211c298d72a95d810bdaf9428021051'
IDS=['FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91004_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91006_fullcoverage']
def reused():
    seal = read(PRIOR/'DELIVERY_FILES_SHA256.json')
    assert sha(PRIOR/'run_once.py') == seal['files']['run_once.py']['sha256']
    ns = runpy.run_path(str(PRIOR/'run_once.py'), run_name='sealed_after28_transport')
    g = ns['reused']()
    original_cpu106 = g['cpu106']
    def cpu110(text):
        text = original_cpu106(text)
        for a,b in [('CPU106','CPU110'),('{106}','{110}'),('106 in a','110 in a'),
                    ('106 not in aff','110 not in aff'),('106 in aff','110 in aff'),
                    ('CPU=106','CPU=110'),('-c 106 ','-c 110 '),("'106'","'110'"),
                    ('(106,1,10)','(110,1,10)'),('helper_CPU=106','helper_CPU=110')]:
            text = text.replace(a,b)
        return text
    g.update(H=H, S=S, PREVIOUS=PREVIOUS, ROOTPROOF=ROOTPROOF, cpu106=cpu110)
    source = (ORIGINAL/'run_once.py').read_text('utf8'); tree = ast.parse(source)
    changes = {
        'parent':[('{22+len(ids)}','{38+len(ids)}',1)],
        'verify':[('{22+len(ids)}','{38+len(ids)}',1)],
        'finalize':[
            ('total=22+n','total=38+n',1),
            ("['INITIAL_GUIDE','INITIAL_OWNER','SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']", "['SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']",1),
            ("prior['accepted_total']==22","prior['accepted_total']==38",1),
            ("p['accepted_job_ids'][:22]","p['accepted_job_ids'][:38]",1),
            ('prior_accepted_new=22','prior_accepted_new=38',1),
            ('old22_ordered_prefix_exact','old38_ordered_prefix_exact',2),
            ('accepted_before=22','accepted_before=38',1),
            ('# Actual FL96 after22:','# Actual FL96 after38:',1),
            ('ordered prior22','ordered prior38',1)]}
    for name,rebindings in changes.items():
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        body = ast.get_source_segment(source,node)
        for before,after,count in rebindings:
            assert body.count(before)==count,(name,before)
            body = body.replace(before,after)
        exec(compile(cpu110(body),str(ORIGINAL/'run_once.py')+'[COUNT_CPU110_PATH_ONLY]', 'exec'),g)
    return g

if __name__=='__main__':
 try:
  p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','verify','finalize']);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args()
  assert sha(a.review)==a.review_sha256
  review=read(a.review);assert review['source_adoptable'] is True and review['exact_selected_ids']==IDS and review['prepared_seal_sha256']==sha(H/'PREPARED_FILES_SHA256.json')
  for n,pin in read(H/'PREPARED_FILES_SHA256.json')['files'].items():assert sha(H/n)==pin['sha256']
  g=reused();g['source_check']()
  assert sha(g['B']/'LATEST_BACKUP.json')==sha(H/'PREVIOUS_LATEST.json')
  if a.phase=='collect':
   save('F_VOLUME_BEFORE_COLLECT.json',g['storage'](0));g['parent'](IDS).collect()
  else:g[a.phase]()
 except BaseException as e:
  if not isinstance(e,SystemExit):save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
