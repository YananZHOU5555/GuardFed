"""Pure table function draft; caller must supply strict/offserver accepted records.
No reader, CLI, acceptance, runtime or output action exists in this draft.
Original statistic/summarize implementations and seed panels remain unchanged.
"""
from types import SimpleNamespace

def need(ok,message):
    if not ok:raise ValueError(message)
source=SimpleNamespace(VIEWS=('native','raw','shared_calibration'),need=need)
PANELS=[('All 10 seeds',list(range(91001,91011))),('Exclude selection seed: 9 seeds',list(range(91002,91011))),('Seeds 91005–91010: 6 seeds',list(range(91005,91011)))]

def panels(records, original):
    expected=[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign'),('non-IID','F Flip')]
    output=[];coverage={};paired={}
    for view in source.VIEWS:
        flat=[{k:r[k] for k in ('id','variant','distribution','attack','seed')} |
            dict(accuracy_pct=100*r['views'][view]['accuracy'],aeod=r['views'][view]['aeod'],aspd=r['views'][view]['aspd']) for r in records]
        summary=original.summarize(flat)
        complete=[(r['distribution'],r['attack']) for r in summary['paired_per_scene'] if r['variant']=='minus_C' and r['complete']]
        source.need(complete==expected,'Only five IID plus non-IID Benign/F Flip10 scenes are publishable')
        coverage[view]=[{k:r[k] for k in ('variant','distribution','attack','n','expected_n','complete','seeds')} for r in summary['paired_per_scene'] if r['variant']=='minus_C']
        paired[view]=summary['paired_per_seed']
        for label,seeds in PANELS:
            rows=[]
            for dist,attack in complete:
                for variant in ('Full','minus_C','minus_C minus Full'):
                    candidates=summary['paired_per_seed'] if variant.endswith(' minus Full') else flat
                    selected=[r for r in candidates if r['distribution']==dist and r['attack']==attack and r['seed'] in seeds
                        and r['variant']==('minus_C' if variant.endswith(' minus Full') else variant)]
                    source.need({r['seed'] for r in selected}==set(seeds),'Matched seed panel incomplete')
                    rows.append(dict(distribution=dist,attack=attack,variant=variant,**original.statistic(selected,len(seeds))))
            output.append(dict(view=view,label=label,seeds=seeds,rows=rows))
    return output,coverage,paired


def aggregate_panels(records, original):
    """Each seed contributes once, after an equal mean over its five IID scenes."""
    import math
    scenes=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
    by={(r['variant'],r['attack'],r['seed']):r for r in records}
    need(len(by)==100 and len(records)==100,'Exact Full50/C50 required')
    output=[]
    for view in source.VIEWS:
        within={}
        for variant in ['Full','minus_C']:
            for seed in range(91001,91011):
                rs=[by[variant,attack,seed] for attack in scenes]
                need(all(r['distribution']=='IID' for r in rs),'Only five IID scenes')
                within[variant,seed]={m:math.fsum((100*r['views'][view]['accuracy'] if m=='accuracy_pct' else r['views'][view][m]) for r in rs)/5 for m in ['accuracy_pct','aeod','aspd']}
        for label,seeds in PANELS:
            rows=[]
            for variant in ['Full','minus_C','minus_C minus Full']:
                selected=[]
                for seed in seeds:
                    values={m:within['minus_C',seed][m]-within['Full',seed][m] for m in ['accuracy_pct','aeod','aspd']} if variant.endswith(' minus Full') else within[variant,seed]
                    selected.append(dict(seed=seed,**values))
                rows.append(dict(distribution='IID',attack='Five-scene equal mean within seed',variant=variant,**original.statistic(selected,len(seeds))))
            output.append(dict(view=view,label=label,seeds=seeds,rows=rows))
    return output
