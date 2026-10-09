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
    expected=[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA')]
    output=[];coverage={};paired={}
    for view in source.VIEWS:
        flat=[{k:r[k] for k in ('id','variant','distribution','attack','seed')} |
            dict(accuracy_pct=100*r['views'][view]['accuracy'],aeod=r['views'][view]['aeod'],aspd=r['views'][view]['aspd']) for r in records]
        summary=original.summarize(flat)
        complete=[(r['distribution'],r['attack']) for r in summary['paired_per_scene'] if r['variant']=='minus_C' and r['complete']]
        source.need(complete==expected,'Only exact C IID Benign10, F Flip10, FedSA10 and S-DFA10 scenes are publishable')
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
