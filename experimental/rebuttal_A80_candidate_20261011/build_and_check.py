"""A80 author-review increment; original document invariant checks reused unchanged."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,re,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
PINS=json.loads((HERE/'SOURCE_PINS.json').read_bytes())
DOCS=('rebuttal_integrated_20261011.md','manuscript_insertions_integrated_20261011.md')
BASE=str(Path(PINS['tables']['path']).parent.as_posix())+'/'
def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(key):
    pin = PINS[key]
    data = Path(pin["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin["sha256"] and len(data) == pin["bytes"], key
    return json.loads(data)

def save(name, value):
    (HERE / name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))

def original_checks():
    """Load exact helper functions and the original document-invariant loop, not its old cohort execution."""
    source_path = PINS["A50_build_checker"]["path"]
    source = Path(source_path).read_bytes().decode()
    assert file_sha(source_path) == PINS["A50_build_checker"]["sha256"]
    tree = ast.parse(source)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("sha", "text_sha", "require", "pointer")]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), source_path, "exec"), globals())
    writing_path = PINS["original_writing_checker"]["path"]
    writing = Path(writing_path).read_bytes().decode()
    assert file_sha(writing_path) == PINS["original_writing_checker"]["sha256"]
    functions = [node for node in ast.parse(writing).body if isinstance(node, ast.FunctionDef) and node.name in ("sentences", "quotations")]
    scope = dict(re=re)
    exec(compile(ast.Module(body=functions, type_ignores=[]), writing_path, "exec"), scope)
    globals()["old"] = SimpleNamespace(**{node.name: scope[node.name] for node in functions})
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check")
    first = next(i for i, node in enumerate(original.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "manifest" for target in node.targets))
    body = original.body[first:-1]
    assert isinstance(original.body[-1], ast.Return)
    # These statements are executed unchanged; the old A50-specific facts/old checker.run() are not rerun.
    loop_sha = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    returned = ast.parse("return {'documents': reports, 'new_link_targets_checked_locally': sorted(new_targets)}").body[0]
    fn = ast.FunctionDef(name="check_documents_original", args=original.args, body=body + [returned],
                         decorator_list=[], returns=None, type_comment=None)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, source_path + " [unchanged document checks]", "exec"), globals())
    return {"source_sha256": PINS["A50_build_checker"]["sha256"], "document_check_AST_sha256": loop_sha,
            "document_check_statements_reused_unchanged": len(body),
            "original_quotation_sentence_helpers_sha256": PINS["original_writing_checker"]["sha256"],
            "old_cohort_checks_rerun": False}

REUSE=original_checks()

def interpretation_guard(panels):
    expected={
        ('F Flip','native'):['lower','lower','higher'],
        ('F Flip','shared_calibration'):['lower','lower','higher'],
        ('F Flip','raw'):['lower','higher','higher'],
        ('FedSA','native'):['higher','higher','higher'],
        ('FedSA','shared_calibration'):['higher','higher','higher'],
        ('FedSA','raw'):['higher','lower','lower'],
    }
    selected=[p for p in panels if p['seed_count']==10]
    require(len(selected)==6 and {(p['attack'],p['view']) for p in selected}==set(expected),'Six ten-seed interpretation bindings')
    for p in selected:
        require(p['directions']==expected[(p['attack'],p['view'])],'Actual A80 signs must support interpretation')
    return selected

def build():
    root, table = read('A80_root'), read('tables')
    require(root['root_adoption'] and root['paired_models']==80 and root['preserved_records']==160, 'Require actual adopted A80')
    facts, directions = [], []
    metrics=('accuracy_pct','aeod','aspd')
    def panel_index(view,n):
        found=[i for i,p in enumerate(table['panels']) if p['view']==view and len(p['seeds'])==n]
        require(len(found)==1,'Unique fixed view/panel');return found[0]
    def row_index(panel,attack):
        found=[i for i,r in enumerate(table['panels'][panel]['rows']) if (r['distribution'],r['attack'],r['variant'])==('non-IID',attack,'minus_A minus Full')]
        require(len(found)==1,'Unique adopted paired-difference row');return found[0]
    def pair(view,attack,metric):
        i=panel_index(view,10);j=row_index(i,attack);path=f'/panels/{i}/rows/{j}/{metric}'
        value=pointer(table,path);digits=3 if metric=='accuracy_pct' else 5
        display=f"{value['mean']:+.{digits}f} ± {value['sample_sd_ddof1']:.{digits}f}"
        shared=None
        if view=='native':
            k=panel_index('shared_calibration',10);z=row_index(k,attack)
            shared=f'/panels/{k}/rows/{z}/{metric}';require(pointer(table,shared)==value,'Native/shared display equality')
        facts.append(dict(source='tables',attack=attack,view=view,mean_pointer=path+'/mean',sd_pointer=path+'/sample_sd_ddof1',
                          display=display,digits=digits,mean=value['mean'],sd=value['sample_sd_ddof1'],shared_pointer=shared))
        return display
    def direction(value):return 'lower' if value<0 else 'higher' if value>0 else 'unchanged'
    evidence=[]
    for attack in ('F Flip','FedSA'):
        native=' / '.join(pair('native',attack,m) for m in metrics)
        raw=' / '.join(pair('raw',attack,m) for m in metrics)
        clauses=[]
        for view in ('native','raw','shared_calibration'):
            for n in (10,9,6):
                i=panel_index(view,n);j=row_index(i,attack)
                paths={m:f'/panels/{i}/rows/{j}/{m}/mean' for m in metrics}
                words=[direction(pointer(table,paths[m])) for m in metrics]
                directions.append(dict(attack=attack,view=view,seed_count=n,seeds_pointer=f'/panels/{i}/seeds',mean_pointers=paths,directions=words))
                clauses.append(f"{view.replace('_',' ')} ({n} seeds): {' / '.join(words)}")
        evidence.append(f"**Added non-IID {attack} evidence.** The ten-seed paired differences, minus_A−Full, are {native} for native/shared calibration and {raw} for raw, in ΔACC / ΔAEOD / ΔASPD order. Across the fixed panels, the corresponding paired-mean directions are: {'; '.join(clauses)}. These directions describe panel means, not every seed, significance or component necessity.\n\n")
    interpretation=interpretation_guard(directions)
    def counts(mapping):return ', '.join(f'{key}: {value}' for key,value in mapping.items())
    devices='; '.join(f'{variant} ({counts(values)})' for variant,values in root['replay_devices'].items())
    training='; '.join(f'{variant} ({counts(values)})' for variant,values in root['training_torch'].items())
    block=f"""**Current A80 extension: three complete non-IID A scenes.** The [accepted A80 table]({BASE}TABLES.md) and [root verification]({BASE}ROOT_VERIFICATION.json) retain all six A60 scenes and add non-IID F Flip and FedSA, giving eight complete scenes and 80 Full/minus_A pairs (160 records). Each scene uses ten matched seeds. All three prediction views use the same terminal round-70 checkpoint for each model and the same 19,867 validation images. Deleting A removes its score contribution while geometric hard filtering and prediction calibration remain enabled.

"""+''.join(evidence)+f"""**Prediction-rule-dependent evidence.** In the ten-seed paired means for non-IID F Flip, deleting A lowers ACC and AEOD but raises ASPD under native and shared calibration: the utility loss accompanies an improvement in one disparity measure and a deterioration in the other. Under raw prediction, the same deletion lowers ACC and raises both gaps. For non-IID FedSA, raw prediction instead improves in all three paired-mean metrics after deleting A (higher ACC and lower AEOD/ASPD), providing a counterexample to a universal benefit from A. Native and shared calibration yield a small mean ACC increase but larger gaps in that scene. These contrasts show that the observed utility–disparity trade-offs depend on the prediction rule. They describe ten-seed paired means, not all individual seeds or statistically significant effects, and do not establish that A is indispensable. The linked table and saved values retain the separate 9- and 6-seed sensitivity panels.

**Interpretation and fixed sensitivity panels.** Values above are means ± sample SD (ddof=1); ΔACC is in percentage points and disparity differences are absolute gaps. The linked [saved values]({BASE}tables.json) retain all fixed 10/9/6-seed panels and unfavorable effects. Lower AEOD/ASPD means smaller measured disparity, while lower ACC means less utility; neither an apparent trade-off nor a small gap by itself establishes an indispensable component or a causal mechanism. Native/shared equality is a property of these saved records, not independent confirmation or an additional calibration gain. AEOD remains the absolute TPR gap, not full equalized odds.

**Preserved aggregation and evaluation boundary.** The [five-IID seed-first aggregate]({BASE}IID_SEED_FIRST.json) remains byte-identical to its earlier accepted version: scenes are averaged within each seed before the across-seed mean and sample SD. The three non-IID scenes are reported separately and do not enter that aggregate; no mean over the imbalanced eight-scene set is supplied. The nine-seed panel excludes configuration-selection seed 91001, and the six-seed panel retains 91005–91010, identically for Full and minus_A. Root-only calibration, exposed validation and prior official-test exposure remain disclosed. These subsets do not restore an untouched confirmation set.

**Current A80 comparability and remaining scope.** The adopted replay-device counts are {devices}; the recorded training-build counts are {training}. The A60 environment counts above describe its historical subset. Mixed devices, training builds and driver provenance do not establish numerical or complete-trajectory equivalence. Native/shared metric values and saved group counts coincide within all 160 records. The remaining two non-IID A scenes, S-DFA and Sp-DFA, and the other five image-control variants remain incomplete. U100 and C100 are unchanged. This is not a complete 17-method benchmark or all-component study. The final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending; no final test is claimed.

"""
    manifest=dict(status='REVERSIBLE_A80_INTEGRATION_FOR_AUTHOR_REVIEW',input_reader_root=PINS['reader_root'],
                  input_A80_root=PINS['A80_root'],documents=[],historical_scope_normalization={},new_science_block=block,new_scientific_measurements=0)
    labels={
        '**Current A60 extension: the first complete non-IID A scene.**':'**Historical A60 extension: the first complete non-IID A scene.**',
        '**Added non-IID Benign evidence.**':'**Historical A60 non-IID Benign evidence.**',
        '**Current A60 comparability and remaining scope.**':'**Historical A60 comparability and remaining scope.**',
        'The added A60 non-IID Benign comparison supplies a further counterexample':'The historical A60 non-IID Benign comparison supplies a further counterexample',
        'The current evidence therefore supports bounded utility–disparity trade-offs, without isolating a causal mechanism or completing the other four non-IID A scenes.':'The historical A60 evidence therefore supports bounded utility–disparity trade-offs, without isolating a causal mechanism or completing the other four non-IID A scenes.',
    }
    manifest['historical_scope_normalization']={after:before for before,after in labels.items()}
    for name in DOCS:
        source=Path(PINS[name]['path']).read_bytes().decode('utf-8');value=source;operations=[]
        def edit(before,after,reason):
            nonlocal value
            require(value.count(before)==1,'Nonunique exact span: '+reason);at=value.index(before)
            op=dict(reason=reason,offset_codepoints=at,before=before,after=after,document_sha256_before=text_sha(value))
            value=value[:at]+after+value[at+len(before):];op['document_sha256_after']=text_sha(value);operations.append(op)
        for before,after in labels.items():
            if before in value:edit(before,after,'Historical A60 scope; old evidence unchanged')
        anchor='**Root/heterogeneity paragraph.**' if name.startswith('manuscript') else next(line for line in value.splitlines() if line.startswith('### R3.3 —'))
        edit(anchor,block+anchor,'Add actual adopted A80 values and bounded interpretation')
        history=next(line for line in value.splitlines() if line.startswith('The [accepted A60 comparison]'))+'\n\n'
        current=f'The [accepted A80 comparison]({BASE}TABLES.md) covers eight complete scenes: all five IID scenes plus non-IID Benign, F Flip and FedSA, with 80 matched Full/minus_A pairs and fixed 10/9/6-seed panels. S-DFA and Sp-DFA remain the two incomplete non-IID A scenes; the other five image controls remain pending. The five-IID aggregate, U100 and C100 are unchanged; no mixed-distribution aggregate or final test is supplied.\n\n'
        edit(history,current+'**Historical A60 accepted snapshot.**\n\n'+history,'Update current snapshot and preserve entire old paragraph')
        if name.startswith('rebuttal'):
            ae=next(line for line in value.splitlines() if line.startswith('The A-deletion extension now covers six complete CelebA scenes'))
            newae='The A-deletion extension now covers eight complete CelebA scenes and 80 matched Full/minus_A pairs, including the new non-IID F Flip and FedSA comparisons. R3.2 reports the actual paired differences and every fixed sensitivity panel; R3.7 retains their bounded interpretation and unfavorable effects. Two non-IID A scenes and five other image-control variants remain pending; no all-component necessity or complete benchmark claim is made.'
            edit(ae,newae,'Update current Associate Editor scope')
            history_anchor='**Historical A60 accepted snapshot.**\n\n'+history
            edit(history_anchor,history_anchor+'**Historical A60 Associate Editor summary.**\n\n'+ae+'\n\n','Preserve old AE text and all numbers')
            overview=next(line for line in value.splitlines() if line.startswith('The current A comparison covers five IID scenes plus non-IID Benign'))
            overview_new='The current A comparison covers five IID scenes plus non-IID Benign, F Flip and FedSA, with 80 paired checkpoints; S-DFA and Sp-DFA and five other control variants remain pending.'
            edit(overview,overview_new+'\n\n**Historical A60 mechanism overview.**\n\n'+overview,'Synchronize overview while retaining old scope')
            r37='The tabular COMPAS counterexamples above remain unchanged.'
            edit(r37,'The added A80 F Flip and FedSA evidence in R3.2 reports each view and fixed-panel direction explicitly. These added comparisons do not convert the earlier counterexamples into evidence of universal component benefit or identify an isolated causal mechanism. '+r37,'Add bounded A80 interpretation without predicting directions')
            p2=next(line for line in value.splitlines() if line.startswith('| P2 —'))
            before='A60 adds non-IID Benign, giving six complete scenes and 60 pairs; complete the other four non-IID A scenes and the other five incomplete image controls.'
            after='The historical A60 snapshot adds non-IID Benign, giving six complete scenes and 60 pairs; its then-pending scope was the other four non-IID A scenes and the other five incomplete image controls. A80 now adds non-IID F Flip and FedSA, giving eight scenes and 80 pairs; complete non-IID S-DFA/Sp-DFA and the other five image controls.'
            require(before in p2,'Original P2 exact span');edit(p2,p2.replace(before,after),'Update only P2 completed/pending scope')
            manifest['historical_scope_normalization'][after]=before
        target=name.replace('_20261011.md','_A80_reader_20261011.md')
        require(not (HERE/target).exists(),'Do not overwrite prior candidate')
        (HERE/target).write_bytes(value.encode())
        manifest['documents'].append(dict(source=PINS[name]['path'],source_sha256=text_sha(source),candidate=(HERE/target).as_posix(),candidate_sha256=text_sha(value),operations=operations))
    save('EDIT_MANIFEST.json',manifest)
    paths=('/paired_models','/preserved_records','/complete_scenes','/complete_IID_scenes','/complete_nonIID_scenes','/replay_devices','/training_torch','/seed_panels','/IID_seed_first_JSON_bytes_exact','/aggregate_scope','/test','/primary_endpoint_selected')
    save('FACT_BINDINGS.json',dict(mean_sd_pairs=facts,new_nonIID_direction_panels=directions,
        interpretation_ten_seed_bindings=interpretation,
        scope=[dict(source='A80_root',pointer=p,expected=pointer(root,p)) for p in paths],
        native_shared_equality=dict(source='summary',pointer='/native_shared_metrics_and_counts_exact',expected=True),source_statistics_recomputed=False))

def check():
    for key,pin in PINS.items():
        data=Path(pin['path']).read_bytes();require(hashlib.sha256(data).hexdigest()==pin['sha256'] and len(data)==pin['bytes'],'Input drift: '+key)
    root,reader,table,summary=read('A80_root'),read('reader_root'),read('tables'),read('summary')
    require(reader['author_review_only'] and reader['A60_incorporated'] and reader['documents_sha256']=={n:PINS[n]['sha256'] for n in DOCS},'A60 reader identity')
    require(root['root_adoption'] and root['paired_models']==80 and root['preserved_records']==160 and root['complete_scenes']==8 and root['complete_IID_scenes']==5 and root['complete_nonIID_scenes']==['Benign','F Flip','FedSA'],'Actual A80 adoption/scope')
    require(root['seed_panels']==[10,9,6] and root['IID_seed_first_JSON_bytes_exact'] and not root['test'] and not root['primary_endpoint_selected'],'Fixed panels and boundary')
    for key in ('tables','iid_seed_first','summary','official_table','saved_verification'):
        require(root['files_sha256'][Path(PINS[key]['path']).name]==PINS[key]['sha256'],'A80 canonical/root SHA')
    facts=json.loads((HERE/'FACT_BINDINGS.json').read_bytes())
    require(len(facts['mean_sd_pairs'])==12 and len(facts['new_nonIID_direction_panels'])==18,'Exact added evidence scope')
    for fact in facts['mean_sd_pairs']:
        mean,sd=pointer(table,fact['mean_pointer']),pointer(table,fact['sd_pointer'])
        require((mean,sd)==(fact['mean'],fact['sd']),'Mean/SD pointer')
        require(f"{mean:+.{fact['digits']}f} ± {sd:.{fact['digits']}f}"==fact['display'],'Numeric display')
        if fact['shared_pointer']:require(pointer(table,fact['shared_pointer'])=={'mean':mean,'sample_sd_ddof1':sd},'Shared/native equality')
    for panel in facts['new_nonIID_direction_panels']:
        words=['lower' if pointer(table,p)<0 else 'higher' if pointer(table,p)>0 else 'unchanged' for p in panel['mean_pointers'].values()]
        require(words==panel['directions'],'Actual fixed-panel directions')
        require(len(pointer(table,panel['seeds_pointer']))==panel['seed_count'],'Fixed panel seed identity')
    require(interpretation_guard(facts['new_nonIID_direction_panels'])==facts['interpretation_ten_seed_bindings'],'Six actual ten-seed interpretation facts')
    for fact in facts['scope']+[facts['native_shared_equality']]:require(pointer(root if fact['source']=='A80_root' else summary,fact['pointer'])==fact['expected'],'Root/scope fact')
    globals()['bindings']=facts
    result=check_documents_original()
    result.update(status='PASS_A80_READER_INTEGRATION_AUTHOR_REVIEW_ONLY',original_document_checks_actually_executed_once=True,
        original_check_reuse=REUSE,new_mean_sd_pairs_bound_to_JSON_pointers=12,fixed_10_9_6_direction_panels_bound=18,explicit_ten_seed_interpretation_sign_bindings=6,
        root_reader_sha256=PINS['reader_root']['sha256'],A80_root_sha256=PINS['A80_root']['sha256'],
        source_statistics_recomputed=False,self_check=True,independent_review=False,canonical_written=False,
        manuscript_applied=False,author_review=True,primary_endpoint_selected=False,final_test=False,new_CNN_fit_training_SSH_Git_bulk_writes=0,
        manifest_sha256=file_sha(HERE/'EDIT_MANIFEST.json'),fact_bindings_sha256=file_sha(HERE/'FACT_BINDINGS.json'),checker_sha256=file_sha(__file__))
    return result

if __name__=='__main__':
    require(sys.argv[1:] in ([],['--check']),'Only --check accepted')
    started=datetime.now(timezone.utc).isoformat()
    if not sys.argv[1:]:
        require(not any((HERE/n).exists() for n in ('EDIT_MANIFEST.json','SELF_CHECK.json','FACT_BINDINGS.json')),'One fresh editorial build only')
        build()
    result=check()
    if not sys.argv[1:]:
        save('SELF_CHECK.json',result)
        save('ACTUAL_COMMAND.json',dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),command='python -B tmp/rebuttal_A80_candidate_20261011/build_and_check.py',exit_code=0,complete_successful_candidate_checks=1,science_statistics_recomputed=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))
