"""Bounded, reversible paragraph patch; stdlib text/source checks only."""
from pathlib import Path
import collections,difflib,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
R=D.parents[3]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C100_20261010'
SCOPE=R/'tmp/celeba_added_baseline_three_view_scope_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(name,value):
    with (D/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def text(name,value):
    with (D/name).open('x',encoding='utf8',newline='\n') as f:f.write(value)
def quotes(value):return re.findall(r'\*\*Original comment \(verbatim\)\.\*\*\n\n((?:>[^\n]*\n)+)',value)

def prepare():
    assert sha(SCOPE/'FILES_SHA256.json')=='1cc32cee3f41abcc644510251a79f5751a60fdf8185533c9f0352ba89c150564'
    for name,pin in read(SCOPE/'FILES_SHA256.json')['files'].items():
        assert sha(SCOPE/name)==pin['sha256'] and (SCOPE/name).stat().st_size==pin['bytes']
    decisions=R/'docs/server_deployment_20260923/training_20260923/AUTHOR_DECISIONS_20261010.json'
    assert sha(decisions)=='aee89e8210b5aa83d8ee814d5afde4bb655f3ba559bb53ae6ebcd1ca0d46851d'
    pins=read(SCOPE/'SOURCE_PINS.json')['sources'];bridge=Path(pins['logofair_bridge']['path'])
    assert sha(bridge)==pins['logofair_bridge']['sha256']=='c1dfa5b5881407c16b0c7c37cfa17cb27c62f9d9fe125a51bdca0c77a0865e81'
    adopted=R/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
    evidence=R/'tmp/celeba_logofair32_root_adoption_20261010/COMPLETE32_REVIEW.json'
    assert sha(adopted)=='145f30628270b457d84d47aeab60825c9d4fdb23a0ae27fcd44a35a19e62ae90'
    assert sha(evidence)==read(adopted)['independent_acceptance_sha256']=='70f8cf59f11d505bd5d6fd1f9eaf7e9f64be629349bfeF3b8fbd995486c06d64'.lower()
    root=read(OLD/'ROOT_REVIEW.json')
    paths=[OLD/NAMES[0],OLD/NAMES[1],OLD/'ROOT_REVIEW.json',OLD/'FILES_SHA256.json',decisions,SCOPE/'REPORT.md',SCOPE/'REVIEW.json',SCOPE/'SOURCE_PINS.json',SCOPE/'FILES_SHA256.json',bridge,adopted,evidence]
    source_pins={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    author_link=decisions.as_posix();scope_link=(SCOPE/'REPORT.md').as_posix()
    edits=[];diff=[]
    for name in NAMES:
        old=(OLD/name).read_bytes().decode('utf8')
        assert sha(OLD/name)==root['documents_sha256'][name]
        paragraphs=[p for p in old.split('\n\n') if p.startswith('**Remaining-baseline implementation status at this evidence cutoff.**')]
        assert len(paragraphs)==1
        before=paragraphs[0]
        first="The [author decision](E:/OneDrive/文档/GuardFed/tmp/celeba_gradient_screen64_v2_20261010/AUTHOR_DECISIONS.json) accepts Huber’s practical CNN adaptation with parameter domain R^p and identity projection; it does not inherit a convex-domain or covering-number guarantee."
        updated=f"The [recorded author decisions]({author_link}) explicitly accept Huber’s practical CNN adaptation with parameter domain R^p and identity projection. We use it as an empirical CNN adaptation without transferring the original convex-domain or covering-number guarantee."
        population='It uses the author-authorized image-ID hash into 20 declared virtual cohorts, the official DP objective and root-only fitting, not real training-client fairness.'
        delegated="Following the author's delegation, we fixed image-ID hashing into 20 declared virtual cohorts, the official DP objective and root-only fitting; this adaptation does not evaluate fairness across real training clients."
        assert before.count(first)==before.count(population)==1
        after=before.replace(first,updated).replace(population,delegated)
        clarification=f"**LoGoFair prediction scope.** As distinguished in the [source compatibility review]({scope_link}), LoGoFair native is the saved `prediction` reproduced from the SHA-bound fitted `post_state.pkl`, settings, cohort mapping and FedAvg checkpoint/score cache. The cache field `valid_native_prediction` instead stores the FedAvg backbone's original argmax decision; it is not the fitted LoGoFair output. Raw/shared outputs from that same backbone and the existing shared calibration are backbone/postprocessing-replacement diagnostics and must be labelled separately from LoGoFair native. They do not preserve the official DP postprocessor. This clarification leaves its existing strict source/state/checkpoint and saved-prediction evidence unchanged; it neither defines a new postprocessor-plus-calibration composition nor adopts a combined comparison. The adaptation decisions do not complete the remaining experiments or select the final primary endpoint; the final test under a frozen protocol has not been run."
        replacement=after+'\n\n'+clarification
        anchor_line=old[:old.index(before)].count('\n')+1
        new=old.replace(before,replacement,1)
        edits.append(dict(document=(OLD/name).relative_to(R).as_posix(),source_sha256=sha(OLD/name),anchor_line=anchor_line,anchor_heading='R3.8 — Explain attacks and comparison methods' if name==NAMES[0] else 'Remaining-baseline implementation status at this evidence cutoff',old=before,new=replacement,old_sha256=hashlib.sha256(before.encode()).hexdigest(),new_sha256=hashlib.sha256(replacement.encode()).hexdigest(),proposed_document_sha256=hashlib.sha256(new.encode()).hexdigest()))
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=(OLD/name).relative_to(R).as_posix(),tofile=(OLD/name).relative_to(R).as_posix()+' [UNAPPLIED_AUTHOR_REVIEW_CANDIDATE]',n=1))
        label='REBUTTAL_PATCH.md' if name==NAMES[0] else 'MANUSCRIPT_INSERTION_PATCH.md'
        text(label,f"# {'Rebuttal response' if name==NAMES[0] else 'Manuscript insertion'} candidate — adaptation scope\n\n**AUTHOR_REVIEW — DO_NOT_SUBMIT. UNAPPLIED PATCH; NO EXPERIMENTAL ACCEPTANCE.**\n\nReplace the single paragraph beginning **Remaining-baseline implementation status at this evidence cutoff.** in [{name}]({(OLD/name).as_posix()}:{anchor_line}), then append the prediction-scope paragraph below. This leaves every original reviewer comment and all other passages unchanged.\n\n"+replacement+'\n')
    save('SOURCE_CHANGES.json',dict(status='AUTHOR_REVIEW_REVERSIBLE_UNAPPLIED_PATCH',source_pins=source_pins,changes=edits,full_documents_written=False,original_comments_modified=False))
    text('UPDATE_DIFF.patch',''.join(diff))
    source=bridge.read_text('utf8').splitlines()
    facts=[
        dict(id='H_projection',source=decisions.relative_to(R).as_posix(),pointer='/H/projection',value='identity'),
        dict(id='H_domain',source=decisions.relative_to(R).as_posix(),pointer='/H/parameter_domain',value='R^p'),
        dict(id='H_claim_limit',source=decisions.relative_to(R).as_posix(),pointer='/H/claims',value='empirical only; no inherited convex or covering-number guarantee'),
        dict(id='L_delegation',source=decisions.relative_to(R).as_posix(),pointer='/L/user_reply',value='你自己来决策'),
        dict(id='L_population',source=decisions.relative_to(R).as_posix(),pointer='/L/root_decision',value='fixed image-ID hash into 20 virtual cohorts'),
        dict(id='L_fit',source=decisions.relative_to(R).as_posix(),pointer='/L/objective',value='official DP variant with root-only fitting'),
        dict(id='L_claim_limit',source=decisions.relative_to(R).as_posix(),pointer='/L/claims',value='not real training-client fairness'),
        dict(id='LoGo_existing_search',source=adopted.relative_to(R).as_posix(),pointer='/accepted_count',value=32),
        dict(id='LoGo_existing_recipe',source=adopted.relative_to(R).as_posix(),pointer='/selected_recipe/id',value='LoGoFair-DP_07'),
        dict(id='LoGo_saved_prediction_evidence',source=evidence.relative_to(R).as_posix(),pointer='/original_strict_rechecked_n',value=32),
        dict(id='LoGo_root_only_fit_evidence',source=evidence.relative_to(R).as_posix(),pointer='/root_only_fit_source_checked',value=True),
        dict(id='primary_still_pending',source=(OLD/'ROOT_REVIEW.json').relative_to(R).as_posix(),pointer='/primary_endpoint',value='PENDING_AUTHOR'),
        dict(id='manuscript_still_unapplied',source=(OLD/'ROOT_REVIEW.json').relative_to(R).as_posix(),pointer='/manuscript_applied',value=False),
        dict(id='final_test_not_run',source=(OLD/'ROOT_REVIEW.json').relative_to(R).as_posix(),pointer='/final_test',value=False)
    ]
    segments=[]
    for identity,start,end in [('root_only_fit',174,192),('saved_prediction_distinction',265,283),('strict_reload_saved_prediction',333,362)]:
        value='\n'.join(source[start-1:end])
        segments.append(dict(id=identity,source=bridge.relative_to(R).as_posix(),first_line=start,last_line=end,text=value,sha256=hashlib.sha256(value.encode()).hexdigest()))
    save('SOURCE_POINTERS.json',dict(source_pins=source_pins,json_facts=facts,source_segments=segments,diagnostic_semantics_source=(SCOPE/'REPORT.md').relative_to(R).as_posix(),diagnostic_semantics_anchor='最小、不增推理/拟合的三列对照候选',diagnostic_semantics_is_scope_review_not_execution_or_endpoint_approval=True))
    text('DECISION_SUMMARY_ZH.md','''# 两项适配裁定：最小英文补丁

**AUTHOR_REVIEW / DO_NOT_SUBMIT；未应用正文，未改旧C100完整回复。**

Huber：作者已明确接受参数域R^p、恒等投影的CNN经验适配。补丁明确“不继承原凸域/covering-number理论保证”，不再把这一方向列为待作者裁定。

LoGoFair：作者已委托root决定，实际固定image-ID hash的20个虚拟cohort，原official DP目标、仅root标签参与拟合。这里不代表真实训练client公平性；valid用于原选参/报告，不能回流拟合。

原C100两个文件已有适配段落，本次每文件仅替换该一段（明确权威来源/委托关系）并追加一段预测语义。LoGo `valid_native_prediction`是FedAvg底座argmax；LoGo native是受checkpoint/cache/post_state/settings/mapping绑定的拟合后`prediction`。raw/shared底座诊断须单独命名，不能冒充保留LoGo官方DP后处理的原生结果，也没有据此采用新的“后处理再共享校准”组合。原bridge strict和已保存预测验收不失效，未重新执行。

保留原搜索接受/recipe、常量和不利结果、未完成完整方法覆盖、validation选择与环境史、final-primary待定及final test未运行。没有新增性能数字、统计、预测、fit或实验接受。24条原评论与所有未改段落可由精确逆替换恢复，实际pins/检查见JSON。SOURCE_CHANGES只含两个局部替换；UPDATE_DIFF可人工审阅，不自动应用，不纳入已冻结Git47。
''')

def check(write_result=False):
    delta=read(D/'SOURCE_CHANGES.json');bindings=read(D/'SOURCE_POINTERS.json')
    assert delta['source_pins']==bindings['source_pins']
    for path,pin in bindings['source_pins'].items():assert sha(R/path)==pin['sha256'] and (R/path).stat().st_size==pin['bytes']
    for fact in bindings['json_facts']:
        value=read(R/fact['source'])
        for key in fact['pointer'].strip('/').split('/'):value=value[key]
        assert value==fact['value'],fact['id']
    for segment in bindings['source_segments']:
        value='\n'.join((R/segment['source']).read_text('utf8').splitlines()[segment['first_line']-1:segment['last_line']])
        assert value==segment['text'] and hashlib.sha256(value.encode()).hexdigest()==segment['sha256']
    checks=[];links=[];patch=[]
    for change in delta['changes']:
        before=(R/change['document']).read_bytes().decode('utf8')
        assert sha(R/change['document'])==change['source_sha256'] and before.count(change['old'])==1
        assert all(hashlib.sha256(change[key].encode()).hexdigest()==change[key+'_sha256'] for key in ('old','new'))
        after=before.replace(change['old'],change['new'],1)
        assert after.count(change['new'])==1 and after.replace(change['new'],change['old'],1).encode()==(R/change['document']).read_bytes()
        assert hashlib.sha256(after.encode()).hexdigest()==change['proposed_document_sha256']
        assert quotes(before)==quotes(after)
        if change['document'].endswith('/'+NAMES[0]):assert len(quotes(after))==24
        numeric=lambda t:collections.Counter(re.findall(r'(?<![\w])[-+]?\d+(?:\.\d+)?',re.sub(r'\]\([^)]*\)',']()',t)))
        assert numeric(before)==numeric(after),'No new or removed numeric result/scope count allowed'
        for phrase in ('AUTHOR_REVIEW — DO_NOT_SUBMIT_BEFORE_FULL_COHORT','six other image-control variants remain incomplete','official-test exposure','identity projection','constant-prediction and unfavorable results','final test under a frozen final protocol has not been run'):
            assert phrase in after
        assert '`valid_native_prediction`' in change['new'] and '`prediction`' in change['new'] and 'must be labelled separately' in change['new']
        for link in re.findall(r'\]\(([^)]+)\)',change['new']):
            assert Path(link).is_absolute() and Path(link).is_file(),link
            assert str(Path(link).relative_to(R)).replace('\\','/') in bindings['source_pins']
            links.append(link)
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=change['document'],tofile=change['document']+' [UNAPPLIED_AUTHOR_REVIEW_CANDIDATE]',n=1))
        checks.append(dict(document=change['document'],anchor_line=change['anchor_line'],reverse_bytes_exact=True,original_comments_exact=True,all_numeric_results_and_counts_preserved=True))
    assert ''.join(patch)==(D/'UPDATE_DIFF.patch').read_text('utf8')
    result=dict(status='PASS_LOCAL_AUTHOR_REVIEW_PATCH_CHECKS_NO_SCIENTIFIC_ACCEPTANCE',changed_documents=2,changed_spans=2,original_comments_verbatim_and_order=24,reverse_exact_documents=2,original_source_files_unchanged=True,source_pins=len(bindings['source_pins']),json_fact_pointers=len(bindings['json_facts']),source_segments=len(bindings['source_segments']),links_checked=len(links),links=links,checks=checks,no_new_numeric_results=True,full_documents_written=False,manuscript_applied=False,new_statistics=False,new_inference=0,new_fit=0,new_experiment_acceptance=0,STATE_Git_or_shared_source_modified=False,author_decisions_resolved=True,final_primary_endpoint='PENDING_AUTHOR',DO_NOT_SUBMIT=True)
    if write_result:save('CHECK_RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('links','checks')}))

if __name__=='__main__':
    if sys.argv[1:] == ['prepare']:
        prepare();check(write_result=True)
    elif sys.argv[1:] == ['check']:
        check()
    else:raise ValueError('Use prepare once, or read-only check; all generated outputs refuse overwrite.')
