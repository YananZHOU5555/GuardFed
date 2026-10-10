"""Reuse unchanged metadata binding; preserve all original fresh write-boundary guards."""
import argparse, hashlib, importlib.util, json, sys, types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OPERATIONS=ROOT/'tmp/celeba_logofair100_root_operations_20261010'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    assert not sys.flags.optimize
    source=OPERATIONS/'prepare_inputs_and_bind_approval.py'
    source_seal=json.loads((OPERATIONS/'FILES_SHA256.json').read_bytes())
    assert H(OPERATIONS/'FILES_SHA256.json')=='1f27eafe56693216ca30f6bd03e92f4c9e3c9e0b14d0248a28e6b29db9a7514f'
    assert H(source)==source_seal['files'][source.name]['sha256']
    sys.path.insert(0,str(OPERATIONS))
    spec=importlib.util.spec_from_file_location('unchanged_logofair_root_bind',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.sources()
    assert H(a.cancellation)==a.cancellation_sha256
    cancellation=json.loads(a.cancellation.read_bytes())
    assert cancellation['status']=='ROOT_READONLY_BIND_CANCELLED_BEFORE_STAGE_CREATION'
    assert cancellation['original_process_stopped'] and cancellation['stage_absent_after_stop']
    assert cancellation['CNN_calls']==cancellation['fits']==0
    original_attempt=OPERATIONS/'bind_attempt001'
    assert H(original_attempt/'STARTED.json')==cancellation['original_started_sha256']
    approved=module.FROOT/'root_approved_inputs001'
    inputs,approval=approved/'ROOT_INPUTS.json',approved/'BIND_APPROVAL.json'
    assert H(inputs)==cancellation['inputs_sha256'] and H(approval)==cancellation['approval_sha256']
    assert not (module.FROOT/'stage001').exists()
    assert a.attempt.resolve().parent==OPERATIONS.resolve() and not a.attempt.exists()
    original_bulk=module.bulk_path
    _,volume=original_bulk('F:/YananResearchStorage/GuardFed/logofair_fullcoverage_20261010/inputs001',0)
    storage_root=Path('F:/YananResearchStorage/GuardFed').resolve()
    counts={'read_checks':0,'fresh_positive_write_checks':0}
    def read_guard(path,required):
        if required:
            counts['fresh_positive_write_checks']+=1
            return original_bulk(path,required)
        p=Path(path).resolve()
        module.require(p.drive.upper()=='F:' and p.is_relative_to(storage_root) and p!=storage_root,
            'Read input must remain on the checked external volume; no internal fallback')
        counts['read_checks']+=1
        return p,volume
    # Only metadata.bind's storage helper changes; its bytecode and all scientific functions remain unchanged.
    original=module.original
    binding_globals=dict(original.bind.__globals__,bulk_path=read_guard)
    binding=types.FunctionType(original.bind.__code__,binding_globals,original.bind.__name__,original.bind.__defaults__,original.bind.__closure__)
    assert binding.__code__ is original.bind.__code__
    adoption=module.pinned(module.ADOPTION,module.ADOPTION_SHA)
    a.attempt.mkdir()
    module.save(a.attempt/'STARTED.json',dict(operation='EXISTING_APPROVED_METADATA_BIND_CONTINUATION',
        cancellation_sha256=a.cancellation_sha256,original_prevalidation_preserved=True,new_fits=0,new_CNN=0))
    binding(adoption['summary_path'],adoption['summary_sha256'],module.ADOPTION,module.ADOPTION_SHA,
        adoption['strict_index_path'],adoption['strict_index_sha256'],inputs,approval,H(approval),module.FROOT/'stage001')
    stage=module.FROOT/'stage001'
    module.save(a.attempt/'BIND_RESULT.json',dict(status='ROOT_LOGOFAIR100_METADATA_BOUND_NOT_STARTED',stage=stage.as_posix(),
        manifest_sha256=H(stage/'manifest.json'),source_sha256=H(stage/'SOURCE_SHA256.json'),inputs=inputs.as_posix(),
        inputs_sha256=H(inputs),approval=approval.as_posix(),approval_sha256=H(approval),source_review_sha256=module.REVIEW_SHA,
        summary_adoption_sha256=module.ADOPTION_SHA,staging_handoff_sha256=json.loads((original_attempt/'STARTED.json').read_bytes())['handoff_sha256'],
        cancellation_sha256=a.cancellation_sha256,metadata_reader_source_sha256=H(__file__),new_fits=0,new_CNN=0))
    target=a.attempt/'READ_GUARD_RECEIPT.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(status='UNCHANGED_BIND_WITH_BOUNDED_READ_PREFLIGHT_PASS',
        original_operations_seal_sha256=H(OPERATIONS/'FILES_SHA256.json'),original_bind_bytecode_identical=True,
        read_path_rule='F project subtree only; actual bytes rehashed; missing/detached files fail with no fallback',
        all_original_write_boundary_guards_preserved=True,initial_volume=volume,**counts,new_fits=0,new_CNN=0),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(bind_result=str(a.attempt/'BIND_RESULT.json'),bind_result_sha256=H(a.attempt/'BIND_RESULT.json'),**counts)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--cancellation',type=Path,required=True);p.add_argument('--cancellation-sha256',required=True)
    p.add_argument('--execute-bind',action='store_true',required=True)
    run(p.parse_args())
