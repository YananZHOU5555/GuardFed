"""Run only after root adoption and proof copies exist on Linux; metadata/SHA only."""
from pathlib import Path
import argparse,json
import cache_verifier as cv

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv.need(cv.sha(a.inputs)==a.inputs_sha256 and not a.output.exists(),'Changed inputs or overwrite')
    x=cv.read(a.inputs);cv.need(x['root_authorized_index'] is True,'Unbound index authorization')
    root=cv.read(cv.pinned(x['root']));selection=cv.read(cv.pinned(x['selection']));transport=cv.read(cv.pinned(x['transport']));audit=cv.read(cv.pinned(x['audit']))
    cv.need(root['status']==cv.ROOT_STATUS and root['accepted']==100 and root['ids']==selection['exact_ids'],'Incomplete root adoption')
    cv.need(root['audit']['sha256']==x['audit']['sha256'] and root['transport']['sha256']==x['transport']['sha256'],'Root proof pins differ')
    cv.need(cv.sha(cv.pinned(x['selection']))==transport['members']['base/COHORT_SELECTION.json']['sha256'],'Wrong exported selection')
    records=[]
    for record in selection['records']:
        identity=record['id'];producer=selection['record_producers'][identity];base=Path(selection['producers'][producer]);out=base/'runs'/identity
        def pin(path):return {'path':str(path),'sha256':cv.sha(path)}
        accepted=base/'tasks'/identity/'ACCEPTED.json';aa=cv.read(accepted)
        cv.need(cv.sha(accepted)==root['records'][identity]['task_acceptance_sha256'],'Root task link differs')
        binding=cv.read(out/'formal_binding.json');original=[Path(path) for path,m in binding['artifact_before'].items() if m['sha256']==record['result']['sha256']]
        cv.need(len(original)==1,'Original result path not unique')
        members={role:pin(out/name) for role,name in [('receipt','receipt.json'),('predictions','validation_predictions.npz'),('formal_binding','formal_binding.json'),('saved_acceptance','formal_saved_acceptance.json')]};members['original_result']=pin(original[0])
        for role,key in [('receipt','receipt'),('predictions','arrays'),('formal_binding','binding'),('saved_acceptance','acceptance')]:cv.need(members[role]==aa[key],'Task actual pin mismatch')
        records.append(dict(id=identity,inventory_record=record,members=members,metadata=x['metadata']))
    index=dict(status='ROOT_ADOPTED_FEDAVG100_CACHE_INDEX',root=x['root'],audit=x['audit'],transport=x['transport'],selection=x['selection'],metadata=x['metadata'],records=records,order='original800 FedAvg subset; verifier returns explicit caller order',new_CNN=0,new_fit=0)
    with a.output.open('x',encoding='utf8') as f:json.dump(index,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    checked=cv.verify_root_index(x['root'],{'path':str(a.output),'sha256':cv.sha(a.output)},selection['exact_ids'])
    print(json.dumps({'status':'ROOT_INDEX_WRITTEN_AND_VERIFIED','records':len(checked['records']),'index_sha256':cv.sha(a.output)}))
if __name__=='__main__':main()
