"""Load untouched Hybrid science; seed and serialization identities are the only overrides."""
import ast,importlib.util,sys,types
from pathlib import Path
from identity import require
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def functions(stage,repo,bindings):
    old=Path(bindings['original_screen']);body=load(stage/'body.py','hybrid100_original_body')
    body.HERE=stage;body.REPO=repo;body.SEALED=old/'scientific_snapshot'
    # Exactly one fixed-seed comparison is an identity check, not RNG handling.
    text=(stage/'body.py').read_text('utf8');tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checked')
    old_expr=ast.parse("result['seed'] == 91001",mode='eval').body;new_expr=ast.parse("result['seed'] == job['config']['seed']",mode='eval').body
    count=0
    class SeedIdentity(ast.NodeTransformer):
        def visit_Compare(self,n):
            nonlocal count
            if ast.dump(n)==ast.dump(old_expr):count+=1;return ast.copy_location(new_expr,n)
            return self.generic_visit(n)
    node=SeedIdentity().visit(node);require(count==1,'Original checker seed boundary changed')
    env=dict(body.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(stage/'body.py')+':seed_identity','exec'),env);body.checked=env['checked']
    driver=load(old/'driver.py','hybrid100_original_driver')
    source=(old/'driver.py').read_text('utf8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='functions')
    function=ast.get_source_segment(source,node)
    require(function.count("job['attack']=='S-DFA'")==2 and function.count("job['attack']=='Benign'")==1,'Original writer branches changed')
    function=function.replace("job['attack']=='S-DFA'","job['attack'] in {'F Flip','S-DFA','Sp-DFA'}").replace("job['attack']=='Benign'","job['attack'] in {'Benign','FedSA'}")
    writer=load(stage/'writer_policy.py','writer_policy')
    env=dict(driver.__dict__,HERE=stage);exec(compile(function,str(old/'driver.py')+':attack_writer_identity','exec'),env)
    return body,env['functions']

def unchanged_science_segments(path):
    text=Path(path).read_text('utf8');return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
