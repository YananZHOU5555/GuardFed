"""Use original verifier body unchanged, with new explicit finalizer output namespace."""
from pathlib import Path
import hashlib,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import verify_staged as original
assert hashlib.sha256((HERE/'verify_staged.py').read_bytes()).hexdigest()=='ac6cc26dbb5c7c1521c8e1763d6a04f320927837c8c72ebdff8a1bf41670b727'
original.HERE=HERE/'v2'
if __name__=='__main__':original.main()
