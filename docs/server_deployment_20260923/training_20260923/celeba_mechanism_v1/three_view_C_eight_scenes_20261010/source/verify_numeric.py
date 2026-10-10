"""Thin entry reusing pinned accepted C70 source; no scientific metric implementation."""
from loader import load,expose
_body=load('verify_numeric')
expose(_body,globals())
if __name__=="__main__":
    _body.main()
