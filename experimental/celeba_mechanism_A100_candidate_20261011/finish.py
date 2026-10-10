"""Reuse original IID/non-IID/balanced seed-first arithmetic after actual build."""
from source_adapter import namespace, preflight
_RUN = (__name__ == '__main__')
if _RUN:
    preflight('finish.py')
globals().update(namespace('finish.py', __file__))
if _RUN:
    main()
