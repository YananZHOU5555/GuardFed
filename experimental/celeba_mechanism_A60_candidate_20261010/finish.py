"""Scoped original A50 source; no scientific function copied or rewritten."""
from source_adapter import namespace, preflight
_RUN = (__name__ == "__main__")
globals().update(namespace('finish.py', __file__))
if _RUN:
    preflight('finish.py')
    main()
