"""Scoped original A50 source; no scientific function copied or rewritten."""
from source_adapter import namespace, preflight
_RUN = (__name__ == "__main__")
globals().update(namespace('verify_saved.py', __file__))
if _RUN:
    preflight('verify_saved.py')
    main()
