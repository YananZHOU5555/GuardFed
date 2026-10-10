"""Scoped original A80 source; saved-evidence statistics only."""
from source_adapter import namespace, preflight
_RUN=(__name__=="__main__")
globals().update(namespace('verify_saved.py',__file__))
if _RUN:
    preflight('verify_saved.py')
    main()
