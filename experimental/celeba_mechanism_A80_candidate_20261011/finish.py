"""Scoped original A60 source; saved-evidence statistics only."""
from source_adapter import namespace, preflight
_RUN=(__name__=="__main__")
globals().update(namespace('finish.py',__file__))
if _RUN:
    preflight('finish.py')
    main()
