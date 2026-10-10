"""Future saved-record verifier; no inference, fit or model/array reads."""
from source_adapter import namespace, preflight
_RUN = (__name__ == '__main__')
if _RUN:
    preflight('verify_saved.py')
globals().update(namespace('verify_saved.py', __file__))
if _RUN:
    main()
