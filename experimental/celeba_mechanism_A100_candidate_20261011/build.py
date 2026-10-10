"""A100 source candidate; no result exists before actual external adoption."""
from source_adapter import namespace, preflight
_RUN = (__name__ == '__main__')
if _RUN:
    preflight('build.py')
globals().update(namespace('build.py', __file__))
if _RUN:
    main()
