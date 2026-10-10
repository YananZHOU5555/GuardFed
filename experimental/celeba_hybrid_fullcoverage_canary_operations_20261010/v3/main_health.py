"""Exact existing main800 health guard; no scientific or recovery imports."""
import datetime

def need(ok, message):
    if not ok:
        raise ValueError(message)

def main_health(service, queue, phase):
    active, failed = queue.get('active'), queue.get('failed')
    fields = service.get('stdout', '').split()
    inputs = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase,
              'service': service, 'queue_snapshot': queue, 'active_n': len(active) if isinstance(active, list) else None,
              'failed_n': len(failed) if isinstance(failed, list) else None,
              'required': {'service': 'RUNNING', 'failed_n': 0, 'active_min': 1, 'active_max': 8}}
    if not (service.get('returncode') == 0 and len(fields) > 1 and fields[1] == 'RUNNING'
            and failed == [] and isinstance(active, list) and 1 <= len(active) <= 8):
        error = ValueError('Protected main800 health failed')
        error.resource_guard_inputs = inputs
        raise error
    return inputs

