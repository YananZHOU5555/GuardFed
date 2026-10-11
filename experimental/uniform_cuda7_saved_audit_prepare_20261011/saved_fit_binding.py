"""Metadata-only group-key restoration; original FL helper adapted to fixed AD2+ native calibration."""
import copy,math
def need(ok,message):
    if not ok:raise ValueError(message)

def saved_fits_for_original_predict(saved, evaluator, native_objective):
    need(set(saved) == {'native', 'raw', 'shared_calibration'}, 'Require all three saved views')
    fits = copy.deepcopy(saved)
    for view, fit in fits.items():
        common = {'view', 'method', 'rule', 'thresholds', 'fit_data', 'fit_sha256'}
        need(set(fit) == common | ({'fit_diagnostics'} if view in ('native', 'shared_calibration') else set()), 'Unknown/missing fit payload field')
        need(fit['view'] == view and fit['method'] == 'GuardFed-AD2+', 'Saved fit method/view identity differs')
        payload = {k: v for k, v in fit.items() if k != 'fit_sha256'}
        need(evaluator._seal_fit(payload) == fit, 'Saved fit payload SHA differs')
        if view in ('native', 'shared_calibration'):
            need(set(fit['thresholds']) == {'0', '1'}, 'Require exact JSON group keys')
            need(all(type(v) in (int, float) and math.isfinite(v) for v in fit['thresholds'].values()), 'Nonfinite or invalid threshold')
            diagnostic = fit['fit_diagnostics']
            expected = {'server_calibration_metrics', 'server_calibration_score', 'server_calibration_risk',
                        'server_base_calibration_metrics', 'server_base_calibration_risk', 'server_adaptive_lambda',
                        'server_calibration_budget', 'server_calibration_acc_floor',
                        'server_calibration_max_acc_drop', 'server_calibration_objective'}
            need(set(diagnostic) == expected, 'Unknown/missing original calibration diagnostic field')
            for key in ('server_calibration_metrics', 'server_base_calibration_metrics'):
                need(set(diagnostic[key]) == {'accuracy', 'aeod', 'aspd'} and
                     all(type(v) in (int, float) and math.isfinite(v) for v in diagnostic[key].values()), 'Invalid root calibration metrics')
            for key in expected - {'server_calibration_metrics', 'server_base_calibration_metrics', 'server_calibration_objective'}:
                value = diagnostic[key]
                need(type(value) in (int, float) and math.isfinite(value), 'Nonfinite calibration diagnostic')
                if key != 'server_calibration_score':
                    need(value >= 0, 'Negative nonnegative calibration diagnostic')
            need(diagnostic['server_calibration_objective'] == ('acc_floor' if view == 'shared_calibration' else native_objective), 'Wrong frozen shared objective')
            fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
        typed_payload = {k: v for k, v in fit.items() if k != 'fit_sha256'}
        need(evaluator._seal_fit(typed_payload) == fit, 'JSON group-key restoration changed fit SHA')
    return fits
