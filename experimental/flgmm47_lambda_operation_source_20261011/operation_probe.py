"""Stdlib arithmetic trace for one pinned lambda; no arrays, fitting or acceptance."""
from pathlib import Path
import argparse, hashlib, json, math, platform, sys, types

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return {'value': value, 'hex': value.hex()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-seal-sha256', required=True)
    a = p.parse_args()
    if sha(HERE / 'FILES_SHA256.json') != a.source_seal_sha256:
        raise ValueError('Wrong source seal')
    seal = json.loads((HERE / 'FILES_SHA256.json').read_bytes())
    for rel, pin in seal['files'].items():
        path = (HERE / rel).resolve()
        if not path.is_relative_to(HERE) or sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
            raise ValueError('Source member changed: ' + rel)
    evidence = json.loads((HERE / 'INPUTS.json').read_bytes())
    operands = {name: float.fromhex(pin['hex']) for name, pin in evidence['operands'].items()}
    if any(operands[k] != pin['value'] for k, pin in evidence['operands'].items()):
        raise ValueError('Decimal/hex operand disagreement')
    aeod, aspd = operands['base_aeod'], operands['base_aspd']
    direct_add, builtin_sum, accurate_sum = aeod + aspd, sum([aeod, aspd]), math.fsum([aeod, aspd])
    config = types.SimpleNamespace(ad2_calibration_base_weight=operands['base_weight'],
                                   ad2_calibration_budget=operands['budget'],
                                   ad2_calibration_temperature=operands['temperature'])
    temp = max(config.ad2_calibration_temperature, 1e-6)
    risks = {'original_binary_add': 0.5 * direct_add,
             'builtin_sum_control_not_original': 0.5 * builtin_sum,
             'fsum_control_not_original': 0.5 * accurate_sum,
             'captured_identical_base_risk': operands['captured_base_risk']}
    traces = {}
    for name, base_risk in risks.items():
        numerator = base_risk - config.ad2_calibration_budget
        quotient = numerator / temp
        exponential = math.exp(quotient)
        log_one_plus = math.log1p(exponential)
        result = config.ad2_calibration_base_weight * log_one_plus
        original_expression = eval(evidence['original_lambda_expression'], {'math': math},
                                   {'base_risk': base_risk, 'config': config, 'temp': temp})
        if result != original_expression:
            raise ValueError('Staged trace differs from original expression')
        traces[name] = {k: number(v) for k, v in {'base_risk': base_risk, 'temperature_after_max': temp,
                        'risk_minus_budget': numerator, 'quotient': quotient, 'exp': exponential,
                        'log1p': log_one_plus, 'weight_times_log1p': result}.items()}
        traces[name]['matches_captured_windows_lambda'] = result.hex() == evidence['captured_lambdas']['windows']['hex']
        traces[name]['matches_captured_linux_lambda'] = result.hex() == evidence['captured_lambdas']['linux']['hex']
    report = {'status': 'STDLIB_OPERATION_TRACE_DIAGNOSTIC_NOT_ACCEPTANCE', 'id': evidence['id'],
              'source_seal_sha256': a.source_seal_sha256, 'inputs_sha256': sha(HERE / 'INPUTS.json'),
              'operands': evidence['operands'], 'captured_lambdas': evidence['captured_lambdas'],
              'sum_operands_in_order': [number(aeod), number(aspd)],
              'sum_operations': {'original_binary_add': number(direct_add), 'builtin_sum_control': number(builtin_sum), 'fsum_control': number(accurate_sum)},
              'traces': traces, 'runtime': {'python': sys.version, 'implementation': platform.python_implementation(),
                                          'platform': platform.platform(), 'libc': platform.libc_ver(), 'math_module': getattr(math, '__file__', None)},
              'original_lambda_calls_sum_or_fsum': False, 'fit_calls': 0, 'CNN_calls': 0, 'test': False,
              'scientific_acceptances': 0, 'root_adopted': False,
              'interpretation': 'Compare the first differing operation hex across measured runtimes. Sum controls are not the original lambda path. No cause or portability claim is preset.'}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
