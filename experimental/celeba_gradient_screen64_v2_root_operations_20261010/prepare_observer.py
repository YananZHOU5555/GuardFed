from pathlib import Path
p=Path(__file__).resolve().parent
t=(p/'observe.py').read_text()
t=t.replace('guardfed_celeba_gradient_screen64_v2','guardfed_celeba_gradient_screen64_v2a')
t=t.replace('queue.log','queue_attempt2.log')
t=t.replace("HERE/'LATEST_OBSERVATION.json'", "HERE/'LATEST_ATTEMPT2_OBSERVATION.json'")
t=t.replace("'OBSERVATION_'", "'ATTEMPT2_OBSERVATION_'")
t=t.replace('if any(str(base) in x for x in a):', "if any(Path(x).name == 'run_queue.py' for x in a):")
(p/'observe_attempt2.py').write_bytes(t.encode())
