"""Separate module-import RNG evidence from actual training RNG state.

Returns NumPy's actual Generator unchanged; never seeds, wraps or draws from it.
Imported generators reintroduced through default_rng or observed to advance after
the import boundary enter the training registry. All import states are retained.
"""
import copy
import traceback
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


class RNGCapture:
    def __init__(self):
        self.original = np.random.default_rng
        self.phase = 'module_import'
        self.imported = []
        self.training = []
        self.import_boundary = []
        self.training_origins = []
        self.import_metadata = []

    @staticmethod
    def state(generator):
        return plain(copy.deepcopy(generator.bit_generator.state))

    def _register_training(self, generator, reason):
        if all(generator is not g for g in self.training):
            self.training.append(generator)
            index = next((i for i, g in enumerate(self.imported) if generator is g), None)
            self.training_origins.append(dict(reason=reason, import_index=index))

    def tracked(self, *args, **kwargs):
        generator = self.original(*args, **kwargs)
        if self.phase == 'module_import':
            if all(generator is not g for g in self.imported):
                self.imported.append(generator)
                frames = traceback.extract_stack(limit=10)[:-1]
                seed = args[0] if args else kwargs.get('seed')
                self.import_metadata.append(dict(index=len(self.imported) - 1,
                    seed_kind='generator' if isinstance(seed, np.random.Generator) else type(seed).__name__,
                    explicit_seed_repr=None if seed is None else str(seed) if isinstance(seed, (int, np.integer)) else type(seed).__name__,
                    initial_state=self.state(generator),
                    callsite=[dict(file=f.filename, line=f.lineno, function=f.name, source=f.line) for f in frames]))
        else:
            self._register_training(generator, 'default_rng_called_after_import')
        return generator

    def install(self):
        assert np.random.default_rng is self.original
        np.random.default_rng = self.tracked

    def begin_training(self):
        assert self.phase == 'module_import'
        self.import_boundary = [self.state(g) for g in self.imported]
        self.phase = 'training'

    def snapshot(self):
        assert self.phase == 'training'
        import_now = [self.state(g) for g in self.imported]
        advanced = [i for i, state in enumerate(import_now) if state != self.import_boundary[i]]
        for i in advanced:
            self._register_training(self.imported[i], 'imported_generator_advanced_after_boundary')
        return dict(training_states=[self.state(g) for g in self.training],
            training_origins=copy.deepcopy(self.training_origins),
            import_states=import_now, import_advanced_indices=advanced)

    def import_evidence(self):
        assert self.phase == 'training'
        return dict(scope='module_import_only_not_training_rng', records=copy.deepcopy(self.import_metadata),
            boundary_states=copy.deepcopy(self.import_boundary),
            limitation='Imported generator state advancement is checked at each recorded boundary; no claim of method-call interception or whole-training resume')

    def restore(self):
        np.random.default_rng = self.original
