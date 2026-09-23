import contextlib
import io
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.data_loader import DatasetLoader
import reproduce_paper_tables as core

class ACSIncomeTest(unittest.TestCase):
    def raw(self):
        n = 100
        return pd.DataFrame(dict(AGEP=np.full(n, 35), COW=np.ones(n), SCHL=np.full(n, 20),
            MAR=np.ones(n), OCCP=np.arange(n) * 10, POBP=np.full(n, 6), RELP=np.zeros(n),
            WKHP=np.full(n, 40), RAC1P=np.ones(n), SEX=np.tile([1,1,2,2], 25),
            PINCP=np.tile([50000,50001,50000,60000], 25), PWGTP=np.ones(n)))
    def load(self, raw, **kwargs):
        with patch('pandas.read_csv', return_value=raw.copy(deep=True)):
            return DatasetLoader('acs_income', **kwargs)
    def test_official_filter_threshold_sensitive_and_feature_schema(self):
        raw = self.raw()
        for i, (col, value) in enumerate([('AGEP',16),('PINCP',100),('WKHP',0),('PWGTP',0)]):
            raw.loc[i,col] = value
        loader = self.load(raw)
        self.assertEqual(loader.preprocessing_audit['filtered_rows'],96)
        ids = np.concatenate([loader.train_original_row_ids,loader.test_original_row_ids])
        self.assertEqual(set(ids),set(range(4,100)))
        rows = pd.concat([loader.train_df,loader.test_df])
        np.testing.assert_array_equal(rows.income,(raw.loc[ids].PINCP > 50000).astype(int))
        np.testing.assert_array_equal(rows.sex,(raw.loc[ids].SEX == 1).astype(int))
        self.assertEqual(list(loader.X_train.columns),['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','RAC1P'])
        self.assertEqual(loader.preprocessing_audit['train_test_overlap'],0)
        self.assertEqual(set(loader.preprocessing_audit['train_strata']),set(loader.preprocessing_audit['test_strata']))
    def test_heldout_changes_do_not_change_scaler_or_train(self):
        raw=self.raw(); first=self.load(raw)
        raw.loc[first.test_original_row_ids,'OCCP'] += 999999
        second=self.load(raw)
        for name in ['mean_','var_','scale_']:
            np.testing.assert_array_equal(getattr(first.scaler,name),getattr(second.scaler,name))
        pd.testing.assert_frame_equal(first.train_df,second.train_df,check_exact=True)
        self.assertFalse(first.test_df.equals(second.test_df))
        self.assertEqual(first.scaler.n_samples_seen_,len(first.train_df))
    def test_root_is_train_only_disjoint_and_sensitive_is_excluded(self):
        raw=self.raw()
        with patch('pandas.read_csv',return_value=raw):
            b=core.load_bundle('acs_income',5,core.ExperimentConfig(device='cpu',server_ratio=.1,synthetic_method='none'),torch.device('cpu'))
        a=b['loader'].preprocessing_audit
        self.assertTrue(a['partition']['root_is_training_subset'])
        self.assertEqual(a['partition']['root_client_overlap'],0)
        self.assertEqual(sum(c['n'] for c in b['clients'].values()) if all('n' in c for c in b['clients'].values()) else sum(len(c['y']) for c in b['clients'].values()),len(b['loader'].train_df)-b['root_clean_rows'])
        self.assertNotIn('sex',b['feature_cols']); self.assertNotIn('income',b['feature_cols'])
    def test_adult_default_unchanged_against_frozen_parent(self):
        source=subprocess.check_output(['git','show','47eb2f6:src/data_loader.py'],cwd=ROOT,text=True)
        old=types.ModuleType('adult_reference');old.__file__=str(ROOT/'src/data_loader.py')
        exec(compile(source,old.__file__,'exec'),old.__dict__)
        with contextlib.redirect_stdout(io.StringIO()):
            expected=old.DatasetLoader('adult'); actual=DatasetLoader('adult')
        for name in ['train_df','test_df','X_train','X_test']:
            pd.testing.assert_frame_equal(getattr(expected,name),getattr(actual,name),check_exact=True)
        for name in ['mean_','var_','scale_']:
            np.testing.assert_array_equal(getattr(expected.scaler,name),getattr(actual.scaler,name))
        self.assertEqual(expected.get_info(),actual.get_info())

if __name__=='__main__':unittest.main()
