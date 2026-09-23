import contextlib
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_paper_tables as core

class RootShareTest(unittest.TestCase):
    def test_fixed_clients_and_nested_root_real_data(self):
        for dataset, protected in [("adult", 0), ("compas", 1)]:
            with contextlib.redirect_stdout(io.StringIO()):
                loader = core.DatasetLoader(dataset_name=dataset, seed=123, device="cpu", **({"preprocessing_version":"train_only"} if dataset=="compas" else {}))
            label = "income" if dataset=="adult" else "two_year_recid"
            previous = None
            for share in [0.5, 0.1, 0.02, 0.0]:
                cfg=core.ExperimentConfig(seed=123,server_ratio=.1,root_protected_share=share)
                root, clients, audit=core.sample_reservoir_root(loader.train_df,label,loader.sensitive_column,cfg,dataset)
                self.assertEqual(len(root),audit["server_rows"])
                self.assertEqual(int((root[loader.sensitive_column]==protected).sum()),round(len(root)*share))
                self.assertEqual(audit["protected_group_value"],protected)
                self.assertEqual(audit["protected_group_present"],share>0)
                if previous is not None:
                    pd.testing.assert_frame_equal(clients,previous[1])
                    self.assertEqual(len(root),len(previous[0]))
                    self.assertEqual(audit["reserve_index_sha256"],previous[2]["reserve_index_sha256"])
                    self.assertTrue(set(root.index[root[loader.sensitive_column]==protected]) <= set(previous[0].index[previous[0][loader.sensitive_column]==protected]))
                previous=(root,clients,audit)
            bundles=[]
            for share in [.5,0.0]:
                cfg=core.ExperimentConfig(seed=123,server_ratio=.1,root_protected_share=share,compas_preprocessing_version="train_only")
                with contextlib.redirect_stdout(io.StringIO()): b=core.load_bundle(dataset,5.0,cfg,torch.device("cpu"))
                bundles.append(b)
            for key in ["X_test","y_test"]: self.assertTrue(torch.equal(bundles[0][key],bundles[1][key]))
            self.assertEqual(bundles[0]["rw_weights"],bundles[1]["rw_weights"])
            for client_id in bundles[0]["clients"]:
                for key in ["X","y"]:self.assertTrue(torch.equal(bundles[0]["clients"][client_id][key],bundles[1]["clients"][client_id][key]))
    def test_reject_mixed_factors(self):
        for kw in [{"root_protected_share":.7},{"root_protected_share":.1,"root_label_noise":.1},{"root_protected_share":.1,"synthetic_ratio":.1}]:
            with self.assertRaises(ValueError):core.ExperimentConfig(**kw)

if __name__=="__main__":unittest.main()
