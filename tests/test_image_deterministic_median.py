import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import reproduce_paper_tables as core

@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MedianNumerics(unittest.TestCase):
    def test_values_match_original_even_odd_ties_and_nan(self):
        for n in [3,20]:
            gen=torch.Generator().manual_seed(10)
            a=torch.randint(-3,4,(n,31),generator=gen).float().cuda()
            a[0,0]=float("nan")
            a[:,1]=torch.arange(n,device="cuda").float()
            updates=[{"w":row} for row in a]
            torch.use_deterministic_algorithms(False)
            expected=core.median_update(updates)["w"]
            try:
                torch.use_deterministic_algorithms(True)
                actual=core.median_update(updates)["w"]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
                self.assertEqual(actual.device,a.device)
                self.assertEqual(actual[1].item(),(n-1)//2)
            finally:torch.use_deterministic_algorithms(False)

if __name__=="__main__":unittest.main()
