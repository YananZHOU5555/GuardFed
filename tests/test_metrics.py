import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reproduce_paper_tables import compute_metrics


class MetricsTest(unittest.TestCase):
    def test_group_rates_match_server_definition(self):
        result = compute_metrics([1, 1, 0, 0, 1, 1, 0, 0],
                                 [1, 0, 0, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['aeod'], 0.5)
        self.assertEqual(result['aspd'], 0.5)
        self.assertEqual(result['prediction_count'], 8)

    def test_constant_predictions_are_preserved(self):
        result = compute_metrics([0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['aeod'], 0)
        self.assertEqual(result['aspd'], 0)
        self.assertEqual(result['positive_rate'], 0)

    def test_undefined_denominator_is_not_reported_as_zero(self):
        result = compute_metrics([0, 1], [0, 1], [0, 0])
        self.assertTrue(math.isnan(result['aeod']))
        self.assertTrue(math.isnan(result['aspd']))
        self.assertTrue(result['warnings'])


if __name__ == '__main__':
    unittest.main()
