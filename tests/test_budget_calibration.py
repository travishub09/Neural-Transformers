import unittest
from pathpred.budget_calibration import calibrate_threshold


class CalibrationTests(unittest.TestCase):
    def test_both_search_directions_and_native(self):
        for target in [.4, .5, .7]:
            calls=[]
            def measure(x):
                calls.append(x)
                return min(1., max(0., .5-x*.2))
            r=calibrate_threshold(measure,target)
            self.assertLessEqual(r['absolute_error'],r['tolerance'])
            self.assertEqual(len(calls),len(r['trials']))
            self.assertEqual(r['nonmonotonic_pairs'],[])
            if target==.5: self.assertEqual(calls,[0.])

    def test_unreachable_or_invalid_retention_fails(self):
        with self.assertRaises(RuntimeError): calibrate_threshold(lambda x:.9,.5,max_evaluations=4)
        with self.assertRaises(ValueError): calibrate_threshold(lambda x:float('nan'),.5)


if __name__=='__main__': unittest.main()


def test_discrete_unmatched_calibration_is_explicit_without_stopping_other_arms():
    result=calibrate_threshold(lambda shift: .8 if shift<.2 else .2,.5,tolerance=.0025,max_evaluations=12,strict=False)
    assert result['matched'] is False
    assert result['absolute_error']>.0025
    assert len(result['trials'])==12
