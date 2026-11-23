"""Validation-only scalar threshold search; no quality measurements enter it."""
import math


def calibrate_threshold(measure, target, tolerance=.0005, max_evaluations=16, strict=True):
    if not 0 < target < 1 or tolerance <= 0 or max_evaluations < 3:
        raise ValueError('Invalid calibration settings')
    trials = []

    def evaluate(shift):
        used = float(measure(shift))
        if not math.isfinite(used) or not 0 <= used <= 1:
            raise ValueError('Invalid measured retention')
        trials.append({'shift': float(shift), 'retention': used})
        return used

    value = evaluate(0.)
    best = trials[-1]
    if abs(value-target) > tolerance:
        direction = 1 if value > target else -1
        previous = 0.
        for exponent in range(max_evaluations-1):
            point = direction * .0625 * 2**exponent
            current = evaluate(point)
            if (value-target)*(current-target) <= 0:
                low, high = sorted([previous, point])
                break
            previous, value = point, current
        else:
            raise RuntimeError('Could not bracket target retention')
        while len(trials) < max_evaluations:
            best = min(trials, key=lambda t: (abs(t['retention']-target), t['shift']))
            if abs(best['retention']-target) <= tolerance:
                break
            point = (low+high)/2
            current = evaluate(point)
            if current > target:
                low = point
            else:
                high = point
    best = min(trials, key=lambda t: (abs(t['retention']-target), t['shift']))
    ordered = sorted(trials, key=lambda t: t['shift'])
    violations = [{'lower': a, 'higher': b} for a,b in zip(ordered, ordered[1:])
                  if b['retention'] > a['retention'] + tolerance]
    result = {'target': target, 'shift': best['shift'], 'validation_retention': best['retention'],
              'absolute_error': abs(best['retention']-target), 'tolerance': tolerance,
              'trials': trials, 'nonmonotonic_pairs': violations}
    result['matched'] = result['absolute_error'] <= tolerance
    if not result['matched'] and strict:
        raise RuntimeError(f'Calibration failed tolerance: {result}')
    return result
