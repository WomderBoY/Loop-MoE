import copy
import itertools
import json
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest

from loop_moe.residency import (History, Schedule, Prepared, POLICIES, Predictor, allocate,
                                fit_predictors, partition, prepare, run, simulate)
from loop_moe.replay import Visit
from test_experiments import make_request


def configuration(**updates):
    config = json.loads(Path('configs/residency.json').read_text())
    config['policies'] = list(POLICIES)
    config.update(updates)
    return config


def metadata(order=(('loop:0', 0), ('loop:0', 1))):
    return {'model': 'synthetic', 'pools': {pool: {'experts': 4, 'bytes_per_expert': 100}
                                         for pool, _ in order}, 'execution_order': order}


def visit(pool, phase, ids):
    return Visit(pool, phase, tuple((e, .5) for e in ids), tuple((e, .5) for e in ids))


class ResidencyTests(unittest.TestCase):
    def test_structural_distance_and_boundary(self):
        order = [('head', None)] + [(p, r) for r in range(2) for p in ('a', 'b', 'c', 'd')] + [('tail', None)]
        schedule = Schedule(order, metadata(order)['pools'])
        self.assertEqual(schedule.delta[1], 4)
        self.assertEqual(schedule.delta[5], 6)  # Includes head/tail at token boundary.
        self.assertEqual(schedule.delta[0], 10)
        self.assertEqual(schedule.delta[-1], 10)
        self.assertFalse(schedule.boundary[1])
        self.assertTrue(schedule.boundary[5])

    def test_stage_history_arrives_causally_and_boundary_a_is_missing(self):
        meta = metadata()
        schedule = Schedule(meta['execution_order'], meta['pools'])
        history = History(schedule, meta, .2)
        first = history.observe(0, visit('loop:0', 0, [0]))
        self.assertEqual(first.dual[0], (1, -1, -1))
        boundary = history.observe(1, visit('loop:0', 1, [1]))
        self.assertEqual(boundary.dual[0], (-1, 1, 0))
        self.assertEqual(boundary.dual[1], (-1, 0, 1))
        next_token = history.observe(2, visit('loop:0', 0, [2]))
        self.assertEqual(next_token.dual[1], (0, 1, -1))
        self.assertEqual(next_token.generic[1], (2, 2, 2))

    def test_fitting_denominator_includes_every_physical_expert(self):
        meta = metadata()
        schedule = Schedule(meta['execution_order'], meta['pools'])
        history = History(schedule, meta, .2)
        features = history.observe(0, visit('loop:0', 0, [0]))
        predictor = Predictor('phase_ab', configuration(), {('loop:0', e): .25 for e in range(4)})
        predictor.add(features, {1})
        for table in predictor.tables:
            self.assertEqual(sum(c[1] for c in table.values()), 4)
            self.assertEqual(sum(c[0] for c in table.values()), 1)

    def test_future_routes_do_not_change_source_predictions(self):
        meta = metadata()
        schedule = Schedule(meta['execution_order'], meta['pools'])
        fitting = [make_request('fit', [(0, 1), (1, 2)], [(1, 2), (2, 3)])]
        config = configuration()
        predictors, _ = fit_predictors(meta, fitting, schedule, config)
        request = make_request('eval', [(0, 1), (1, 2)], [(1, 2), (2, 3)])
        before = copy.deepcopy([p.tables for p in predictors.values()])
        original = prepare(meta, request, schedule, config, predictors)
        altered = copy.deepcopy(request)
        altered['steps'][1]['visits'][1]['routes'][0][0] = [0, 3]
        changed = prepare(meta, altered, schedule, config, predictors)
        self.assertEqual(original[0].scores, changed[0].scores)
        self.assertEqual(before, [p.tables for p in predictors.values()])

    def test_belady_optimal_under_both_admission_models(self):
        meta = metadata((('loop:0', 0),))
        schedule = Schedule(meta['execution_order'], meta['pools'])
        for sequence in itertools.product(range(3), repeat=4):
            rows = [Prepared(visit('loop:0', 0, [e]), i+1, {}, {}, frozenset())
                    for i, e in enumerate(sequence)]
            for mode, capacity in itertools.product(('peak', 'resident'), (1, 2)):
                @lru_cache(None)
                def optimal(index, resident):
                    if index == len(sequence):
                        return 0
                    e = sequence[index]
                    if e in resident:
                        return optimal(index+1, resident)
                    candidates = resident | {e}
                    keep = min(capacity, len(candidates))
                    choices = list(itertools.combinations(candidates, keep))
                    if mode == 'peak':
                        choices = [c for c in choices if e in c]
                    return 1 + min(optimal(index+1, frozenset(c)) for c in choices)
                outcome = simulate(meta, rows, schedule, configuration(memory_mode=mode),
                                   'belady', capacity, set())
                self.assertEqual(outcome.misses, optimal(0, frozenset()), (mode, capacity, sequence))
                self.assertLessEqual(outcome.peak_bytes, (capacity + (mode == 'resident')) * 100)

    def test_discount_uses_topology_and_remaining_distance(self):
        meta = metadata((('a', 0), ('b', 0), ('c', 0)))
        schedule = Schedule(meta['execution_order'], meta['pools'])
        # At c, a is next and b is further away. Both use the exact same probabilities.
        rows = [Prepared(visit('a', 0, [0]), 3, {'phase_ab': [.5, 0, 0, 0]}, {}, frozenset()),
                Prepared(visit('b', 0, [0]), 4, {'phase_ab': [.6, 0, 0, 0]}, {}, frozenset()),
                Prepared(visit('c', 0, [0]), 5, {'phase_ab': [.1, 0, 0, 0]}, {}, frozenset()),
                Prepared(visit('a', 0, [0]), 6, {'phase_ab': [.5, 0, 0, 0]}, {}, frozenset())]
        ordinary = simulate(meta, rows, schedule, configuration(), 'phase_ab', 2, set())
        discount = simulate(meta, rows, schedule, configuration(), 'phase_ab_distance', 2, set())
        self.assertEqual(ordinary.hit_bits[-1], 0)
        self.assertEqual(discount.hit_bits[-1], 1)

    def test_staging_memory_changes_bypass_and_is_reported(self):
        meta = metadata((('loop:0', 0),))
        schedule = Schedule(meta['execution_order'], meta['pools'])
        rows = [Prepared(visit('loop:0', 0, [e]), i+1,
                         {'phase_ab': [1., 0., 0., 0.]}, {}, frozenset())
                for i, e in enumerate((0, 1, 0))]
        peak = simulate(meta, rows, schedule, configuration(), 'phase_ab', 1, set())
        resident = simulate(meta, rows, schedule, configuration(memory_mode='resident'),
                            'phase_ab', 1, set())
        self.assertEqual((peak.misses, peak.peak_bytes), (3, 100))
        self.assertEqual((resident.misses, resident.peak_bytes), (2, 200))

    def test_zero_bonus_and_oled_runtime_update_interval(self):
        meta = metadata()
        schedule = Schedule(meta['execution_order'], meta['pools'])
        rows = [Prepared(visit('loop:0', i % 2, [i % 3]), i+1, {},
                         {i % 3: .5}, frozenset([i % 3])) for i in range(24)]
        config = configuration(loop_bonus=0)
        ema = simulate(meta, rows, schedule, config, 'score_ema', 2, set())
        bonus = simulate(meta, rows, schedule, config, 'score_ema_loop', 2, set())
        self.assertEqual(ema.hit_bits, bonus.hit_bits)
        adaptive = simulate(meta, rows, schedule, config, 'oled_adaptive', 2, set())
        self.assertEqual(adaptive.quota_updates, 2)
        self.assertLessEqual(adaptive.peak_bytes, 200)

    def test_integer_adaptive_quotas_preserve_budget_and_allow_recovery(self):
        for capacity in range(12):
            quotas = allocate(capacity, {'a': 0., 'b': .8, 'c': .1}, {'a': 2, 'b': 3, 'c': 5})
            self.assertEqual(sum(quotas.values()), min(capacity, 10))
            self.assertLessEqual(quotas['b'], 3)
        self.assertEqual(allocate(1, {'a': 0, 'b': 0}, {'a': 1, 'b': 1}), {'a': 1, 'b': 0})

    def test_split_rejects_leakage_and_leaves_validation_out(self):
        requests = [{'request_id': x} for x in ('a', 'b', 'c')]
        with self.assertRaises(ValueError):
            partition(requests, {'fit_request_ids': ['a'], 'evaluation_request_ids': ['a', 'b', 'c']})
        fit, validation, evaluation = partition(requests, {
            'fit_request_ids': ['a'], 'validation_request_ids': ['b'], 'evaluation_request_ids': ['c']})
        self.assertEqual([x[0]['request_id'] for x in (fit, validation, evaluation)], ['a', 'b', 'c'])

    def test_full_pipeline_budget_oracle_and_gain_loss_conservation(self):
        meta = metadata()
        requests = [make_request(x, [(0, 1), (1, 2)], [(1, 2), (2, 3)]) for x in ('a', 'b', 'c')]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summaries = run(meta, requests, configuration(budgets_percent=[25, 50, 100]), root)
            for row in summaries:
                oracle = next(r for r in summaries if r['budget_percent'] == row['budget_percent']
                              and r['policy'] == 'belady')
                self.assertGreaterEqual(row['misses'], oracle['misses'])
                if row['budget_percent'] == 100:
                    self.assertEqual(row['misses'], 8)  # Four unique experts per cold-start request.
            import csv
            with (root/'paired_requests.csv').open() as stream:
                for row in csv.DictReader(stream):
                    self.assertEqual(int(row['gained_hit_bytes'])-int(row['lost_hit_bytes']),
                                     int(row['net_saved_h2d_bytes']))
            with (root/'residency_requests.csv').open() as stream:
                for row in csv.DictReader(stream):
                    self.assertLessEqual(int(row['peak_weight_bytes']), int(row['budget_bytes']))
            self.assertEqual(json.loads((root/'residency_config.json').read_text())['validation_request_ids'], [])


if __name__ == '__main__':
    unittest.main()
