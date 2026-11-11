"""Focused arithmetic and schedule checks for the authorized 20-hour profile."""
from pathlib import Path
import unittest

from scripts.neural import curriculum_profiles as profiles


PROFILE = 'ironclad-twenty-hour-20261103'


class TwentyHourCampaignTests(unittest.TestCase):
    def test_campaign_caps_are_the_finite_stream_plus_qualification_and_panels(self):
        from scripts.neural import curriculum_controls as controls

        root = Path(__file__).resolve().parents[2]
        with profiles.scope(PROFILE):
            campaign = controls._read(root / profiles.current().campaign_path)
            expected = dict(environment_attempts=4899328, sample_rows=4719104,
                optimizer_attempts=294920, reset_slots=4737984, fixed_random_choices=0)
            self.assertEqual(controls._effective_global_effect_caps(root, campaign), expected)
            caps = controls.release_caps(root)
            self.assertEqual(caps['whole_seconds'], 72000)
            self.assertEqual(caps['training_stop_second'], 70200)
            self.assertEqual(caps['evaluation_seconds'], 7200)
            self.assertIsNone(caps['matched_arm'])

    def test_absolute_shard_indices_map_to_closed_rows(self):
        from scripts.neural.curriculum_controls import _training_binding

        package = {'campaign': {'training_start_indices': {'mixed': 256}},
            'training': {'mixed': [{'id': 0}, {'id': 1}, {'id': 2}]}}
        self.assertEqual(_training_binding(package, 'mixed', 256), {'id': 0})
        self.assertEqual(_training_binding(package, 'mixed', 258), {'id': 2})
        for index in (255, 259, True):
            with self.assertRaises(ValueError):
                _training_binding(package, 'mixed', index)

    def test_schedule_is_mixed_only_and_panels_follow_charged_training_time(self):
        from scripts.neural.curriculum_controls import TwentyHourSchedule

        campaign = {'training_caps': {'environment_attempts': 10000,
                'optimizer_attempts': 10000}, 'training_start_indices': {'mixed': 256},
            'training_shards': {'mixed': 4}, 'evaluation': {'milestone_training_seconds': [10, 20]}}
        package = {'campaign': campaign,
            'evaluation_slots': ['warm-start-baseline', 'mixed-time-10', 'mixed-time-20', 'final-mixed']}
        charged = dict(fixed=0., randomized=0., mixed=0.)
        with profiles.scope(PROFILE):
            plan = TwentyHourSchedule(package, start=100., training_seconds=charged)
            self.assertEqual(plan.next(100.), {'kind': 'qualification'})
            plan.accept_qualification()
            self.assertEqual(plan.next(100.), dict(kind='panel', slot='warm-start-baseline', arm=None))
            plan.finish_panel('warm-start-baseline', status='complete', reason='test')
            self.assertEqual(plan.next(100.), dict(kind='segment', arm='mixed', index=256,
                arm_deadline=100. + profiles.TWENTY_HOUR.training_stop_second))
            plan.accept_segment('mixed', {'checkpoint': {'counters': {'attempted_environment_slots': 1,
                'optimizer_attempted_steps': 1}}, 'receipt_binding': {'path': 'unused', 'sha256': '0'*64}})
            charged['mixed'] = 10.
            self.assertEqual(plan.next(101.), dict(kind='panel', slot='mixed-time-10', arm='mixed'))
            plan.finish_panel('mixed-time-10', status='complete', reason='test')
            charged['mixed'] = 20.
            self.assertEqual(plan.next(101.), dict(kind='panel', slot='mixed-time-20', arm='mixed'))

    def test_panel_lease_preserves_final_reserve(self):
        from scripts.neural.curriculum_controls import CampaignBudget

        now = [1000.]
        caps = dict(environment_attempts=100, sample_rows=100, optimizer_attempts=10,
            reset_slots=100, fixed_random_choices=0)
        with profiles.scope(PROFILE):
            budget = CampaignBudget(caps, start=900., deadline=72900., clock=lambda: now[0],
                evaluation_cap=7200, final_panel_reserve=600)
            soft, whole = budget.job_deadlines('panel')
            self.assertEqual((soft, whole), (1480., 1600.))
            budget.evaluation_seconds = 6100.
            soft, whole = budget.job_deadlines('panel')
            self.assertEqual((soft, whole), (1400., 1500.))
            soft, whole = budget.job_deadlines('panel', final_panel=True)
            self.assertEqual((soft, whole), (1480., 1600.))

    def test_cutoff_skips_late_milestones_and_nonfinal_panel_lease_ends_at_cutoff(self):
        from scripts.neural.curriculum_controls import CampaignBudget, TwentyHourSchedule

        campaign = {'training_caps': {'environment_attempts': 10000,
                'optimizer_attempts': 10000}, 'training_start_indices': {'mixed': 256},
            'training_shards': {'mixed': 4}, 'evaluation': {'milestone_training_seconds': [10, 20]}}
        package = {'campaign': campaign,
            'evaluation_slots': ['warm-start-baseline', 'mixed-time-10', 'mixed-time-20', 'final-mixed']}
        charged = dict(fixed=0., randomized=0., mixed=20.)
        now = [900. + profiles.TWENTY_HOUR.training_stop_second - 350.]
        caps = dict(environment_attempts=100, sample_rows=100, optimizer_attempts=10,
            reset_slots=100, fixed_random_choices=0)
        with profiles.scope(PROFILE):
            budget = CampaignBudget(caps, start=900., deadline=72900., clock=lambda: now[0],
                evaluation_cap=7200, final_panel_reserve=600)
            soft, whole = budget.job_deadlines('panel')
            self.assertEqual(whole, 900. + profiles.TWENTY_HOUR.training_stop_second)
            self.assertEqual(soft, whole - 100.)
            now[0] = 900. + profiles.TWENTY_HOUR.training_stop_second - 50.
            with self.assertRaises(TimeoutError):
                budget.job_deadlines('panel')
            now[0] = 900. + profiles.TWENTY_HOUR.training_stop_second
            _, final_whole = budget.job_deadlines('panel', final_panel=True)
            self.assertEqual(final_whole, now[0] + 600.)

            plan = TwentyHourSchedule(package, start=900., training_seconds=charged)
            self.assertEqual(plan.next(900.), {'kind': 'qualification'})
            plan.accept_qualification()
            plan.finish_panel('warm-start-baseline', status='complete', reason='test')
            request = plan.next(now[0])
            self.assertEqual(request, dict(kind='panel', slot='final-mixed', arm='mixed'))
            for slot in package['evaluation_slots'][1:-1]:
                self.assertEqual(plan.availability[slot]['status'], 'not_started')
                self.assertIn('training cutoff', plan.availability[slot]['reason'])

            plan.stop_arm('mixed', reason='time')
            self.assertEqual(plan.stopped, {'mixed': 'time'})


if __name__ == '__main__':
    unittest.main()
