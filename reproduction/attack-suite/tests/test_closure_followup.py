"""Test delivery, receipt and failure boundaries without another Codex writer."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('closure_followup', Path(__file__).resolve().parents[1] / 'experiments/run_closure_followup.py')
followup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(followup)
THREAD = '01a0ae71-32a2-73b0-801a-b2d62eaa2a21'


class ClosureFollowupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.queue = self.base / 'queue.json'
        self.queue.write_text(json.dumps({'active': True}))
        prompt = self.base / 'prompt.txt'
        prompt.write_text('Continue the authorised queue.')
        self.config = self.base / 'config.json'
        self.config.write_text(json.dumps({
            'queue_state': str(self.queue), 'codex': '/mock/codex',
            'workspace': str(self.base), 'thread': THREAD, 'prompt': str(prompt),
            'not_before_epoch': 100, 'interval_seconds': 1800,
        }))
        self.calls = []

    def runner(self, command, **kwargs):
        self.calls.append(command)
        self.assertEqual(command[1:4], ['queue', '--thread', THREAD])
        self.assertNotIn('resume', command)
        self.assertNotIn('fork', command)
        kwargs['stdout'].write(f'Queued message 01a0bfeb-6d5b-7bf2-b817-74cae9f3551b for thread {THREAD}.\n')
        return SimpleNamespace(returncode=0)

    def status(self):
        return json.loads((self.base / 'scheduler-status.json').read_text())

    def test_future_and_inactive_do_not_launch(self):
        self.assertEqual(followup.tick(self.config, self.runner, 99), 'WAITING_FOR_SCHEDULE')
        self.queue.write_text(json.dumps({'active': False}))
        self.assertEqual(followup.tick(self.config, self.runner, 100), 'QUEUE_INACTIVE')
        self.assertEqual(self.calls, [])

    def test_acceptance_is_not_completion_and_never_duplicates(self):
        self.assertEqual(followup.tick(self.config, self.runner, 100), 'FOLLOWUP_QUEUED')
        self.assertEqual(followup.tick(self.config, self.runner, 101), 'FOLLOWUP_QUEUED')
        self.assertEqual(len(self.calls), 1)
        ticket = self.status()['pending_ticket']
        followup.receipt(self.config, ticket, 'ack', now=200)
        self.assertEqual(followup.tick(self.config, self.runner, 201), 'FOLLOWUP_RUNNING')
        followup.receipt(self.config, ticket, 'complete', now=300)
        self.assertEqual(followup.tick(self.config, self.runner, 301), 'WAITING_FOR_SCHEDULE')
        self.assertEqual(self.status()['next_check_epoch'], 2100)
        self.assertEqual(followup.tick(self.config, self.runner, 2100), 'FOLLOWUP_QUEUED')
        self.assertEqual(len(self.calls), 2)

    def test_missing_delivery_receipt_pauses_instead_of_retrying_forever(self):
        followup.tick(self.config, self.runner, 100)
        self.assertEqual(followup.tick(self.config, self.runner, 1900), 'PAUSED_UNACKNOWLEDGED_DELIVERY')
        self.assertFalse(json.loads(self.queue.read_text())['active'])
        self.assertEqual(len(self.calls), 1)

    def test_missing_completion_receipt_pauses(self):
        followup.tick(self.config, self.runner, 100)
        followup.receipt(self.config, self.status()['pending_ticket'], 'ack', now=200)
        self.assertEqual(followup.tick(self.config, self.runner, 7400), 'PAUSED_UNACKNOWLEDGED_EXECUTION')
        self.assertEqual(len(self.calls), 1)

    def test_three_rejections_pause(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('thread already has an active writer')
            return SimpleNamespace(returncode=1)
        for now in (100, 220, 340):
            followup.tick(self.config, fail, now)
        self.assertEqual(self.status()['status'], 'PAUSED_AFTER_THREE_FAILURES')
        self.assertFalse(json.loads(self.queue.read_text())['active'])

    def test_timeout_does_not_retry_ambiguous_delivery(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 30)
        self.assertEqual(followup.tick(self.config, timeout, 100), 'PAUSED_DELIVERY_UNCERTAIN')
        self.assertFalse(json.loads(self.queue.read_text())['active'])

    def test_completion_requires_started_receipt(self):
        with self.assertRaises(ValueError):
            followup.receipt(self.config, '00000000-0000-0000-0000-000000000000', 'complete')

    def test_lock_prevents_duplicate_local_delivery(self):
        with (self.base / 'scheduler.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(followup.tick(self.config, self.runner, 100), 'FOLLOWUP_ALREADY_RUNNING')
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
