"""Fake-CLI/restart/evidence fixtures. No real ORX, SSH, model or GPU calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bounded_controller as controller


class ControllerFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.manifest=json.loads(controller.DEFAULT_MANIFEST.read_text())
        self.manifest_path=self.root/"manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.pair=self.root/"pair.py";self.pair.write_text("def verify_pair(*args): return {'status':'VERIFIED'}\n")
        self.calls=[];self.now=1000000
        self.c=controller.Controller(self.manifest_path,self.root/"state.json",self.pair,controller.sha(self.pair),
                                     execute=self.fake,clock=lambda:self.now,free=lambda:100*2**30)
        self.key=next(iter(self.c.specs));self.rid="11111111-1111-4111-8111-111111111111"

    def tearDown(self):self.temporary.cleanup()

    def fake(self,args,timeout=120):
        self.calls.append(args)
        return ""

    def running(self,key=None):
        key=key or self.key
        self.c.state['entries'][key].update(state='running',run_id=self.rid,intent_unix=self.now-100,
                                           last_progress_unix=self.now-100)

    def test_exact_fresh_reservation_and_failure_charge(self):
        self.c.state['failed_attempts']=[{'experiment_id':self.key,'run_id':self.rid,'charged_seconds':60}]
        spent,reserved=self.c.budget()
        self.assertEqual(spent,60);self.assertEqual(reserved,45*3600+60)

    def test_budget_blocks_even_if_individual_guards_exist(self):
        for entry in self.c.state['entries'].values():entry.update(state='verified',charged_seconds=30000)
        with self.assertRaisesRegex(RuntimeError,'ceiling'):self.c.budget()

    def test_invalid_historical_budget_rejected(self):
        for value in [float('nan'),float('inf'),-1,True]:
            self.c.state['failed_attempts']=[{'experiment_id':self.key,'run_id':self.rid,'charged_seconds':value}]
            with self.assertRaises(ValueError):self.c.budget()

    def test_ambiguous_launch_never_resubmits(self):
        self.c.state['entries'][self.key].update(state='launch_intent',intent_unix=self.now-10)
        with mock.patch.object(self.c,'runs',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'Ambiguous'):self.c.reconcile()
        self.assertFalse(any(args[:3]==['orx','exp','run'] for args in self.calls))

    def test_unique_recorded_intent_is_adopted_without_launch(self):
        self.c.state['entries'][self.key].update(state='launch_intent',intent_unix=self.now-10)
        with mock.patch.object(self.c,'runs',side_effect=lambda key:{self.rid:'running'} if key==self.key else {}),mock.patch.object(self.c,'binding'):
            self.c.reconcile()
        self.assertEqual(self.c.state['entries'][self.key]['run_id'],self.rid)
        self.assertFalse(any(args[:3]==['orx','exp','run'] for args in self.calls))

    def test_pending_unrecorded_run_is_not_adopted(self):
        with mock.patch.object(self.c,'runs',return_value={self.rid:'running'}):
            with self.assertRaisesRegex(RuntimeError,'Unrecorded'):self.c.reconcile()

    def test_duplicate_run_halts(self):
        self.running()
        with mock.patch.object(self.c,'runs',return_value={self.rid:'running','22222222-2222-4222-8222-222222222222':'done'}):
            with self.assertRaisesRegex(RuntimeError,'Multiple'):self.c.reconcile()

    def test_launch_intent_is_durable_before_cli_failure(self):
        def fake(args,timeout=120):
            self.calls.append(args)
            if args[:3]==['orx','exp','run']:
                state=json.loads(self.c.state_path.read_text())
                self.assertEqual(state['entries'][self.key]['state'],'launch_intent')
                self.assertIn('LAUNCH_INTENT',self.c.journal.read_text())
                raise TimeoutError('synthetic CLI timeout')
            return ''
        self.c.execute=fake
        with mock.patch.object(self.c,'runs',return_value={}),mock.patch.object(self.c,'binding'),mock.patch.object(self.c,'source_config',return_value={}):
            with self.assertRaises(TimeoutError):self.c.launch(self.key)
        self.assertEqual(sum(args[:3]==['orx','exp','run'] for args in self.calls),1)

    def test_same_host_second_model_job_rejected(self):
        self.running()
        second=next(key for key,entry in self.c.state['entries'].items() if key!=self.key and entry['host']==self.c.state['entries'][self.key]['host'])
        with mock.patch.object(self.c,'runs',return_value={}),mock.patch.object(self.c,'binding'),mock.patch.object(self.c,'source_config',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'already recorded'):self.c.launch(second)

    def test_stop_is_latched_and_cancels_only_recorded_study_runs(self):
        self.running()
        with mock.patch.object(self.c,'runs',return_value={self.rid:'running'}),mock.patch.object(self.c,'binding'):
            self.c.stop(RuntimeError('synthetic failure'))
        self.assertTrue(json.loads(self.c.state_path.read_text())['stop'])
        self.assertEqual([args for args in self.calls if args[:3]==['orx','exp','cancel']],[['orx','exp','cancel',self.key]])
        with self.assertRaisesRegex(RuntimeError,'STOP'):self.c.tick()

    def test_restart_preserves_stop_and_manifest_binding(self):
        self.c.state['stop']=True;self.c.save('fixture_stop')
        restarted=controller.Controller(self.manifest_path,self.c.state_path,self.pair,controller.sha(self.pair),execute=self.fake)
        self.assertTrue(restarted.state['stop'])
        data=json.loads(self.manifest_path.read_text());data['authorization']['user_answer']='changed'
        self.manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'binding'):controller.Controller(self.manifest_path,self.c.state_path,self.pair,controller.sha(self.pair),execute=self.fake)

    def test_stall_and_progress_growth(self):
        self.running();log=self.root/'run.log';log.write_text('first')
        entry=self.c.state['entries'][self.key];entry.update(log_bytes=5,last_progress_unix=self.now-1801)
        self.c.execute=lambda args,timeout=120:'This is the path to the full log file: '+str(log)+'\n'
        with self.assertRaisesRegex(TimeoutError,'progress'):self.c.progress(self.key)
        log.write_text('first second');self.c.progress(self.key)
        self.assertEqual(entry['last_progress_unix'],self.now)

    def test_exclusive_lock(self):
        with controller.exclusive_lock(self.root/'controller.lock'):
            with self.assertRaises(BlockingIOError):
                with controller.exclusive_lock(self.root/'controller.lock'):pass

    def test_index_hash_count_escape_and_corruption(self):
        out=self.root/'evidence';out.mkdir();(out/'checkpoint').write_bytes(b'checkpoint');(out/'capture').write_bytes(b'capture')
        def item(name):return {'path':name,'bytes':(out/name).stat().st_size,'sha256':controller.sha(out/name)}
        (out/'checkpoint-index.json').write_text(json.dumps([item('checkpoint')]))
        (out/'capture-index.json').write_text(json.dumps([item('capture')]))
        self.assertEqual(len(controller.verify_indices(out,expected_captures=1,expected_total=2)),2)
        with self.assertRaisesRegex(ValueError,'Incomplete'):controller.verify_indices(out,expected_captures=400)
        (out/'capture').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash/path'):controller.verify_indices(out,expected_captures=1)
        (out/'capture-index.json').write_text(json.dumps([{'path':'../outside','bytes':0,'sha256':'wrong'}]))
        with self.assertRaises(ValueError):controller.verify_indices(out,expected_captures=1)

    def test_operator_stop_file_checked_before_any_tick_io(self):
        self.c.stop_request.touch()
        with self.assertRaisesRegex(RuntimeError,'STOP_REQUESTED'):self.c.tick()
        self.assertFalse(self.calls)

    def test_operator_hold_between_preflight_and_submission(self):
        def fake(args,timeout=120):
            self.calls.append(args)
            if args[0]=='ssh':self.c.stop_request.touch()
            return ''
        self.c.execute=fake
        with mock.patch.object(self.c,'runs',return_value={}),mock.patch.object(self.c,'binding'),mock.patch.object(self.c,'source_config',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'STOP_REQUESTED'):self.c.launch(self.key)
        self.assertFalse(any(args[:3]==['orx','exp','run'] for args in self.calls))

    def test_stop_reconciles_unique_ambiguous_submission_then_cancels(self):
        self.c.state['entries'][self.key].update(state='launch_intent',intent_unix=self.now-10)
        with mock.patch.object(self.c,'runs',return_value={self.rid:'running'}),mock.patch.object(self.c,'binding'):
            self.c.stop(RuntimeError('synthetic reply loss'))
        self.assertEqual(self.c.state['entries'][self.key]['run_id'],self.rid)
        self.assertEqual(sum(args[:3]==['orx','exp','cancel'] for args in self.calls),1)

    def test_unchanged_pair_receipts_skip_reaudit_including_restart(self):
        counter=self.root/'pair-count'
        self.pair.write_text('from pathlib import Path\nimport json\ndef verify_pair(mid,none,receipt):\n'
                             f' p=Path({str(counter)!r});p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\n'
                             ' result={"status":"VERIFIED"};Path(receipt).write_text(json.dumps(result));return result\n')
        self.c.pair_sha=controller.sha(self.pair);self.c.state['pair_audit_sha256']=self.c.pair_sha
        for key,entry in self.c.state['entries'].items():
            if entry['seed']!=11:continue
            out=self.root/entry['condition'];out.mkdir();receipt=out/'controller-retention.json';receipt.write_text('{"retained":true}')
            entry.update(state='verified',output_directory=str(out),retention_receipt_sha256=controller.sha(receipt),charged_seconds=100)
        self.c.pairs();self.c.pairs();self.assertEqual(counter.read_text(),'1')
        self.c.save('fixture_pair_verified')
        restarted=controller.Controller(self.manifest_path,self.c.state_path,self.pair,self.c.pair_sha,execute=self.fake)
        restarted.pairs();self.assertEqual(counter.read_text(),'1')
        entry=next(x for x in restarted.state['entries'].values() if x['seed']==11)
        Path(entry['output_directory'],'controller-retention.json').write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError,'receipt changed'):restarted.pairs()

    def test_wire_auditor_is_cpu_hidden_and_requires_full_pinned_result(self):
        self.c.capture_module=self.root/'wire.py';self.c.capture_module.write_text('# fixture, never executed\n')
        self.c.capture_sha=controller.sha(self.c.capture_module)
        def fake(args,timeout=120):
            self.calls.append(args);receipt=Path(args[args.index('--receipt-out')+1])
            receipt.write_text(json.dumps({'status':'PASS','capture_files_verified':400,'wire_tensors_verified':400,'source_sha256':self.c.capture_sha}))
            return ''
        self.c.execute=fake
        self.assertTrue(self.c.wire_audit(self.root).is_file())
        self.assertIn('CUDA_VISIBLE_DEVICES=',self.calls[-1])
        old=self.c.execute
        def incomplete(args,timeout=120):
            old(args,timeout);receipt=Path(args[args.index('--receipt-out')+1]);data=json.loads(receipt.read_text());data['capture_files_verified']=399;receipt.write_text(json.dumps(data));return ''
        self.c.execute=incomplete
        with self.assertRaisesRegex(ValueError,'all400'):self.c.wire_audit(self.root)

    def test_cpu_completion_allowance_does_not_extend_gpu_guard(self):
        self.running();log=self.root/'run.log';log.write_text('progress')
        self.c.execute=lambda args,timeout=120:'This is the path to the full log file: '+str(log)+'\n'
        entry=self.c.state['entries'][self.key]
        entry.update(intent_unix=self.now-27700,last_progress_unix=self.now,log_bytes=8)
        self.c.progress(self.key)
        entry['intent_unix']=self.now-27721
        with self.assertRaisesRegex(TimeoutError,'guard'):self.c.progress(self.key)
        self.assertEqual(self.c.specs[self.key]['wall_seconds'],27000)

    def test_actual_running_failed_reason_table_is_not_a_duplicate(self):
        text='''ID STATUS EXPERIMENT COMMIT DURATION UPDATED
84ba32e6-8c1e-4494-8aaa-350e36b461e2 running Paper B bounded FP32 MID seed11 e238c4d 20m now
498bc32d-37b0-488f-9408-64d508111583 failed Paper B bounded FP32 MID seed11 203fe43 0s earlier
498bc32d-37b0-488f-9408-64d508111583 reason: Job failed: exited with code1
'''
        self.assertEqual(controller.parse_runs(text),{'84ba32e6-8c1e-4494-8aaa-350e36b461e2':'running',
                                                      '498bc32d-37b0-488f-9408-64d508111583':'failed'})

    def test_reason_diagnostic_cannot_hide_an_unlisted_run(self):
        with self.assertRaisesRegex(ValueError,'corresponding'):
            controller.parse_runs(self.rid+' reason: Job failed: exited with code1\n')

    def test_duplicate_primary_status_rows_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            controller.parse_runs(self.rid+' running example\n'+self.rid+' failed example\n')


if __name__=='__main__':unittest.main(verbosity=2)
