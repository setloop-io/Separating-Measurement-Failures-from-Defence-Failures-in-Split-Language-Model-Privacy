"""CPU-only selector/configuration fixtures; no model, training or CUDA calls."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import types

import bounded_study as study
import native_pipeline

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
UPSTREAM = REPO/'artifacts/paper-b-review-round2-20261021/all-review-closure/cpu-closure-20261022/framework/upstream'
FROZEN = REPO/'artifacts/paper-b-vflair-replication-20261022/worktrees/timing-amp-mid/experiments/vflair_native_replication_20261022/native_pipeline.py'


class Rows:
    def __init__(self,indices):
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        return {'text':f'synthetic book {self.indices[index]}','domain':'fixture'}
    def select(self,indices):
        return Rows(self.indices[i] for i in indices)


def profile(condition='mid',seed=11):
    return json.loads((TASK/'profiles'/f'bounded-{condition}-seed{seed}.json').read_text())


def inputs_fixture():
    receipt = study.load_selection(profile())
    return {'dataset':{'revision':receipt['dataset_revision'],'manifest_sha256':'synthetic-fixture-only',
                       'splits':{s:copy.deepcopy(receipt['dataset_files'][s]) for s in ['train','test']}},
            'model':{'revision':receipt['model_revision'],'files':copy.deepcopy(receipt['model_files'])}}


class BoundedStudyContracts(unittest.TestCase):
    def test_proposal_and_window_source_bindings_match_retained_files(self):
        receipt = study.load_selection(profile())
        self.assertEqual(study.sha256_file(REPO/receipt['proposal']['path']),receipt['proposal']['sha256'])
        countsource = receipt['window_count_source']
        self.assertEqual(study.sha256_file(REPO/countsource['path']),countsource['sha256'])
        rows = [json.loads(x) for x in (REPO/countsource['path']).read_text().splitlines()]
        indexed = {x['index']:x for x in rows if x['split']=='train'}
        self.assertEqual([indexed[i] for i in study.SOURCE_INDICES],receipt['selected_books'])

    def test_all_six_profiles_blocked_without_budget(self):
        for seed in [11,22,33]:
            for condition in ['mid','none']:
                cfg = profile(condition,seed)
                study.validate_configuration(cfg)
                self.assertEqual(cfg['launch_status'],'NOT_AUTHORIZED_TO_LAUNCH')
                self.assertIsNone(cfg['guards']['wall_seconds'])
                self.assertIsNone(cfg['authorization']['approved_aggregate_gpu_hours'])
                with self.assertRaises(PermissionError):
                    study.assert_launch_authorized(cfg)

    def test_phase_override_cannot_remove_block(self):
        cfg = profile()
        cfg['phase'] = 'full'
        with self.assertRaises(PermissionError):
            study.assert_launch_authorized(cfg)
        with self.assertRaises(PermissionError):
            native_pipeline.execute(None,cfg,None,None,None)

    def test_direct_runner_and_dispatch_block_before_host_or_output_access(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = profile(); cfg['output_root'] = str(root/'never-created')
            config = root/'blocked.json'; config.write_text(json.dumps(cfg))
            commands = [[sys.executable,str(TASK/'dispatch.py'),'--config',str(config),'--phase','full'],
                        [sys.executable,str(TASK/'runner.py'),'--config',str(config),'--output',str(root/'never-created')]]
            for command in commands:
                result = subprocess.run(command,text=True,capture_output=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('NOT_AUTHORIZED_TO_LAUNCH',result.stderr)
                self.assertFalse((root/'never-created').exists())

    def test_recorded_scope_and_positive_wall_budget_required(self):
        cfg = profile()
        cfg['launch_status'] = study.AUTHORIZED
        cfg['authorization'].update(status=study.AUTHORIZED,user_scope_choice='synthetic-fixture-only',
                                    allocation_reference='unit-test-no-real-authorization',approved_aggregate_gpu_hours=1,
                                    per_run_wall_seconds=60)
        with self.assertRaises(PermissionError):
            study.assert_launch_authorized(cfg)
        cfg['guards']['wall_seconds'] = 60
        study.assert_launch_authorized(cfg)  # Validates data only; no launch.
        for invalid in [float('nan'),float('inf'),float('-inf'),True,0]:
            cfg['authorization']['approved_aggregate_gpu_hours'] = invalid
            with self.assertRaises(PermissionError):
                study.assert_launch_authorized(cfg)

    def test_uniform_selection_preserves_all_test_rows_and_hashes_actual_rows(self):
        dataset = {'train':Rows(range(2662)),'test':Rows(range(5153))}
        selected,receipt = study.select_fixed_books(dataset,profile(),inputs_fixture())
        self.assertEqual(selected['train'].indices,study.SOURCE_INDICES)
        self.assertIs(selected['test'],dataset['test'])
        self.assertEqual(receipt['all_test_rows_retained'],5153)
        self.assertEqual(receipt['native_training_windows_expected'],1655)
        self.assertEqual(len(receipt['selected_books']),8)
        self.assertEqual(sum(row['native_windows'] for row in receipt['selected_books']),1655)
        self.assertTrue(all(len(row['row_text_domain_sha256'])==64 for row in receipt['selected_books']))

    def test_source_asset_identity_and_order_are_enforced(self):
        for change in ['hash','order','test_count','tokenizer']:
            inputs = inputs_fixture()
            if change=='hash': inputs['dataset']['splits']['train'][0]['sha256']='wrong'
            if change=='order': inputs['dataset']['splits']['train'].reverse()
            if change=='test_count': inputs['dataset']['splits']['test'][0]['rows']=5152
            if change=='tokenizer': inputs['model']['files'][0]['sha256']='wrong'
            with self.assertRaises(ValueError):
                study.validate_inputs(profile(),inputs)

    def test_incomplete_source_splits_rejected(self):
        with self.assertRaises(ValueError):
            study.select_fixed_books({'train':Rows(range(2661)),'test':Rows(range(5153))},profile(),inputs_fixture())

    def test_actual_native_counts_must_equal_frozen_counts(self):
        counts = {'train_windows':1655,'test_rows':5153,'native_batches_per_epoch':104,'native_max_training_steps':3120}
        self.assertEqual(study.validate_native_counts(profile(),counts)['status'],'PASS')
        for field in counts:
            bad = dict(counts); bad[field]-=1
            with self.assertRaises(ValueError):
                study.validate_native_counts(profile(),bad)

    def test_bounded_native_configuration_retains_original_budgets(self):
        for condition in ['mid','none']:
            original,actual = native_pipeline.native_config(profile(condition),UPSTREAM,Path('/fixture/model'))
            self.assertEqual(actual['epochs'],30)
            self.assertEqual(actual['batch_size'],16)
            self.assertEqual(actual['early_stop_threshold'],1)
            self.assertEqual(actual['attack_list']['0']['parameters']['epochs'],400)
            self.assertEqual(actual['attack_list']['1']['parameters']['epochs'],300)
            self.assertEqual(actual['attack_list']['1']['parameters']['T'],.05)
            self.assertTrue(all(a['parameters']['attack_sample_num']==200 for a in actual['attack_list'].values()))
            self.assertEqual(actual['model_list']['0']['head']['trainable'],1)
            self.assertEqual(actual['model_list']['1']['tail']['trainable'],1)
            if condition=='mid': self.assertEqual(actual['defense'],original['defense'])
            else: self.assertNotIn('defense',actual)

    def test_fp32_and_original_native_limits_required(self):
        for change in ['precision','epochs','guard','proposal']:
            cfg = profile()
            if change=='precision': cfg['training_autocast']='bfloat16'
            if change=='epochs': cfg['full']['epochs']=2
            if change=='guard': cfg['guards']['maximum_optimizer_steps']=62904
            if change=='proposal': cfg['bounded_study']['protocol_sha256']='wrong'
            with self.assertRaises(ValueError): study.validate_configuration(cfg)

    @unittest.skipUnless(FROZEN.is_file(),'Frozen timing snapshot unavailable in this source-only fixture environment')
    def test_existing_full_canary_outputs_match_frozen_implementation(self):
        spec = importlib.util.spec_from_file_location('frozen_native_pipeline',FROZEN)
        frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
        for mode in ['full','canary']:
            for condition in ['mid','none','at']:
                cfg = json.loads((TASK/'profiles'/f'{mode}-{condition}.json').read_text())
                self.assertEqual(native_pipeline.native_config(cfg,UPSTREAM,Path('/fixture/model')),
                                 frozen.native_config(cfg,UPSTREAM,Path('/fixture/model')))

    @unittest.skipUnless(FROZEN.is_file(),'Frozen timing snapshot unavailable in this source-only fixture environment')
    def test_existing_full_canary_data_selection_matches_frozen_implementation(self):
        spec = importlib.util.spec_from_file_location('frozen_native_io',FROZEN)
        frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
        def exercise(function,cfg):
            dataset = {'train':Rows(range(2662)),'test':Rows(range(5153))}
            datasets = types.ModuleType('datasets'); datasets.load_dataset=lambda *a,**k:dataset
            load = types.ModuleType('load'); load.__path__=[]
            data = types.ModuleType('load.LoadDataset'); data.load_dataset=lambda *a,**k:'original-loader'
            load.LoadDataset = data
            inputs = inputs_fixture()
            for split,files in inputs['dataset']['splits'].items():
                for index,item in enumerate(files): item['path']=f'/fixture/{split}{index}.parquet'
            with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules,{'datasets':datasets,'load':load,'load.LoadDataset':data}):
                function(cfg,inputs,Path(td))
                result = data.load_dataset('__PINNED_LAMBADA_MANIFEST__')
                return {'train':result['train'].indices,'test':result['test'].indices,
                        'receipt':json.loads((Path(td)/'data-selection.json').read_text())}
        for mode in ['full','canary']:
            cfg = json.loads((TASK/'profiles'/f'{mode}-mid.json').read_text())
            self.assertEqual(exercise(native_pipeline.prepare_data_io,cfg),exercise(frozen.prepare_data_io,cfg))


if __name__=='__main__':
    unittest.main(verbosity=2)
