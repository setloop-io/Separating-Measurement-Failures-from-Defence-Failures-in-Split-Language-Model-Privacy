"""Campaign integrity checks independent of GPU model execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from attacker.bisr_fidelity import write_predictions
from experiments.bisr_campaign_score import verified_rows


def test_prediction_journal_rejects_truncation_and_mutation(tmp_path):
    path=tmp_path/'cell';path.mkdir()
    record={'schema':'bisr.predictions.v1','sample_ids':['article'],
            'input_manifest_sha256':'manifest','arms':{'sip':[[1]]}}
    digest=write_predictions(path/'00000.json',record)
    event={'index':0,'sample_id':'article','prediction_sha256':digest,'previous':'0'*64}
    tip=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest()
    event['hash']=tip
    (path/'journal.jsonl').write_text(json.dumps(event)+'\n')
    (path/'result.json').write_text(json.dumps({'status':'PREDICTIONS_COMPLETE','documents':1,
        'manifest_sha256':'manifest','journal_tip':tip}))
    assert verified_rows(path,['article'],'manifest')==[record]
    with pytest.raises(ValueError):verified_rows(path,['article','missing'],'manifest')
    (path/'00000.json').write_text('{}')
    with pytest.raises(ValueError,match='digest'):verified_rows(path,['article'],'manifest')


def test_new_predictor_rejects_truth_before_loading_models(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'schema':'bisr.campaign_prediction.v1','truth':'secret'}))
    p=subprocess.run([sys.executable,'experiments/bisr_campaign_predict.py','--manifest',str(manifest),
        '--output',str(tmp_path/'out'),'--engine','candidate','--device','cpu'],capture_output=True,text=True)
    assert p.returncode and 'truth is not accepted' in p.stderr
    assert not (tmp_path/'out').exists()
