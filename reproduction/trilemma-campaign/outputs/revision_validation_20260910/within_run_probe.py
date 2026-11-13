"""Exploratory linear token probe on existing captures; leave one whole sequence out.
Known-plaintext assumption: labels for seven captured sequences from target run.
This does not represent a zero-knowledge attack or a new training cohort.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path('/Users/geo/Paper 3 - Privacy-Constrained Remote');sys.path.insert(0,str(ROOT/'src'))
from pcr.data import load_corpus,split_corpus,make_probe_batch
_,region=split_corpus(load_corpus(ROOT/'data/raw/tinyshakespeare.txt'))
y=make_probe_batch(region,8,128)[0].numpy()
records=[]
for q in (4,32):
 for seed in (7,17,27,37,47,57,67,87):
  rid=f'trained-medium-high-q{q}-s{seed}';p=ROOT/'outputs/rate_sweep'/rid/'transcript_tensors.pt'
  events=torch.load(p,map_location='cpu',weights_only=True)
  x=max(events,key=lambda e:e['step'])['forward_values'].numpy().astype('float64')
  acc=[]; null=[]; base=[]
  for held in range(8):
   tr=np.arange(8)!=held; X=x[tr].reshape(-1,256); labels=y[tr].reshape(-1); V=x[held]
   center=X.mean(0);X=X-center;V=V-center
   onehot=np.eye(256)[labels]; shuffled=np.random.default_rng(12000+held).permutation(onehot)
   Y=np.concatenate((onehot,shuffled),axis=1);prior=Y.mean(0)
   W=np.linalg.solve(X.T@X+np.eye(256),X.T@(Y-prior))
   pred=V@W+prior
   acc.append(float(np.mean(pred[:,:256].argmax(1)==y[held])))
   null.append(float(np.mean(pred[:,256:].argmax(1)==y[held])))
   base.append(float(np.mean(np.bincount(labels,minlength=256).argmax()==y[held])))
  row={'q':q,'seed':seed,'accuracy':float(np.mean(acc)),'null_accuracy':float(np.mean(null)),'majority_baseline':float(np.mean(base)),'accuracy_per_fold':acc,'capture_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};records.append(row);print(row,flush=True)
Path(__file__).with_name('within_run_probe.json').write_text(json.dumps({'status':'post-hoc exploratory known-plaintext leave-one-sequence-out ridge probe, lambda=1, no tuning','records':records},indent=2))
