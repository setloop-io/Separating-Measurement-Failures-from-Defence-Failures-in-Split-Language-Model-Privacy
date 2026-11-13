"""Tiny CPU data-only checks against the exact native party property definitions.

No neural model, forward/backward pass, optimizer update or CUDA call is made.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from native_pipeline import module_inventory,optimizer_inventory,scheduler_inventory


def check(source,output):
    import torch
    tree = ast.parse(source.read_text())
    native = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Party')
    selected = {'local_model','local_model_tail','global_model','local_model_optimizer','local_model_tail_optimizer','global_model_optimizer'}
    methods = [n for n in native.body if isinstance(n,ast.FunctionDef) and n.name in selected]
    synthetic = ast.ClassDef(name='NativeProperties',bases=[],keywords=[],body=methods,decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[synthetic],type_ignores=[]))
    namespace = {}
    exec(compile(module,str(source),'exec'),namespace)
    NativeProperties = namespace['NativeProperties']
    class TinyDataModule:
        def __init__(self):
            self.vector = torch.tensor([1.,2.],requires_grad=True)
            self.scalar = torch.tensor(3.)
        def parameters(self):
            return iter([self.vector])
        def state_dict(self):
            return {'vector':self.vector,'scalar':self.scalar}
    class OptimizerReceipt:
        param_groups = [{'lr':0.001,'betas':(0.9,0.999),'params':[]}]
        def state_dict(self):
            return {'state':{},'param_groups':[{'lr':0.001,'params':[]}]}
    class SchedulerReceipt:
        last_epoch = 0
        def get_last_lr(self):
            return [0.001]
        def state_dict(self):
            return {'last_epoch':0,'_last_lr':[0.001]}
    passive,active,empty = NativeProperties(),NativeProperties(),NativeProperties()
    for party,key in [(passive,0),(active,1)]:
        party.models = {key:TinyDataModule()}
        party.optimizers = {key:OptimizerReceipt()}
        party.lr_schedulers = {key:SchedulerReceipt()}
    empty.models,empty.optimizers,empty.lr_schedulers = {},{0:None},{0:None}
    passive.head_imagined_adversary_optimizer = OptimizerReceipt()
    # Demonstrate that the real upstream property still raises on this fixture;
    # successful reporting therefore proves the observer avoided that property.
    raised = False
    try:
        passive.global_model_optimizer
    except KeyError:
        raised = True
    assert raised
    parties = [passive,active,empty]
    modules = module_inventory(parties,hash_weights=False)
    hashed = module_inventory(parties,hash_weights=True)
    optimizers = optimizer_inventory(parties,include_state=False)
    optimizer_states = optimizer_inventory(parties,include_state=True)
    schedulers = scheduler_inventory(parties,include_state=False)
    scheduler_states = scheduler_inventory(parties,include_state=True)
    assert [(x['party'],x['name']) for x in modules]==[(0,'local_model'),(1,'global_model')]
    assert len(hashed)==2 and all(len(x['state_sha256'])==64 for x in hashed)
    assert [(x['party'],x['name']) for x in optimizers]==[(0,'optimizers[0]'),(0,'head_imagined_adversary_optimizer'),(1,'optimizers[1]')]
    assert all('state_dict' in x for x in optimizer_states)
    assert len(schedulers)==len(scheduler_states)==2 and all('state_dict' in x for x in scheduler_states)
    # Duplicate references in a dictionary/AT alias produce one state receipt.
    passive.tail_imagined_adversary_optimizer = passive.optimizers[0]
    assert len(optimizer_inventory(parties,include_state=True))==3
    result = {'status':'PASS','native_property_source':str(source),
              'native_property_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'native_property_function_AST_sha256':hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in methods).encode()).hexdigest(),
              'native_global_optimizer_property_raises_KeyError_on_passive_fixture':True,
              'observer_avoids_alias_property':True,'all_three_inventory_functions_checked':True,
              'metadata_and_state_paths_checked':True,'passive_key0_active_key1_empty_None_checked':True,
              'imagined_adversary_optimizer_and_identity_deduplication_checked':True,
              'scalar_and_vector_state_hashes_checked':True,
              'neural_models_constructed':0,'optimizer_updates':0,'cuda_calls':0,
              'torch_version':torch.__version__,
              'native_pipeline_sha256':hashlib.sha256(Path(__file__).with_name('native_pipeline.py').read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    print(json.dumps(check(Path(args.upstream)/'src/party/llm_party.py',Path(args.output)),indent=2))
