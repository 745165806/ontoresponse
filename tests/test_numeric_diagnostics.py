import math
from collections import Counter
import pytest,torch,numpy as np
from ontoresponse.train.sampling import origin_uniform_epoch
from ontoresponse.evidence.diagnostics import scaler_diagnostics
from ontoresponse.models.factory import build_model
from ontoresponse.ontology.registry import OntologyRegistry
from reference.core import ExpertStats,FusionOutput,source_weighted_bce


def test_origin_sampler_equal_weight_missing_versions_and_hand_loss():
    origins=['a','a','a','b','c','c'];versions=['a0','a1','a2','b0','c0','c1'];valid=[True,False,True,True,False,True]
    selected=origin_uniform_epoch(origins,versions,valid,17,2)
    assert Counter(origins[i] for i in selected)=={'a':1,'b':1,'c':1}
    logits=torch.tensor([-.5,100.,.8,-1.,100.,1.2]);labels=torch.tensor([0.,0.,0.,1.,1.,1.])
    expected=sum(math.log1p(math.exp(float(logits[i])))-float(labels[i]*logits[i]) for i in selected)/3
    result=source_weighted_bce(FusionOutput(logits[selected],torch.ones(3,dtype=torch.bool)),labels[selected])
    assert abs(float(result)-expected)<1e-6
    assert selected==origin_uniform_epoch(origins,versions,valid,17,2)
    with pytest.raises(RuntimeError):origin_uniform_epoch(origins,versions,[False]*6,17,2)

@pytest.mark.parametrize('name',['original_mlp','response_mlp','all_views_mlp','typed_response','relation_tied','set_attention','untyped_response'])
def test_fusion_eval_batch_size_invariance(name):
    torch.manual_seed(17);reg=OntologyRegistry();model=build_model([3,5],reg,name,16).eval()
    h=[torch.randn(4,7,d) for d in (3,5)];s=torch.randn(4,2,7);v=torch.ones_like(s,dtype=torch.bool);q=torch.zeros(4,4)
    together=model(h,s,v,reg.features,q).logits
    separate=torch.cat([model([x[i:i+1] for x in h],s[i:i+1],v[i:i+1],reg.features,q[i:i+1]).logits for i in range(4)])
    torch.testing.assert_close(together,separate,atol=1e-6,rtol=1e-5)


def test_scaler_diagnostics_train_only():
    from types import SimpleNamespace
    h=[torch.randn(4,7,3)];h[0][:,:,0]=2;score=torch.randn(4,1,7);valid=torch.ones_like(score,dtype=torch.bool)
    batch=SimpleNamespace(hidden=h,scores=score,valid=valid,inputs=lambda:(h,score,valid))
    stats=ExpertStats.fit(h,score,valid,role='fusion_train');report=scaler_diagnostics(batch,stats,['unit'],'fusion_train')
    assert report['records'][0]['hidden_floor_fraction']>=1/3
    assert not report['records'][0]['transform_changed']
    with pytest.raises(ValueError):scaler_diagnostics(batch,stats,['unit'],'test')


def test_cache_quota_is_hard_and_does_not_leave_observation(tmp_path):
    import numpy as np
    from ontoresponse.evidence.cache import Cache
    cache=Cache(tmp_path,max_bytes=1);spec={'unit':'quota'};key=cache.key(spec)
    with pytest.raises(RuntimeError,match='CACHE_QUOTA'):cache.put(key,spec,{'h':np.ones(10,dtype='float32')},{})
    assert cache.get(key) is None and not list(tmp_path.glob('*.npz'))

@pytest.mark.parametrize('alias,old',[('full_attention','set_attention'),('untyped_union','untyped_response')])
def test_accurate_mode_aliases_keep_original_computation(alias,old):
    reg=OntologyRegistry();torch.manual_seed(17);a=build_model([3,5],reg,alias,16).eval()
    torch.manual_seed(17);b=build_model([3,5],reg,old,16).eval()
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])

@pytest.mark.parametrize('name',['no_embedding_response','no_score_response','identity_copy_views'])
def test_information_controls_preserve_missing_before_semantics(name):
    reg=OntologyRegistry();model=build_model([3,5],reg,name,16).eval()
    h=[torch.randn(2,7,d) for d in (3,5)];s=torch.randn(2,2,7);v=torch.ones_like(s,dtype=torch.bool);v[0,:,0]=False
    for i in range(2):h[i][0,0]=float('nan')
    s[0,:,0]=float('nan')
    out=model(h,s,v,reg.features,torch.zeros(2,4))
    assert not out.observed[0] and torch.isfinite(out.logits).all()
