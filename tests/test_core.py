import inspect
import pytest
import torch
from reference.core import (compile_relations, ExpertStats, ResponseTokenizer,
                            RelationalLayer, EvidenceFusion, source_weighted_bce)

torch.set_num_threads(1)
FAMILIES = ["bandwidth", "bandwidth", "noise", "noise", "coding", "coding"]


def fixture(b=4):
    torch.manual_seed(11)
    hidden = [torch.randn(b, 7, d) for d in [5, 7, 9]]
    scores = torch.randn(b, 3, 7)
    valid = torch.ones_like(scores, dtype=torch.bool)
    probe = torch.randn(6, 4)
    quality = torch.randn(b, 4)
    return hidden, scores, valid, probe, quality


def model(mode="typed"):
    torch.manual_seed(13)
    return EvidenceFusion([5,7,9], FAMILIES, 4, width=16, mode=mode, dropout=0).eval()


def test_graph_shape_and_counts():
    a = compile_relations(3, FAMILIES)
    assert a.shape == (4,21,21)
    assert a.sum((1,2)).tolist() == [18,18,36,18]


def test_family_rename_is_not_ablation():
    assert torch.equal(compile_relations(3,FAMILIES), compile_relations(3,[7,7,2,2,9,9]))


def test_random_membership_changes_edges_preserving_degree():
    a = compile_relations(3,FAMILIES)
    b = compile_relations(3,[0,1,0,2,1,2])
    assert not torch.equal(a,b)
    assert torch.equal(a.sum(-1),b.sum(-1))


def test_cross_expert_edges_only_same_probe():
    a=compile_relations(3,FAMILIES)
    for target,source in torch.nonzero(a[2]).tolist():
        assert (target-3)%6==(source-3)%6
        assert (target-3)//6!=(source-3)//6


@pytest.mark.parametrize("n,f", [(0,FAMILIES),(3,[])])
def test_bad_graph(n,f):
    with pytest.raises(ValueError): compile_relations(n,f)


def test_scaler_role_guard():
    h,s,v,_,_=fixture()
    with pytest.raises(ValueError): ExpertStats.fit(h,s,v,role="test")


def test_scaler_shared_between_views():
    h,s,v,_,_=fixture()
    stats=ExpertStats.fit(h,s,v,role="fusion_train")
    hs,ss=stats.transform(h,s,v)
    assert torch.allclose(hs[0][:,1]-hs[0][:,0],(h[0][:,1]-h[0][:,0])/stats.hidden_scales[0],atol=1e-5)
    assert torch.allclose(ss[:,0,1]-ss[:,0,0],(s[:,0,1]-s[:,0,0])/stats.score_scales[0],atol=1e-5)


def test_scaler_is_frozen_after_test_transform():
    h,s,v,_,_=fixture()
    stats=ExpertStats.fit(h,s,v,role="fusion_train")
    before=[x.clone() for x in stats.hidden_means]
    stats.transform([x+100 for x in h],s+100,v)
    assert all(torch.equal(a,b) for a,b in zip(before,stats.hidden_means))


def test_token_count():
    h,s,v,p,_=fixture()
    z,m=ResponseTokenizer([5,7,9],6,4,16)(h,s,v,p)
    assert z.shape==(4,21,16)
    assert m.shape==(4,21)


def test_missing_original_invalidates_responses():
    h,s,v,p,_=fixture()
    v[:,1,0]=False
    z,m=ResponseTokenizer([5,7,9],6,4,16)(h,s,v,p)
    assert not m[:,1].any()
    assert not m[:,9:15].any()
    assert (z[:,9:15]==0).all()


@pytest.mark.parametrize("mode",["typed","untyped","none","flat","set_attention"])
def test_all_modes_forward_backward(mode):
    args=fixture()
    net=model(mode)
    out=net(*args)
    assert out.logits.shape==(4,)
    assert torch.isfinite(out.probabilities()).all()
    loss=source_weighted_bce(out,torch.tensor([0.,1.,0.,1.]))
    loss.backward()
    grads=[p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("mode",["typed","flat","set_attention"])
def test_no_evidence_has_no_probability(mode):
    h,s,v,p,q=fixture()
    v[0]=False
    out=model(mode)(h,s,v,p,q)
    assert torch.isnan(out.probabilities()[0])
    assert torch.isfinite(out.probabilities()[1:]).all()
    with pytest.raises(ValueError): source_weighted_bce(out,torch.zeros(4))


@pytest.mark.parametrize("mode",["typed","untyped","flat","set_attention"])
def test_masked_placeholder_does_not_change_prediction(mode):
    h,s,v,p,q=fixture()
    v[:,0,2]=False
    net=model(mode)
    expected=net(h,s,v,p,q).logits
    h[0][:,2]=float("nan")
    s[:,0,2]=float("nan")
    got=net(h,s,v,p,q).logits
    assert torch.allclose(expected,got,atol=1e-6)


def test_nonfinite_valid_rejected():
    h,s,v,p,q=fixture()
    s[0,0,0]=float("nan")
    with pytest.raises(ValueError): model()(h,s,v,p,q)


def test_identity_views_raw_deltas_zero():
    h,s,v,p,_=fixture()
    h=[x[:,:1].expand(-1,7,-1).clone() for x in h]
    s=s[:,:,:1].expand(-1,-1,7).clone()
    stat=ExpertStats.fit(h,s,v,role="fusion_train")
    hh,ss=stat.transform(h,s,v)
    assert all(torch.equal(x[:,1:]-x[:,:1],torch.zeros_like(x[:,1:])) for x in hh)
    assert (ss[:,:,1:]-ss[:,:,:1]==0).all()


def test_graph_layer_node_permutation_equivariant():
    torch.manual_seed(8)
    layer=RelationalLayer(8,4,0).eval()
    z=torch.randn(2,21,8)
    m=torch.ones(2,21,dtype=torch.bool)
    m[0,7]=False
    a=compile_relations(3,FAMILIES)
    perm=torch.randperm(21)
    expected=layer(z,m,a)[:,perm]
    actual=layer(z[:,perm],m[:,perm],a[:,perm][:,:,perm])
    assert torch.allclose(expected,actual,atol=1e-5)


def test_topology_has_numeric_effect():
    args=fixture()
    net=model()
    expected=net(*args).logits
    with torch.no_grad(): net.adjacency[2].zero_()
    actual=net(*args).logits
    assert not torch.allclose(expected,actual)


def test_save_load_equivalence(tmp_path):
    args=fixture()
    net=model()
    torch.save(net.state_dict(),tmp_path/"head.pt")
    other=model()
    other.load_state_dict(torch.load(tmp_path/"head.pt",weights_only=True))
    assert torch.equal(net(*args).logits,other(*args).logits)


def test_source_weights_math():
    args=fixture()
    out=model()(*args)
    y=torch.tensor([0.,1.,0.,1.])
    w=torch.tensor([1.,.5,.5,1.])
    losses=torch.nn.functional.binary_cross_entropy_with_logits(out.logits,y,reduction="none")
    assert torch.allclose(source_weighted_bce(out,y,w),(losses*w).sum()/w.sum())


def test_direct_classifier_not_convex_score_average():
    h,s,v,p,q=fixture()
    s.fill_(10.)
    net=model()
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.fill_(-5.)
    assert (net(h,s,v,p,q).probabilities()<.01).all()


def test_forward_has_no_truth_or_env_argument():
    names=set(inspect.signature(EvidenceFusion.forward).parameters)
    assert names=={"self","hidden","scores","valid","probe_features","quality"}


def test_bad_label_rejected():
    out=model()(*fixture())
    with pytest.raises(ValueError): source_weighted_bce(out,torch.tensor([0.,2.,0.,1.]))


def test_bad_probe_metadata_rejected():
    h,s,v,p,q=fixture()
    with pytest.raises(ValueError): model()(h,s,v,p[:3],q)
