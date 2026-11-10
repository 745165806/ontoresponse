import pytest
from scripts.recompute_predictions import independent_metrics,sigmoid

def table(scores,labels=None):
    labels=[0,0,1,1] if labels is None else labels
    return [{'id':str(i),'status':'scored','label':y,'logit':s,'p':sigmoid(s),'threshold':.5} for i,(s,y) in enumerate(zip(scores,labels))]

@pytest.mark.parametrize('scores,expected',[([-2,-1,1,2],0.),([2,1,-1,-2],1.),([0,0,0,0],.5)])
def test_perfect_reverse_and_ties(scores,expected):
    measured=independent_metrics(table(scores));assert measured['eer']==expected
    assert measured['auroc']==1-expected
    assert measured['spoof_auprc']>=.4

@pytest.mark.parametrize('records',[[],table([0,1],[1,1]),table([float('nan'),1,2,3])])
def test_undefined_or_nonfinite_not_counted_as_success(records):
    with pytest.raises(ValueError):independent_metrics(records)


def test_positive_monotonic_calibration_does_not_flip_ranking():
    rows=table([-2,-1,1,2]);before=independent_metrics(rows)
    for r in rows:r['logit']=r['logit']*1.4+.2;r['p']=sigmoid(r['logit'])
    after=independent_metrics(rows);assert before['eer']==after['eer'] and before['auroc']==after['auroc']


def test_failed_rows_preserved_in_coverage():
    rows=table([-2,-1,1,2]);rows.append({'id':'failed','status':'review','label':1,'p':None,'logit':None,'threshold':.5})
    result=independent_metrics(rows);assert result['n_requested']==5 and result['coverage']==.8
