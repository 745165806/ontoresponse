"""Run from /tmp using a wheel-installed package and only the dependency layer."""
from pathlib import Path
import argparse,json,torch,yaml,ontoresponse
from ontoresponse.resources import resource_path
from ontoresponse.ontology.registry import OntologyRegistry
from ontoresponse.train.fit import restore,forward
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.models.factory import build_model
from ontoresponse.ontology.validate_graph import validate_numeric_graph
from rdflib import Graph
from pyshacl import validate

p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True);args=p.parse_args()
repo=Path(args.repo);assert Path.cwd()==Path('/tmp')
assert Path(ontoresponse.__file__).is_relative_to('${EXTERNAL_PATH}')
assert resource_path('configs/probes.yaml').is_relative_to('${EXTERNAL_PATH}')
cfg=yaml.safe_load((repo/'configs/local.yaml').read_text());cfg['cache']=str(repo/cfg['cache']);torch.set_num_threads(2)
cp,model,stats,reg=restore(repo/'reports/implementation/runs/typed_response-fixed-seed17/model.pt')
batch=load_batch(cfg,repo/'reports/implementation/extraction-0-of-1.jsonl',repo/'reports/implementation/audit_manifest.jsonl','fusion_val',cp['dims'])
with torch.inference_mode():out=forward(model,batch,stats,reg)
assert torch.isfinite(out.logits).all() and out.observed.all()
tied=build_model([3,5],reg,'relation_tied',width=16);assert tied.adjacency.shape[0]==4
from ontoresponse.train.pilot import operating_point
from ontoresponse.pilot_gate import preflight
assert operating_point([0,1],[20.,40.],'fusion_val',.1)['space']=='calibrated_logit_float64'
graph=Graph().parse(repo/'reports/implementation/evidence.ttl');valid=validate(graph,shacl_graph=str(resource_path('ontology/shapes.ttl')),inference='rdfs')[0];assert valid
result={'status':'passed','cwd':str(Path.cwd()),'package':ontoresponse.__file__,'resource':str(resource_path('configs/probes.yaml')),
        'real_cached_validation_requests':len(batch.ids),'finite_logits':True,'relation_tied_channels':4,'SHACL':valid,
        'numeric_graph':validate_numeric_graph(graph,3),'pilot_api_installed':True,'no_calibration_or_fitting':True}
Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
