"""Check RDF-to-numerical edge identity in addition to SHACL measurement rules."""
from collections import defaultdict
from rdflib import Namespace,RDF
from reference.core import RELATIONS
from .registry import OntologyRegistry

OR=Namespace('https://example.org/ontoresponse#')

def validate_numeric_graph(graph,experts):
    reg=OntologyRegistry();adj=reg.adjacency(experts);groups=defaultdict(dict);node_group={}
    for node,index in graph.subject_objects(OR.numericNodeIndex):
        value=int(index)
        if not 0<=value<adj.shape[-1]:raise ValueError('Numeric node index out of range')
        if (node,RDF.type,OR.ResponseObservation) in graph:
            before=graph.value(node,OR.hasBefore);root=graph.value(before,OR.rootRecording)
        else:root=graph.value(node,OR.rootRecording)
        if root is None or value in groups[root]:raise ValueError('Missing root/duplicate numeric node index')
        if str(graph.value(node,OR.registryHash))!=reg.digest:raise ValueError('Registry hash mismatch')
        groups[root][value]=node;node_group[node]=root
    expected=set()
    for root,nodes in groups.items():
        for ri,name in enumerate(RELATIONS):
            for target,source in adj[ri].nonzero().tolist():
                if target in nodes and source in nodes:expected.add((nodes[source],OR[name],nodes[target]))
    actual={(s,OR[name],t) for name in RELATIONS for s,t in graph.subject_objects(OR[name])}
    if actual!=expected:raise ValueError('RDF edges differ from registry compiled numerical topology')
    return {'registry_hash':reg.digest,'recordings':len(groups),'edges':len(actual)}
