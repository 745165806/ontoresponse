"""Four original directed channels with only their transformation matrix tied.

This is distinct from untyped_response, which unions edges and changes the
normalization. Inherit the typed forward so neighbors, per-relation masked mean,
self map, residual, nonlinearity, dropout, LayerNorm and readout stay identical.
"""
import torch
from torch import nn
from reference.core import EvidenceFusion,RelationalLayer

class RelationTiedLayer(RelationalLayer):
    def __init__(self,width,n_relations=4,dropout=.1):
        if n_relations!=4:raise ValueError('relation_tied keeps all four typed channels')
        super().__init__(width,n_relations,dropout)
        shared=self.maps[0]
        self.maps=nn.ModuleList([shared for _ in range(n_relations)])

class RelationTiedFusion(EvidenceFusion):
    def __init__(self,expert_dims,families,probe_feature_dim,quality_dim=4,width=64,layers=2,dropout=.1):
        super().__init__(expert_dims,families,probe_feature_dim,quality_dim,width,layers,'typed',dropout)
        # Preserve even the seeded self/norm/adapter/head initialization.
        # The sole model change is aliasing each layer's four relation maps.
        for layer in self.layers:
            layer.maps=nn.ModuleList([layer.maps[0] for _ in range(4)])


def copy_tied_to_typed(tied,typed):
    if not isinstance(tied,RelationTiedFusion) or typed.mode!='typed':raise ValueError('Expected tied -> typed models')
    typed.load_state_dict(tied.state_dict(),strict=True)
    return typed

def copy_equal_typed_to_tied(typed,tied):
    if not isinstance(tied,RelationTiedFusion) or typed.mode!='typed':raise ValueError('Expected typed -> tied models')
    for layer in typed.layers:
        if len(layer.maps)!=4 or any(not torch.equal(layer.maps[0].weight,m.weight) for m in layer.maps[1:]):
            raise ValueError('Four typed matrices must be equal before an exact tied copy')
    tied.load_state_dict(typed.state_dict(),strict=True)
    return tied
