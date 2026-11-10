"""Historical environment operators are audit-only, separate from probe metadata."""
from itertools import product
from ..common import digest

# A trace is ordered physical operations, not dataset/environment labels.
def exposure_audit(train_traces,probe_operations,candidates,expert_augmentations=None):
    exposed={tuple(trace) for trace in train_traces}
    exposed|={tuple(trace)+(p,) for trace,p in product(train_traces,probe_operations)}
    operators={op for trace in exposed for op in trace}
    results=[]
    for trace in candidates:
        key=tuple(trace)
        if key in exposed:status='exposed_in_T_or_P_of_T'
        elif any(op not in operators for op in key):status='fusion_unseen_operator'
        else:status='fusion_unseen_ordered_composition'
        results.append({'trace':trace,'classification':status,
                        'all_experts_unseen':None if expert_augmentations is None else all(op not in expert_augmentations for op in key)})
    return {'training_closure':sorted([list(t) for t in exposed]),'candidates':results,
            'expert_augmentations':expert_augmentations,'hash':digest([train_traces,probe_operations,candidates,expert_augmentations])}
