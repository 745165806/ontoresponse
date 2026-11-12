"""Conservative speaker/source connected components for audited split design."""

from ..common import digest


def connected_groups(rows, source_links):
    """source_links maps recording IDs to COMPLETE source IDs, including derivatives.

    Missing coverage blocks a certified split. Speaker labels and source IDs are
    namespaced by corpus in the caller; a coarse corpus name is not a source ID.
    """
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording")
    if any(not source_links.get(r["id"]) or not r.get("speaker_id") for r in rows):
        raise ValueError("BLOCKED: incomplete source/speaker mapping")
    if any(
        not isinstance(source_links[r["id"]], (list, tuple))
        or any(not isinstance(s, str) or not s for s in source_links[r["id"]])
        for r in rows
    ):
        raise ValueError("Source links must be nonempty lists of certified source IDs")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        a = ("recording", row["id"])
        find(a)
        union(a, ("speaker", row["speaker_id"]))
        for speaker in row.get("linked_speaker_ids", []):
            union(a, ("speaker", speaker))
        for source in source_links[row["id"]]:
            union(a, ("source", source))
        union(a, ("origin", row.get("origin_id", row["id"])))
    members = {}
    for node in parent:
        members.setdefault(find(node), []).append(node)
    keys = {root: "cc_" + digest(sorted(nodes))[:20] for root, nodes in members.items()}
    return {r["id"]: keys[find(("recording", r["id"]))] for r in rows}
