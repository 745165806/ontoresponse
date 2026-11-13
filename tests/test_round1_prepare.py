from scripts.prepare_round1 import development_pool, select_fold


def test_source_connections_precede_fold_sampling(tmp_path):
    """A dev VC target joins its train source speaker, including unseen rows."""
    protocol = tmp_path / "ASVspoof2019_LA_cm_protocols"
    protocol.mkdir()
    (protocol / "ASVspoof2019.LA.cm.train.trn.txt").write_text(
        "x a - - bonafide\nx b - A01 spoof\nx c - - bonafide\nx d - A01 spoof\n"
    )
    (protocol / "ASVspoof2019.LA.cm.dev.trl.txt").write_text("x e - A02 spoof\n")

    def entry(bona="-", target="-", vc="-", text="-"):
        return dict(
            VCTK_ID=bona,
            TTS_VC_target_speaker=target,
            VC_source_VCTK_ID=vc,
            TTS_text=text,
        )

    mapping = {
        "a": entry(bona="p1_001"),
        "b": entry(target="p1", text="hello"),
        "c": entry(bona="p2_001"),
        "d": entry(target="p2", text="hello"),
        "e": entry(target="p3", vc="p1_001"),
    }
    pool = development_pool(tmp_path, mapping)
    by_id = {r["id"]: r for r in pool}
    assert by_id["a"]["source_group"] == by_id["e"]["source_group"]
    assert by_id["a"]["source_group"] != by_id["c"]["source_group"]
    first = select_fold(pool, count=4, fold=0)
    second = select_fold(pool, count=4, fold=1)
    assert {r["id"] for r in first} == {r["id"] for r in second}
    assert all(a["role"] != b["role"] for a, b in zip(first, second))
    assert {r["role"] for r in first} == {"fusion_train", "fusion_val"}
    assert all("eval" not in r["audio"] for r in first)
