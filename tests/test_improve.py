import numpy as np

import improve


def test_expand_crosses_ensembles_with_postprocessing_grid():
    spec = {"ensembles": [{"name": "a", "members": ["m1"]},
                          {"name": "ab", "members": ["m1", "m2"], "weights": [2, 1]}],
            "postprocess": [{"threshold": [0.5, 0.4], "bridge_radius": 8, "min_area": [240, 320]}]}
    candidates = improve.expand(spec)
    assert len(candidates) == 8
    assert candidates[0]["members"] == ["m1"] and candidates[0]["weights"] == [1.0]
    assert {c["threshold"] for c in candidates} == {0.5, 0.4}
    assert all(c["postprocess"]["bridge_radius"] == 8 for c in candidates)
    assert candidates[-1]["weights"] == [2, 1]
    assert len({c["name"] for c in candidates}) == len(candidates)


def test_convert_and_load_cached_round_trip(tmp_path):
    source = tmp_path / "predictions"
    source.mkdir()
    probability = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    np.savez_compressed(source / "20120101000000Bh.npz", probability=probability,
                        instances=np.zeros((16, 16), np.uint32), observation="20120101000000Bh")
    improve.convert_command(type("Args", (), dict(predictions=str(source), output=str(tmp_path / "cache"),
                                                  workers=1))())
    restored = improve.load_cached(tmp_path / "cache" / "val", "20120101000000Bh")
    assert restored.shape == probability.shape
    assert np.abs(restored - probability).max() <= 0.5 / improve.QUANT + 1e-7
