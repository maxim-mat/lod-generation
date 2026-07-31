"""Subsampling the cleaned dataset for rapid training runs.

Two properties matter: the sample is reproducible from its seed, and it leans
towards small buildings, since graph cost is quadratic in node count and the
point of a mini set is that it trains quickly.
"""
import json

import numpy as np
import pytest

from src.sample_mini_dataset import (
    object_vertex_counts, sample_dataset, weighted_sample, write_subset,
)


def _solid(n_side, base=0):
    """A prism with 2*n_side vertices, so vertex count is controllable."""
    bottom = [base + i for i in range(n_side)]
    top = [base + n_side + i for i in range(n_side)]
    faces = [[bottom], [top]]
    for i in range(n_side):
        j = (i + 1) % n_side
        faces.append([[bottom[i], bottom[j], top[j], top[i]]])
    types = ["GroundSurface", "RoofSurface"] + ["WallSurface"] * n_side
    return {
        "type": "Solid", "lod": "2.2", "boundaries": [faces],
        "semantics": {"surfaces": [{"type": t} for t in types],
                      "values": [list(range(len(types)))]},
    }


def _cj(sizes):
    """One CityJSON holding an object per entry of ``sizes`` (vertices per side)."""
    objects, verts, base = {}, [], 0
    for i, n in enumerate(sizes):
        objects[f"b{i}"] = {"type": "Building", "geometry": [_solid(n, base)]}
        for k in range(2 * n):
            verts.append([k, i, 0])
        base += 2 * n
    return {"type": "CityJSON", "version": "1.1",
            "CityObjects": objects, "vertices": verts}


# --- counting -------------------------------------------------------------

def test_vertex_counts_are_the_distinct_outer_ring_vertices():
    cj = _cj([3, 5])
    assert object_vertex_counts(cj) == {"b0": 6, "b1": 10}


# --- the weighting --------------------------------------------------------

def test_small_objects_are_favoured():
    counts = {f"o{i}": (4 if i < 500 else 400) for i in range(1000)}
    picked = weighted_sample(counts, fraction=0.1, seed=1, alpha=1.0)
    small = sum(1 for k in picked if counts[k] == 4)
    assert len(picked) == 100
    # inverse-size weighting puts 100x the mass on each small object
    assert small > 90


def test_alpha_zero_is_uniform():
    counts = {f"o{i}": (4 if i < 500 else 400) for i in range(1000)}
    picked = weighted_sample(counts, fraction=0.2, seed=3, alpha=0.0)
    small = sum(1 for k in picked if counts[k] == 4)
    assert 70 < small < 130          # ~100 of 200 either way


def test_sampling_is_reproducible_and_seed_sensitive():
    counts = {f"o{i}": 1 + (i % 50) for i in range(400)}
    a = weighted_sample(counts, fraction=0.25, seed=7, alpha=1.0)
    b = weighted_sample(counts, fraction=0.25, seed=7, alpha=1.0)
    c = weighted_sample(counts, fraction=0.25, seed=8, alpha=1.0)
    assert a == b
    assert a != c


def test_fraction_of_zero_or_one_behaves():
    counts = {f"o{i}": 5 for i in range(10)}
    assert weighted_sample(counts, fraction=1.0, seed=1, alpha=1.0) == set(counts)
    assert weighted_sample(counts, fraction=0.0, seed=1, alpha=1.0) == set()


# --- writing --------------------------------------------------------------

def test_write_subset_keeps_only_the_chosen_objects_and_compacts():
    cj = _cj([3, 5])
    out = write_subset(cj, {"b1"})
    assert set(out["CityObjects"]) == {"b1"}
    assert len(out["vertices"]) == 10          # b0's 6 vertices dropped
    # boundaries must have been reindexed into the smaller array
    ids = {i for g in out["CityObjects"]["b1"]["geometry"]
           for sh in g["boundaries"] for f in sh for r in f for i in r}
    assert max(ids) < len(out["vertices"])


def test_write_subset_returns_none_when_nothing_is_kept():
    assert write_subset(_cj([3]), set()) is None


# --- end to end -----------------------------------------------------------

def _dataset(tmp_path, folders=("LOD2", "LOD1_synth")):
    for f in folders:
        d = tmp_path / "clean" / f / "Source A"
        d.mkdir(parents=True)
        (d / "t.json").write_text(json.dumps(_cj([3, 4, 5, 20])), encoding="utf-8")
    return tmp_path / "clean"


def test_pairing_is_preserved_across_folders(tmp_path):
    root = _dataset(tmp_path)
    stats = sample_dataset(root, tmp_path / "mini", fraction=0.5, seed=5, alpha=1.0)

    def ids(folder):
        p = tmp_path / "mini" / folder / "Source A" / "t.json"
        return set(json.loads(p.read_text(encoding="utf-8"))["CityObjects"])

    assert ids("LOD2") == ids("LOD1_synth")
    assert stats["sampled"] == 2


def test_output_is_readable_by_the_parser(tmp_path):
    from src.dataset.dataset import parse_cityjson_file_to_graphs

    root = _dataset(tmp_path, folders=("LOD2",))
    sample_dataset(root, tmp_path / "mini", fraction=0.75, seed=2, alpha=1.0)
    graphs = parse_cityjson_file_to_graphs(
        str(tmp_path / "mini" / "LOD2" / "Source A" / "t.json"))
    assert graphs
    for g in graphs.values():
        assert g["x"].shape[0] > 0


def test_manifest_records_the_seed(tmp_path):
    root = _dataset(tmp_path, folders=("LOD2",))
    sample_dataset(root, tmp_path / "mini", fraction=0.5, seed=11, alpha=1.0)
    manifest = json.loads(
        (tmp_path / "mini" / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["args"]["seed"] == 11
    assert manifest["args"]["alpha"] == 1.0
