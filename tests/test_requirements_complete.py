"""Every third-party import in src/ must be pinned in requirements.txt.

Two dependencies reached remote unpinned because they happened to be installed
in the dev environment: `mapbox-earcut` (cjio's and trimesh's triangulation
backend) and `rtree` (trimesh's spatial index). Both are *extras* of packages we
do pin, so pinning the parent is not enough.

This catches the direct-import half automatically. The transitive half --
a backend reached *through* a pinned package -- cannot be found by scanning
imports, so those get an explicit runtime guard instead (see
`geometry.triangulate_face` and `mesh_metrics._closest_surface_distance`).
"""
import ast
import sys
from pathlib import Path

import pytest

# import name -> distribution name, where they differ.
ALIAS = {
    "yaml": "pyyaml", "sklearn": "scikit-learn", "PIL": "pillow",
    "cv2": "opencv-python", "mapbox_earcut": "mapbox-earcut",
    "vector_quantize_pytorch": "vector-quantize-pytorch",
}


def normalise(name):
    return name.lower().replace("_", "-")


def third_party_imports():
    out = {}
    for path in Path("src").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "src":
                    out.setdefault(top, set()).add(path.as_posix())
    return out


def pinned():
    text = Path("requirements.txt").read_text(encoding="utf-8")
    names = set()
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        names.add(normalise(line.split("[")[0].split(">")[0].split("=")[0]
                            .split("<")[0].split(";")[0].strip()))
    return names


@pytest.mark.parametrize("module", sorted(third_party_imports()))
def test_import_is_pinned(module):
    have = pinned()
    dist = normalise(ALIAS.get(module, module))
    users = ", ".join(sorted(third_party_imports()[module])[:3])
    assert dist in have, (
        f"src imports {module!r} but requirements.txt does not pin {dist!r} "
        f"(used by {users}). It probably works for you only because it is "
        f"already in your environment.")


def test_the_two_extras_that_bit_us_are_pinned():
    """Transitive backends an import scan cannot see."""
    have = pinned()
    for dist, why in (("mapbox-earcut", "cjio/trimesh triangulation engine"),
                      ("rtree", "trimesh spatial index for closest_point")):
        assert dist in have, f"{dist} unpinned ({why})"
