"""
Layer 5 - Metrics Extractor
===========================

Extracts objective geometry metrics from a single GLB. This is the data engine
behind the 6B objective pipeline: "looks good" != "good geometry".

Toolchain (no Blender needed):
    trimesh    -> counts, watertight, winding, volume, area, bbox, UV stretch
    pymeshlab  -> topology integrity (non-manifold, boundary edges, components,
                  holes, genus) via a geometry-only PLY bridge (pymeshlab's GLB
                  loader chokes on embedded textures, so we hand it clean PLY)
    scipy      -> symmetry score via mirrored nearest-neighbour distance

All metrics are RAW here. Normalization to 0-100 happens later in Layer 6B,
where models are compared against each other.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import trimesh
import pymeshlab
from scipy.spatial import cKDTree


def _safe(v):
    """Convert numpy / non-finite values into JSON-safe Python scalars."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _trimesh_metrics(mesh: trimesh.Trimesh) -> dict:
    extents = mesh.bounding_box.extents          # (x, y, z)
    aspect = float(max(extents) / max(min(extents), 1e-9))
    return {
        "face_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": _safe(mesh.euler_number),
        "volume": _safe(float(mesh.volume)) if mesh.is_watertight else None,
        "surface_area": _safe(float(mesh.area)),
        "bbox_extents": [_safe(float(x)) for x in extents],
        "bbox_aspect_ratio": _safe(aspect),
    }


def _uv_stretch(mesh: trimesh.Trimesh) -> dict:
    """
    UV quality proxy: variation in texel density across faces.

    Compares each face's UV area to its 3D area; high coefficient of variation
    => uneven texture stretching. Returns has_uv + uv_stretch_cv (lower better).
    """
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return {"has_uv": False, "uv_stretch_cv": None}

    f = mesh.faces
    tri3d = mesh.vertices[f]                      # (F,3,3)
    a3 = 0.5 * np.linalg.norm(
        np.cross(tri3d[:, 1] - tri3d[:, 0], tri3d[:, 2] - tri3d[:, 0]), axis=1)

    triuv = np.asarray(uv)[f]                     # (F,3,2)
    e1 = triuv[:, 1] - triuv[:, 0]
    e2 = triuv[:, 2] - triuv[:, 0]
    auv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    valid = a3 > 1e-12
    ratio = auv[valid] / a3[valid]
    if ratio.size == 0 or ratio.mean() == 0:
        return {"has_uv": True, "uv_stretch_cv": None}
    cv = float(ratio.std() / ratio.mean())
    return {"has_uv": True, "uv_stretch_cv": _safe(cv)}


def _pymeshlab_metrics(mesh: trimesh.Trimesh) -> dict:
    """Topology integrity via a geometry-only PLY bridge."""
    tmp = os.path.join(tempfile.gettempdir(), f"_l5_{os.getpid()}.ply")
    try:
        mesh.export(tmp)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp)
        tm = ms.get_topological_measures()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    def clean(key):
        v = tm.get(key)
        # pymeshlab returns -1 when a measure can't be computed (e.g. holes on
        # a non-manifold mesh). Surface that as None rather than a fake number.
        return None if v in (-1, None) else _safe(v)

    return {
        "non_manifold_edges": clean("non_two_manifold_edges"),
        "non_manifold_vertices": clean("non_two_manifold_vertices"),
        "boundary_edges": clean("boundary_edges"),
        "connected_components": clean("connected_components_number"),
        "holes": clean("number_holes"),
        "genus": clean("genus"),
    }


def _symmetry(mesh: trimesh.Trimesh, sample: int = 4000) -> dict:
    """
    Reflective symmetry score in [0,1]: 1 = perfectly symmetric.

    Mirror a sample of vertices across each principal axis (through centroid),
    measure nearest-neighbour distance to the original surface, normalize by the
    bounding-box diagonal. Best axis wins.
    """
    v = mesh.vertices
    if len(v) < 4:
        return {"symmetry_score": None, "symmetry_axis": None}

    center = v.mean(axis=0)
    vc = v - center
    diag = float(np.linalg.norm(mesh.bounding_box.extents)) or 1.0

    idx = np.random.default_rng(0).choice(
        len(vc), size=min(sample, len(vc)), replace=False)
    ref = vc[idx]
    tree = cKDTree(ref)

    best_score, best_axis = -1.0, None
    for axis, name in enumerate("XYZ"):
        mirrored = ref.copy()
        mirrored[:, axis] *= -1.0
        dist, _ = tree.query(mirrored)
        score = 1.0 - float(np.mean(dist)) / diag
        score = max(0.0, min(1.0, score))
        if score > best_score:
            best_score, best_axis = score, name
    return {"symmetry_score": _safe(best_score), "symmetry_axis": best_axis}


def extract_metrics(glb_path: str) -> dict:
    """
    Load a GLB and return the full raw metric dict. Raises on unloadable mesh
    (this is also the de-facto mesh-validity gate for the pipeline).
    """
    mesh = trimesh.load(glb_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Not a usable mesh: {glb_path}")

    metrics = {}
    metrics.update(_trimesh_metrics(mesh))
    metrics.update(_uv_stretch(mesh))
    metrics.update(_pymeshlab_metrics(mesh))
    metrics.update(_symmetry(mesh))
    return metrics


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "../TEST CASES/TC001 GLB/TC001_Trellis 2.glb"
    print(json.dumps(extract_metrics(path), indent=2))
