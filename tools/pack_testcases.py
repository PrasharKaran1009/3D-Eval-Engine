#!/usr/bin/env python3
"""
Benchmark mesh packer — TEXTURE-ONLY GLB compression + per-case zips
===================================================================

Turns the loose benchmark meshes in ``TEST CASES/TC00x GLB/`` into five
git-shippable archives ``TEST CASES/TC00x.zip``, each holding that case's 7
compressed ``.glb`` files and each comfortably under GitHub's 100 MB hard
file-size limit.

Why texture-only
----------------
The 5B objective pipeline reads every mesh with ``trimesh.load()`` and derives
``face_count`` / ``is_watertight`` / ``non_manifold_*`` / ``symmetry_score`` /
``uv_stretch_cv`` from it. Two consequences drive every choice below:

1. **No geometry may change.** Only the *textures* are recompressed, so every
   vertex, index and UV survives byte-for-byte and the objective metrics stay
   bit-identical to the original benchmark. Nothing here welds, joins,
   simplifies, quantizes or reorders. In particular ``gltf-transform optimize``
   is deliberately NOT used: it defaults to simplify + weld + join + palette
   AND to Meshopt geometry compression.

2. **No compression trimesh cannot read.** trimesh cannot decode Draco,
   ``EXT_meshopt_compression`` or KTX2/Basis textures — any of those would turn
   the whole benchmark into unloadable files. WebP is safe: it rides on
   ``EXT_texture_webp`` (an image-codec extension, not a geometry one) and
   Pillow decodes it, so ``has_texture`` / ``texture_pixels`` keep working.

So the per-mesh recipe is exactly two texture passes:

    gltf-transform resize <in> <tmp> --width N --height N   # 4096 -> N, Lanczos3
    gltf-transform webp   <tmp> <out> --quality Q           # PNG/JPEG -> WebP

...then the 7 results are zipped with plain stdlib DEFLATE (level 9). Deflate,
not LZMA/BZIP2: ``zipfile``, ``unzip``, macOS Archive Utility and every CI
runner read it without special-casing. LZMA would save ~17 MB per case but is
not portably extractable, and deflate already clears the budget.

The only metric this recipe moves is ``texture_pixels`` for meshes whose maps
were larger than ``--texture-size`` (see the note in ``main()``).

Usage
-----
    python tools/pack_testcases.py pack             # build all five zips
    python tools/pack_testcases.py pack --cases TC001 TC002
    python tools/pack_testcases.py verify           # re-check existing zips
    python tools/pack_testcases.py unpack --cases TC001

``unpack`` mirrors ``main.py:ensure_testcases()`` for standalone use; the app
bootstraps the same zips on its own, so a fresh clone needs no manual step.

Requires ``npx`` (Node) for the glTF-Transform CLI. Everything else is stdlib.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTCASES_DIR = os.path.join(ROOT, "TEST CASES")

CASES = ["TC001", "TC002", "TC003", "TC004", "TC005"]
EXPECTED_MESHES = 7

# GitHub hard-rejects pushes containing a file > 100 MB. Stay well clear so a
# future re-texture or an extra model can't quietly cross the line.
MAX_ZIP_MB = 95.0

CLI = ["npx", "--yes", "@gltf-transform/cli@latest"]

# Codecs trimesh cannot decode. If any of these show up in a packed mesh the
# 5B extractor would fail on it, so treat their presence as a hard error.
FORBIDDEN_EXTENSIONS = {
    "KHR_draco_mesh_compression",
    "EXT_meshopt_compression",
    "KHR_texture_basisu",
}

MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def case_dir(tc: str) -> str:
    return os.path.join(TESTCASES_DIR, f"{tc} GLB")


def case_zip(tc: str) -> str:
    return os.path.join(TESTCASES_DIR, f"{tc}.zip")


def meshes_in(folder: str) -> list[str]:
    """The .glb files in `folder`, sorted, ignoring macOS turds."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, e.name)
        for e in os.scandir(folder)
        if e.is_file()
        and e.name.lower().endswith(".glb")
        and not e.name.startswith("._")
    )


def run_cli(args: list[str]) -> None:
    """Run a glTF-Transform CLI command, surfacing its output on failure."""
    proc = subprocess.run(CLI + args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "gltf-transform failed: " + " ".join(args) + "\n"
            + (proc.stdout or "") + (proc.stderr or "")
        )


def glb_extensions(path: str) -> set[str]:
    """Read `extensionsUsed` straight out of a GLB's JSON chunk (no deps)."""
    import json
    import struct

    with open(path, "rb") as f:
        magic, _version, length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:                    # 'glTF'
            raise ValueError(f"not a GLB: {path}")
        while f.tell() < length:
            head = f.read(8)
            if len(head) < 8:
                break
            chunk_len, chunk_type = struct.unpack("<II", head)
            data = f.read(chunk_len)
            if chunk_type == 0x4E4F534A:           # 'JSON'
                return set(json.loads(data.decode("utf-8")).get(
                    "extensionsUsed", []))
    return set()


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def compress_mesh(src: str, dst: str, texture_size: int, quality: int) -> None:
    """
    Texture-only recompress: downscale oversized maps, then re-encode as WebP.

    Geometry (positions, normals, tangents, UVs, indices) is passed through
    untouched, which is what keeps the 5B metrics identical to the originals.
    """
    tmp_fd, tmp = tempfile.mkstemp(suffix=".glb", prefix="_pack_")
    os.close(tmp_fd)
    try:
        run_cli(["resize", src, tmp,
                 "--width", str(texture_size), "--height", str(texture_size)])
        run_cli(["webp", tmp, dst, "--quality", str(quality)])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    bad = glb_extensions(dst) & FORBIDDEN_EXTENSIONS
    if bad:
        raise RuntimeError(
            f"{os.path.basename(dst)} uses codecs trimesh cannot read: "
            f"{sorted(bad)}. The 5B extractor would fail on this mesh."
        )


def write_zip(files: list[str], out: str) -> None:
    """Zip `files` flat (basenames only), DEFLATE level 9.

    Flat entries match what `main.py:ensure_testcases()` expects, and keep the
    archive free of any `..`/absolute paths.
    """
    tmp = out + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    os.replace(tmp, out)


def pack_case(tc: str, texture_size: int, quality: int,
              keep_dir: str | None = None) -> dict:
    """Compress one case's meshes and write `TEST CASES/<TC>.zip`."""
    folder = case_dir(tc)
    sources = meshes_in(folder)
    if not sources:
        raise RuntimeError(f"no .glb files found in {folder}")

    before = sum(os.path.getsize(p) for p in sources)
    stage = keep_dir or tempfile.mkdtemp(prefix=f"pack_{tc}_")
    os.makedirs(stage, exist_ok=True)
    try:
        packed = []
        for src in sources:
            dst = os.path.join(stage, os.path.basename(src))
            compress_mesh(src, dst, texture_size, quality)
            packed.append(dst)
            print(f"    {os.path.basename(src):<34s} "
                  f"{os.path.getsize(src)/MB:7.1f} MB -> "
                  f"{os.path.getsize(dst)/MB:7.1f} MB")

        out = case_zip(tc)
        write_zip(packed, out)
        return {
            "tc": tc,
            "n": len(packed),
            "src_bytes": before,
            "glb_bytes": sum(os.path.getsize(p) for p in packed),
            "zip_bytes": os.path.getsize(out),
        }
    finally:
        if keep_dir is None:
            shutil.rmtree(stage, ignore_errors=True)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify_case(tc: str, deep: bool = True) -> list[str]:
    """
    Gate one case's zip. Returns a list of failure strings ([] == pass).

    Checks, in order: the zip exists; it is under the size budget; it opens with
    stdlib zipfile and holds exactly EXPECTED_MESHES .glb entries with no stray
    members or unsafe paths; its CRCs are intact; and (deep) one extracted mesh
    loads in trimesh with a non-zero face count and no forbidden codecs.
    """
    problems: list[str] = []
    zpath = case_zip(tc)

    if not os.path.exists(zpath):
        return [f"{tc}: {zpath} does not exist"]

    size_mb = os.path.getsize(zpath) / MB
    if size_mb >= MAX_ZIP_MB:
        problems.append(f"{tc}: zip is {size_mb:.2f} MB "
                        f"(budget {MAX_ZIP_MB:.0f} MB)")

    try:
        with zipfile.ZipFile(zpath) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            globs = [n for n in names if n.lower().endswith(".glb")]
            if len(globs) != EXPECTED_MESHES:
                problems.append(f"{tc}: {len(globs)} .glb entries, "
                                f"expected {EXPECTED_MESHES} ({names})")
            extra = [n for n in names if n not in globs]
            if extra:
                problems.append(f"{tc}: non-.glb entries present: {extra}")
            unsafe = [n for n in names
                      if os.path.isabs(n) or ".." in n.replace("\\", "/").split("/")]
            if unsafe:
                problems.append(f"{tc}: unsafe member paths: {unsafe}")
            if z.testzip() is not None:
                problems.append(f"{tc}: CRC failure in archive")

            if deep and globs:
                member = sorted(globs)[0]
                tmpdir = tempfile.mkdtemp(prefix=f"verify_{tc}_")
                try:
                    out = os.path.join(tmpdir, os.path.basename(member))
                    with z.open(member) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    bad = glb_extensions(out) & FORBIDDEN_EXTENSIONS
                    if bad:
                        problems.append(f"{tc}: {member} uses {sorted(bad)}")
                    faces = _trimesh_faces(out)
                    if faces is None:
                        problems.append(f"{tc}: {member} did not load in trimesh")
                    elif faces <= 0:
                        problems.append(f"{tc}: {member} loaded with 0 faces")
                    else:
                        print(f"  {tc}: {member} -> trimesh OK, "
                              f"{faces:,} faces")
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
    except zipfile.BadZipFile as exc:
        problems.append(f"{tc}: unreadable zip ({exc})")

    return problems


def _trimesh_faces(path: str) -> int | None:
    """Face count via trimesh, or None if it fails to load as a mesh."""
    try:
        import trimesh
    except ImportError:
        print("  ! trimesh not importable in this interpreter; "
              "skipping the load check", file=sys.stderr)
        return None
    try:
        mesh = trimesh.load(path, force="mesh")
        return int(len(mesh.faces))
    except Exception as exc:                      # noqa: BLE001 - gate, report all
        print(f"  ! trimesh.load failed on {path}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# unpack (standalone mirror of main.py:ensure_testcases)
# --------------------------------------------------------------------------- #
def unpack_case(tc: str) -> str:
    """Extract `TEST CASES/<TC>.zip` -> `TEST CASES/<TC> GLB/` (idempotent)."""
    folder = case_dir(tc)
    if os.path.isdir(folder) and meshes_in(folder):
        print(f"  {tc}: already extracted, leaving alone")
        return folder

    zpath = case_zip(tc)
    if not os.path.exists(zpath):
        raise RuntimeError(f"{zpath} not found")

    prefix = f"{tc} GLB/"
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            rel = name[len(prefix):] if name.startswith(prefix) else name
            base = os.path.basename(rel)
            if not base or base.startswith("._") or base == ".DS_Store":
                continue
            if os.path.isabs(rel) or ".." in rel.split("/"):
                continue                          # zip-slip guard
            dest = os.path.join(folder, base)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"  {tc}: extracted -> {os.path.relpath(folder, ROOT)}/")
    return folder


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_cases(values: list[str] | None) -> list[str]:
    if not values:
        return list(CASES)
    out = []
    for v in values:
        tc = v.strip().upper()
        if not re.fullmatch(r"TC\d{3}", tc):
            raise SystemExit(f"bad case id: {v!r} (expected e.g. TC001)")
        out.append(tc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Texture-only compress + zip the benchmark test cases.")
    ap.add_argument("mode", choices=["pack", "verify", "unpack"],
                    nargs="?", default="pack")
    ap.add_argument("--cases", nargs="*",
                    help="case ids to process (default: all five)")
    ap.add_argument("--texture-size", type=int, default=2048,
                    help="max texture dimension in px (default: 2048). Meshes "
                         "whose maps are already smaller are left as-is, so "
                         "only 4096-px maps lose resolution.")
    ap.add_argument("--quality", type=int, default=85,
                    help="WebP quality 1-100 (default: 85)")
    ap.add_argument("--keep-staging", metavar="DIR",
                    help="keep the compressed .glb files in DIR (for metric "
                         "before/after comparison)")
    ap.add_argument("--no-deep-verify", action="store_true",
                    help="skip the trimesh load check")
    args = ap.parse_args()

    cases = parse_cases(args.cases)

    if args.mode == "unpack":
        for tc in cases:
            unpack_case(tc)
        return 0

    if args.mode == "verify":
        failures = []
        for tc in cases:
            failures += verify_case(tc, deep=not args.no_deep_verify)
        if failures:
            print("\nFAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print("\nAll gates passed.")
        return 0

    # pack
    results = []
    for tc in cases:
        print(f"\n[{tc}] compressing (textures -> WebP q{args.quality}, "
              f"max {args.texture_size}px; geometry untouched)")
        stage = os.path.join(args.keep_staging, tc) if args.keep_staging else None
        results.append(pack_case(tc, args.texture_size, args.quality, stage))

    print(f"\n{'case':<7s} {'before':>10s} {'glb after':>10s} "
          f"{'zip':>10s} {'saved':>8s}")
    tot_b = tot_z = 0
    for r in results:
        tot_b += r["src_bytes"]
        tot_z += r["zip_bytes"]
        print(f"{r['tc']:<7s} {r['src_bytes']/MB:9.1f}M "
              f"{r['glb_bytes']/MB:9.1f}M {r['zip_bytes']/MB:9.1f}M "
              f"{100*(1-r['zip_bytes']/r['src_bytes']):7.1f}%")
    print(f"{'TOTAL':<7s} {tot_b/MB:9.1f}M {'':10s} {tot_z/MB:9.1f}M "
          f"{100*(1-tot_z/tot_b):7.1f}%")

    print("\nVerifying...")
    failures = []
    for tc in cases:
        failures += verify_case(tc, deep=not args.no_deep_verify)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
