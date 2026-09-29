"""Is this repository self-contained?

Answers one question: if you copied this directory to another machine, would it
still work? The failure modes are all silent — a symlink into a repository that
is not shipped, an absolute path baked into a config, a mesh referenced by a
demonstration but never committed.

    python tutorial/00_setup/check_portable.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {".git", "__pycache__", "logs", "wandb", "outputs", "runs", "dumps",
             ".egg-info", "node_modules"}
TEXT_EXT = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".md"}
# Filesystem roots that would not exist on another machine.
ABS_RE = re.compile(r"(?<![\w.])/(?:home|Users|mnt|media|opt|data)/[\w./-]+")

problems: list[str] = []


def walk() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for f in filenames:
            out.append(os.path.join(dirpath, f))
    return out


def main() -> int:
    problems.clear()
    files = walk()

    print("=== symlinks")
    n_sym = 0
    for p in files + [os.path.join(dp, d) for dp, dns, _ in os.walk(ROOT) for d in dns]:
        if not os.path.islink(p):
            continue
        n_sym += 1
        rel = os.path.relpath(p, ROOT)
        target = os.path.realpath(p)
        if not os.path.exists(target):
            problems.append(f"broken symlink: {rel}")
            print(f"  [BROKEN]   {rel} -> {os.readlink(p)}")
        elif not target.startswith(ROOT + os.sep):
            problems.append(f"symlink leaves the repository: {rel}")
            print(f"  [EXTERNAL] {rel} -> {os.readlink(p)}")
    print(f"  {n_sym} symlink(s) examined"
          + ("" if n_sym else "  (none — good)"))

    # A path written in prose cannot break a copy; a path in code or config can.
    # Markdown is reported and not counted against self-containment.
    print("\n=== absolute paths in shipped text files")
    hits = warns = 0
    for p in files:
        if os.path.splitext(p)[1] not in TEXT_EXT:
            continue
        rel = os.path.relpath(p, ROOT)
        if rel.startswith("tutorial/") and rel.endswith(".py"):
            continue  # this file documents the pattern it searches for
        try:
            with open(p, encoding="utf-8", errors="ignore") as source:
                text = source.read()
        except OSError:
            continue
        for m in ABS_RE.finditer(text):
            frag = m.group(0)
            if frag.startswith("/opt/ros"):
                continue  # a system location, legitimately absolute
            line = text[:m.start()].count("\n") + 1
            if rel.endswith(".md"):
                warns += 1
                print(f"  [note] {rel}:{line}  {frag}   (prose — not a dependency)")
                continue
            hits += 1
            problems.append(f"absolute path in {rel}:{line}")
            print(f"  [ABS]  {rel}:{line}  {frag}")
    if not hits:
        print(f"  no absolute paths in code or config"
              + (f"  ({warns} in prose, listed above)" if warns else ""))

    print("\n=== demonstrations resolve to real data")
    rt_dir = os.path.join(ROOT, "data", "retargeting", "robotool_batch", "mano2sharpa_rh")
    src_dir = os.path.join(ROOT, "data", "robotool_batch")
    demo_count = 0
    if os.path.isdir(rt_dir):
        for task in sorted(os.listdir(rt_dir)):
            if not os.path.isdir(os.path.join(rt_dir, task)):
                continue
            for pkl in sorted(x for x in os.listdir(os.path.join(rt_dir, task))
                              if x.endswith("@0.pkl")):
                demo_count += 1
                seq = pkl[:-len("@0.pkl")]
                need = os.path.join(src_dir, task, seq)
                ok = os.path.isdir(need) and not os.path.islink(need)
                meta = os.path.join(need, "meta.json")
                mesh_ok = True
                try:
                    with open(meta, encoding="utf-8") as source:
                        metadata = json.load(source)
                    meshes = metadata.get("obj_mesh_paths")
                    if not isinstance(meshes, dict) or not meshes:
                        raise ValueError("obj_mesh_paths must be a nonempty object")
                    for rel_mesh in meshes.values():
                        if not isinstance(rel_mesh, str) or not rel_mesh or os.path.isabs(rel_mesh):
                            raise ValueError("mesh paths must be nonempty relative paths")
                        candidate = os.path.realpath(os.path.join(src_dir, rel_mesh))
                        if (os.path.commonpath((os.path.realpath(src_dir), candidate)) != os.path.realpath(src_dir)
                                or not os.path.isfile(candidate)):
                            raise ValueError(f"mesh missing or outside the data directory: {rel_mesh}")
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    mesh_ok = False
                    problems.append(f"{task}/{seq}: invalid metadata or mesh ({error})")
                good = ok and mesh_ok
                if not ok:
                    problems.append(f"{task}/{seq}: source directory missing or a symlink")
                print(f"  [{'ok' if good else 'FAIL'}] {task}/{seq}"
                      + ("" if good else "   source dir or mesh unavailable"))

    if not demo_count:
        problems.append("no retargeted demonstrations found")

    print("\n=== assets referenced by the robot config exist")
    for rel in ("assets/robot", "assets/generated", "checkpoints/teacher_poseobs.pth",
                "calib/camera_align/current.npy"):
        p = os.path.join(ROOT, rel)
        good = os.path.exists(p)
        if not good:
            problems.append(f"missing: {rel}")
        print(f"  [{'ok' if good else 'MISSING'}] {rel}")

    print()
    if problems:
        print(f"NOT SELF-CONTAINED — {len(problems)} problem(s):")
        for x in problems[:20]:
            print(f"  - {x}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1
    print("SELF-CONTAINED: this directory can be copied to another machine as is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
