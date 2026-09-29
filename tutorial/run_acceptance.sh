#!/usr/bin/env bash
# Acceptance test for the release: does the pipeline still work end to end?
#
# Run this before and after any change to src/. It is the difference between
# "the imports still resolve" and "the thing still trains and evaluates".
#
#   bash tutorial/run_acceptance.sh [outdir]
#
# Run in an activated Linux Isaac Lab environment. A stage must exit zero and
# produce its required artefact. Shutdown hangs time out and fail the acceptance.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${1:-$ROOT/logs/acceptance_$(date +%Y%m%d_%H%M%S)}
if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || [ -n "$(find "$OUT" -mindepth 1 -print -quit)" ]; }; then
  echo "ERROR: acceptance needs a new or empty output directory: $OUT"
  exit 1
fi
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)
DEMOS='["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]'
EXTR=calib/camera_align/current.npy
fail=0

say () { echo; echo "=== $*"; }

# Fail fast and legibly rather than three stages deep. Activate your Isaac Lab
# conda env before running this; the script deliberately does not guess which
# one it is.
if ! python -c "import dexx, torch" 2>/dev/null; then
  echo 'ERROR: dexx and torch must be importable.'
  echo "       activate your Isaac Lab env (conda activate <env> or source <venv>/bin/activate)"
  exit 1
fi

active_pid=""
stop_group () {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup () {
  if [ -n "$active_pid" ]; then stop_group "$active_pid"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

watch_for () {   # watch_for <artefact> <logfile> <timeout_units_of_10s> -- cmd...
  local art="$1" log="$2" lim="$3"; shift 4
  local pid rc tick
  [ ! -e "$art" ] || { echo "ERROR: stale artefact: $art"; return 1; }
  setsid "$@" >"$log" 2>&1 </dev/null & pid=$!
  active_pid="$pid"
  for ((tick=0; tick<lim*10; tick++)); do
    # Check the producer first: an artefact never overrides a failed command.
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      stop_group "$pid"
      active_pid=""
      [ "$rc" -eq 0 ] && [ -s "$art" ] && return 0
      echo "ERROR: stage exited $rc (required artefact: $art)"
      return 1
    fi
    sleep 1
  done
  echo "ERROR: stage timed out (required artefact: $art)"
  stop_group "$pid"
  active_pid=""
  return 2
}

say "1/5  static checks"
for check in tutorial/00_setup/check_install.py \
             tutorial/00_setup/check_versions.py \
             tutorial/01_frames_and_constants/check_frames.py \
             tutorial/06_camera_calibration/inspect_extrinsic.py \
             tutorial/00_setup/check_portable.py \
             tutorial/00_setup/check_imports.py; do
  log="$OUT/$(basename "$check" .py).log"
  if ! python "$check" >"$log" 2>&1; then
    cat "$log"
    echo "STATIC CHECK FAILED: $check"
    exit 1
  fi
  tail -1 "$log"
done
# A binary Sim puts its own bundled packages on PYTHONPATH; check this env.
if ! PYTHONPATH= python -m pip check >"$OUT/pip_check.log" 2>&1; then
  cat "$OUT/pip_check.log"
  echo "DEPENDENCY CHECK FAILED"
  exit 1
fi
tail -1 "$OUT/pip_check.log"

say "2/5  headless runtime preflight"
watch_for "$OUT/runtime.json" "$OUT/runtime.log" 30 -- \
  python -u tutorial/00_setup/check_runtime.py --headless --result "$OUT/runtime.json" \
  || { echo "RUNTIME CHECK FAILED"; exit 1; }
python - "$OUT/runtime.json" <<'PY_RUNTIME' || exit 1
import json, sys
with open(sys.argv[1]) as source:
    result = json.load(source)
if result.get("ok") is not True:
    raise SystemExit("RUNTIME CHECK FAILED: missing success marker")
PY_RUNTIME

say "3/5  train a lean student (3 DAgger iterations)"
watch_for "$OUT/student/dagger_final.pth" "$OUT/train.log" 90 -- \
  python -u scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right --data_idx "$DEMOS" \
    --num_envs 64 --dagger_iters 3 --rollout_steps 2048 --train_epochs 2 \
    --batch_size 256 --hand_body_subset minimal6 \
    --student_drop_slots obj_bps,tips_distance,obj_pose_tail \
    --camera_extrinsic "$EXTR" --seed 0 --out_dir "$OUT/student" --headless \
  || { echo "TRAIN FAILED"; exit 1; }

say "4/5  evaluate it with physics DR on and a per-demo quota"
watch_for "$OUT/eval/summary.json" "$OUT/eval.log" 90 -- \
  python -u scripts/eval.py --load_path "$OUT/student/dagger_final.pth" \
    --out_dir "$OUT/eval" --side right --data_idx "$DEMOS" \
    --num_envs 64 --max_episodes 40 --max_steps 4000 \
    --no-expand_aug --per_demo_quota -1 \
    --keep_physics_dr --camera_extrinsic "$EXTR" --headless \
  || { echo "EVAL FAILED"; exit 1; }

say "5/5  assert the artefacts"
python - "$OUT" "$DEMOS" <<'PY' || fail=1
import collections, json, math, os, sys, torch
out = sys.argv[1]
expected_demos = set(json.loads(sys.argv[2]))
ok = True
def chk(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}" + ("" if good else f"  expected {want}"))

ck = torch.load(os.path.join(out, "student", "dagger_final.pth"),
                map_location="cpu", weights_only=False)
cfg = ck.get("cfg"); cfg = cfg if isinstance(cfg, dict) else vars(cfg)
chk("student proprio_dim", cfg.get("proprio_dim"), 417)
chk("hand keypoints", cfg.get("n_hand"), 6)
chk("scene points", cfg.get("n_scene"), 1024)
chk("tactile points", cfg.get("n_tactile"), 25)
chk("dropped slots", list(ck.get("student_drop_slots") or []),
    ["obj_bps", "tips_distance", "obj_pose_tail"])
chk("slot map", ck.get("student_obs_slots"),
    {"proprioception": (0, 79), "ref_tracking": (79, 390),
     "target_obj_pose": (390, 397), "tips_distance": (397, 402),
     "obj_bps": (402, 530), "tactile": (530, 550), "obj_pose_tail": (550, 557)})
w = [v for k, v in ck["model"].items() if k.endswith("mlp.0.weight") and v.shape[1] > 100][0]
chk("student MLP in_features", int(w.shape[1]), 481)

with open(os.path.join(out, "eval", "summary.json")) as source:
    summ = json.load(source)
# strict3 is the metric the protocol says to report; assert it is computed.
_strict = summ.get("strict") or {}
_has = all(k in _strict for k in ("strict2", "strict3", "strict5"))
ok &= _has
print(f"  [{'ok' if _has else 'FAIL'}] strict2/3/5 present in summary"
      + (f": strict3 = {100*_strict['strict3']['rate']:.1f}%" if _has else ""))
per = summ.get("success_rate_per_demo") or {}
chk("summary demo names", set(per), expected_demos)
chk("summary per-demo episodes", {d: v["episodes"] for d, v in per.items()},
    {d: 10 for d in expected_demos})
chk("summary actual episodes", summ.get("actual_episodes"), 40)
chk("summary per-demo quota", summ.get("per_demo_quota"), 10)
with open(os.path.join(out, "eval", "records.json")) as source:
    records = json.load(source)["single"]
chk("record count", len(records), 40)
chk("record per-demo episodes", dict(collections.Counter(r["demo_idx"] for r in records)),
    {d: 10 for d in expected_demos})
# Verify that the summary actually represents the recorded measurements.
valid = all(isinstance(r.get("survival_len"), int) and r["survival_len"] >= 0
            and isinstance(r.get("end_final_dist"), (int, float))
            and math.isfinite(r["end_final_dist"]) and r["end_final_dist"] >= 0
            and isinstance(r.get("fail_causes"), list)
            and all(isinstance(cause, str) for cause in r["fail_causes"])
            for r in records)
chk("record metric fields valid", valid, True)
if valid:
    def strict_counts(recs, cm):
        kept = [r for r in recs if r["survival_len"] > 5]
        successes = sum(r["end_final_dist"] < cm / 100.0
                        and "obj_pos_drift" not in r["fail_causes"] for r in kept)
        return {"rate": successes / max(1, len(kept)), "successes": successes, "episodes": len(kept)}
    for cm in (2, 3, 5):
        chk(f"strict{cm} matches records", _strict.get(f"strict{cm}"), strict_counts(records, cm))
    chk("bad-init count", summ.get("bad_init_excluded"), sum(r["survival_len"] <= 5 for r in records))
    chk("at least one valid evaluation episode", any(r["survival_len"] > 5 for r in records), True)
    for demo, result in per.items():
        expected = strict_counts([r for r in records if r["demo_idx"] == demo], 3)
        # Zero policy success is valid; a demo with no usable episode is not.
        chk(f"{demo} has a valid evaluation episode", expected["episodes"] > 0, True)
        chk(f"{demo} strict3 matches records",
            {"rate": result.get("strict3"), "successes": result.get("strict3_successes"),
             "episodes": result.get("strict3_episodes")}, expected)
print("\nACCEPTANCE: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY

echo
if [ "$fail" -eq 0 ]; then echo "ACCEPTANCE TEST PASSED   ($OUT)"; else echo "ACCEPTANCE TEST FAILED   ($OUT)"; fi
exit $fail
