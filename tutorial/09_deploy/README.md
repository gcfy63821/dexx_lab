# 09 — Deploying on the real robot

**Goal:** run the student on hardware, with the checks that stop you from finding
out the hard way.

[DEPLOY.md](../../docs/DEPLOY.md) is the operational reference — launch
commands, flags, safety limits; this lesson is the reasoning around it.

## Three processes on three machines

```
camera host              NUC                         workstation
-----------              ---                         -----------
RealSense depth PUB      Polymetis server            deploy_pc.py (the policy)
  :5562  ──────────────────────────────────────────►  depth SUB
                         joint bridge  :5560 state ─►  arm client
                                       :5561 cmd   ◄─  arm client
                                                     Sharpa hand over USB (SDK)
```

The split is not incidental. Polymetis wants to sit next to the arm on a
real-time-ish machine; the policy wants a GPU. ZMQ between them means either side
can be restarted without the other noticing.

The reference path is **Polymetis + ZMQ**; nothing in it uses ROS. A ROS2
backend (a `ros2_control` joint-impedance controller for the arm, the stock
RealSense driver for depth) is available behind `--arm_backend ros2
--depth_backend ros2` as an experimental path —
see [DEPLOY.md](../../docs/DEPLOY.md#ros2-backend-experimental).

## Launch order

Order matters because each stage assumes the previous one is up: Polymetis server
and joint bridge on the NUC, depth publisher on the camera host, then on the
workstation a bridge sanity check (`deploy/test_polymetis_arm.py`), an optional
move to the demo's start frame, and `deploy/deploy_pc.py`. The exact commands are
in [DEPLOY.md](../../docs/DEPLOY.md#launch-order).

The sanity check exists so that a bridge problem surfaces as a failed check rather
than as a policy that appears to be misbehaving. For the same reason
`deploy_pc.py` refuses to start without arm state (10 s), a depth frame (5 s) or
the Sharpa hand, and e-stops on stale input once running.

Run the first closed-loop attempts with a hand on the FR3 e-stop. `deploy_pc.py`
ramps the action in over the first 15 steps (`--action_ramp_steps`).

## Before the first run

Work through these in order. Each one catches a failure that otherwise looks
like a policy problem.

| check | lesson |
|---|---|
| `check_frames.py` passes and matches your physical setup | 01 |
| the extrinsic you are deploying is the one you **trained** with | 06 |
| the real cropped cloud overlays the sim cropped cloud | 06 |
| joint ordering verified end to end — the hand, not just the arm | 02 |
| arm impedance gains: a deliberate choice (training used `ARM_TUNED_KP/KD`; the Polymetis defaults differ) | 07 |
| `ARM_BASE_Z` matches the mount the student was trained on ([checkpoints/README.md](../../checkpoints/README.md)) | 01 |

The extrinsic one deserves emphasis: **train and deploy must use the same file.**
Even two calibrations of the same mount can differ by centimetres and degrees, and
a policy trained on one and deployed on the other meets that viewpoint shift at
exactly the moment it matters.

## What the student receives at deploy

417 proprioceptive dims plus a 64-d point-cloud feature. The dropped channels
(lesson 05) are precisely the ones unavailable on hardware — object shape encoding,
demo-precomputed fingertip distances, and the object pose estimate. That is the
point of dropping them: **the student's input vector is the same shape in sim and
on the robot**, so there is no deploy-time substitution to get wrong.

`deploy_pc.py` enforces this: a checkpoint that reads the object-pose tail (dims
550+) is refused, because the deploy env does not produce one, and so is a PPO
checkpoint.

What is *not* dropped is `target_obj_pose` — the demonstration's target object
pose, which comes from the demo file at deploy just as it does in sim. The policy
is told where the object should go, not where it is.

## When it behaves differently than in simulation

Work down lesson 07's symptom table first — dynamics gaps are common and are
easily mistaken for perception or policy problems. Then check the extrinsic. Then check that the
cropped point cloud actually contains the object, by dumping it and looking at it.

Resist retraining until you have a measurement that says which of the three it is.
