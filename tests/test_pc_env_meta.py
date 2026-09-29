"""Checkpoint-side point-cloud settings; no simulator, Torch or GPU needed."""
import io
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexx.algo.dagger import pc_env_meta  # noqa: E402
from dexx import deploy_config  # noqa: E402


def env_cfg():
    return SimpleNamespace(pc_workspace_min=deploy_config.PC_WORKSPACE_MIN,
                           pc_workspace_max=deploy_config.PC_WORKSPACE_MAX)


class CropBoxTests(unittest.TestCase):
    def test_checkpoint_crop_is_restored(self):
        cfg = env_cfg()
        ckpt = {"pc_env_meta": {"pc_workspace_min": (0.0, -0.4, 0.5),
                                "pc_workspace_max": (0.8, 0.25, 0.9)}}
        pc_env_meta.apply_pc_env_meta(cfg, ckpt)
        self.assertEqual(cfg.pc_workspace_min, (0.0, -0.4, 0.5))
        self.assertEqual(cfg.pc_workspace_max, (0.8, 0.25, 0.9))

    def test_checkpoint_without_crop_gets_the_legacy_box(self):
        cfg = env_cfg()
        pc_env_meta.apply_pc_env_meta(cfg, {"pc_env_meta": {"pc_force_scale": 1.0}})
        self.assertEqual(cfg.pc_workspace_min, (0.00, -0.40, 0.420))
        self.assertEqual(cfg.pc_workspace_max, (0.80, 0.25, 1.30))

    def test_cli_override_wins(self):
        cfg = env_cfg()
        ckpt = {"pc_env_meta": {"pc_workspace_min": (0.0, -0.4, 0.417)}}
        pc_env_meta.apply_pc_env_meta(cfg, ckpt, overrides={"pc_workspace_min": (0.0, -0.4, 0.422)})
        self.assertEqual(cfg.pc_workspace_min, (0.0, -0.4, 0.422))

    def test_scene_ablation_is_restored(self):
        cfg = env_cfg()
        pc_env_meta.apply_pc_env_meta(cfg, {"pc_env_meta": {"pc_ablate_scene_pc": True}})
        self.assertTrue(cfg.pc_ablate_scene_pc)

    def test_legacy_vec3_checkpoint_keeps_vec3(self):
        cfg = env_cfg()
        cfg.pc_tactile_use_vec3 = False
        ckpt = {"tactile_feat_dim": 3}                       # no pc_env_meta at all
        pc_env_meta.align_pc_dims_to_ckpt(cfg, ckpt)
        pc_env_meta.apply_pc_env_meta(cfg, ckpt)
        self.assertTrue(cfg.pc_tactile_use_vec3)
        self.assertEqual(cfg.pc_tactile_feature_dim, 3)

    def test_training_default_is_the_final_box(self):
        self.assertEqual(deploy_config.PC_WORKSPACE_MIN, (0.00, -0.40, 0.417))
        self.assertEqual(deploy_config.PC_WORKSPACE_MAX, (0.80, 0.25, 0.70))


class ResearchCheckpointTests(unittest.TestCase):
    def test_research_module_paths_resolve_to_dexx(self):
        # Protocol 0 is text, so the module path can be rewritten in place.
        data = pickle.dumps(pc_env_meta.read_pc_env_meta, protocol=0)
        data = data.replace(b"dexx.algo", b"rl_isaaclab.algo")
        loaded = pc_env_meta._RenamingUnpickler(io.BytesIO(data)).load()
        self.assertIs(loaded, pc_env_meta.read_pc_env_meta)

    def test_other_modules_are_untouched(self):
        data = pickle.dumps(SimpleNamespace, protocol=0)
        self.assertIs(pc_env_meta._RenamingUnpickler(io.BytesIO(data)).load(), SimpleNamespace)


class CheckpointKindTests(unittest.TestCase):
    DAGGER = {"proprio_dim": 557, "model": {"mlp.0.weight": 0, "pc_encoder.x": 0}}
    PPO = {"cfg": SimpleNamespace(proprio_dim=557),
           "model": {"actor_mlp.0.weight": 0, "mu.weight": 0}}

    def test_dagger_vs_ppo(self):
        self.assertTrue(pc_env_meta.is_dagger_student_ckpt(self.DAGGER))
        self.assertFalse(pc_env_meta.is_dagger_student_ckpt(self.PPO))
        self.assertFalse(pc_env_meta.is_dagger_student_ckpt({"model": {}}))

    def test_full_student_can_warm_start_ppo(self):
        self.assertIsNone(pc_env_meta.lean_student_refusal(self.DAGGER))
        self.assertIsNone(pc_env_meta.lean_student_refusal(
            dict(self.DAGGER, student_keep_idx=[], student_drop_slots=[])))

    def test_lean_student_is_refused(self):
        lean = dict(self.DAGGER, proprio_dim=3, student_keep_idx=[0, 1, 2],
                    student_drop_slots=["obj_bps"])
        why = pc_env_meta.lean_student_refusal(lean)
        self.assertIsNotNone(why)
        self.assertIn("obj_bps", why)


if __name__ == "__main__":
    unittest.main()
