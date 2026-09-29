from .base import DexHand
from .factory import register_dexhand
from abc import ABC
import numpy as np
from dexx.tasks.hand_imitation.dataset.transform import aa_to_rotmat

class Sharpa(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "sharpa"
        self.self_collision = False
        self.body_names = [
            "hand_C_MC",
            # thumb
            "thumb_CMC_VL",
            "thumb_MC",
            "thumb_MCP_VL",
            "thumb_PP",
            "thumb_DP",
            "thumb_elastomer",
            "thumb_fingertip",
            # index
            "index_MCP_VL",
            "index_PP",
            "index_MP",
            "index_DP",
            "index_elastomer",
            "index_fingertip",
            # middle
            "middle_MCP_VL",
            "middle_PP",
            "middle_MP",
            "middle_DP",
            "middle_elastomer",
            "middle_fingertip",
            # ring
            "ring_MCP_VL",
            "ring_PP",
            "ring_MP",
            "ring_DP",
            "ring_elastomer",
            "ring_fingertip",
            # pinky
            "pinky_MC",
            "pinky_MCP_VL",
            "pinky_PP",
            "pinky_MP",
            "pinky_DP",
            "pinky_elastomer",
            "pinky_fingertip",
        ]
        self.dof_names = [
            # thumb
            "thumb_CMC_FE",
            "thumb_CMC_AA",
            "thumb_MCP_FE",
            "thumb_MCP_AA",
            "thumb_IP",
            # index
            "index_MCP_FE",
            "index_MCP_AA",
            "index_PIP",
            "index_DIP",
            # middle
            "middle_MCP_FE",
            "middle_MCP_AA",
            "middle_PIP",
            "middle_DIP",
            # ring
            "ring_MCP_FE",
            "ring_MCP_AA",
            "ring_PIP",
            "ring_DIP",
            # pinky
            "pinky_CMC",
            "pinky_MCP_FE",
            "pinky_MCP_AA",
            "pinky_PIP",
            "pinky_DIP",
        ]
        self.hand2dex_mapping = {
            "wrist": ["hand_C_MC"],
            "thumb_proximal": ["thumb_CMC_VL", "thumb_MC"],  # one-to-many mapping
            "thumb_intermediate": ["thumb_MCP_VL", "thumb_PP"],
            "thumb_distal": ["thumb_DP"],
            "thumb_tip": ["thumb_elastomer", "thumb_fingertip"],
            "index_proximal": ["index_MCP_VL", "index_PP"],
            "index_intermediate": ["index_MP"],
            "index_distal": ["index_DP"],
            "index_tip": ["index_elastomer", "index_fingertip"],
            "middle_proximal": ["middle_MCP_VL", "middle_PP"],
            "middle_intermediate": ["middle_MP"],
            "middle_distal": ["middle_DP"],
            "middle_tip": ["middle_elastomer", "middle_fingertip"],
            "ring_proximal": ["ring_MCP_VL", "ring_PP"],
            "ring_intermediate": ["ring_MP"],
            "ring_distal": [ "ring_DP"],
            "ring_tip": ["ring_elastomer", "ring_fingertip"],
            "pinky_proximal": ["pinky_MC", "pinky_MCP_VL", "pinky_PP"],
            "pinky_intermediate": ["pinky_MP"],
            "pinky_distal": [ "pinky_DP"],
            "pinky_tip": ["pinky_elastomer", "pinky_fingertip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        # self.contact_body_names = [
        #     "thumb_DP",
        #     "index_DP",
        #     "middle_DP",
        #     "ring_DP",
        #     "pinky_DP",
        # ]
        self.contact_body_names = [
            "thumb_elastomer",
            "index_elastomer",
            "middle_elastomer",
            "ring_elastomer",
            "pinky_elastomer",
        ]
        self.gym_bone_links = [
            [0, 1],  # base to thumb
            [1, 2],  # thumb_CMC_VL to thumb_MC
            [2, 3],  # thumb_MC to thumb_MCP_VL
            [3, 4],  # thumb_MCP_VL to thumb_PP
            [4, 5],  # thumb_PP to thumb_DP
            [5, 6],  # thumb_DP to thumb_elastomer
            [6, 7],  # thumb_elastomer to thumb_fingertip
            [0, 8],  # base to index
            [8, 9],  # index_MCP_VL to index_PP
            [9, 10],  # index_PP to index_MP
            [10, 11],  # index_MP to index_DP
            [11, 12],  # index_DP to index_elastomer
            [12, 13],  # index_elastomer to index_fingertip
            [0, 14],  # base to middle
            [14, 15],  # middle_MCP_VL to middle_PP
            [15, 16],  # middle_PP to middle_MP
            [16, 17],  # middle_MP to middle_DP
            [17, 18],  # middle_DP to middle_elastomer
            [18, 19],  # middle_elastomer to middle_fingertip
            [0, 20],  # base to ring
            [20, 21],  # ring_MCP_VL to ring_PP
            [21, 22],  # ring_PP to ring_MP
            [22, 23],  # ring_MP to ring_DP
            [23, 24],  # ring_DP to ring_elastomer
            [24, 25],  # ring_elastomer to ring_fingertip
            [0, 26],  # base to pinky
            [26, 27],  # pinky_MC to pinky_MCP_VL
            [27, 28],  # pinky_MCP_VL to pinky_PP
            [28, 29],  # pinky_PP to pinky_MP
            [29, 30],  # pinky_MP to pinky_DP
            [30, 31],  # pinky_DP to pinky_elastomer
            [31, 32],  # pinky_elastomer to pinky_fingertip
        ]
        self.bone_links = [
            # ---- Thumb chain ----
            [0, 5],      # base → thumb_CMC_VL
            [5, 10],     # thumb_CMC_VL → thumb_MC
            [10, 15],    # thumb_MC → thumb_MCP_VL
            [15, 20],    # thumb_MCP_VL → thumb_PP
            [20, 25],    # thumb_PP → thumb_DP
            [25, 30],    # thumb_DP → thumb_elastomer
            [30, 32],    # thumb_elastomer → thumb_fingertip

            # ---- Index chain ----
            [0, 1],      # base → index_MCP_VL
            [1, 6],      # MCP_VL → PP
            [6, 11],     # PP → MP
            [11, 16],    # MP → DP
            [16, 21],    # DP → elastomer
            [21, 26],    # elastomer → fingertip

            # ---- Middle chain ----
            [0, 2],      # base → middle_MCP_VL
            [2, 7],      # MCP_VL → PP
            [7, 12],     # PP → MP
            [12, 17],    # MP → DP
            [17, 22],    # DP → elastomer
            [22, 27],    # elastomer → fingertip

            # ---- Ring chain ----
            [0, 4],      # base → ring_MCP_VL
            [4, 9],      # MCP_VL → PP
            [9, 14],     # PP → MP
            [14, 19],    # MP → DP
            [19, 24],    # DP → elastomer
            [24, 29],    # elastomer → fingertip

            # ---- Pinky chain ----
            [0, 3],      # base → pinky_MC
            [3, 8],      # MC → MCP_VL
            [8, 13],     # MCP_VL → PP
            [13, 18],    # PP → MP
            [18, 23],    # MP → DP
            [23, 28],    # DP → elastomer
            [28, 31],    # elastomer → fingertip
        ]

        self.gym_weight_idx = {
            "thumb_tip": [7],
            "index_tip": [13],
            "middle_tip": [19],
            "ring_tip": [25],
            "pinky_tip": [32],
            "level_1_joints": [1, 2, 4, 8, 9, 14, 15, 20, 21, 26, 27, 28],
            "level_2_joints": [3, 5, 10, 11, 16, 17, 22, 23, 29, 30],
        }
        self.weight_idx = {
            # fingertip indices
            "thumb_tip": [32],
            "index_tip": [26],
            "middle_tip": [27],
            "ring_tip": [29],
            "pinky_tip": [31],

            # first layer joints (MCP, PP, + pinky MC)
            "level_1_joints": [
                15, 20,    # thumb
                1,  6,     # index
                2,  7,     # middle
                4,  9,     # ring
                3,  8, 13  # pinky MC, MCP, PP
            ],

            # second layer joints (MP + DP)
            "level_2_joints": [
                25,        # thumb DP
                11, 16,    # index
                12, 17,    # middle
                14, 19,    # ring
                18, 23     # pinky
            ],
        }

        # Same content as `weight_idx` but addressed by BODY NAME (unprefixed —
        # the env resolver tacks on `{hand_side}_` per side). Source of truth;
        # the env builds `self.dexhand_weight_idx` by resolving
        # these against the actual `hand_body_names` order PhysX gives. Keeps
        # rewards robust to URDF / USD parse-order changes and to left-hand vs
        # right-hand switches.
        self.weight_idx_body_names = {
            "thumb_tip":  ["thumb_fingertip"],
            "index_tip":  ["index_fingertip"],
            "middle_tip": ["middle_fingertip"],
            "ring_tip":   ["ring_fingertip"],
            "pinky_tip":  ["pinky_fingertip"],
            "level_1_joints": [
                "thumb_MCP_VL", "thumb_PP",
                "index_MCP_VL", "index_PP",
                "middle_MCP_VL", "middle_PP",
                "ring_MCP_VL",  "ring_PP",
                "pinky_MC", "pinky_MCP_VL", "pinky_PP",
            ],
            "level_2_joints": [
                "thumb_DP",
                "index_MP",  "index_DP",
                "middle_MP", "middle_DP",
                "ring_MP",   "ring_DP",
                "pinky_MP",  "pinky_DP",
            ],
        }


        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1
        # ? <<<<<<<<<<


    def __str__(self):
        return self.name


@register_dexhand("sharpa_rh")
class SharpaRH(Sharpa):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/sharpa_wave/right_sharpa_wave/right_sharpa_wave.urdf"
        self.side = "rh"
        # Apply "right_" prefix to all body and dof names
        self.body_names = ["right_" + name for name in self.body_names]
        self.dof_names = ["right_" + name for name in self.dof_names]
        # Apply prefix to hand2dex_mapping
        self.hand2dex_mapping = {k: ["right_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["right_" + name for name in self.contact_body_names]
        # Note: relative_rotation and relative_translation may need manual adjustment
        # self.relative_rotation = aa_to_rotmat(torch.tensor([np.pi / 2, 0, 0], device=self.device)) @ aa_to_rotmat(torch.tensor([0, -np.pi / 2, 0], device=self.device))
        self.relative_rotation = aa_to_rotmat(np.array([np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        self.relative_translation = np.array([0.0, 0.0, 0.0])  # Adjust based on URDF wrist position
        # self.relative_translation = torch.tensor([0.0, 0.0, 0.0], device = device)
    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("sharpa_lh")
class SharpaLH(Sharpa):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/sharpa_wave/left_sharpa_wave/left_sharpa_wave.urdf"
        self.side = "lh"
        # Apply "left_" prefix to all body and dof names
        self.body_names = ["left_" + name for name in self.body_names]
        self.dof_names = ["left_" + name for name in self.dof_names]
        # Apply prefix to hand2dex_mapping
        self.hand2dex_mapping = {k: ["left_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["left_" + name for name in self.contact_body_names]
        # Note: relative_rotation and relative_translation may need manual adjustment
        # Based on the hand's URDF coordinate frame relative to MANO
        # self.relative_rotation = aa_to_rotmat(np.array([0, np.pi / 2, -np.pi / 2]))
        # self.relative_rotation = aa_to_rotmat(np.array([np.pi * 3/2, np.pi / 2, -np.pi / 2]))
        # self.relative_rotation = aa_to_rotmat(np.array([np.pi * 2, np.pi / 2, -np.pi / 2]))
        # self.relative_rotation = aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))
        self.relative_rotation = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, np.pi / 2, 0]))
        # self.relative_rotation = aa_to_rotmat(torch.tensor([-np.pi / 2, 0, 0], device=self.device)) @ aa_to_rotmat(torch.tensor([0, np.pi / 2, 0], device=self.device))
        
        self.relative_translation = np.array([0.0, 0.0, 0.0])  # Adjust based on URDF wrist position
        # self.relative_translation = torch.tensor([0.0, 0.0, 0.0], device = device)

    def __str__(self):
        return super().__str__() + "_lh"

