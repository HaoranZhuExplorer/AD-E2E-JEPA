import argparse
import sys
import os
import time
import warnings

# Suppress harmless nuplan map parsing warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import logging
import copy

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dino_wm.dino_wm_agent import consecutive_relative_se2_actions
from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene, Trajectory, TrajectorySampling, Frame
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.common.dataloader import SceneLoader
from navsim.agents.dino_wm.dino_wm_config import DINO_WMConfig

from navsim.visualization.plots import configure_bev_ax, add_configured_bev_on_ax
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLANNING_VOCABULARY_PATH = Path(__file__).resolve().parents[2] / "planning_vocabulary" / "8192.npy"
EVALUATION_HORIZONS = ((1, "0p5s"), (2, "1s"), (4, "2s"), (8, "4s"))

DINO_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _local_se2_to_global(origin: StateSE2, local_poses: np.ndarray) -> np.ndarray:
    """Convert ego-local SE(2) poses into global SE(2) poses."""
    local_poses = np.asarray(local_poses, dtype=np.float64)
    global_poses = np.empty_like(local_poses, dtype=np.float64)

    cos_h, sin_h = np.cos(origin.heading), np.sin(origin.heading)
    x_local = local_poses[:, 0]
    y_local = local_poses[:, 1]

    global_poses[:, 0] = origin.x + x_local * cos_h - y_local * sin_h
    global_poses[:, 1] = origin.y + x_local * sin_h + y_local * cos_h
    global_poses[:, 2] = _wrap_angle(origin.heading + local_poses[:, 2])
    return global_poses


def _local_xy_to_global(origin: StateSE2, local_xy: np.ndarray) -> np.ndarray:
    """Convert ego-local XY points into global XY points."""
    local_xy = np.asarray(local_xy, dtype=np.float64)
    global_xy = np.empty_like(local_xy, dtype=np.float64)

    cos_h, sin_h = np.cos(origin.heading), np.sin(origin.heading)
    global_xy[:, 0] = origin.x + local_xy[:, 0] * cos_h - local_xy[:, 1] * sin_h
    global_xy[:, 1] = origin.y + local_xy[:, 0] * sin_h + local_xy[:, 1] * cos_h
    return global_xy


def _global_xy_to_local(origin: StateSE2, global_xy: np.ndarray) -> np.ndarray:
    """Convert global XY points into ego-local XY points."""
    global_xy = np.asarray(global_xy, dtype=np.float64)
    local_xy = np.empty_like(global_xy, dtype=np.float64)

    cos_h, sin_h = np.cos(origin.heading), np.sin(origin.heading)
    dx = global_xy[:, 0] - origin.x
    dy = global_xy[:, 1] - origin.y
    local_xy[:, 0] = dx * cos_h + dy * sin_h
    local_xy[:, 1] = -dx * sin_h + dy * cos_h
    return local_xy


def _latent_distance_for_step(
    z_pred_step: torch.Tensor,
    z_tgt_step: torch.Tensor,
) -> torch.Tensor:
    """Compute per-candidate Euclidean latent distance to a target latent."""
    z_tgt_loss = z_tgt_step[:, -1:].detach()

    z_tgt_loss = z_tgt_loss.expand(z_pred_step.shape[0], *([-1] * (z_pred_step.ndim - 1)))
    diff = z_pred_step - z_tgt_loss
    return torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)


def _collect_gt_visual_features(
    dino_wm: torch.nn.Module,
    gt_seq: torch.Tensor,
    num_future_steps: int = 8,
) -> torch.Tensor:
    """Collect each GT frame once in the original encoder patch space."""
    raw_windows = []

    for step in range(num_future_steps):
        obs = {
            "visual": gt_seq[:, step : step + dino_wm.num_hist + 1],
        }
        raw_grid = dino_wm.encode_obs(obs)["visual"]
        raw_windows.append(raw_grid)

    # The first window supplies history; each window supplies one future frame.
    return torch.cat(
        [raw_windows[0][:, : dino_wm.num_hist]] + [window[:, -1:] for window in raw_windows],
        dim=1,
    )


def _visual_prediction_tokens(z: torch.Tensor, dino_wm: torch.nn.Module) -> torch.Tensor:
    """Extract only the visual channels from a rollout-compatible latent."""
    visual_dim = int(dino_wm.backbone.emb_dim)
    if z.shape[-1] < visual_dim:
        raise ValueError(
            f"Predicted latent width {z.shape[-1]} is smaller than visual width {visual_dim}."
        )
    return z[..., :visual_dim]


def _pca_rgb_maps(*feature_sequences: torch.Tensor, grid_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Fit one shared 3D PCA basis and return consistently normalized RGB maps."""
    if not feature_sequences:
        raise ValueError("At least one feature sequence is required for PCA visualization.")

    feature_dim = feature_sequences[0].shape[-1]
    num_patches = feature_sequences[0].shape[-2]
    grid_height, grid_width = grid_shape
    for features in feature_sequences:
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError(
                "Expected each PCA feature sequence to have shape (1, T, P, D), "
                f"got {tuple(features.shape)}."
            )
        if features.shape[-2:] != (num_patches, feature_dim):
            raise ValueError("All feature sequences in a shared PCA must use the same P and D.")

    flattened = torch.cat(
        [features.detach().float().reshape(-1, feature_dim) for features in feature_sequences],
        dim=0,
    )
    if not torch.isfinite(flattened).all():
        raise ValueError("PCA input contains non-finite patch features.")

    # q=3 computes only the components needed for RGB and avoids introducing a
    # scikit-learn dependency. The same mean and basis are used for every row.
    mean = flattened.mean(dim=0, keepdim=True)
    cuda_devices = []
    if flattened.is_cuda:
        device_index = flattened.device.index
        cuda_devices = [torch.cuda.current_device() if device_index is None else device_index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(0)
        _, _, components = torch.pca_lowrank(flattened - mean, q=3, center=False, niter=4)
    projected = (flattened - mean) @ components

    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    scale = (high - low).clamp_min(1e-6)
    projected = ((projected - low) / scale).clamp(0.0, 1.0)

    rgb_sequences = []
    offset = 0
    for features in feature_sequences:
        count = features.shape[1] * num_patches
        rgb = projected[offset : offset + count].reshape(
            features.shape[1], grid_height, grid_width, 3
        )
        rgb_sequences.append(rgb.cpu().numpy())
        offset += count
    return rgb_sequences


def _imagenet_to_rgb_sequence(images: torch.Tensor) -> np.ndarray:
    """Convert normalized model inputs from (1, T, C, H, W) to displayable RGB."""
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
        raise ValueError(
            "Expected normalized images shaped (1, T, 3, H, W), "
            f"got {tuple(images.shape)}."
        )
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    return rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy()


def _save_pca_grid(
    rgb_rows: List[np.ndarray],
    row_labels: List[str],
    output_prefix: str,
    title: str,
    num_history: int,
) -> str:
    """Save the canonical PC1/PC2/PC3-to-RGB visualization grid."""
    if len(rgb_rows) != len(row_labels):
        raise ValueError("Each visualization row needs one label.")
    num_frames = rgb_rows[0].shape[0]
    if any(row.shape[0] != num_frames for row in rgb_rows):
        raise ValueError("All PCA visualization rows must have the same number of frames.")

    fig, axes = plt.subplots(
        len(rgb_rows),
        num_frames,
        figsize=(1.65 * num_frames, 1.9 * len(rgb_rows)),
        dpi=200,
        squeeze=False,
    )
    for row_idx, (rgb_sequence, row_label) in enumerate(zip(rgb_rows, row_labels)):
        for frame_idx in range(num_frames):
            ax = axes[row_idx, frame_idx]
            ax.imshow(rgb_sequence[frame_idx], interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                phase = "H" if frame_idx < num_history else "P"
                phase_idx = frame_idx if frame_idx < num_history else frame_idx - num_history + 1
                ax.set_title(f"{phase}{phase_idx}", fontsize=9)
            if frame_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    fig.suptitle(f"{title} | RGB = PC123", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.02, top=0.84, wspace=0.03, hspace=0.08)
    output_path = f"{output_prefix}_rgb-pc123.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def _generate_pca_visualizations(
    dino_wm: torch.nn.Module,
    gt_seq: torch.Tensor,
    predicted_encodings: torch.Tensor,
    output_path: str,
    sequence_name: str,
    token: str,
) -> List[str]:
    """Generate temporal PCA grids for encoder and AR-predicted patch embeddings."""
    raw_gt = _collect_gt_visual_features(dino_wm, gt_seq)
    camera_rgb = _imagenet_to_rgb_sequence(gt_seq[:, : raw_gt.shape[1]])

    # BatchedWMUnroll returns (future_steps, B, 1, patches, dim). Convert its
    # single selected trajectory into (1, future_steps, patches, visual_dim).
    if predicted_encodings.ndim != 5 or predicted_encodings.shape[1] != 1:
        raise ValueError(
            "Expected selected AR encodings shaped (T, 1, 1, P, D), "
            f"got {tuple(predicted_encodings.shape)}."
        )
    predicted_future = predicted_encodings[:, 0].permute(1, 0, 2, 3)
    predicted_future = _visual_prediction_tokens(predicted_future, dino_wm)
    ar_sequence = torch.cat(
        [raw_gt[:, : dino_wm.num_hist], predicted_future],
        dim=1,
    )

    output_stem = os.path.splitext(output_path)[0]
    scene_prefix = f"{output_stem}{sequence_name}_{token}_pca"
    saved_paths = []

    gt_rgb, ar_rgb = _pca_rgb_maps(
        raw_gt, ar_sequence,
        grid_shape=dino_wm.grid_size,
    )
    saved_paths.append(
        _save_pca_grid(
            [gt_rgb, ar_rgb, camera_rgb],
            ["GT encoder", "AR rollout", "GT image"],
            f"{scene_prefix}_encoder",
            "Original DINOv2 patch embeddings (shared PCA)",
            dino_wm.num_hist,
        )
    )

    return saved_paths


def _remap_frame_bev_to_ego_pose(frame: Frame, target_ego_pose: np.ndarray) -> Frame:
    """
    Remap ego-local BEV layers from the frame's recorded ego pose into a target ego pose.

    Map geometry is drawn from global coordinates by NAVSIM's BEV helper, but annotations and
    lidar are stored in ego-local coordinates. If we replace only ego_pose for a simulated ego
    trajectory, those local layers stay stuck in the recorded ego frame. This helper first lifts
    them to global coordinates through the recorded pose, then projects them into the target pose.
    """
    source_origin = StateSE2(*np.asarray(frame.ego_status.ego_pose, dtype=np.float64))
    target_origin = StateSE2(*np.asarray(target_ego_pose, dtype=np.float64))

    if frame.annotations is not None and frame.annotations.boxes is not None and len(frame.annotations.boxes) > 0:
        boxes = frame.annotations.boxes.copy()
        local_box_poses = boxes[:, [BoundingBoxIndex.X, BoundingBoxIndex.Y, BoundingBoxIndex.HEADING]]
        global_box_poses = _local_se2_to_global(source_origin, local_box_poses)
        remapped_box_poses = convert_absolute_to_relative_se2_array(target_origin, global_box_poses)

        boxes[:, BoundingBoxIndex.X] = remapped_box_poses[:, 0]
        boxes[:, BoundingBoxIndex.Y] = remapped_box_poses[:, 1]
        boxes[:, BoundingBoxIndex.HEADING] = remapped_box_poses[:, 2]
        frame.annotations.boxes = boxes

    if frame.lidar is not None and frame.lidar.lidar_pc is not None and frame.lidar.lidar_pc.size > 0:
        lidar_pc = frame.lidar.lidar_pc.copy()
        local_points = lidar_pc[[LidarIndex.X, LidarIndex.Y], :].T
        global_points = _local_xy_to_global(source_origin, local_points)
        remapped_points = _global_xy_to_local(target_origin, global_points)

        lidar_pc[LidarIndex.X, :] = remapped_points[:, 0]
        lidar_pc[LidarIndex.Y, :] = remapped_points[:, 1]
        frame.lidar.lidar_pc = lidar_pc

    object.__setattr__(frame.ego_status, 'ego_pose', np.asarray(target_ego_pose, dtype=np.float64))
    return frame


def evaluate_dino_wm_seq(
    checkpoint_path,
    output_path,
    pdms_output_path,
    num_samples=50,
    split="navtest",
    compute_human_pdms=False,
    save_plot=False,
    save_pca_visualization=False,
    interval=1,
    num_candidate_traj=128,
    num_chunks=32,
    target_token=None,
    original_sensor_path_override=None,
    agent_config="dino_wm_agent",
    planning_warmup_runs=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load Hydra Config
    if not os.path.exists("navsim"):
        raise RuntimeError("Please run this script from the project root (where 'navsim' folder exists).")

    config_path = os.path.abspath("navsim/planning/script/config/training")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_path, version_base=None)
    
    overrides = [
        f"agent={agent_config}",
        "experiment_name=evaluate_dino_wm_seq",
        f"train_test_split={split}",
    ]
    print("Composing DINO-WM eval Hydra config...", flush=True)
    cfg = hydra.compose(
        config_name="default_training", 
        overrides=overrides,
    )
    print("Finished composing DINO-WM eval Hydra config.", flush=True)

    # 2. Instantiate Agent (World Model)
    print("Building DINO-WM Agent...", flush=True)
    agent: AbstractAgent = instantiate(cfg.agent)
    print("Finished building DINO-WM Agent.", flush=True)
    
    # 3. Load Checkpoint 
    print(f"Loading DINO-WM from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('agent.'):
            k = k.replace('agent.', '')
        new_state_dict[k] = v
    
    agent.load_state_dict(new_state_dict)

    agent._dino_wm_model.to(device)
    agent._dino_wm_model.eval()

    # 4. Scene Loading
    print("Initializing SceneLoader...")
    if split == 'train':
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = cfg.train_logs
    elif split == 'val':
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = cfg.val_logs
    elif split == 'test':
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = cfg.test_logs
    else: 
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(original_sensor_path_override or cfg.original_sensor_path)

    scene_filter.num_history_frames = 4
    scene_filter.num_future_frames = 10 
    
    sensor_config = agent.get_sensor_config()
    sensor_config.cam_f0 = True 

    scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config, 
    )

    print(f"Found {len(scene_loader.tokens)} scenes.")
    if target_token is not None:
        if target_token not in scene_loader.tokens:
            raise ValueError(f"Token {target_token} was not found in split '{split}'.")
        print(f"Evaluating only token: {target_token}")
    
    # Load PDM Scoring
    config_path_pdm = os.path.abspath("navsim/planning/script/config/pdm_scoring")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_path_pdm, version_base=None)
    cfg_pdm = hydra.compose(config_name="default_run_pdm_score", overrides=[])
    
    pdm_simulator: PDMSimulator = instantiate(cfg_pdm.simulator)
    pdm_scorer: PDMScorer = instantiate(cfg_pdm.scorer)
    if cfg_pdm.traffic_agents == "non_reactive":
        pdm_traffic_agents_policy = instantiate(cfg_pdm.traffic_agents_policy.non_reactive, pdm_simulator.proposal_sampling)
    else:
        pdm_traffic_agents_policy = instantiate(cfg_pdm.traffic_agents_policy.reactive, pdm_simulator.proposal_sampling)
        
    metric_cache_path = Path(os.environ.get("NAVSIM_EXP_ROOT")) / "metric_cache"
    metric_cache_loader = MetricCacheLoader(metric_cache_path)

    if target_token is None:
        print(f"Visualizing {num_samples} seqs...")
    else:
        print("Visualizing 1 requested seq...")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 12})
    selected_tokens = list(enumerate(scene_loader.tokens))
    if target_token is not None:
        selected_tokens = [(idx, token) for idx, token in selected_tokens if token == target_token]
    else:
        selected_tokens = selected_tokens[::interval][:num_samples]

    # Prepare the fixed candidate trajectories once, before warm-up and timing.
    clustered_trajectories = np.load(PLANNING_VOCABULARY_PATH)
    num_trajs_to_plot = num_candidate_traj
    trajectory_interval = len(clustered_trajectories) // num_trajs_to_plot
    sampled_trajs = clustered_trajectories[::trajectory_interval][:num_trajs_to_plot]
    # sampled_trajs are [N, 40, 3] at 0.1s,...,4.0s. Select 0.5s,...,4.0s.
    downsampled_trajs = sampled_trajs[:, 4::5, :]
    traj_8_pts_all = downsampled_trajs[:, :8, :]
    actions_tensor = torch.tensor(traj_8_pts_all, dtype=torch.float32, device=device)
    actions_tensor_T = actions_tensor.transpose(0, 1).detach()

    # Repeat the first selected scene for warm-up without consuming evaluated scenes.
    warmup_tokens = selected_tokens[:1] * planning_warmup_runs
    planning_times_s = []
    print(
        f"Planning timing: {len(warmup_tokens)} warm-up runs, {len(selected_tokens)} measured scenes; "
        "preprocessing through trajectory selection, excluding scene loading and subsequent evaluation.",
        flush=True,
    )
    epdms_stage1_avg = 0.0 # EPDMS Stage 1 Average (without extended comfort)
    epdms_stage1_weighted_avg = 0.0 # EPDMS Stage 1 Average (without extended comfort), without considering critical safety check

    if compute_human_pdms:
        epdms_human_stage1_avg = 0.0
        epdms_human_stage1_weighted_avg = 0.0

    csv_results = []
    csv_results_human = []
    import pandas as pd

    with torch.no_grad():
        for is_planning_warmup, tokens in ((True, warmup_tokens), (False, selected_tokens)):
            for token_idx, token in tokens:
                if not is_planning_warmup:
                    print("Number of processed samples:", len(csv_results))
                print(f"Loading scene {token_idx}: {token}")
                scene = scene_loader.get_scene_from_token(token)
                sequence_name = scene.scene_metadata.log_name
                print(f"Current Sequence (Log) Name: {sequence_name}")
                
                num_total_frames = len(scene.frames)
                print("debug: total frames in scene", num_total_frames)    

                # Include CPU preprocessing and completed GPU work in planning time.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                planning_start = time.perf_counter()

                # 1. Get GT observations
                imgs_tensors = []
                for k in range(num_total_frames):
                    frame_idx =k
                    f0 = scene.frames[frame_idx].cameras.cam_f0.image
                    tensor_img = transforms.ToTensor()(f0[28:-28])
                    tensor_img = DINO_IMAGE_TRANSFORM(tensor_img)
                    imgs_tensors.append(tensor_img)
                    
                gt_seq = torch.stack(imgs_tensors, dim=0).unsqueeze(0).to(device) # (1, 14, 3, 256, 512)
                global_ego_poses = [scene.frames[k].ego_status.ego_pose for k in range(num_total_frames)]

                # 2. Precompute GT Z_tgts
                # Visual patch encoding is frame-independent; actions are attached during rollout.
                dino_wm = agent._dino_wm_model
                frame_indices = list(range(dino_wm.num_hist)) + [
                    dino_wm.num_hist + future_step - 1 for future_step, _ in EVALUATION_HORIZONS
                ]
                curr_obs = {"visual": gt_seq[:, frame_indices]}
                z_patch = dino_wm.encode_obs(curr_obs)["visual"]
                gt_z_src_list = [z_patch[:, : dino_wm.num_hist]]
                # Only the four scored horizons need target embeddings during planning.
                gt_z_tgt_list = [None] * 8
                for horizon_idx, (future_step, _) in enumerate(EVALUATION_HORIZONS):
                    target_idx = dino_wm.num_hist + horizon_idx
                    gt_z_tgt_list[future_step - 1] = z_patch[:, target_idx : target_idx + 1]

                # 3. Evaluate candidate trajectories in a batched manner
                class BatchedWMUnroll:
                    def __init__(
                        self,
                        agent,
                        global_ego_poses,
                        device,
                        chunk_size=128,
                    ):
                        self.agent = agent
                        self.global_ego_poses = global_ego_poses
                        self.device = device
                        self.chunk_size = chunk_size
                        
                    def __call__(self, z_init, actions=None, act_suffix=None):
                        if act_suffix is not None:
                            actions = act_suffix
                        actions = actions.permute(1, 0, 2) # [B, 8, 3]
                        B = actions.shape[0] if actions.dim() == 3 else 1
                        
                        origin_se2 = StateSE2(*self.global_ego_poses[3])
                        origin_x, origin_y, origin_yaw = origin_se2.x, origin_se2.y, origin_se2.heading
                        c, s = np.cos(origin_yaw), np.sin(origin_yaw)
                        x_local, y_local, yaw_local = actions[:, :, 0], actions[:, :, 1], actions[:, :, 2]
                        x_g = origin_x + x_local * c - y_local * s
                        y_g = origin_y + x_local * s + y_local * c
                        yaw_g = origin_yaw + yaw_local
                        future_poses = torch.stack([x_g, y_g, yaw_g], dim=-1) # [B, 8, 3]
                        history_poses = torch.tensor(np.array(self.global_ego_poses[:4]), dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, 4, 3)
                        candidate_simulated_poses = torch.cat([history_poses, future_poses], dim=1) # [B, 12, 3]
                        
                        gt_z_src_expanded = z_init.repeat(B, *([1]*(z_init.ndim-1)))
                        rollout_context = gt_z_src_expanded
                        predicted_encs = []
                        
                        for k in range(8):
                            curr_ego_poses = candidate_simulated_poses[:, k:k+5, :] # [B, 5, 3]
                            curr_actions = consecutive_relative_se2_actions(curr_ego_poses) # [B, 4, 3]

                            with torch.no_grad():
                                dino_wm = self.agent._dino_wm_model
                                cur_act_embed = dino_wm.encode_act(curr_actions)

                                z_pred = dino_wm.predict(
                                    dino_wm.concat_obs_act(rollout_context, cur_act_embed),
                                )

                            z_p_loss = z_pred[:, -1:, :, :-dino_wm.action_dim]
                            predicted_encs.append(z_p_loss)
                            rollout_context = torch.cat(
                                [rollout_context[:, 1:], z_p_loss], dim=1
                            )
                            
                        predicted_encs = torch.stack(predicted_encs, dim=0) # [8, B, ...]
                        return predicted_encs
                        
                class GCZSPEvalObjective:
                    def __init__(
                        self,
                        gt_z_tgt_last,
                        agent,
                    ):
                        self.gt_z_tgt_last = gt_z_tgt_last
                        self.agent = agent
                        
                    def __call__(self, encodings, actions):
                        z_pred_last = encodings[-1] # [B, ...]
                        z_t_loss = self.gt_z_tgt_last[:, -1:].detach()
                        
                        z_t_loss = z_t_loss.expand(z_pred_last.shape[0], *([-1]*(z_pred_last.ndim-1)))
                        diff = (z_pred_last - z_t_loss) ** 2
                        return diff.mean(dim=list(range(1, diff.ndim)))
                        
                unroll_fn = BatchedWMUnroll(
                    agent,
                    global_ego_poses,
                    device=device,
                )
                objective_fns = {
                    future_step: GCZSPEvalObjective(
                        gt_z_tgt_list[future_step - 1],
                        agent,
                    )
                    for future_step, _ in EVALUATION_HORIZONS
                }
                
                # Evaluate pre-clustered trajectories for GCZSP
                print(f"Evaluating {num_trajs_to_plot} candidate trajectories directly...")
                
                # Unroll and score in chunks to avoid OOM
                chunk_size = num_chunks
                all_costs = []
                all_step_latent_dists = [] if save_plot else None
                with torch.no_grad():
                    z_init = gt_z_src_list[0].detach()
                    final_gt_z_tgt = gt_z_tgt_list[-1].detach()
                    
                    for i in range(0, num_trajs_to_plot, chunk_size):
                        print(f"debug eval traj chunk {i}")
                        chunk_actions = actions_tensor_T[:, i:i+chunk_size, :]
                        predicted_encs = unroll_fn(z_init, actions=chunk_actions) # [8, chunk_size, ...]
                        costs = torch.stack([
                            objective_fns[future_step](predicted_encs[:future_step], chunk_actions[:future_step])
                            for future_step, _ in EVALUATION_HORIZONS
                        ], dim=1) # [chunk_size, 4]
                        all_costs.append(costs.detach().cpu())

                        if save_plot:
                            step_latent_dists = []
                            for step_idx in range(predicted_encs.shape[0]):
                                step_latent_dists.append(
                                    _latent_distance_for_step(
                                        predicted_encs[step_idx],
                                        final_gt_z_tgt,
                                    )
                                )
                            all_step_latent_dists.append(torch.stack(step_latent_dists, dim=1).detach().cpu())
                            del step_latent_dists
                        
                        del predicted_encs
                        del costs
                        del chunk_actions
                
                traj_losses = torch.cat(all_costs, dim=0).numpy() # [num_trajs_to_plot, 4]
                traj_step_latent_dists = torch.cat(all_step_latent_dists, dim=0).numpy() if save_plot else None
                best_traj_indices = np.argmin(traj_losses, axis=0)
                best_traj_idx = best_traj_indices[-1] # 4s selection for EPDMS and visualization
                best_traj_8_pts = traj_8_pts_all[best_traj_idx]
                best_traj = best_traj_8_pts

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                planning_time_s = time.perf_counter() - planning_start
                if is_planning_warmup:
                    print(f"[{sequence_name} | {token}] Planning warm-up complete (excluded from statistics).", flush=True)
                    continue
                planning_times_s.append(planning_time_s)
                print(
                    f"[{sequence_name} | {token}] Planning time: {planning_time_s:.6f} s "
                    f"({planning_time_s * 1000:.2f} ms)",
                    flush=True,
                )

                if save_pca_visualization:
                    print("Generating temporal PCA patch-embedding visualizations...")
                    selected_actions = actions_tensor_T[:, best_traj_idx : best_traj_idx + 1, :]
                    selected_predicted_encs = unroll_fn(
                        gt_z_src_list[0].detach(),
                        actions=selected_actions,
                    )
                    pca_paths = _generate_pca_visualizations(
                        dino_wm=agent._dino_wm_model,
                        gt_seq=gt_seq,
                        predicted_encodings=selected_predicted_encs,
                        output_path=output_path,
                        sequence_name=sequence_name,
                        token=token,
                    )
                    print(f"Saved {len(pca_paths)} PCA visualizations under {os.path.dirname(pca_paths[0])}")
                    del selected_predicted_encs
                    del selected_actions

                pose_origin_se2 = StateSE2(*global_ego_poses[3])
                gt_future_poses = np.array(global_ego_poses[4:12], dtype=np.float64)
                gt_future_local = convert_absolute_to_relative_se2_array(pose_origin_se2, gt_future_poses)
                # Roll out recorded actions from the same history, exclusively for GT hit rates.
                gt_actions_tensor_T = torch.tensor(gt_future_local, dtype=torch.float32, device=device).unsqueeze(1)
                gt_predicted_encs = unroll_fn(gt_z_src_list[0].detach(), actions=gt_actions_tensor_T)
                gt_traj_losses = torch.stack([
                    objective_fns[future_step](gt_predicted_encs[:future_step], gt_actions_tensor_T[:future_step])
                    for future_step, _ in EVALUATION_HORIZONS
                ], dim=1).squeeze(0).detach().cpu().numpy()
                del gt_predicted_encs, gt_actions_tensor_T
                if not (np.isfinite(traj_losses).all() and np.isfinite(gt_traj_losses).all()):
                    raise ValueError("GT hit rates require finite anchor and GT latent distances.")

                pose_metrics = {}
                hit_metrics = {}
                for horizon_idx, (future_step, horizon_label) in enumerate(EVALUATION_HORIZONS):
                    selected_idx = best_traj_indices[horizon_idx]
                    selected_pose = traj_8_pts_all[selected_idx, future_step - 1:future_step, :]
                    # Signed errors in the GT pose's frame at this horizon; SE2 conversion wraps heading.
                    pose_delta = convert_absolute_to_relative_se2_array(
                        StateSE2(*gt_future_local[future_step - 1]), selected_pose
                    )[0]
                    fde = float(np.linalg.norm(pose_delta[:2]))
                    delta_x, delta_y = float(pose_delta[0]), float(pose_delta[1])
                    delta_heading = float(np.rad2deg(pose_delta[2]))
                    pose_metrics.update({
                        f"fde_{horizon_label}_m": fde,
                        f"delta_x_{horizon_label}_m": delta_x,
                        f"delta_y_{horizon_label}_m": delta_y,
                        f"delta_heading_{horizon_label}_deg": delta_heading,
                        f"abs_delta_x_{horizon_label}_m": abs(delta_x),
                        f"abs_delta_y_{horizon_label}_m": abs(delta_y),
                        f"abs_delta_heading_{horizon_label}_deg": abs(delta_heading),
                    })
                    gt_loss = float(gt_traj_losses[horizon_idx])
                    anchor_losses = traj_losses[:, horizon_idx]
                    num_better = int(np.count_nonzero(anchor_losses < gt_loss))
                    num_tied = 1 + int(np.count_nonzero(anchor_losses == gt_loss))
                    # Share available top-k places among exact ties, including GT itself.
                    hit_at_1 = float(np.clip((1 - num_better) / num_tied, 0.0, 1.0))
                    hit_at_5 = float(np.clip((5 - num_better) / num_tied, 0.0, 1.0))
                    hit_metrics.update({
                        f"gt_latent_mse_{horizon_label}": gt_loss,
                        f"gt_hit_at_1_{horizon_label}": hit_at_1,
                        f"gt_hit_at_5_{horizon_label}": hit_at_5,
                    })
                    print(
                        f"[{future_step * 0.5:g}s] Best idx: {selected_idx}, "
                        f"latent MSE: {traj_losses[selected_idx, horizon_idx]:.6f}, "
                        f"FDE: {fde:.4f} m, delta_x: {delta_x:.4f} m, "
                        f"delta_y: {delta_y:.4f} m, delta_heading: {delta_heading:.4f} deg"
                    )
                    print(
                        f"[{future_step * 0.5:g}s] GT latent MSE: {gt_loss:.6f}, "
                        f"GT Hit@1: {100 * hit_at_1:.2f}%, GT Hit@5: {100 * hit_at_5:.2f}%"
                    )
                
                final_score_val = None
                human_score_val = None
                
                # --- PDM Score Calculation ---
                #try:
                best_traj_8_pts = best_traj
                trajectory_obj = Trajectory(best_traj_8_pts, TrajectorySampling(num_poses=8, interval_length=0.5))
                metric_cache = metric_cache_loader.get_from_token(token)
                
                score_row_df, _ = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory_obj,
                    future_sampling=pdm_simulator.proposal_sampling,
                    simulator=pdm_simulator,
                    scorer=pdm_scorer,
                    traffic_agents_policy=pdm_traffic_agents_policy,
                )
                
                row = score_row_df.iloc[0]
                weighted_metrics = row["weighted_metrics"].copy()
                weighted_metrics_array = row["weighted_metrics_array"].copy()
                
                two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
                weighted_metrics[two_frame_idx] = 0.0
                weighted_metrics_array[two_frame_idx] = 0.0
                
                total_weight = np.sum(weighted_metrics_array)
                weighted_score = np.sum(weighted_metrics * weighted_metrics_array) / total_weight if total_weight > 0 else 0.0
                final_score = row["multiplicative_metrics_prod"] * weighted_score
                final_score_val = final_score
                
                print(f"[{sequence_name} | {token}] PDM Score: {final_score:.4f} (Mult: {row['multiplicative_metrics_prod']:.4f}, Weighted: {weighted_score:.4f})")
                print("DINO-WM all metrics", row)

                epdms_stage1_avg += final_score
                epdms_stage1_weighted_avg += weighted_score
                
                result_dict = {
                    "idx": token_idx,
                    "token": token,
                    "planning_time_s": planning_time_s,
                    "final_score": final_score,
                    "weighted_score": weighted_score,
                    "no_at_fault_collisions": row["no_at_fault_collisions"],
                    "drivable_area_compliance": row["drivable_area_compliance"],
                    "driving_direction_compliance": row["driving_direction_compliance"],
                    "traffic_light_compliance": row["traffic_light_compliance"],
                    "ego_progress": row["ego_progress"],
                    "time_to_collision_within_bound": row["time_to_collision_within_bound"],
                    "lane_keeping": row["lane_keeping"],
                    "history_comfort": row["history_comfort"],
                    **pose_metrics,
                    **hit_metrics,
                }

                if compute_human_pdms:
                    human_origin_se2 = StateSE2(*global_ego_poses[3])
                    human_future_poses = np.array(global_ego_poses[4:12], dtype=np.float64)
                    human_local_poses = convert_absolute_to_relative_se2_array(human_origin_se2, human_future_poses)
                    
                    if len(human_local_poses) == 8:
                        human_trajectory_obj = Trajectory(human_local_poses, TrajectorySampling(num_poses=8, interval_length=0.5))
                        human_score_row_df, _ = pdm_score(
                            metric_cache=metric_cache,
                            model_trajectory=human_trajectory_obj,
                            future_sampling=pdm_simulator.proposal_sampling,
                            simulator=pdm_simulator,
                            scorer=pdm_scorer,
                            traffic_agents_policy=pdm_traffic_agents_policy,
                        )
                        
                        h_row = human_score_row_df.iloc[0]
                        h_weighted_metrics = h_row["weighted_metrics"].copy()
                        h_weighted_metrics_array = h_row["weighted_metrics_array"].copy()
                        
                        h_weighted_metrics[two_frame_idx] = 0.0
                        h_weighted_metrics_array[two_frame_idx] = 0.0

                        h_total_weight = np.sum(h_weighted_metrics_array)
                        h_weighted_score = np.sum(h_weighted_metrics * h_weighted_metrics_array) / h_total_weight if h_total_weight > 0 else 0.0
                        h_final_score = h_row["multiplicative_metrics_prod"] * h_weighted_score
                        human_score_val = h_final_score
                        
                        print(f"[{sequence_name} | {token}] Human PDM Score: {h_final_score:.4f} (Mult: {h_row['multiplicative_metrics_prod']:.4f}, Weighted: {h_weighted_score:.4f})")
                        print("Human all metrics", h_row)

                        epdms_human_stage1_avg += h_final_score
                        epdms_human_stage1_weighted_avg += h_weighted_score
                        
                        human_result_dict = {
                            "idx": token_idx,
                            "token": token,
                            "final_score": h_final_score,
                            "weighted_score": h_weighted_score,
                            "no_at_fault_collisions": h_row["no_at_fault_collisions"],
                            "drivable_area_compliance": h_row["drivable_area_compliance"],
                            "driving_direction_compliance": h_row["driving_direction_compliance"],
                            "traffic_light_compliance": h_row["traffic_light_compliance"],
                            "ego_progress": h_row["ego_progress"],
                            "time_to_collision_within_bound": h_row["time_to_collision_within_bound"],
                            "lane_keeping": h_row["lane_keeping"],
                            "history_comfort": h_row["history_comfort"],
                        }
                        csv_results_human.append(human_result_dict)

                csv_results.append(result_dict)

                # except Exception as e:
                #     print(f"Failed to compute PDM score for token {token}: {e}")
                # # --- End PDM Score Calculation ---

                if save_plot:
                    # Plot candidate waypoints only, without the BEV map background.
                    fig_waypoints, ax_waypoints = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
                    frame_idx_bev = 3
                    current_ego_pose_bev = global_ego_poses[frame_idx_bev]
                    origin_se2_bev = StateSE2(*current_ego_pose_bev)
                    
                    latent_values = traj_step_latent_dists[np.isfinite(traj_step_latent_dists)]
                    if latent_values.size == 0:
                        raise ValueError("No finite latent distances available for BEV waypoint coloring.")
                    vmin = np.percentile(latent_values, 2)
                    vmax = np.percentile(latent_values, 98)
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin = float(np.min(latent_values))
                        vmax = float(np.max(latent_values))
                    if vmax <= vmin:
                        vmax = vmin + 1e-12

                    latent_cmap = plt.get_cmap("coolwarm")
                    latent_norm = Normalize(vmin=vmin, vmax=vmax)

                    # gray curves for trajs
                    for idx in range(num_trajs_to_plot):
                        pose_array = np.concatenate([np.array([[0, 0]]), traj_8_pts_all[idx, :, :2]])
                        ax_waypoints.plot(
                            pose_array[:, 1],
                            pose_array[:, 0],
                            color="black",
                            alpha=0.08,
                            linewidth=0.6,
                            zorder=3,
                        )

                    waypoint_x = traj_8_pts_all[:, :, 1].reshape(-1)
                    waypoint_y = traj_8_pts_all[:, :, 0].reshape(-1)
                    waypoint_colors = traj_step_latent_dists.reshape(-1)
                    valid_waypoints = np.isfinite(waypoint_x) & np.isfinite(waypoint_y) & np.isfinite(waypoint_colors)
                    waypoint_scatter = ax_waypoints.scatter(
                        waypoint_x[valid_waypoints],
                        waypoint_y[valid_waypoints],
                        c=waypoint_colors[valid_waypoints],
                        cmap=latent_cmap,
                        norm=latent_norm,
                        s=5,
                        alpha=0.92,
                        edgecolors="none",
                        zorder=4,
                    )
                    cbar = fig_waypoints.colorbar(waypoint_scatter, ax=ax_waypoints, fraction=0.046, pad=0.04)
                    cbar.set_label("Euclidean latent distance to final GT future latent", fontsize=10)
                    
                    best_traj_8_pts = best_traj
                    best_sim_poses_bev = list(global_ego_poses[:4])
                    for i in range(8):
                        x_local, y_local, yaw_local = best_traj_8_pts[i]
                        c, s = np.cos(origin_se2_bev.heading), np.sin(origin_se2_bev.heading)
                        x_g = origin_se2_bev.x + x_local * c - y_local * s
                        y_g = origin_se2_bev.y + x_local * s + y_local * c
                        yaw_g = origin_se2_bev.heading + yaw_local
                        best_sim_poses_bev.append((x_g, y_g, yaw_g))

                    best_pose_array = np.concatenate([np.array([[0, 0]]), best_traj_8_pts[:, :2]])

                    # plot best traj
                    # ax_waypoints.plot(
                    #     best_pose_array[:, 1],
                    #     best_pose_array[:, 0],
                    #     color="lime",
                    #     alpha=0.95,
                    #     linewidth=2.0,
                    #     zorder=5,
                    #     label="Selected",
                    # )

                    # Mark GT future waypoints as a reference for the latent-distance colors.
                    future_human_poses_bev = np.array(global_ego_poses[4:12], dtype=np.float64)
                    local_human_poses_bev = None
                    if len(future_human_poses_bev) > 0:
                        local_human_poses_bev = convert_absolute_to_relative_se2_array(origin_se2_bev, future_human_poses_bev)
                        if len(local_human_poses_bev) > 0:
                            human_pose_array = np.concatenate([np.array([[0, 0]]), local_human_poses_bev[:, :2]])
                            # plot GT Human future waypoints
                            # ax_waypoints.plot(
                            #     human_pose_array[:, 1],
                            #     human_pose_array[:, 0],
                            #     color="gold",
                            #     alpha=0.9,
                            #     linewidth=1.4,
                            #     zorder=5,
                            #     label="Human",
                            # )
                            # ax_waypoints.scatter(
                            #     local_human_poses_bev[:, 1],
                            #     local_human_poses_bev[:, 0],
                            #     marker="*",
                            #     s=5,
                            #     c="yellow",
                            #     edgecolors="black",
                            #     linewidths=0.5,
                            #     zorder=6,
                            # )
                            final_future_xy = local_human_poses_bev[-1, :2]
                            ax_waypoints.scatter(
                                final_future_xy[1],
                                final_future_xy[0],
                                marker="*",
                                s=180,
                                c="yellow",
                                edgecolors="black",
                                linewidths=0.8,
                                zorder=7,
                                label="Final GT future",
                            )

                    all_plot_x = waypoint_x[valid_waypoints]
                    all_plot_y = waypoint_y[valid_waypoints]
                    if local_human_poses_bev is not None and len(local_human_poses_bev) > 0:
                        all_plot_x = np.concatenate([all_plot_x, local_human_poses_bev[:, 1]])
                        all_plot_y = np.concatenate([all_plot_y, local_human_poses_bev[:, 0]])
                    x_pad = max(2.0, 0.08 * (np.max(all_plot_x) - np.min(all_plot_x)))
                    y_pad = max(2.0, 0.08 * (np.max(all_plot_y) - np.min(all_plot_y)))

                    ax_waypoints.set_xlim(np.max(all_plot_x) + x_pad, np.min(all_plot_x) - x_pad)
                    ax_waypoints.set_ylim(np.min(all_plot_y) - y_pad, np.max(all_plot_y) + y_pad)
                    ax_waypoints.set_aspect("equal", adjustable="box")
                    ax_waypoints.grid(True, color="0.88", linewidth=0.5)
                    ax_waypoints.set_xlabel("local y (m)")
                    ax_waypoints.set_ylabel("local x (m)")
                    ax_waypoints.set_title(f"Candidate waypoint latent distances (N={num_trajs_to_plot})", fontsize=14, fontweight='bold', pad=3)
                    ax_waypoints.legend(loc="lower right", frameon=True, fontsize=9)
                    
                    if final_score_val is not None and human_score_val is not None:
                        pdms_text = f"Human: {human_score_val*100:.2f} (PDMS)\nDINO-WM: {final_score_val*100:.2f} (PDMS)"
                        ax_waypoints.text(0.95, 0.95, pdms_text, transform=ax_waypoints.transAxes,
                                    color='red', fontsize=12, fontweight='bold',
                                    ha='right', va='top')
                    
                    waypoints_output_path = output_path.replace(".png", f"{sequence_name}_{token}_candidate_waypoints_heatmap.png")
                    os.makedirs(os.path.dirname(waypoints_output_path), exist_ok=True)
                    fig_waypoints.savefig(waypoints_output_path, bbox_inches='tight', pad_inches=0.1)

                    ax_waypoints.plot(
                        best_pose_array[:, 1],
                        best_pose_array[:, 0],
                        color="lime",
                        alpha=0.95,
                        linewidth=2.0,
                        zorder=5,
                        label="Selected",
                    )
                    if local_human_poses_bev is not None and len(local_human_poses_bev) > 0:
                        ax_waypoints.plot(
                            human_pose_array[:, 1],
                            human_pose_array[:, 0],
                            color="gold",
                            alpha=0.9,
                            linewidth=1.4,
                            zorder=5,
                            label="Human",
                        )
                    ax_waypoints.set_title(
                        f"Candidate waypoint latent distances with trajectories (N={num_trajs_to_plot})",
                        fontsize=14,
                        fontweight='bold',
                        pad=3,
                    )
                    ax_waypoints.legend(loc="lower right", frameon=True, fontsize=9)
                    waypoints_with_trajs_output_path = output_path.replace(".png", f"{sequence_name}_{token}_candidate_waypoints_heatmap_with_trajs.png")
                    fig_waypoints.savefig(waypoints_with_trajs_output_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig_waypoints)

                    fig_final_waypoints, ax_final_waypoints = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
                    final_waypoint_colors = traj_step_latent_dists[:, -1]
                    valid_final_waypoints = (
                        np.isfinite(traj_8_pts_all[:, -1, 1])
                        & np.isfinite(traj_8_pts_all[:, -1, 0])
                        & np.isfinite(final_waypoint_colors)
                    )
                    final_latent_values = final_waypoint_colors[valid_final_waypoints]
                    if final_latent_values.size == 0:
                        raise ValueError("No finite final-waypoint latent distances available for plotting.")
                    final_vmin = np.percentile(final_latent_values, 2)
                    final_vmax = np.percentile(final_latent_values, 98)
                    if not np.isfinite(final_vmin) or not np.isfinite(final_vmax) or final_vmax <= final_vmin:
                        final_vmin = float(np.min(final_latent_values))
                        final_vmax = float(np.max(final_latent_values))
                    if final_vmax <= final_vmin:
                        final_vmax = final_vmin + 1e-12
                    final_latent_norm = Normalize(vmin=final_vmin, vmax=final_vmax)

                    for idx in range(num_trajs_to_plot):
                        pose_array = np.concatenate([np.array([[0, 0]]), traj_8_pts_all[idx, :, :2]])
                        ax_final_waypoints.plot(
                            pose_array[:, 1],
                            pose_array[:, 0],
                            color="black",
                            alpha=0.08,
                            linewidth=0.6,
                            zorder=3,
                        )

                    final_waypoint_scatter = ax_final_waypoints.scatter(
                        traj_8_pts_all[valid_final_waypoints, -1, 1],
                        traj_8_pts_all[valid_final_waypoints, -1, 0],
                        c=final_waypoint_colors[valid_final_waypoints],
                        cmap=latent_cmap,
                        norm=final_latent_norm,
                        s=16,
                        alpha=0.95,
                        edgecolors="none",
                        zorder=4,
                    )
                    final_cbar = fig_final_waypoints.colorbar(final_waypoint_scatter, ax=ax_final_waypoints, fraction=0.046, pad=0.04)
                    final_cbar.set_label("Euclidean latent distance to final GT future latent", fontsize=10)

                    all_final_plot_x = waypoint_x[np.isfinite(waypoint_x) & np.isfinite(waypoint_y)]
                    all_final_plot_y = waypoint_y[np.isfinite(waypoint_x) & np.isfinite(waypoint_y)]
                    final_x_pad = max(2.0, 0.08 * (np.max(all_final_plot_x) - np.min(all_final_plot_x)))
                    final_y_pad = max(2.0, 0.08 * (np.max(all_final_plot_y) - np.min(all_final_plot_y)))
                    ax_final_waypoints.set_xlim(np.max(all_final_plot_x) + final_x_pad, np.min(all_final_plot_x) - final_x_pad)
                    ax_final_waypoints.set_ylim(np.min(all_final_plot_y) - final_y_pad, np.max(all_final_plot_y) + final_y_pad)
                    ax_final_waypoints.set_aspect("equal", adjustable="box")
                    ax_final_waypoints.grid(True, color="0.88", linewidth=0.5)
                    ax_final_waypoints.set_xlabel("local y (m)")
                    ax_final_waypoints.set_ylabel("local x (m)")
                    ax_final_waypoints.set_title(
                        f"Candidate final-waypoint latent distances (N={num_trajs_to_plot})",
                        fontsize=14,
                        fontweight='bold',
                        pad=3,
                    )
                    # visualize final GT future waypoint
                    final_future_xy = local_human_poses_bev[-1, :2]
                    ax_final_waypoints.scatter(
                        final_future_xy[1],
                        final_future_xy[0],
                        marker="*",
                        s=15,
                        c="yellow",
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=7,
                        label="Final GT future",
                    )
                    ax_final_waypoints.legend(loc="lower right", frameon=True, fontsize=9)                    

                    final_waypoints_output_path = output_path.replace(".png", f"{sequence_name}_{token}_candidate_final_waypoints_heatmap.png")
                    fig_final_waypoints.savefig(final_waypoints_output_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig_final_waypoints)

                    # Per-frame planning visualization
                    def unprocess_imagenet(tensor):
                        tensor = tensor.detach().cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = tensor * std + mean
                        img = img.permute(1, 2, 0).numpy()
                        return np.clip(img, 0, 1)

                    frames_output_dir = output_path.replace(".png", f"{sequence_name}_{token}_frames")
                    os.makedirs(frames_output_dir, exist_ok=True)
                    gt_images_output_dir = os.path.join(frames_output_dir, "ground_truth")
                    os.makedirs(gt_images_output_dir, exist_ok=True)

                    # Frame index 3 is the last history frame and the shared planning origin.
                    # Export each 4-second plan on the same BEV background and scale.
                    plans_output_dir = os.path.join(frames_output_dir, "trajectories")
                    os.makedirs(plans_output_dir, exist_ok=True)
                    for plan_name, plan_poses in (
                        ("human", local_human_poses_bev),
                        ("agent", best_traj_8_pts),
                    ):
                        fig_plan, ax_plan = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
                        add_configured_bev_on_ax(ax_plan, scene.map_api, scene.frames[frame_idx_bev])
                        add_trajectory_to_bev_ax(
                            ax_plan,
                            Trajectory(
                                plan_poses,
                                TrajectorySampling(num_poses=len(plan_poses), interval_length=0.5),
                            ),
                            {**TRAJECTORY_CONFIG[plan_name], "line_width": 6.0, "marker_size": 8},
                        )
                        configure_bev_ax(ax_plan)
                        ax_plan.set_xlim(64, -64)
                        ax_plan.set_ylim(-64, 64)
                        ax_plan.set_axis_off()
                        fig_plan.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        plan_path = os.path.join(
                            plans_output_dir, f"frame_{frame_idx_bev:04d}_{plan_name}.png"
                        )
                        fig_plan.savefig(plan_path, bbox_inches="tight", pad_inches=0)
                        plt.close(fig_plan)

                    for frame_idx in range(12): # 4 history + 8 predicted
                        img_gt_np = unprocess_imagenet(gt_seq[0, frame_idx])
                        # Save the camera pixels directly, without figure decorations or padding.
                        plt.imsave(
                            os.path.join(gt_images_output_dir, f"frame_{frame_idx:04d}.png"),
                            img_gt_np,
                        )
                        is_hist = frame_idx < agent._dino_wm_model.num_hist
                        text_top = f"Step {frame_idx} / 11"

                        gt_label = "Ground Truth (History)" if is_hist else "Ground Truth (Future)"
                        color = "green" if is_hist else "red"

                        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

                        # BEV Trajectory
                        frame_for_bev = copy.deepcopy(scene.frames[frame_idx])
                        current_ego_pose = best_sim_poses_bev[frame_idx]
                        frame_for_bev = _remap_frame_bev_to_ego_pose(frame_for_bev, np.array(current_ego_pose, dtype=np.float64))
                        add_configured_bev_on_ax(axes[0], scene.map_api, frame_for_bev)

                        if frame_idx < 11:
                            origin_se2 = StateSE2(*current_ego_pose)

                            future_optimal_poses = np.array(best_sim_poses_bev[frame_idx + 1: 12], dtype=np.float64)
                            local_optimal_poses = convert_absolute_to_relative_se2_array(origin_se2, future_optimal_poses)
                            if len(local_optimal_poses) > 0 and frame_idx >= 3:
                                add_trajectory_to_bev_ax(axes[0], Trajectory(local_optimal_poses, TrajectorySampling(num_poses=len(local_optimal_poses), interval_length=0.5)), TRAJECTORY_CONFIG["agent"])

                            future_human_poses = np.array(global_ego_poses[frame_idx + 1: 12], dtype=np.float64)
                            local_human_poses = convert_absolute_to_relative_se2_array(origin_se2, future_human_poses)
                            if len(local_human_poses) > 0:
                                add_trajectory_to_bev_ax(axes[0], Trajectory(local_human_poses, TrajectorySampling(num_poses=len(local_human_poses), interval_length=0.5)), TRAJECTORY_CONFIG["human"])

                        configure_bev_ax(axes[0])
                        axes[0].set_xlim(64, -64)
                        axes[0].set_ylim(-64, 64)
                        axes[0].set_title("BEV Map & Trajectories", color=color, fontsize=14, fontweight='bold', pad=3)

                        # if frame_idx >= 3 and final_score_val is not None and human_score_val is not None:
                        #     pdms_text = f"Human: {human_score_val*100:.2f} (PDMS)\nDINO-WM: {final_score_val*100:.2f} (PDMS)"
                        #     axes[0].text(0.95, 0.95, pdms_text, transform=axes[0].transAxes,
                        #                 color='red', fontsize=10, fontweight='bold',
                        #                 ha='right', va='top')

                        # GT Camera
                        axes[1].imshow(img_gt_np)
                        axes[1].set_title(gt_label, color=color, fontsize=14, fontweight='bold', pad=3)
                        axes[1].axis('off')

                        fig.suptitle(text_top, fontsize=16, fontweight='bold', y=1.0)

                        if 'result_dict' in locals():
                            metric_keys = [
                                ("NC", "no_at_fault_collisions"),
                                ("DAC", "drivable_area_compliance"),
                                ("DDC", "driving_direction_compliance"),
                                ("TTC", "traffic_light_compliance"),
                                ("EP", "ego_progress"),
                                ("TTC", "time_to_collision_within_bound"),
                                ("LK", "lane_keeping"),
                                ("HC", "history_comfort"),
                                ("EPDMS", "final_score"),
                                ("EPDMS'", "weighted_score")
                            ]
                            has_human = 'human_result_dict' in locals()
                            x_coords = np.linspace(0.12, 0.98, len(metric_keys))

                            y_name = 0.09 if has_human else 0.07
                            y_human = 0.05
                            y_dino = 0.01 if has_human else 0.03
                            bottom_margin = 0.24 if has_human else 0.18

                            if has_human:
                                fig.text(0.01, y_human, "Human:", ha='left', va='center', fontsize=12, fontweight='bold', color='black')
                            fig.text(0.01, y_dino, "DINO-WM:", ha='left', va='center', fontsize=12, fontweight='bold', color='black')

                            for (name, key), x in zip(metric_keys, x_coords):
                                color_m = 'red' if name == 'EP' else 'black'
                                fig.text(x, y_name, name, ha='center', va='center', fontsize=12, fontweight='bold', color=color_m)
                                if has_human:
                                    val_human = f"{human_result_dict.get(key, 0.0) * 100:.2f}"
                                    fig.text(x, y_human, val_human, ha='center', va='center', fontsize=12, fontweight='bold', color=color_m)
                                val_dino = f"{result_dict.get(key, 0.0) * 100:.2f}"
                                fig.text(x, y_dino, val_dino, ha='center', va='center', fontsize=12, fontweight='bold', color=color_m)

                            fig.subplots_adjust(wspace=0.005, hspace=0.01, left=0.01, right=0.99, top=0.88, bottom=bottom_margin)
                        else:
                            fig.subplots_adjust(wspace=0.005, hspace=0.01, left=0.01, right=0.99, top=0.88, bottom=0.01)

                        frame_path = os.path.join(frames_output_dir, f"frame_{frame_idx:04d}.png")
                        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0.0)
                        plt.close(fig)
                    print(f"Saved 12 planning frames for sequence {sequence_name} under {frames_output_dir}")
                    print(f"Saved 12 clean ground-truth camera images under {gt_images_output_dir}")
                    print(f"Saved separate human and agent plans at frame {frame_idx_bev} under {plans_output_dir}")
        
        if not planning_times_s:
            print("No scenes evaluated.")
            return
        print(
            f"Planning time over {len(planning_times_s)} scenes (warm-up excluded): "
            f"mean={np.mean(planning_times_s):.6f} s, "
            f"median={np.median(planning_times_s):.6f} s, "
            f"P95={np.percentile(planning_times_s, 95):.6f} s",
            flush=True,
        )

        if target_token is not None:
            print(f"Single-token evaluation complete for {target_token}; skipped CSV save and metric-average reporting.")
            return

        # compute average epdms
        epdms_stage1_avg /= len(csv_results)
        epdms_stage1_weighted_avg /= len(csv_results)
        print("EPDMS Stage 1 Average:", epdms_stage1_avg)
        print("EPDMS Stage 1 Weighted Average:", epdms_stage1_weighted_avg)

        if compute_human_pdms and csv_results_human:
            epdms_human_stage1_avg /= len(csv_results_human)
            epdms_human_stage1_weighted_avg /= len(csv_results_human)
            print("EPDMS Human Stage 1 Average:", epdms_human_stage1_avg)
            print("EPDMS Human Stage 1 Weighted Average:", epdms_human_stage1_weighted_avg)

        if len(csv_results) > 0:
            df = pd.DataFrame(csv_results)
            avg_row = df.mean(numeric_only=True).to_dict()
            avg_row["idx"] = "average"
            avg_row["token"] = "average"
            df.loc[len(df)] = avg_row
            for future_step, horizon_label in EVALUATION_HORIZONS:
                print(
                    f"[{future_step * 0.5:g}s] Mean FDE: {avg_row[f'fde_{horizon_label}_m']:.4f} m, "
                    f"mean |delta_x|: {avg_row[f'abs_delta_x_{horizon_label}_m']:.4f} m, "
                    f"mean |delta_y|: {avg_row[f'abs_delta_y_{horizon_label}_m']:.4f} m, "
                    f"mean |delta_heading|: {avg_row[f'abs_delta_heading_{horizon_label}_deg']:.4f} deg"
                )
                print(
                    f"[{future_step * 0.5:g}s] Mean GT latent MSE: {avg_row[f'gt_latent_mse_{horizon_label}']:.6f}, "
                    f"GT Hit@1: {100 * avg_row[f'gt_hit_at_1_{horizon_label}']:.2f}%, "
                    f"GT Hit@5: {100 * avg_row[f'gt_hit_at_5_{horizon_label}']:.2f}%"
                )
            
            # update output_path, such that it include num_candidate_traj, num_samples 
            pdms_output_path_dino_wm = pdms_output_path.replace(".csv", f"dinowm-{num_candidate_traj}-{num_samples}.csv")
            os.makedirs(os.path.dirname(pdms_output_path_dino_wm), exist_ok=True)
            df.to_csv(pdms_output_path_dino_wm, index=False)
            print(f"Saved PDMS results to {pdms_output_path_dino_wm}")

        if compute_human_pdms and len(csv_results_human) > 0:
            df_h = pd.DataFrame(csv_results_human)
            avg_row_h = df_h.mean(numeric_only=True).to_dict()
            avg_row_h["idx"] = "average"
            avg_row_h["token"] = "average"
            df_h.loc[len(df_h)] = avg_row_h
            
            # update output_path, such that it include num_candidate_traj, num_samples 
            human_output_path = pdms_output_path.replace(".csv", f"human-{num_candidate_traj}-{num_samples}.csv")
            os.makedirs(os.path.dirname(human_output_path), exist_ok=True)
            df_h.to_csv(human_output_path, index=False)
            print(f"Saved Human PDMS results to {human_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate autoregressive latent prediction for DINO-WM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a vanilla DINO-WM checkpoint")
    parser.add_argument("--agent_config", type=str, default="dino_wm_agent", help="Hydra agent config used to construct the DINO-WM model")
    parser.add_argument("--extra_tag", type=str, default=None, help="Optional suffix for the default result directory, e.g. 0426_0030 -> dino_seq_gczsp_dino_wm_0426_0030")
    parser.add_argument("--output_path", type=str, default=None, help="Base path for saving figures and frame PNGs.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sequences to evaluate")
    parser.add_argument(
        "--planning_warmup_runs", type=int, default=5,
        help="Extra planning-only runs on the first selected scene before timing statistics (0 disables warm-up)",
    )
    parser.add_argument("--split", type=str, default="navtest", help="Data split to evaluate on (train, val, mini, etc)")
    parser.add_argument("--original_sensor_path", type=str, default=None, help="Override the sensor blob root")
    parser.add_argument("--compute_human_pdms", type=bool, default=False, help="Whether to compute human PDMs")
    parser.add_argument("--save_plot", type=bool, default=False, help="Whether to save plots")
    parser.add_argument(
        "--save_pca_visualization",
        action="store_true",
        help="Save temporal patch-embedding PCA maps using RGB = PC123",
    )
    parser.add_argument("--interval", type=int, default=100, help="Interval between samples")   
    parser.add_argument("--pdms_output_path", type=str, default=None, help="Base path for pdms to a .csv file")
    parser.add_argument("--num_candidate_traj", type=int, default=256, help="Number of candidate trajectories to plot")
    parser.add_argument("--num_chunks", type=int, default=32, help="Maximum candidate trajectories per rollout chunk")
    parser.add_argument("--token", type=str, default=None, help="Evaluate only the scene with this token; skips CSV and metric-average reporting")
    args = parser.parse_args()

    result_dir_name = "dino_seq_gczsp_dino_wm"
    if args.extra_tag:
        result_dir_name = f"{result_dir_name}_{args.extra_tag}"
    result_dir = os.path.join("scripts/evaluation_jepa/evaluation_dino_results", result_dir_name)
    output_path = args.output_path or os.path.join(result_dir, ".png")
    pdms_output_path = args.pdms_output_path or os.path.join(result_dir, "pdms-.csv")

    evaluate_dino_wm_seq(
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        pdms_output_path=pdms_output_path,
        num_samples=args.num_samples,
        split=args.split,
        compute_human_pdms=args.compute_human_pdms,
        save_plot=args.save_plot,
        save_pca_visualization=args.save_pca_visualization,
        interval=args.interval,
        num_candidate_traj=args.num_candidate_traj,
        num_chunks=args.num_chunks,
        target_token=args.token,
        original_sensor_path_override=args.original_sensor_path,
        agent_config=args.agent_config,
        planning_warmup_runs=args.planning_warmup_runs,
    )
