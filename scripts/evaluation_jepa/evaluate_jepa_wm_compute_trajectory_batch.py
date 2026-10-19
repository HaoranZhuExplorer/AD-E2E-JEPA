"""Evaluate batched JEPA-WM (vanilla/v2/v3) and LE-WM planning with stage-one EPDMS, horizon pose errors, and GT hit rates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.run_pdm_score_one_stage import (
    compute_final_scores,
    create_scene_aggregators,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.state_representation import StateSE2

EVALUATION_HORIZONS = ((1, "0p5s"), (2, "1s"), (4, "2s"), (8, "4s"))


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = [
        f"agent={args.agent_config}",
        "experiment_name=evaluate_jepa_wm_compute_trajectory_batch",
        f"train_test_split={args.split}",
    ]

    config_root = PROJECT_ROOT / "navsim/planning/script/config"
    with hydra.initialize_config_dir(config_dir=str(config_root / "training"), version_base=None):
        cfg = hydra.compose(config_name="default_training", overrides=overrides)

    print(f"Building {args.agent_config} on {device}", flush=True)
    agent = instantiate(cfg.agent)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.removeprefix("agent."): value for key, value in state_dict.items()}
    agent.load_state_dict(state_dict)
    agent.to(device).eval()
    del checkpoint, state_dict

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    if args.split in {"train", "val", "test"}:
        scene_filter.log_names = cfg[f"{args.split}_logs"]
    scene_filter.num_history_frames = 4
    scene_filter.num_future_frames = 10
    scene_filter.include_synthetic_scenes = False
    scene_loader = SceneLoader(
        original_sensor_path=args.original_sensor_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(args.cache_path)
    cached_tokens = set(metric_cache_loader.tokens)
    eligible_tokens = [(idx, token) for idx, token in enumerate(scene_loader.tokens) if token in cached_tokens]
    selected_tokens = eligible_tokens[::args.interval][:args.num_samples]
    selected_token_order = [token for _, token in selected_tokens]

    # Frame 3 is the planning timestamp; frame 2 identifies the preceding scene.
    # Include available predecessors even when sampling would omit them.
    evaluation_token_set = set(selected_token_order)
    for token in selected_token_order:
        frames = scene_loader.scene_frames_dicts[token]
        previous_token = frames[2]["token"]
        time_gap_us = frames[3]["timestamp"] - frames[2]["timestamp"]
        if (
            previous_token in scene_loader.scene_frames_dicts
            and previous_token in cached_tokens
            and 0 < time_gap_us < 550_000
        ):
            evaluation_token_set.add(previous_token)
    evaluation_tokens = [(idx, token) for idx, token in eligible_tokens if token in evaluation_token_set]
    print(
        f"Evaluating {len(selected_tokens)} requested scenes and "
        f"{len(evaluation_tokens) - len(selected_tokens)} supporting predecessors; "
        f"scene batch size={args.scene_batch_size}, candidates per scene={args.num_candidate_traj}, "
        f"rollout chunk size={args.rollout_chunk_size or 'all pairs in the scene batch'}",
        flush=True,
    )

    with hydra.initialize_config_dir(config_dir=str(config_root / "pdm_scoring"), version_base=None):
        cfg_pdm = hydra.compose(config_name="default_run_pdm_score")
    pdm_simulator = instantiate(cfg_pdm.simulator)
    pdm_scorer = instantiate(cfg_pdm.scorer)
    pdm_traffic_agents_policy = instantiate(
        cfg_pdm.traffic_agents_policy[cfg_pdm.traffic_agents],
        pdm_simulator.proposal_sampling,
    )

    def load_batch(batch_start):
        batch_tokens = evaluation_tokens[batch_start:batch_start + args.scene_batch_size]
        scenes = [scene_loader.get_scene_from_token(token) for _, token in batch_tokens]
        agent_inputs = [scene.get_agent_input() for scene in scenes]
        return batch_tokens, scenes, agent_inputs

    def score_batch(batch_tokens, scenes, trajectories, diagnostics):
        batch_score_rows = []
        for (token_idx, token), scene, trajectory, scene_diagnostics in zip(batch_tokens, scenes, trajectories, diagnostics):
            metric_cache = metric_cache_loader.get_from_token(token)
            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=pdm_simulator.proposal_sampling,
                simulator=pdm_simulator,
                scorer=pdm_scorer,
                traffic_agents_policy=pdm_traffic_agents_policy,
            )
            score_row["idx"] = token_idx
            score_row["token"] = token
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            score_row["ego_simulated_states"] = [ego_simulated_states]

            origin = StateSE2(*scene.frames[3].ego_status.ego_pose)
            gt_future_poses = np.array([frame.ego_status.ego_pose for frame in scene.frames[4:12]], dtype=np.float64)
            gt_future_local = convert_absolute_to_relative_se2_array(origin, gt_future_poses)
            traj_losses = scene_diagnostics["anchor_latent_mse"]
            gt_traj_losses = scene_diagnostics["gt_latent_mse"]
            if not (np.isfinite(traj_losses).all() and np.isfinite(gt_traj_losses).all()):
                raise ValueError("GT hit rates require finite anchor and GT latent distances.")
            pose_metrics = {}
            hit_metrics = {}
            for horizon_idx, (future_step, horizon_label) in enumerate(EVALUATION_HORIZONS):
                selected_pose = scene_diagnostics["selected_poses"][horizon_idx:horizon_idx + 1]
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
            score_row = score_row.assign(**pose_metrics, **hit_metrics)
            batch_score_rows.append(score_row)
        return batch_score_rows

    score_rows = []
    # Prefetch one batch and score one batch while the main thread runs GPU planning.
    # A single scoring worker owns the mutable simulator, scorer, and traffic policy.
    with ThreadPoolExecutor(max_workers=1) as loader_pool, ThreadPoolExecutor(max_workers=1) as scoring_pool:
        print("CPU/GPU overlap enabled: prefetch=1 batch, scoring=1 worker", flush=True)
        load_future = loader_pool.submit(load_batch, 0) if evaluation_tokens else None
        score_future = None
        for batch_start in range(0, len(evaluation_tokens), args.scene_batch_size):
            batch_tokens, scenes, agent_inputs = load_future.result()
            next_batch_start = batch_start + args.scene_batch_size
            if next_batch_start < len(evaluation_tokens):
                load_future = loader_pool.submit(load_batch, next_batch_start)
            trajectories, diagnostics = agent.compute_trajectory(
                agent_inputs,
                scenes,
                num_candidate_traj=args.num_candidate_traj,
                chunk_size=args.rollout_chunk_size,
                return_diagnostics=True,
            )

            if score_future is not None:
                score_rows.extend(score_future.result())
                print(f"Processed {len(score_rows)}/{len(evaluation_tokens)} scenes including predecessors", flush=True)
            score_future = scoring_pool.submit(score_batch, batch_tokens, scenes, trajectories, diagnostics)

        if score_future is not None:
            score_rows.extend(score_future.result())
            print(f"Processed {len(score_rows)}/{len(evaluation_tokens)} scenes including predecessors", flush=True)

    if score_rows:
        # Use NAVSIM's stage-one aggregation after all batches, including cross-batch pairs.
        df = pd.concat(score_rows, ignore_index=True)
        adjacent_mapping = infer_start_adjacent_mapping(df)
        if adjacent_mapping:
            df = create_scene_aggregators(adjacent_mapping, df, pdm_simulator.proposal_sampling)
        else:
            df["two_frame_extended_comfort"] = np.nan
            df = df.drop(columns=["ego_simulated_states"])
        df = compute_final_scores(df)
        df = df.set_index("token").loc[selected_token_order].reset_index()
        df["final_score"] = df["score"]
        df["extended_comfort_available"] = df["two_frame_extended_comfort"].notna()
        df["previous_token"] = df["token"].map(adjacent_mapping)
        df = df.drop(columns=["frame_type", "start_time"])
        print(
            f"Extended comfort available for {df['extended_comfort_available'].sum()}/{len(df)} requested scenes. "
            "Missing pairs follow NAVSIM's stage-one policy: comfort is NaN and its weight is excluded.",
            flush=True,
        )
        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row.update(idx="average", token="average")
        df.loc[len(df)] = avg_row
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_path, index=False)
        print(f"NAVSIM v2 Stage 1 EPDMS Average: {avg_row['score']:.4f}")
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
        print(f"Saved EPDMS results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_config", default="jepa_wm_agent", choices=[
        "jepa_wm_agent",
        "jepa_wm_agent_v2", "jepa_wm_agent_v3",
        "le_wm_agent",
    ])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim_log_path", type=Path, required=True)
    parser.add_argument("--original_sensor_path", type=Path, required=True)
    parser.add_argument("--cache_path", type=Path, default=PROJECT_ROOT / "navsim_experiments/metric_cache")
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--extra_tag", default=None)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--scene_batch_size", type=int, default=1)
    parser.add_argument("--num_candidate_traj", type=int, default=256)
    parser.add_argument(
        "--rollout_chunk_size", "--num_chunks", type=int, default=None,
        help="Maximum total scene/candidate pairs per rollout; defaults to the whole scene batch.",
    )
    args = parser.parse_args()
    if min(args.scene_batch_size, args.num_samples, args.interval, args.num_candidate_traj) < 1:
        parser.error("scene_batch_size, num_samples, interval, and num_candidate_traj must be positive.")
    if args.rollout_chunk_size is not None and args.rollout_chunk_size < 1:
        parser.error("rollout_chunk_size must be positive.")
    if args.output_path is None:
        tag = f"_{args.extra_tag}" if args.extra_tag else ""
        args.output_path = (
            PROJECT_ROOT / "scripts/evaluation_jepa/evaluation_jepa_results"
            / f"{args.agent_config}_compute_trajectory_batch{tag}_{args.num_candidate_traj}_{args.num_samples}.csv"
        )
    evaluate(args)
