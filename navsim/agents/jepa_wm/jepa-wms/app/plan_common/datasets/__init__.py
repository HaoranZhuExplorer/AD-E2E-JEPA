# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset dimension and normalization registry for JEPA-WMs environments.

This module provides hardcoded action_dim, proprio_dim, and normalization
statistics (mean/std) for each environment. This enables model loading via
torchhub without requiring access to the actual datasets.

To regenerate these values, run:
    python src/scripts/extract_dataset_stats.py

Note: The normalization statistics below are computed from the training datasets
and match the values used during model training.
"""

# Environment data dimensions and normalization statistics
# These are used by hubconf.py to load models without needing the datasets
# fmt: off
DATA_STATS = {
    # Simulation environments
    "metaworld": {
        "action_dim": 4,
        "proprio_dim": 4,
        "state_dim": 39,
        "action_mean": [0.005723577458411455, 0.15735651552677155, -0.1396457850933075, 0.1998193860054016],
        "action_std": [0.7359239459037781, 0.73408043384552, 0.7182977199554443, 0.743205726146698],
        "proprio_mean": [-0.0029392109718173742, 0.6544238924980164, 0.15410053730010986, 0.6402314901351929],
        "proprio_std": [0.10431358963251114, 0.11230525374412537, 0.07981759309768677, 0.26138949394226074],
        "state_mean": [-0.0029392109718173742, 0.6544238924980164, 0.15410053730010986, 0.6402314901351929, 0.008108077570796013, 0.6766929626464844, 0.09434457868337631, 0.3859776258468628, -0.050619181245565414, 0.03434790298342705, 0.4750967025756836, 0.017891764640808105, 0.04413251578807831, 0.008904759772121906, 0.02380952425301075, 0.0, 0.0, 0.0, -0.0029707495123147964, 0.6529995203018188, 0.1545131504535675, 0.6445063948631287, 0.008061803877353668, 0.6762475371360779, 0.09425390511751175, 0.3860057294368744, -0.050728388130664825, 0.03427530452609062, 0.4753906726837158, 0.017830422148108482, 0.04414031654596329, 0.008904759772121906, 0.02380952425301075, 0.0, 0.0, 0.0, 0.015675902366638184, 0.7268462181091309, 0.12401046603918076],
        "state_std": [0.10431358963251114, 0.11230525374412537, 0.07981759309768677, 0.26138949394226074, 0.1050795242190361, 0.10192868858575821, 0.0880839079618454, 0.4468156695365906, 0.18397848308086395, 0.14911998808383942, 0.47220379114151, 0.06688503921031952, 0.16022595763206482, 0.03222600743174553, 0.1524554044008255, 0.0, 0.0, 0.0, 0.10383506864309311, 0.11242222785949707, 0.07946359366178513, 0.2625894844532013, 0.10486840456724167, 0.10173390805721283, 0.08810629695653915, 0.4467460811138153, 0.18375301361083984, 0.14870719611644745, 0.4721624553203583, 0.0666341558098793, 0.16022934019565582, 0.03222600743174553, 0.1524554044008255, 0.0, 0.0, 0.0, 0.14182789623737335, 0.12113576382398605, 0.09631838649511337],
    },
    "pusht": {
        "action_dim": 2,
        "proprio_dim": 4,  # with_velocity=True: 2 pos + 2 vel
        "state_dim": 7,
        "action_mean": [-0.008700000122189522, 0.006800000090152025],
        "action_std": [0.20190000534057617, 0.20020000636577606],
        "proprio_mean": [236.61549377441406, 264.5674133300781, -2.9303202629089355, 2.543079137802124],
        "proprio_std": [101.12020111083984, 87.01119995117188, 74.8455581665039, 74.14009094238281],
        "state_mean": [236.61549377441406, 264.5674133300781, 255.13070678710938, 266.3721008300781, 1.958400011062622, -2.9303202629089355, 2.543079137802124],
        "state_std": [101.12020111083984, 87.01119995117188, 52.70539855957031, 57.497100830078125, 1.7555999755859375, 74.8455581665039, 74.14009094238281],
    },
    "pointmaze": {
        "action_dim": 2,
        "proprio_dim": 4,
        "state_dim": 4,
        "action_mean": [7.821338658686727e-05, 0.0006003659800626338],
        "action_std": [0.5769129991531372, 0.577587902545929],
        "proprio_mean": [1.8130563497543335, 1.9377127885818481, -0.0039275349117815495, -0.019777977839112282],
        "proprio_std": [1.0055010318756104, 0.9630089402198792, 1.6802769899368286, 1.9319307804107666],
        "state_mean": [1.8130563497543335, 1.9377127885818481, -0.0039275349117815495, -0.019777977839112282],
        "state_std": [1.0055010318756104, 0.9630089402198792, 1.6802769899368286, 1.9319307804107666],
    },
    "wall": {
        "action_dim": 2,
        "proprio_dim": 2,
        "state_dim": 2,
        "action_mean": [-0.005564616061747074, 0.013197818771004677],
        "action_std": [0.752332866191864, 0.7466897964477539],
        "proprio_mean": [31.89451026916504, 31.81402015686035],
        "proprio_std": [17.663522720336914, 17.27516746520996],
        "state_mean": [31.894489288330078, 31.81402587890625],
        "state_std": [17.663522720336914, 17.27516746520996],
    },
    # Real robot environments (normalize_action=False, so mean=0, std=1)
    "droid": {
        "action_dim": 7,  # 3 pos + 3 euler + 1 gripper
        "proprio_dim": 7,
        "state_dim": 7,
        "action_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "proprio_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "proprio_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "robocasa": {
        "action_dim": 7,  # Same format as DROID
        "proprio_dim": 7,
        "state_dim": 7,
        "action_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "proprio_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "proprio_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
}
# fmt: on


def get_data_dims(env_name: str) -> tuple[int, int]:
    """
    Get the action_dim and proprio_dim for a given environment.

    Args:
        env_name: Environment name (e.g., 'metaworld', 'droid', 'pusht')

    Returns:
        Tuple of (action_dim, proprio_dim)

    Raises:
        KeyError: If the environment is not in the registry
    """
    env_lower = env_name.lower()
    if env_lower not in DATA_STATS:
        raise KeyError(
            f"Unknown environment: {env_name}. "
            f"Available environments: {list(DATA_STATS.keys())}"
        )
    stats = DATA_STATS[env_lower]
    return stats["action_dim"], stats["proprio_dim"]


def get_data_stats(env_name: str) -> dict:
    """
    Get all data statistics for a given environment.

    Args:
        env_name: Environment name (e.g., 'metaworld', 'droid', 'pusht')

    Returns:
        Dictionary with action_dim, proprio_dim, state_dim, and normalization stats

    Raises:
        KeyError: If the environment is not in the registry
    """
    env_lower = env_name.lower()
    if env_lower not in DATA_STATS:
        raise KeyError(
            f"Unknown environment: {env_name}. "
            f"Available environments: {list(DATA_STATS.keys())}"
        )
    return DATA_STATS[env_lower].copy()
