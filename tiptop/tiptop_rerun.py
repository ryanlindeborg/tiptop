"""Re-run TiPToP perception + planning from a saved run directory."""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tyro
from omegaconf import OmegaConf

from tiptop.perception.cameras import Frame
from tiptop.tiptop_h5 import run_tiptop
from tiptop.tiptop_run import Observation
from tiptop.utils import print_tiptop_banner, setup_logging

_log = logging.getLogger(__name__)


def load_observation_from_run(run_dir: Path) -> tuple[Observation, np.ndarray | None, dict]:
    """Load an Observation and optional gripper mask from a saved TiPToP run directory.

    Returns the observation, gripper mask (or None), and the parsed metadata dict.
    """
    run_dir = Path(run_dir)

    # Load and validate metadata first — file layout may differ across versions
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata.json in {run_dir}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata["version"] != "1.0.0":
        raise ValueError(
            f"Unsupported metadata version '{metadata['version']}' (expected '1.0.0'). "
            f"The run at {run_dir} may have been saved with an incompatible TiPToP version."
        )

    perception_dir = run_dir / "perception"

    # RGB
    rgb_path = run_dir / "rgb.png"
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Depth: saved as uint16 in millimeters, convert back to float32 meters
    depth_path = perception_dir / "depth.png"
    depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_uint16 is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    depth = depth_uint16.astype(np.float32) / 1000.0

    # Intrinsics
    intrinsics_path = perception_dir / "intrinsics.json"
    with open(intrinsics_path) as f:
        intrinsics = np.array(json.load(f)["intrinsics"], dtype=np.float32)

    # Gripper mask (optional — only present for runs with a hand camera)
    gripper_mask = None
    gripper_mask_path = perception_dir / "gripper_mask.png"
    if gripper_mask_path.exists():
        gripper_mask_img = cv2.imread(str(gripper_mask_path), cv2.IMREAD_GRAYSCALE)
        if gripper_mask_img is not None:
            gripper_mask = gripper_mask_img > 0

    # Observation fields from metadata
    obs_meta = metadata["observation"]
    q_init = np.array(obs_meta["q_at_capture"], dtype=np.float32)
    world_from_cam = np.array(obs_meta["world_from_cam"], dtype=np.float32)

    frame = Frame(serial="rerun", timestamp=0.0, rgb=rgb, intrinsics=intrinsics, depth=depth)
    observation = Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_init)
    return observation, gripper_mask, metadata


def _load_cutamp_config(run_dir: Path) -> dict:
    """Load the cuTAMP config from a saved run directory."""
    config_path = run_dir / "cutamp" / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find cuTAMP config at {config_path}")
    return OmegaConf.to_container(OmegaConf.load(config_path))


def rerun_tiptop(
    run_dir: str,
    task_instruction: str | None = None,
    output_dir: str = "tiptop_rerun_outputs",
    max_planning_time: float | None = None,
    opt_steps_per_skeleton: int | None = None,
    num_particles: int | None = None,
    cutamp_visualize: bool = False,
    rr_spawn: bool = True,
):
    """Re-run TiPToP from a saved run directory.

    Loads the observation from a previous run and runs perception + planning.
    The task instruction and planning parameters default to those from the original run
    but can be overridden.

    Args:
        run_dir: Path to a saved TiPToP run directory (contains metadata.json, rgb.png, etc.).
        task_instruction: Task instruction override. If not provided, uses the original instruction.
        output_dir: Top-level directory to save outputs; a timestamped subdirectory is created per run.
        max_planning_time: Override max planning time. Defaults to the original run's value.
        opt_steps_per_skeleton: Override optimization steps per skeleton. Defaults to the original run's value.
        num_particles: Override number of particles. Defaults to the original run's value.
        cutamp_visualize: Whether to visualize cuTAMP optimization.
        rr_spawn: Whether to spawn a Rerun viewer.
    """
    setup_logging(level=logging.INFO)
    print_tiptop_banner()
    run_dir_path = Path(run_dir)
    observation, gripper_mask, metadata = load_observation_from_run(run_dir_path)
    cutamp_config = _load_cutamp_config(run_dir_path)

    if task_instruction is None:
        task_instruction = metadata["task_instruction"]
        _log.info(f"Using task instruction from original run: '{task_instruction}'")
    if max_planning_time is None:
        max_planning_time = cutamp_config["max_loop_dur"]
        _log.info(f"Using max_planning_time from original run: {max_planning_time}")
    if opt_steps_per_skeleton is None:
        opt_steps_per_skeleton = cutamp_config["num_opt_steps"]
        _log.info(f"Using opt_steps_per_skeleton from original run: {opt_steps_per_skeleton}")
    if num_particles is None:
        num_particles = cutamp_config["num_particles"]
        _log.info(f"Using num_particles from original run: {num_particles}")

    run_tiptop(
        observation=observation,
        task_instruction=task_instruction,
        output_dir=output_dir,
        max_planning_time=max_planning_time,
        opt_steps_per_skeleton=opt_steps_per_skeleton,
        num_particles=num_particles,
        gripper_mask=gripper_mask,
        cutamp_visualize=cutamp_visualize,
        rr_spawn=rr_spawn,
    )


def entrypoint():
    """CLI entrypoint wrapper."""
    try:
        tyro.cli(rerun_tiptop)
    except Exception:
        _log.exception("TiPToP rerun failed")
        os._exit(1)
    else:
        os._exit(0)


if __name__ == "__main__":
    entrypoint()
