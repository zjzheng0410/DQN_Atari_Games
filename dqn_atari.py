import argparse
import csv
import json
import os
import random
import re
import shlex
import sys
import time
from collections import deque
from datetime import datetime
from distutils.util import strtobool

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

try:
    import ale_py

    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)

    if "ALE/MsPacman-v5" not in gym.envs.registry:
        from ale_py.registration import register_v5_envs

        register_v5_envs()
except ImportError:
    pass


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save the final model into the checkpoint folder")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--async-vector-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="use AsyncVectorEnv to run Atari workers in separate processes")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="the number of gradient updates after each training trigger")
    parser.add_argument("--torch-threads", type=int, default=1,
        help="the number of CPU threads used by PyTorch in the learner process")
    parser.add_argument("--allow-tf32", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="allow TF32 matmul/cuDNN kernels on NVIDIA Ampere+ GPUs")
    parser.add_argument("--checkpoint-frequency", type=int, default=0,
        help="save a checkpoint every N environment steps; 0 disables checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=10,
        help="the number of evaluation episodes after training")
    parser.add_argument("--eval-frequency", type=int, default=5000000,
        help="run a no-video evaluation every N environment steps; 0 disables periodic eval")
    parser.add_argument("--progress-interval", type=int, default=5000,
        help="refresh tqdm postfix every N environment steps")
    args = parser.parse_args()
    # fmt: on

    return args


def make_env(env_id, seed, idx, capture_video, run_name, record_every_episode=False, video_dir=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            video_kwargs = {}
            if record_every_episode:
                video_kwargs["episode_trigger"] = lambda episode_id: True
            env = gym.wrappers.RecordVideo(env, video_dir or f"videos/{run_name}", **video_kwargs)
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        grayscale_wrapper = getattr(gym.wrappers, "GrayScaleObservation", None)
        if grayscale_wrapper is None:
            grayscale_wrapper = gym.wrappers.GrayscaleObservation
        env = grayscale_wrapper(env)
        frame_stack_wrapper = getattr(gym.wrappers, "FrameStack", None)
        if frame_stack_wrapper is None:
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        else:
            env = frame_stack_wrapper(env, 4)
        env.action_space.seed(seed)

        return env
    
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_vector_env(args, run_name, video_dir=None):
    env_fns = [
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, video_dir=video_dir)
        for i in range(args.num_envs)
    ]
    if args.async_vector_env:
        return gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    return gym.vector.SyncVectorEnv(env_fns)


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def get_episode_returns(infos):
    returns = []

    if "final_info" in infos:
        final_infos = infos["final_info"]
        final_info_mask = infos.get("_final_info", np.ones(len(final_infos), dtype=bool))
        for present, info in zip(final_info_mask, final_infos):
            if not present or info is None or "episode" not in info:
                continue
            returns.append(scalar(info["episode"]["r"]))

    if "episode" in infos:
        episode_info = infos["episode"]
        if isinstance(episode_info, dict) and "r" in episode_info:
            episode_mask = infos.get("_episode", np.ones_like(episode_info["r"], dtype=bool))
            for present, episode_return in zip(episode_mask, episode_info["r"]):
                if present:
                    returns.append(scalar(episode_return))

    return returns


def get_real_next_obs(next_obs, truncated, infos):
    real_next_obs = next_obs.copy()
    final_observations = infos.get("final_observation")
    if final_observations is None:
        return real_next_obs

    final_observation_mask = infos.get("_final_observation", np.ones(len(final_observations), dtype=bool))
    for idx, is_truncated in enumerate(truncated):
        if is_truncated and final_observation_mask[idx] and final_observations[idx] is not None:
            real_next_obs[idx] = final_observations[idx]
    return real_next_obs


def update_target_network(target_network, q_network, tau):
    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
        target_network_param.data.copy_(
            tau * q_network_param.data + (1.0 - tau) * target_network_param.data
        )


def safe_path_part(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned or "run"


def make_run_name(exp_name, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_path_part(exp_name)}__{timestamp}__seed{seed}"


def make_artifact_dirs(run_name):
    checkpoint_dir = os.path.join("log_checkpoint", "checkpoint", run_name)
    log_dir = os.path.join("log_checkpoint", "log", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return checkpoint_dir, log_dir


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_scores(scores):
    if not scores:
        return {
            "episodes": 0,
            "mean_score": None,
            "best_score": None,
            "highest_score": None,
            "worst_score": None,
            "std_score": None,
            "median_score": None,
        }

    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "episodes": int(len(scores)),
        "mean_score": float(np.mean(score_array)),
        "best_score": float(np.max(score_array)),
        "highest_score": float(np.max(score_array)),
        "worst_score": float(np.min(score_array)),
        "std_score": float(np.std(score_array)),
        "median_score": float(np.median(score_array)),
    }


def save_checkpoint(q_network, checkpoint_dir, exp_name, global_step, label=None):
    safe_exp_name = safe_path_part(exp_name)
    checkpoint_label = label or f"step_{global_step}"
    model_path = os.path.join(checkpoint_dir, f"{safe_exp_name}.{checkpoint_label}.pth")
    torch.save(q_network.state_dict(), model_path)
    return model_path


def evaluate_current_model(q_network, env_id, eval_episode, run_name, device, epsilon=0.05, seed=0):
    if eval_episode <= 0:
        return {"returns": [], "episodes": [], "video_dir": None}

    from dqn_eval import build_eval_env, get_episode_stats

    envs = gym.vector.SyncVectorEnv(
        [build_eval_env(make_env, env_id, seed, False, run_name, video_dir=None)]
    )
    was_training = q_network.training
    q_network.eval()
    eval_random = random.Random(seed)
    episode_records = []

    try:
        obs, _ = envs.reset()
        while len(episode_records) < eval_episode:
            if eval_random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    q_values = q_network(torch.as_tensor(obs, device=device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, _, _, _, infos = envs.step(actions)
            for episode_stat in get_episode_stats(infos):
                episode_records.append(
                    {
                        "episode": len(episode_records) + 1,
                        "score": episode_stat["score"],
                        "length": episode_stat.get("length"),
                        "time": episode_stat.get("time"),
                    }
                )
                if len(episode_records) >= eval_episode:
                    break
            obs = next_obs
    finally:
        envs.close()
        if was_training:
            q_network.train()

    return {
        "returns": [episode["score"] for episode in episode_records],
        "episodes": episode_records,
        "video_dir": None,
    }


def make_eval_record(phase, global_step, eval_result):
    eval_returns = [scalar(episode_return) for episode_return in eval_result["returns"]]
    eval_episode_lengths = [
        episode["length"]
        for episode in eval_result["episodes"]
        if episode.get("length") is not None
    ]
    eval_score_summary = summarize_scores(eval_returns)
    return {
        "phase": phase,
        "global_step": int(global_step),
        "episodes": eval_score_summary["episodes"],
        "eval_mean_score": eval_score_summary["mean_score"],
        "eval_best_score": eval_score_summary["best_score"],
        "eval_highest_score": eval_score_summary["highest_score"],
        "eval_worst_score": eval_score_summary["worst_score"],
        "eval_std_score": eval_score_summary["std_score"],
        "eval_median_score": eval_score_summary["median_score"],
        "scores": eval_returns,
        "episode_lengths": eval_episode_lengths,
    }


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """On going migration: run the following command to install new dependencies
        pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1"
        """
        )
    
    args = parse_args()
    run_name = make_run_name(args.exp_name, args.seed)
    checkpoint_dir, log_dir = make_artifact_dirs(run_name)
    train_video_dir = os.path.join(log_dir, "videos", "train")
    eval_video_dir = os.path.join(log_dir, "videos", "eval")
    os.makedirs(train_video_dir, exist_ok=True)
    os.makedirs(eval_video_dir, exist_ok=True)
    run_config_path = os.path.join(log_dir, "run_config.json")
    save_json(
        run_config_path,
        {
            "run_name": run_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "env_id": args.env_id,
            "args": vars(args),
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "train_video_dir": train_video_dir if args.capture_video else None,
            "eval_video_dir": eval_video_dir,
        },
    )

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.torch_deterministic:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda" and args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"run_name={run_name}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"log_dir={log_dir}")
    print(f"device={device}, num_envs={args.num_envs}, async_vector_env={args.async_vector_env}")

    envs = make_vector_env(args, run_name, video_dir=train_video_dir)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )
    start_time = time.time()
    recent_returns = deque(maxlen=20)
    train_scores = []
    train_best_score = None
    train_episode_count = 0
    last_loss = None
    last_checkpoint_step = 0
    train_episode_log_path = os.path.join(log_dir, "train_episodes.jsonl")
    train_progress_log_path = os.path.join(log_dir, "train_progress.csv")
    periodic_eval_log_path = os.path.join(log_dir, "periodic_eval.jsonl")
    train_episode_log = open(train_episode_log_path, "w", encoding="utf-8")
    train_progress_log = open(train_progress_log_path, "w", encoding="utf-8", newline="")
    periodic_eval_log = open(periodic_eval_log_path, "w", encoding="utf-8")
    best_eval_mean_score = None
    best_eval_step = None
    best_model_path = None
    best_eval_record = None
    progress_writer = csv.DictWriter(
        train_progress_log,
        fieldnames=[
            "global_step",
            "sps",
            "epsilon",
            "loss",
            "recent_return20_mean",
            "recent_return20_best",
            "train_best_score",
            "episodes",
        ],
    )
    progress_writer.writeheader()

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    progress = tqdm(
        total=args.total_timesteps,
        dynamic_ncols=True,
        desc="training",
        unit="step",
        mininterval=5,
        maxinterval=30,
    )

    while global_step < args.total_timesteps:
        previous_step = global_step
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(obs, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        global_step += args.num_envs

        for episodic_return in get_episode_returns(infos):
            score = scalar(episodic_return)
            recent_returns.append(score)
            train_scores.append(score)
            train_best_score = score if train_best_score is None else max(train_best_score, score)
            train_episode_count += 1
            train_episode_log.write(
                json.dumps(
                    {
                        "episode": train_episode_count,
                        "global_step": global_step,
                        "score": score,
                        "best_score_so_far": train_best_score,
                    }
                )
                + "\n"
            )
            train_episode_log.flush()

        real_next_obs = get_real_next_obs(next_obs, truncated, infos)
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            updates_due = max(
                0,
                global_step // args.train_frequency
                - max(previous_step, args.learning_starts) // args.train_frequency,
            )
            for _ in range(updates_due * args.gradient_steps):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(old_val, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            if global_step // args.target_network_frequency > previous_step // args.target_network_frequency:
                update_target_network(target_network, q_network, args.tau)

        if (
            args.eval_frequency > 0
            and global_step < args.total_timesteps
            and global_step // args.eval_frequency > previous_step // args.eval_frequency
        ):
            periodic_eval_result = evaluate_current_model(
                q_network,
                args.env_id,
                args.eval_episodes,
                run_name=f"{run_name}-periodic-eval-step-{global_step}",
                device=device,
                epsilon=0.05,
            )
            eval_record = make_eval_record("periodic", global_step, periodic_eval_result)
            periodic_eval_log.write(json.dumps(eval_record) + "\n")
            periodic_eval_log.flush()
            eval_mean_score = eval_record["eval_mean_score"]
            if (
                eval_mean_score is not None
                and (best_eval_mean_score is None or eval_mean_score > best_eval_mean_score)
            ):
                best_eval_mean_score = eval_mean_score
                best_eval_step = global_step
                best_eval_record = eval_record
                if args.save_model:
                    best_model_path = save_checkpoint(
                        q_network,
                        checkpoint_dir,
                        args.exp_name,
                        global_step,
                        label="best_eval_mean",
                    )
                    progress.write(f"best eval model saved to {best_model_path}")
            progress.write(
                f"periodic_eval step={global_step} "
                f"mean={eval_record['eval_mean_score']} best={eval_record['eval_best_score']}"
            )

        if (
            args.checkpoint_frequency > 0
            and global_step - last_checkpoint_step >= args.checkpoint_frequency
            and global_step > 0
        ):
            checkpoint_path = save_checkpoint(q_network, checkpoint_dir, args.exp_name, global_step)
            last_checkpoint_step = global_step
            print(f"checkpoint saved to {checkpoint_path}")

        progress.update(global_step - previous_step)
        if (
            args.progress_interval <= 0
            or global_step // args.progress_interval > previous_step // args.progress_interval
            or global_step >= args.total_timesteps
        ):
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            postfix = {"sps": sps, "eps": f"{epsilon:.3f}"}
            if last_loss is not None:
                postfix["loss"] = f"{last_loss:.4f}"
            if recent_returns:
                postfix["return20"] = f"{np.mean(recent_returns):.1f}"
            progress.set_postfix(postfix, refresh=False)
            progress_writer.writerow(
                {
                    "global_step": global_step,
                    "sps": sps,
                    "epsilon": epsilon,
                    "loss": last_loss,
                    "recent_return20_mean": float(np.mean(recent_returns)) if recent_returns else None,
                    "recent_return20_best": float(np.max(recent_returns)) if recent_returns else None,
                    "train_best_score": train_best_score,
                    "episodes": train_episode_count,
                }
            )
            train_progress_log.flush()

    progress.close()
    train_episode_log.close()
    train_progress_log.close()
    envs.close()

    if args.save_model:
        model_path = save_checkpoint(q_network, checkpoint_dir, args.exp_name, global_step, label=f"final_step_{global_step}")
        print(f"model saved to {model_path}")

        from dqn_eval import evaluate

        eval_log_path = os.path.join(log_dir, "eval_episodes.jsonl")
        eval_result = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episode=args.eval_episodes,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=True,
            video_dir=eval_video_dir,
            log_path=eval_log_path,
            return_details=True,
        )

        eval_record = make_eval_record("final", global_step, eval_result)
        periodic_eval_log.write(json.dumps(eval_record) + "\n")
        periodic_eval_log.flush()
        eval_mean_score = eval_record["eval_mean_score"]
        if (
            eval_mean_score is not None
            and (best_eval_mean_score is None or eval_mean_score > best_eval_mean_score)
        ):
            best_eval_mean_score = eval_mean_score
            best_eval_step = global_step
            best_eval_record = eval_record
            best_model_path = save_checkpoint(
                q_network,
                checkpoint_dir,
                args.exp_name,
                global_step,
                label="best_eval_mean",
            )
            print(f"best eval model saved to {best_model_path}")

        eval_returns = eval_record["scores"]
        eval_episode_lengths = eval_record["episode_lengths"]
        eval_score_summary = summarize_scores(eval_returns)
        train_score_summary = summarize_scores(train_scores)
        best_eval_episode = None
        if eval_result["episodes"]:
            best_eval_episode = max(eval_result["episodes"], key=lambda episode: episode["score"])
        quality = {
            "run_name": run_name,
            "env_id": args.env_id,
            "final_model_path": model_path,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "eval_video_dir": eval_video_dir,
            "periodic_eval_log": periodic_eval_log_path,
            "best_model_path": best_model_path,
            "best_eval_mean_score": best_eval_mean_score,
            "best_eval_step": best_eval_step,
            "score_source": "raw episode scores from Gymnasium RecordEpisodeStatistics",
            "survival_signal": "Use eval episode lengths and videos to judge whether MsPacman keeps eating without dying.",
            "eval": {
                **eval_score_summary,
                "scores": eval_returns,
                "episode_lengths": eval_episode_lengths,
                "longest_episode_length": int(max(eval_episode_lengths)) if eval_episode_lengths else None,
                "best_episode": best_eval_episode,
            },
            "training": {
                **train_score_summary,
                "recent20_mean_score": float(np.mean(recent_returns)) if recent_returns else None,
                "recent20_best_score": float(np.max(recent_returns)) if recent_returns else None,
            },
            "best_eval": best_eval_record,
        }
        quality_path = os.path.join(log_dir, "quality.json")
        save_json(quality_path, quality)
        summary = {
            "run_name": run_name,
            "env_id": args.env_id,
            "model_path": model_path,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "train_episode_log": train_episode_log_path,
            "train_progress_log": train_progress_log_path,
            "periodic_eval_log": periodic_eval_log_path,
            "eval_episode_log": eval_log_path,
            "train_video_dir": train_video_dir if args.capture_video else None,
            "eval_video_dir": eval_video_dir,
            "total_timesteps": args.total_timesteps,
            "num_envs": args.num_envs,
            "async_vector_env": args.async_vector_env,
            "batch_size": args.batch_size,
            "gradient_steps": args.gradient_steps,
            "buffer_size": args.buffer_size,
            "learning_rate": args.learning_rate,
            "eval_returns": eval_returns,
            "eval_mean_score": eval_score_summary["mean_score"],
            "eval_best_score": eval_score_summary["best_score"],
            "eval_highest_score": eval_score_summary["highest_score"],
            "best_eval_mean_score": best_eval_mean_score,
            "best_eval_step": best_eval_step,
            "best_model_path": best_model_path,
            "train_best_score": train_score_summary["best_score"],
            "train_highest_score": train_score_summary["highest_score"],
            "quality_path": quality_path,
            "wall_time_seconds": time.time() - start_time,
        }
        summary_path = os.path.join(log_dir, "summary.json")
        save_json(summary_path, summary)
        print(f"quality saved to {quality_path}")
        print(f"summary saved to {summary_path}")

    periodic_eval_log.close()
    
