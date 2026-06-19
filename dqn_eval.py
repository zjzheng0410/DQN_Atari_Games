import json
import os
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def make_episode_stat(info):
    episode = info["episode"]
    stat = {"score": scalar(episode["r"])}
    if "l" in episode:
        stat["length"] = int(scalar(episode["l"]))
    if "t" in episode:
        stat["time"] = scalar(episode["t"])
    return stat


def get_episode_stats(infos):
    episode_stats = []

    if "final_info" in infos:
        final_infos = infos["final_info"]
        final_info_mask = infos.get("_final_info", np.ones(len(final_infos), dtype=bool))
        for present, info in zip(final_info_mask, final_infos):
            if not present or info is None or "episode" not in info:
                continue
            episode_stats.append(make_episode_stat(info))

    if "episode" in infos:
        episode_info = infos["episode"]
        if isinstance(episode_info, dict) and "r" in episode_info:
            episode_mask = infos.get("_episode", np.ones_like(episode_info["r"], dtype=bool))
            lengths = episode_info.get("l", [None] * len(episode_info["r"]))
            times = episode_info.get("t", [None] * len(episode_info["r"]))
            for present, episode_return, episode_length, episode_time in zip(
                episode_mask,
                episode_info["r"],
                lengths,
                times,
            ):
                if present:
                    stat = {"score": scalar(episode_return)}
                    if episode_length is not None:
                        stat["length"] = int(scalar(episode_length))
                    if episode_time is not None:
                        stat["time"] = scalar(episode_time)
                    episode_stats.append(stat)

    return episode_stats


def get_episode_returns(infos):
    return [episode["score"] for episode in get_episode_stats(infos)]


def build_eval_env(make_env, env_id, seed, capture_video, run_name, video_dir=None):
    try:
        return make_env(
            env_id,
            seed,
            0,
            capture_video,
            run_name,
            record_every_episode=True,
            video_dir=video_dir,
        )
    except TypeError:
        return make_env(env_id, seed, 0, capture_video, run_name)


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    video_dir: str = None,
    log_path: str = None,
    return_details: bool = False,
):
    if log_path is not None:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    envs = gym.vector.SyncVectorEnv([build_eval_env(make_env, env_id, 0, capture_video, run_name, video_dir)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episode_records = []
    log_file = open(log_path, "w", encoding="utf-8") if log_path is not None else None
    while len(episode_records) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = model(torch.as_tensor(obs, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        for episode_stat in get_episode_stats(infos):
            episode_record = {
                "episode": len(episode_records) + 1,
                "score": episode_stat["score"],
                "length": episode_stat.get("length"),
                "time": episode_stat.get("time"),
            }
            print(
                f"eval_episode={len(episode_records)}, "
                f"score={episode_record['score']}, "
                f"length={episode_record['length']}"
            )
            episode_records.append(episode_record)
            if log_file is not None:
                log_file.write(json.dumps(episode_record) + "\n")
                log_file.flush()
            if len(episode_records) >= eval_episode:
                break
        obs = next_obs

    if log_file is not None:
        log_file.close()
    envs.close()
    if return_details:
        return {
            "returns": [episode["score"] for episode in episode_records],
            "episodes": episode_records,
            "video_dir": video_dir,
        }
    return [episode["score"] for episode in episode_records]


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from dqn_atari import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="dqn.cleanrl_model")
    # model_path = ".pth"
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episode=0,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )           
