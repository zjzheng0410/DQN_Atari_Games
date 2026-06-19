import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


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


def build_eval_env(make_env, env_id, seed, capture_video, run_name):
    try:
        return make_env(env_id, seed, 0, capture_video, run_name, record_every_episode=True)
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
    capture_video: bool = True
):
    envs = gym.vector.SyncVectorEnv([build_eval_env(make_env, env_id, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = model(torch.as_tensor(obs, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        for episodic_return in get_episode_returns(infos):
            print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
            episodic_returns.append(episodic_return)
        obs = next_obs

    envs.close()
    return episodic_returns


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
