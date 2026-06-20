import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class GpuReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs):
        if device.type != "cuda":
            raise ValueError("GpuReplayBuffer requires a CUDA device")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("GpuReplayBuffer only supports discrete action spaces")

        self.device = device
        self.n_envs = int(n_envs)
        self.action_dim = 1
        self.obs_shape = tuple(observation_space.shape)
        self.transition_capacity = int(buffer_size)
        self.buffer_size = max(self.transition_capacity // self.n_envs, 2)
        self.full = False
        self.pos = 0

        self.observations = torch.empty(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.empty((self.buffer_size, self.n_envs, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)

    @property
    def capacity(self):
        return self.buffer_size * self.n_envs

    @property
    def transition_count(self):
        rows = self.buffer_size if self.full else self.pos
        return rows * self.n_envs

    def add(self, obs, next_obs, action, reward, done, infos=None):
        obs_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.uint8, device=self.device)
        next_obs_tensor = torch.as_tensor(np.asarray(next_obs), dtype=torch.uint8, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(action).reshape(self.n_envs, 1), dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(np.asarray(done), dtype=torch.float32, device=self.device)

        self.observations[self.pos].copy_(obs_tensor, non_blocking=True)
        self.observations[(self.pos + 1) % self.buffer_size].copy_(next_obs_tensor, non_blocking=True)
        self.actions[self.pos].copy_(action_tensor, non_blocking=True)
        self.rewards[self.pos].copy_(reward_tensor, non_blocking=True)
        self.dones[self.pos].copy_(done_tensor, non_blocking=True)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        samples = self.sample_many(batch_size, 1)
        return slice_replay_samples(samples, 0)

    def sample_many(self, batch_size, num_batches):
        if self.transition_count <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        total_samples = int(batch_size) * int(num_batches)
        if self.full:
            batch_inds = (
                torch.randint(1, self.buffer_size, (total_samples,), device=self.device) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = torch.randint(0, self.pos, (total_samples,), device=self.device)
        env_indices = torch.randint(0, self.n_envs, (total_samples,), device=self.device)
        next_batch_inds = (batch_inds + 1) % self.buffer_size

        batch_shape = (int(num_batches), int(batch_size))
        observations = self.observations[batch_inds, env_indices].reshape(*batch_shape, *self.obs_shape)
        next_observations = self.observations[next_batch_inds, env_indices].reshape(*batch_shape, *self.obs_shape)
        actions = self.actions[batch_inds, env_indices].reshape(*batch_shape, 1)
        dones = self.dones[batch_inds, env_indices].reshape(*batch_shape, 1)
        rewards = self.rewards[batch_inds, env_indices].reshape(*batch_shape, 1)
        return ReplayBufferSamples(observations, actions, next_observations, dones, rewards)

    def stats(self):
        tensors = (self.observations, self.actions, self.rewards, self.dones)
        bytes_used = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return {
            "replay_device": str(self.device),
            "replay_rows": self.buffer_size,
            "replay_capacity": self.capacity,
            "replay_transitions": self.transition_count,
            "replay_memory_gib": bytes_used / (1024 ** 3),
        }


def slice_replay_samples(samples, index):
    return ReplayBufferSamples(
        samples.observations[index],
        samples.actions[index],
        samples.next_observations[index],
        samples.dones[index],
        samples.rewards[index],
    )


def make_replay_buffer(args, envs, device):
    requested_device = args.replay_buffer_device
    use_gpu_replay = requested_device == "cuda" or (requested_device == "auto" and device.type == "cuda")
    if use_gpu_replay:
        if device.type != "cuda":
            if requested_device == "cuda":
                raise ValueError("--replay-buffer-device cuda requires --cuda and a visible CUDA device")
        else:
            try:
                rb = GpuReplayBuffer(
                    args.buffer_size,
                    envs.single_observation_space,
                    envs.single_action_space,
                    device,
                    args.num_envs,
                )
                return rb, "cuda", None
            except RuntimeError as exc:
                if requested_device == "cuda":
                    raise
                print(f"GPU replay allocation failed, falling back to CPU replay: {exc}")

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    return rb, "cpu", "cuda unavailable or not requested"


def get_replay_stats(rb, replay_device, args):
    if hasattr(rb, "stats"):
        return rb.stats()
    rows = getattr(rb, "buffer_size", None)
    pos = getattr(rb, "pos", 0)
    full = getattr(rb, "full", False)
    row_count = rows if full else pos
    return {
        "replay_device": replay_device,
        "replay_rows": rows,
        "replay_capacity": rows * args.num_envs if rows is not None else args.buffer_size,
        "replay_transitions": row_count * args.num_envs if rows is not None else None,
        "replay_memory_gib": None,
    }


def resolve_async_learner(args, rb, device):
    if args.async_learner == "auto":
        return device.type == "cuda" and isinstance(rb, GpuReplayBuffer) and args.num_envs > 1
    return bool(args.async_learner)
