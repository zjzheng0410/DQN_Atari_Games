# DQN Atari MsPacman 实验报告

> **郑智君 23307130062** 

## 实验目的

本实验根据 `RL实践 .pdf` 的要求，阅读并理解 Gymnasium 强化学习训练流程，在 Atari MsPacman 环境中实现 DQN 的卷积 Q 网络，训练智能体并提交最终代码、实践报告和吃豆人游戏视频。最新一次实验 `MsPacman-v5-640envs__20260620_165534__seed1` 已完成 50,000,000 environment steps，最终评估结果明显优于原始短训练配置。

## 实验方法

环境使用 `ALE/MsPacman-v5`。观测先经过 Atari 常用预处理：随机 no-op reset、frame skip、episodic life、reward clipping、84x84 resize、灰度化和 4 帧堆叠。策略网络输入为 `(4, 84, 84)` 的图像帧，输出每个动作的 Q 值；训练采用 epsilon-greedy 探索、experience replay、target network 和 DQN 的 TD target 更新。

本次最终训练使用的主要参数为：`--total-timesteps 50000000`、`--num-envs 640`、`--async-vector-env`、`--replay-buffer-device cuda`、`--async-learner`、`--batch-size 512`、`--buffer-size 400000`、`--target-network-frequency 8000`。相比原始 5,000,000 steps 的配置，本次训练时间更长，并通过并行环境和异步学习器提升采样与更新吞吐。

## 遇到的问题与代码修改

首先，课程代码中需要补全 `QNetwork`。我按照 DQN 论文和 PPT 中的 Atari CNN 结构实现了三层卷积网络，并在前向传播中将像素归一化到 `[0, 1]`：

```python
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
```

第二个问题是 MsPacman 的学习信号比较稀疏，原始 5M steps 训练后策略不稳定；同时多环境训练下，CPU replay buffer 和同步 learner 容易成为瓶颈。为此我增加了 CUDA replay buffer，将经验直接存储在 GPU 上，并支持一次采样多个 batch，减少 CPU/GPU 数据搬运：

```python
class GpuReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs):
        self.observations = torch.empty(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.empty((self.buffer_size, self.n_envs, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)

    def sample_many(self, batch_size, num_batches):
        total_samples = int(batch_size) * int(num_batches)
        batch_inds = torch.randint(0, self.pos, (total_samples,), device=self.device)
        env_indices = torch.randint(0, self.n_envs, (total_samples,), device=self.device)
        next_batch_inds = (batch_inds + 1) % self.buffer_size
        batch_shape = (int(num_batches), int(batch_size))
        return ReplayBufferSamples(
            self.observations[batch_inds, env_indices].reshape(*batch_shape, *self.obs_shape),
            self.actions[batch_inds, env_indices].reshape(*batch_shape, 1),
            self.observations[next_batch_inds, env_indices].reshape(*batch_shape, *self.obs_shape),
            self.dones[batch_inds, env_indices].reshape(*batch_shape, 1),
            self.rewards[batch_inds, env_indices].reshape(*batch_shape, 1),
        )
```

第三个问题是需要在长训练中持续确认模型质量，并保存可提交证据。因此我增加了异步 learner、周期评估、best checkpoint、最终评估视频和 `quality.json` 摘要。主训练循环中 actor 负责环境交互，learner 后台执行梯度更新，并通过 backlog 控制避免策略权重滞后过多：

```python
if total_updates_due > 0:
    if learner is not None:
        learner.add_updates(total_updates_due)
    else:
        last_loss = run_gradient_updates(...)

if global_step // args.target_network_frequency > previous_step // args.target_network_frequency:
    if learner is not None:
        learner.request_target_update()
    else:
        update_target_network(target_network, q_network, args.tau)

if learner is not None:
    learner.wait_for_backlog(args.max_learner_lag_updates)
```

## 最终实验结果
最终 10 局评估分数为：`7090, 5350, 1960, 1960, 4980, 4350, 3150, 4000, 3610, 4360`。平均分 `4081.0`，最高分 `7090.0`，最低分 `1960.0`，中位数 `4175.0`。训练过程中最高 episode score 为 `10661.0`，最后 20 个训练 episode 的平均分为 `4636.5`。周期评估显示最终 50M steps 的均分 `4081.0` 是本次训练的 best eval mean。

