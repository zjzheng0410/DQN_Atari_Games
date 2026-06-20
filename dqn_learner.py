import threading
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from dqn_replay import slice_replay_samples


def maybe_lock(lock):
    return lock if lock is not None else nullcontext()


def maybe_profile(profiler, stage_name):
    return profiler.stage(stage_name) if profiler is not None else nullcontext()


def update_target_network(target_network, q_network, tau):
    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
        target_network_param.data.copy_(
            tau * q_network_param.data + (1.0 - tau) * target_network_param.data
        )


def train_replay_batch(samples, q_network, target_network, optimizer, args, profiler=None):
    with maybe_profile(profiler, "learner.update_total"):
        actions = samples.actions.long()
        with torch.no_grad():
            with maybe_profile(profiler, "learner.forward_target"):
                target_max, _ = target_network(samples.next_observations).max(dim=1)
            td_target = samples.rewards.flatten() + args.gamma * target_max * (1 - samples.dones.flatten())

        with maybe_profile(profiler, "learner.forward_q"):
            old_val = q_network(samples.observations).gather(1, actions).squeeze()
        loss = F.mse_loss(old_val, td_target)

        optimizer.zero_grad(set_to_none=True)
        with maybe_profile(profiler, "learner.backward"):
            loss.backward()
        with maybe_profile(profiler, "optimizer.step"):
            optimizer.step()
        return loss.item()


def run_gradient_updates(
    rb,
    q_network,
    target_network,
    optimizer,
    args,
    num_updates,
    replay_lock=None,
    network_lock=None,
    profiler=None,
):
    last_loss = None
    updates_left = int(num_updates)
    sample_chunk_updates = max(int(args.replay_sample_chunk_updates), 1)

    while updates_left > 0:
        chunk_updates = min(updates_left, sample_chunk_updates)
        if hasattr(rb, "sample_many"):
            with maybe_lock(replay_lock):
                with maybe_profile(profiler, "rb.sample_many"):
                    sample_block = rb.sample_many(args.batch_size, chunk_updates)
            with maybe_lock(network_lock):
                for batch_idx in range(chunk_updates):
                    last_loss = train_replay_batch(
                        slice_replay_samples(sample_block, batch_idx),
                        q_network,
                        target_network,
                        optimizer,
                        args,
                        profiler=profiler,
                    )
        else:
            with maybe_lock(network_lock):
                for _ in range(chunk_updates):
                    with maybe_lock(replay_lock):
                        with maybe_profile(profiler, "rb.sample"):
                            samples = rb.sample(args.batch_size)
                    last_loss = train_replay_batch(
                        samples,
                        q_network,
                        target_network,
                        optimizer,
                        args,
                        profiler=profiler,
                    )
        updates_left -= chunk_updates

    return last_loss


class AsyncLearner:
    def __init__(self, rb, q_network, target_network, optimizer, args, replay_lock, network_lock, profiler=None):
        self.rb = rb
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.args = args
        self.replay_lock = replay_lock
        self.network_lock = network_lock
        self.profiler = profiler
        self.condition = threading.Condition()
        self.pending_updates = 0
        self.requested_updates = 0
        self.completed_updates = 0
        self.target_update_milestones = deque()
        self.last_loss = None
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, name="dqn-async-learner", daemon=True)

    def start(self):
        self.thread.start()

    def add_updates(self, num_updates):
        if num_updates <= 0:
            return
        with self.condition:
            self.pending_updates += int(num_updates)
            self.requested_updates += int(num_updates)
            self.condition.notify_all()

    def request_target_update(self):
        with self.condition:
            self.target_update_milestones.append(self.requested_updates)
            self.condition.notify_all()

    def wait_for_backlog(self, max_backlog):
        if max_backlog <= 0:
            return
        with self.condition:
            while self.pending_updates > max_backlog and not self.stop_requested:
                self.condition.wait(timeout=0.01)

    def get_stats(self):
        with self.condition:
            return {
                "learner_backlog": self.pending_updates,
                "learner_updates": self.completed_updates,
                "loss": self.last_loss,
            }

    def stop_and_join(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        self.thread.join()

    def _pop_due_target_updates(self):
        with self.condition:
            due_count = 0
            while (
                self.target_update_milestones
                and self.completed_updates >= self.target_update_milestones[0]
            ):
                self.target_update_milestones.popleft()
                due_count += 1
            return due_count

    def _apply_due_target_updates(self):
        due_count = self._pop_due_target_updates()
        if due_count <= 0:
            return
        with maybe_lock(self.network_lock):
            for _ in range(due_count):
                update_target_network(self.target_network, self.q_network, self.args.tau)

    def _run(self):
        while True:
            with self.condition:
                while (
                    not self.stop_requested
                    and self.pending_updates <= 0
                    and not self.target_update_milestones
                ):
                    self.condition.wait()
                if self.stop_requested and self.pending_updates <= 0:
                    break
                chunk_updates = min(
                    self.pending_updates,
                    max(int(self.args.replay_sample_chunk_updates), 1),
                )
                self.pending_updates -= chunk_updates

            if chunk_updates > 0:
                last_loss = run_gradient_updates(
                    self.rb,
                    self.q_network,
                    self.target_network,
                    self.optimizer,
                    self.args,
                    chunk_updates,
                    replay_lock=self.replay_lock,
                    network_lock=self.network_lock,
                    profiler=self.profiler,
                )
                with self.condition:
                    self.completed_updates += chunk_updates
                    self.last_loss = last_loss
                    self.condition.notify_all()

            self._apply_due_target_updates()

        self._apply_due_target_updates()
