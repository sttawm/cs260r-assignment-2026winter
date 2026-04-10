"""
Full training script for the multi-agent racing environment.

Trains a PPO agent with:
- Curriculum learning (start with fewer opponents, scale up)
- Periodic evaluation with detailed metrics
- Self-play support (optional)
- TensorBoard logging

Usage:
    python train.py
    python train.py --total-timesteps 2000000 --num-agents 6
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import RacingEnv, make_racing_env
from prepare import MAX_TRAIN_DURATION_SECONDS, evaluate

UID = "mikey"  # Replace with your unique UID for submission


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
NAME = "Test"  # Replace with your agent's name

assert UID != "000000000", "Please update the UID"
if NAME == "Your Agent Name":
    print("Consider updating the agent name from the default placeholder.")


class TimeLimitCallback(BaseCallback):
    """Stops training after MAX_TRAIN_DURATION_SECONDS wall-clock seconds from training start."""

    def __init__(self, max_seconds: float, verbose=0):
        super().__init__(verbose)
        self._max_seconds = max_seconds
        self._train_start: float | None = None

    def _on_training_start(self) -> None:
        self._train_start = time.time()
        print(f"Training started. Time limit: {self._max_seconds / 3600:.1f}h")

    def _on_step(self) -> bool:
        elapsed = time.time() - self._train_start
        if elapsed >= self._max_seconds:
            print(f"\nTime limit reached ({elapsed:.0f}s). Stopping training.")
            return False
        return True


class PeriodicEvalCallback(BaseCallback):
    """
    Every EVAL_INTERVAL_S wall-clock seconds, runs evaluate() on all 7 maps.

    - Saves the model to <save_dir>/best_by_winrate whenever win_rate improves.
    - Stops training early when win_rate has not improved for PATIENCE consecutive
      evaluations (default 3 × 20 min = 60 min without progress).
    - Stores each checkpoint's win_rate in self.checkpoint_win_rates
      keyed by the nominal minute (20, 40, 60, …) for TSV logging.
    """

    EVAL_INTERVAL_S = 20 * 60  # 20 minutes
    PATIENCE = 3

    def __init__(self, save_dir: str, seed: int = 42, verbose: int = 0):
        super().__init__(verbose)
        self._save_dir = save_dir
        self._seed = seed
        self._train_start: float | None = None
        self._next_eval_at: float | None = None
        self._eval_num = 0
        self._best_win_rate = -1.0
        self._no_improve_streak = 0
        self.checkpoint_win_rates: dict[int, float] = {}   # {20: 0.48, 40: 0.52, …}

    def _on_training_start(self) -> None:
        self._train_start = time.time()
        self._next_eval_at = self._train_start + self.EVAL_INTERVAL_S

    def _on_step(self) -> bool:
        if time.time() < self._next_eval_at:
            return True

        self._eval_num += 1
        bucket = self._eval_num * 20   # nominal minute: 20, 40, 60, …
        elapsed_min = (time.time() - self._train_start) / 60
        print(f"\n[Periodic eval #{self._eval_num}]  t={elapsed_min:.0f}min")

        results = evaluate(self.model, seed=self._seed)
        win_rate = results["win_rate"]
        self.checkpoint_win_rates[bucket] = win_rate
        # Machine-readable line for grep
        print(f"[{bucket}min] win_rate: {win_rate:.4f}")

        self.logger.record("eval/win_rate", win_rate)
        self.logger.dump(self.num_timesteps)

        self._next_eval_at = time.time() + self.EVAL_INTERVAL_S

        if win_rate > self._best_win_rate:
            self._best_win_rate = win_rate
            self._no_improve_streak = 0
            save_path = os.path.join(self._save_dir, "best_by_winrate")
            self.model.save(save_path)
            print(f"  New best — saved to {save_path}")
        else:
            self._no_improve_streak += 1
            print(f"  No improvement  ({self._no_improve_streak}/{self.PATIENCE}).  "
                  f"Best so far: {self._best_win_rate:.4f}")
            if self._no_improve_streak >= self.PATIENCE:
                print(f"\nEarly stopping: no improvement for "
                      f"{self.PATIENCE} consecutive evals "
                      f"({self.PATIENCE * 20} min).")
                return False

        return True


class RacingMetricsCallback(BaseCallback):
    """Logs additional racing-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._route_completions = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
            if "route_completion" in info:
                self._route_completions.append(info["route_completion"])

        if len(self._episode_rewards) >= 10:
            self.logger.record("racing/mean_reward", np.mean(self._episode_rewards))
            self.logger.record("racing/mean_length", np.mean(self._episode_lengths))
            if self._route_completions:
                self.logger.record("racing/mean_route_completion", np.mean(self._route_completions))
            self._episode_rewards.clear()
            self._episode_lengths.clear()
            self._route_completions.clear()

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a racing agent (full example)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--num-train-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--opponent-policy", type=str, default="aggressive",
                        choices=["random", "aggressive", "still"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Racing - Full Training Example")
    print("=" * 60)
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Train envs: {args.num_train_envs}")
    print(f"  Agents per race: {args.num_agents}")
    print(f"  Opponent: {args.opponent_policy}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    device = _get_device()
    print(f"  Device: {device}")
    print("=" * 60)

    # Create environments
    train_envs = SubprocVecEnv(
        [make_racing_env(
            rank=i,
            num_agents=args.num_agents,
            opponent_policy=args.opponent_policy,
        ) for i in range(args.num_train_envs)]
    )

    periodic_cb = PeriodicEvalCallback(save_dir=args.save_dir, seed=args.seed)

    # Callbacks
    callbacks = [
        TimeLimitCallback(MAX_TRAIN_DURATION_SECONDS),
        periodic_cb,
        CheckpointCallback(
            save_freq=max(args.save_freq // args.num_train_envs, 1),
            save_path=args.save_dir,
            name_prefix="racing_ppo",
        ),
        RacingMetricsCallback(),
    ]

    # Create RL agent
    model = PPO(
        "MlpPolicy",
        train_envs,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=args.log_dir,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=0.5,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    print(f"\nPolicy architecture: {model.policy}")
    print(f"Observation space: {train_envs.observation_space}")
    print(f"Action space: {train_envs.action_space}")
    print()

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # Save final model
    final_path = os.path.join(args.save_dir, "racing_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final model saved to {final_path}")

    train_envs.close()

    # ── Win-rate progression ─────────────────────────────────────────────
    print("\nWin-rate progression:")
    for bucket in sorted(periodic_cb.checkpoint_win_rates):
        print(f"  {bucket:3d}min: {periodic_cb.checkpoint_win_rates[bucket]:.4f}")

    # ── Model selection ──────────────────────────────────────────────────
    # Use the checkpoint that achieved the highest win_rate during training;
    # fall back to the final model if no periodic eval ran yet.
    best_ckpt_path = os.path.join(args.save_dir, "best_by_winrate.zip")
    if os.path.exists(best_ckpt_path) and periodic_cb._best_win_rate >= 0:
        print(f"\nLoading best checkpoint (win_rate={periodic_cb._best_win_rate:.4f})...")
        best_model = type(model).load(best_ckpt_path, device=device)
    else:
        print("\nNo periodic-eval checkpoint found — using final model.")
        best_model = model

    print("\nFinal evaluation of selected model:")
    evaluate(best_model, seed=args.seed)

    # Convert the winning model to submission format
    print("\nConverting to submission format...")
    convert_to_submission(best_model, os.path.join("agents", f"agent_{UID}"))
    print(f"Done! Agent saved to agents/agent_{UID}/")


def convert_to_submission(model, output_dir):
    """Extract policy from SB3 model and save as standalone agent."""
    os.makedirs(output_dir, exist_ok=True)
    policy = model.policy

    obs_dim = policy.observation_space.shape[0]
    action_dim = policy.action_space.shape[0]

    # Extract MLP extractor layers
    pi_layers = policy.mlp_extractor.policy_net
    hidden_sizes = []
    state_dict = {}

    for i, layer in enumerate(pi_layers):
        if isinstance(layer, torch.nn.Linear):
            hidden_sizes.append(layer.out_features)
            state_dict[f"features.{i}.weight"] = layer.weight.data.clone()
            state_dict[f"features.{i}.bias"] = layer.bias.data.clone()

    state_dict["action_mean.weight"] = policy.action_net.weight.data.clone()
    state_dict["action_mean.bias"] = policy.action_net.bias.data.clone()

    checkpoint = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "state_dict": state_dict,
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pt"))

    agent_code = '''"""Example trained racing agent."""

import os
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        return self.action_mean(x)


class Policy:
    CREATOR_NAME = "__CREATOR_NAME__"
    CREATOR_UID = "__CREATOR_UID__"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.action_dim = checkpoint["action_dim"]
        hidden_sizes = checkpoint["hidden_sizes"]

        self.model = PolicyNetwork(self.obs_dim, self.action_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)
'''
    agent_code = agent_code.replace("__CREATOR_NAME__", NAME).replace("__CREATOR_UID__", UID)
    with open(os.path.join(output_dir, "agent.py"), "w") as f:
        f.write(agent_code)


if __name__ == "__main__":
    main()
