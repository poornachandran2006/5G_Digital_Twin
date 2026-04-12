"""
optimizer/agent.py

PPO agent for 5G network load-balancing using Stable-Baselines3.

Wraps the NetworkOptimizationEnv with Monitor + DummyVecEnv, trains
a PPO MlpPolicy, and provides evaluation utilities.

Run from project root: python scripts/train_rl_agent.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from optimizer.rl_env import NetworkOptimizationEnv

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Stable-Baselines3 PPO agent for 5G network optimisation.

    Wraps :class:`~optimizer.rl_env.NetworkOptimizationEnv` with the
    SB3 training loop, periodic evaluation callbacks, and model
    persistence utilities.
    """

    def __init__(
        self,
        env: NetworkOptimizationEnv,
        device: str = "cpu",
    ) -> None:
        """
        Initialise the PPO agent.

        Wraps *env* with :class:`~stable_baselines3.common.monitor.Monitor`
        and :class:`~stable_baselines3.common.vec_env.DummyVecEnv`, then
        constructs a :class:`~stable_baselines3.PPO` with the hyperparameters
        specified in the Phase 5 spec.

        Args:
            env: The :class:`~optimizer.rl_env.NetworkOptimizationEnv` to train on.
            device: Torch device string (``'cpu'`` or ``'cuda'``).
        """
        Path("logs/ppo_tensorboard").mkdir(parents=True, exist_ok=True)

        self._raw_env = env
        self.device = device

        # Wrap for SB3
        monitored = Monitor(env)
        self.vec_env = DummyVecEnv([lambda: monitored])

        # Only enable TensorBoard logging if the package is available
        try:
            import tensorboard  # noqa: F401
            tb_log: str | None = "logs/ppo_tensorboard/"
        except ImportError:
            tb_log = None
            logger.warning("tensorboard not installed — TB logging disabled")

        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=device,
            tensorboard_log=tb_log,
        )
        logger.info("PPOAgent created: device=%s", device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 50_000,
        save_path: str = "models/ppo_agent",
        eval_freq: int = 5_000,
        extra_callbacks: list[BaseCallback] | None = None,
    ) -> dict:
        """
        Train the PPO agent with periodic evaluation callbacks.

        Args:
            total_timesteps: Total environment steps to train for.
            save_path: Path to save the final model checkpoint.
            eval_freq: Evaluate every this many steps; save best model.

        Returns:
            dict: Training summary with keys ``total_timesteps``,
            ``save_path``, ``eval_freq``.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Build a separate eval env capped at max_episode_steps to keep
        # evaluation fast (EvalCallback runs full episodes; 10800 ticks is slow)
        from gymnasium.wrappers import TimeLimit
        from simulation.engine import NetworkSimulation

        eval_raw = NetworkOptimizationEnv(
            sim_engine=NetworkSimulation(),
            ensemble_model=self._raw_env._ensemble,
            device=self._raw_env._device,
            training_mode=False,
        )
        eval_limited = TimeLimit(eval_raw, max_episode_steps=500)
        eval_vec = DummyVecEnv([lambda: Monitor(eval_limited)])

        eval_callback = EvalCallback(
            eval_env=eval_vec,
            best_model_save_path=str(Path(save_path).parent),
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )

        callbacks: list[BaseCallback] = [eval_callback]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        logger.info("Training PPO for %d timesteps", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )

        self.save(save_path)
        logger.info("Training complete. Model saved to %s", save_path)

        return {
            "total_timesteps": total_timesteps,
            "save_path": save_path,
            "eval_freq": eval_freq,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 5) -> dict:
        """
        Run *n_episodes* full episodes with the trained policy and collect metrics.

        Args:
            n_episodes: Number of evaluation episodes to run.

        Returns:
            dict with keys:
            - ``mean_reward`` — mean episode reward.
            - ``std_reward`` — standard deviation of episode rewards.
            - ``mean_congestion_rate`` — fraction of ticks with any cell > 0.9 load.
            - ``mean_handovers_per_tick`` — handovers averaged per tick.
            - ``episodes_data`` — list of per-episode metric dicts.
        """
        episodes_data: list[dict] = []
        env = self._raw_env

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_ticks = 0
            ep_congested = 0
            ep_handovers = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                ep_reward += float(reward)
                ep_ticks += 1
                loads = info.get("cell_loads", [0.0, 0.0, 0.0])
                if any(l > 0.9 for l in loads):
                    ep_congested += 1
                ep_handovers += info.get("handovers", 0)

            ep_data = {
                "episode": ep + 1,
                "total_reward": ep_reward,
                "ticks": ep_ticks,
                "congestion_rate": ep_congested / max(ep_ticks, 1),
                "handovers_per_tick": ep_handovers / max(ep_ticks, 1),
            }
            episodes_data.append(ep_data)
            logger.info(
                "Episode %d/%d: reward=%.1f congestion=%.2f%% ho/tick=%.2f",
                ep + 1, n_episodes,
                ep_reward,
                ep_data["congestion_rate"] * 100,
                ep_data["handovers_per_tick"],
            )

        rewards = [e["total_reward"] for e in episodes_data]
        cong_rates = [e["congestion_rate"] for e in episodes_data]
        ho_rates = [e["handovers_per_tick"] for e in episodes_data]

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_congestion_rate": float(np.mean(cong_rates)),
            "mean_handovers_per_tick": float(np.mean(ho_rates)),
            "episodes_data": episodes_data,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/ppo_agent") -> None:
        """
        Save the trained PPO model to disk.

        Args:
            path: Destination path (without ``.zip`` extension).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("PPO model saved to %s.zip", path)

    def load(self, path: str = "models/ppo_agent") -> None:
        """
        Load a previously saved PPO model from disk.

        Args:
            path: Path to load from (without ``.zip`` extension).
        """
        self.model = PPO.load(path, env=self.vec_env)
        logger.info("PPO model loaded from %s", path)
