"""
Racing environment wrapper for single-agent training.

Wraps MetaDrive's MultiAgentRacingEnv into a standard gymnasium.Env interface.
Supports training with configurable opponent policies including self-play.
"""

import gymnasium
import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv


def random_opponent(obs, agent_id):
    return np.random.uniform(-1, 1, size=(2,)).astype(np.float32)


def aggressive_opponent(obs, agent_id):
    return np.array([0.0, 1.0], dtype=np.float32)


def still_opponent(obs, agent_id):
    return np.array([0.0, 0.0], dtype=np.float32)


OPPONENT_POLICIES = {
    "random": random_opponent,
    "aggressive": aggressive_opponent,
    "still": still_opponent,
}


class SelfPlayOpponent:
    """Opponent that uses a copy of the training policy (loaded from checkpoint)."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        import torch
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def __call__(self, obs, agent_id):
        if self.model is None:
            return aggressive_opponent(obs, agent_id)
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = self.model(obs_t).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)


class RacingEnv(gymnasium.Env):
    """
    Single-agent wrapper around MultiAgentRacingEnv.

    Supports curriculum learning by adjusting opponent count and policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_agents=2,
        opponent_policy="aggressive",
        extra_config=None,
        render_mode=None,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.ego_id = "agent0"

        if isinstance(opponent_policy, str):
            self._opponent_fn = OPPONENT_POLICIES[opponent_policy]
        else:
            self._opponent_fn = opponent_policy

        base_config = {
            "num_agents": num_agents,
            "use_render": False,
            "crash_done": False,
            "crash_vehicle_done": False,
            "out_of_road_done": True,
            "allow_respawn": False,
            "horizon": 3000,
            "map_config": {
                "lane_num": max(2, num_agents),
                "exit_length": 20,
                "bottle_lane_num": max(4, num_agents),
                "neck_lane_num": 1,
                "neck_length": 20,
            },
        }
        # Probe observation/action spaces with a headless env.
        # Do NOT merge extra_config here — unknown keys (e.g. render options
        # not in MetaDrive's default config) would raise an error.  The spaces
        # only depend on num_agents and sensor config, which are in base_config.
        headless_config = dict(base_config)
        headless_config["use_render"] = False

        temp_env = MultiAgentRacingEnv(headless_config)
        temp_obs, _ = temp_env.reset()
        sample_id = list(temp_obs.keys())[0]
        self.observation_space = temp_env.observation_space[sample_id]
        self.action_space = temp_env.action_space[sample_id]
        self._last_ego_obs = temp_obs.get(self.ego_id, np.zeros(self.observation_space.shape))
        self._opponent_obs = {k: v for k, v in temp_obs.items() if k != self.ego_id}
        temp_env.close()

        # Create the real env; apply extra_config then honour render_mode.
        real_config = dict(base_config)
        if extra_config:
            real_config.update(extra_config)
        if render_mode == "human":
            real_config["use_render"] = True
        self.env = MultiAgentRacingEnv(real_config)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.config["start_seed"] = seed
        obs_dict, info_dict = self.env.reset()
        self._last_ego_obs = obs_dict.get(
            self.ego_id, np.zeros(self.observation_space.shape)
        )
        self._opponent_obs = {k: v for k, v in obs_dict.items() if k != self.ego_id}
        ego_info = info_dict.get(self.ego_id, {})
        return self._last_ego_obs.copy(), ego_info

    def step(self, action):
        actions = {}
        for agent_id in list(self.env.agents.keys()):
            if agent_id == self.ego_id:
                actions[agent_id] = action
            else:
                opp_obs = self._opponent_obs.get(agent_id, np.zeros(self.observation_space.shape))
                actions[agent_id] = self._opponent_fn(opp_obs, agent_id)

        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._opponent_obs = {k: v for k, v in obs.items() if k != self.ego_id}

        if self.ego_id in obs:
            ego_obs = obs[self.ego_id]
            self._last_ego_obs = ego_obs
        else:
            ego_obs = self._last_ego_obs

        ego_reward = rewards.get(self.ego_id, 0.0)
        ego_terminated = terms.get(self.ego_id, False)
        ego_truncated = truncs.get(self.ego_id, False)
        ego_info = infos.get(self.ego_id, {})

        return ego_obs.copy(), float(ego_reward), bool(ego_terminated), bool(ego_truncated), ego_info

    def set_opponent_policy(self, policy):
        """Update the opponent policy (e.g., for self-play curriculum).

        Args:
            policy: Either a string name ("random", "aggressive", "still")
                    or a callable with signature (obs, agent_id) -> action.
        """
        if isinstance(policy, str):
            self._opponent_fn = OPPONENT_POLICIES[policy]
        else:
            self._opponent_fn = policy

    def render(self):
        pass

    def close(self):
        self.env.close()


def make_racing_env(rank=0, num_agents=2, opponent_policy="aggressive", extra_config=None):
    def _init():
        env = RacingEnv(
            num_agents=num_agents,
            opponent_policy=opponent_policy,
            extra_config=extra_config,
        )
        return env
    return _init
