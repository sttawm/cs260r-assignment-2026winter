import importlib.util
import os
import sys

import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from racing_maps import set_racing_map

MAX_TRAIN_DURATION_SECONDS = 2 * 60 * 60  # 2 hours

_ALL_MAPS = ["circuit", "hairpin", "oval", "chicane", "technical", "mountain", "street"]
_RACES_PER_MAP = 3  # 7 maps × 3 = 21 total races

_ENV_BASE_CFG = dict(
    use_render=False,
    crash_done=False,
    crash_vehicle_done=False,
    out_of_road_done=True,
    allow_respawn=False,
    horizon=3000,
)

_EXAMPLE_AGENT_DIR = os.path.join(os.path.dirname(__file__), "agents", "example_agent")


def _load_example_agent():
    spec = importlib.util.spec_from_file_location(
        "example_agent", os.path.join(_EXAMPLE_AGENT_DIR, "agent.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(_EXAMPLE_AGENT_DIR))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module.Policy()


def _make_env(num_agents, seed):
    cfg = dict(_ENV_BASE_CFG)
    cfg["num_agents"] = num_agents
    cfg["start_seed"] = seed
    cfg["map_config"] = {
        "lane_num": max(2, num_agents),
        "exit_length": 20,
        "bottle_lane_num": max(4, num_agents),
        "neck_lane_num": 1,
        "neck_length": 20,
    }
    return MultiAgentRacingEnv(cfg)


def _run_episode(env, model, baseline_fn):
    """Run one episode with model as ego (agent0) vs baseline_fn for opponents."""
    ego_id = "agent0"
    obs_dict, _ = env.reset()
    obs_shape = next(iter(env.observation_space.values())).shape

    # Record total track length in meters from the navigation module.
    # navigation.total_length is computed at reset as the sum of lane lengths along the route.
    track_length = env.agents[ego_id].navigation.total_length

    ep_reward = 0.0
    ep_route = 0.0
    ego_arrive = False
    ego_arrive_step = None
    opp_first_arrive_step = None
    opp_arrived = set()
    steps = 0
    ego_done = False
    opp_obs = {k: v for k, v in obs_dict.items() if k != ego_id}

    while not ego_done:
        actions = {}
        for aid in list(env.agents.keys()):
            if aid == ego_id:
                ego_obs = obs_dict.get(ego_id, np.zeros(obs_shape))
                action, _ = model.predict(ego_obs, deterministic=True)
                actions[aid] = action
            else:
                actions[aid] = baseline_fn(opp_obs.get(aid, np.zeros(obs_shape)), aid)

        obs_dict, _, terms, truncs, infos = env.step(actions)
        opp_obs = {k: v for k, v in obs_dict.items() if k != ego_id}
        steps += 1

        ego_info = infos.get(ego_id, {})
        ep_reward = ego_info.get("episode_reward", ep_reward)

        if ego_info.get("arrive_dest", False) and not ego_arrive:
            ego_arrive = True
            ego_arrive_step = steps

        for aid, info in infos.items():
            if aid != ego_id and aid not in opp_arrived and info.get("arrive_dest", False):
                opp_arrived.add(aid)
                if opp_first_arrive_step is None:
                    opp_first_arrive_step = steps

        if ego_id in env.agents:
            try:
                ep_route = env.agents[ego_id].navigation.route_completion
            except Exception:
                pass

        ego_done = terms.get(ego_id, False) or truncs.get(ego_id, False)

    # Win: ego arrived and either opponent never arrived or ego was first
    win = ego_arrive and (
        opp_first_arrive_step is None or ego_arrive_step <= opp_first_arrive_step
    )

    return {
        "win": win,
        "route_completion": ep_route,
        "distance_covered": ep_route * track_length,   # meters traveled along route
        "distance_total": track_length,                # total track length in meters
        "arrive": ego_arrive,
        "arrive_step": ego_arrive_step,
        "steps": steps,
        "reward": ep_reward,
    }


def evaluate(model, seed=42):
    """
    Evaluate model across all 7 maps × 3 races (21 total) vs example_agent.

    Metrics:
      - win_rate            (ego arrives first)
      - avg_route_completion
      - avg_steps_to_finish (for races the ego completed)
      - num_params          (total trainable parameters in model.policy)
    """
    baseline_policy = _load_example_agent()
    baseline_fn = lambda obs, aid: baseline_policy(obs)

    num_params = sum(p.numel() for p in model.policy.parameters())

    map_results = {}
    for map_name in _ALL_MAPS:
        restore = set_racing_map(map_name)
        races = []
        env = _make_env(num_agents=2, seed=seed)
        try:
            for _ in range(_RACES_PER_MAP):
                races.append(_run_episode(env, model, baseline_fn))
        finally:
            env.close()
            restore()
        map_results[map_name] = races

    # ── print table ───────────────────────────────────────────────────── #
    W = 72
    total_races = len(_ALL_MAPS) * _RACES_PER_MAP
    print()
    print("=" * W)
    print(f"{'POST-TRAINING EVALUATION':^{W}}")
    print(f"  {len(_ALL_MAPS)} maps  ·  {_RACES_PER_MAP} races/map  ·  "
          f"{total_races} total  ·  opponent: example_agent")
    print(f"  Model parameters: {num_params:,}")
    print("=" * W)
    print(f"  {'Map':<14}  {'Wins':>6}  {'Win%':>5}  {'Route%':>7}  {'Finish Steps':>12}")
    print("-" * W)

    all_wins = 0
    all_routes = []
    all_arrive_steps = []

    for map_name in _ALL_MAPS:
        races = map_results[map_name]
        wins = sum(1 for r in races if r["win"])
        routes = [r["route_completion"] for r in races]
        arrive_steps = [r["arrive_step"] for r in races if r["arrive_step"] is not None]

        all_wins += wins
        all_routes.extend(routes)
        all_arrive_steps.extend(arrive_steps)

        win_pct = wins / len(races) * 100
        avg_route = np.mean(routes) * 100
        steps_str = f"{np.mean(arrive_steps):.0f}" if arrive_steps else "—"

        print(f"  {map_name:<14}  {wins}/{len(races):>1}   {win_pct:>4.0f}%  "
              f"{avg_route:>6.1f}%  {steps_str:>12}")

    print("=" * W)

    win_rate = all_wins / total_races if total_races else 0.0
    avg_route_pct = np.mean(all_routes) * 100 if all_routes else 0.0
    avg_steps = float(np.mean(all_arrive_steps)) if all_arrive_steps else float("nan")
    n_finished = len(all_arrive_steps)

    # speed_score = completion × velocity, where:
    #   completion = sum(distance_covered_i) / sum(distance_total_i)
    #             = total distance traveled / total possible distance across all tracks (weighted by track length)
    #   velocity   = sum(distance_covered_i) / sum(steps_i)
    #             = total distance traveled / total time steps
    # Uses actual meters from navigation.total_length (set at episode reset).
    all_races_flat = [r for races in map_results.values() for r in races]
    total_distance_covered = sum(r["distance_covered"] for r in all_races_flat)
    total_distance_possible = sum(r["distance_total"] for r in all_races_flat)
    total_steps_all = sum(r["steps"] for r in all_races_flat)
    completion = total_distance_covered / total_distance_possible if total_distance_possible > 0 else 0.0
    velocity = total_distance_covered / total_steps_all if total_steps_all > 0 else 0.0
    speed_score = float(completion * velocity)

    print(f"\n  speed_score:           {speed_score:.6f}")
    print(f"  win_rate:              {all_wins}/{total_races}   ({win_rate:.1%})")
    print(f"  avg_route_completion:  {avg_route_pct:.1f}%")
    if all_arrive_steps:
        print(f"  avg_steps_to_finish:   {avg_steps:.0f}   ({n_finished} of {total_races} races finished)")
    else:
        print(f"  avg_steps_to_finish:   —   (0 of {total_races} races finished)")
    print(f"  num_params:            {num_params:,}")
    print("=" * W)
    print()

    return {
        "speed_score": speed_score,
        "win_rate": win_rate,
        "avg_route_completion": avg_route_pct / 100.0,
        "avg_steps_to_finish": avg_steps,
        "n_finished": n_finished,
        "total_races": total_races,
        "num_params": num_params,
        "map_results": map_results,
    }
