"""
Human demonstration data collection for behavior cloning.

Launches a racing environment with MetaDrive's top-down pygame view and
records (observation, action) pairs using mouse input. The data format
is identical to what the RL training pipeline uses:

  Observation space: Box(0, 1, (161,), float32)  — MetaDrive sensor vector
  Action space:      Box(-1, 1,   (2,), float32)  — [steering, throttle]

panda3D / OpenGL is NOT used; the top-down renderer is pure pygame and
works on all platforms including macOS 15+.

Controls
--------
  Mouse             : move the control dot in the bottom-right box
                        horizontal → steering  (left = -1, right = +1)
                        vertical   → throttle  (up = +1 accel, down = -1 brake)
  SPACE (hold)      : record frames
  (no SPACE held)   : practice — simulation runs but nothing is recorded
  ESC / close window: end the current episode early

After each episode you are prompted in the terminal to save or discard the
collected frames.  Saved data is written as a .npz file containing:
  observations : float32 array of shape (N, obs_dim)
  actions      : float32 array of shape (N, 2)

Multiple runs (or multiple episodes in one run) append to the same output
file when --output is kept the same across runs.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from env import RacingEnv
from racing_maps import set_racing_map, RACING_MAPS


# ---------------------------------------------------------------------------
# Display / control constants
# ---------------------------------------------------------------------------

_DISPLAY_PX = 800    # on-screen window size
_RENDER_PX  = 1600   # offscreen render size — larger so rotation doesn't clip
_CROP_PX    = 400    # how many pixels to crop from centre before scaling up;
                     # smaller = more zoomed in (800→zoom 2×, 400→zoom 4×)

_BOX_SIZE   = 220    # outer control-box pixel size
_BOX_PAD    = 10     # padding from outer border to inner play area
_BOX_MARGIN = 12     # gap from window edge to box outer corner
_BOX_INNER  = _BOX_SIZE - 2 * _BOX_PAD   # 200 — dot travels this many pixels

_SLOW_DELAY = 0.15   # seconds inserted between steps in slow/practice mode


# ---------------------------------------------------------------------------
# Control-box drawing
# ---------------------------------------------------------------------------

def draw_control_box(surface, dot_x: float, dot_y: float, recording: bool):
    """
    Draw the steering/throttle control box in the bottom-right corner of
    *surface*.  The white dot (red when recording) shows current action.
    """
    import pygame

    bx = _DISPLAY_PX - _BOX_MARGIN - _BOX_SIZE
    by = _BOX_MARGIN
    ix = bx + _BOX_PAD   # inner-area top-left
    iy = by + _BOX_PAD

    # Background + border
    pygame.draw.rect(surface, (20, 20, 20), (bx, by, _BOX_SIZE, _BOX_SIZE))
    border_col = (220, 70, 70) if recording else (160, 160, 160)
    pygame.draw.rect(surface, border_col, (bx, by, _BOX_SIZE, _BOX_SIZE), 2)

    # Center crosshair
    cx = ix + _BOX_INNER // 2
    cy = iy + _BOX_INNER // 2
    pygame.draw.line(surface, (55, 55, 55), (ix,              cy), (ix + _BOX_INNER, cy), 1)
    pygame.draw.line(surface, (55, 55, 55), (cx, iy              ), (cx, iy + _BOX_INNER), 1)

    # Dot
    dot_col = (255, 80, 80) if recording else (255, 255, 255)
    pygame.draw.circle(surface, dot_col,
                       (int(ix + dot_x), int(iy + dot_y)), 6)

    # Labels — font initialised lazily and cached on the function object
    if not hasattr(draw_control_box, "_font"):
        pygame.font.init()
        draw_control_box._font = pygame.font.SysFont("monospace", 10)
    font = draw_control_box._font

    def blit(text, x, y, col=(110, 110, 110)):
        surface.blit(font.render(text, True, col), (x, y))

    # Mode tag (top-left of box)
    tag_col = (255, 80, 80) if recording else (180, 180, 180)
    blit("● REC" if recording else "SLOW", bx + 4, by + 3, tag_col)

    # Axis labels (inside inner area, near edges)
    blit("accel", cx - 15, iy + 2)
    blit("brake", cx - 15, iy + _BOX_INNER - 12)
    blit("R",     ix + 2,               cy - 6)
    blit("L",     ix + _BOX_INNER - 8,  cy - 6)


# ---------------------------------------------------------------------------
# Mouse / action helpers
# ---------------------------------------------------------------------------

def mouse_to_dot(mouse_x: int, mouse_y: int):
    """Map absolute mouse position to dot coordinates clamped to [0, _BOX_INNER]."""
    bx = _DISPLAY_PX - _BOX_MARGIN - _BOX_SIZE
    by = _BOX_MARGIN
    return (float(np.clip(mouse_x - (bx + _BOX_PAD), 0, _BOX_INNER)),
            float(np.clip(mouse_y - (by + _BOX_PAD), 0, _BOX_INNER)))


def dot_to_action(dot_x: float, dot_y: float) -> np.ndarray:
    """Map dot position → [steering, throttle] each in [-1, 1].

    Steering uses a square curve (x²) so small movements near centre are
    gentle and you have to push further for aggressive turns.
    """
    raw = -((dot_x / _BOX_INNER) * 2.0 - 1.0)
    steering = float(np.sign(raw) * raw ** 2)
    throttle  = 1.0 - (dot_y / _BOX_INNER) * 2.0
    return np.array([steering, throttle], dtype=np.float32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(env: RacingEnv, dot_x: float, dot_y: float,
                 recording: bool, n_frames: int, step: int):
    """
    Render one frame: ego-centred top-down view + control-box overlay.
    Does NOT drain the event queue — call process_events() separately.
    """
    import cv2
    import pygame

    engine  = env.env.engine
    agents  = env.env.agents
    ego     = agents.get(env.ego_id) or next(iter(agents.values()), None)
    ego_pos = tuple(ego.position[:2]) if ego is not None else None

    # ---- initialise renderer and display once ----
    if engine.top_down_renderer is None:
        pygame.init()
        pygame.font.init()
        pygame.mouse.set_visible(True)
        from metadrive.engine.top_down_renderer import TopDownRenderer
        engine.top_down_renderer = TopDownRenderer(
            camera_position=ego_pos,
            window=False,
            screen_size=(_RENDER_PX, _RENDER_PX),
        )
        pygame.display.set_mode((_DISPLAY_PX, _DISPLAY_PX))
    elif ego_pos is not None:
        engine.top_down_renderer.position = ego_pos

    # ---- get offscreen frame ----
    frame = env.env.render(mode="topdown")   # (H, W, 3) RGB numpy

    # ---- rotate so ego always faces up ----
    if ego is not None:
        angle = float(-np.rad2deg(ego.heading_theta) + 90)
        M = cv2.getRotationMatrix2D(
            (_RENDER_PX // 2, _RENDER_PX // 2), angle, 1.0
        )
        frame = cv2.warpAffine(frame, M, (_RENDER_PX, _RENDER_PX),
                               borderValue=(255, 255, 255))

    # ---- crop centre then scale up to display size ----
    half = _CROP_PX // 2
    cx = cy = _RENDER_PX // 2
    frame = frame[cy - half : cy + half, cx - half : cx + half]
    frame = cv2.resize(frame, (_DISPLAY_PX, _DISPLAY_PX), interpolation=cv2.INTER_LINEAR)

    # ---- blit game frame ----
    screen = pygame.display.get_surface()
    screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))

    # ---- control-box overlay ----
    draw_control_box(screen, dot_x, dot_y, recording)

    # ---- hint text (top-left corner) ----
    if not hasattr(render_frame, "_font"):
        render_frame._font = pygame.font.SysFont("monospace", 11)
    font = render_frame._font
    if recording:
        hint     = f"● RECORDING  ({n_frames} frames)"
        hint_col = (255, 80, 80)
    else:
        hint     = "hold SPACE to record"
        hint_col = (160, 160, 160)
    screen.blit(font.render(hint, True, hint_col), (8, 8))

    pygame.display.flip()
    pygame.display.set_caption(
        f"● REC  frames={n_frames}  step={step}" if recording
        else f"SLOW  step={step}"
    )


def process_events():
    """
    Drain the pygame event queue.
    Returns (quit: bool, mouse_x: int, mouse_y: int, space_held: bool).
    """
    import pygame
    quit_flag = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_flag = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            quit_flag = True
    mouse_x, mouse_y = pygame.mouse.get_pos()
    keys = pygame.key.get_pressed()
    return quit_flag, mouse_x, mouse_y, bool(keys[pygame.K_SPACE])


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(env: RacingEnv, slow_delay: float = _SLOW_DELAY,
                    verbose: bool = True):
    """
    Run one episode driven by mouse input.

    Mouse moves the dot in the bottom-right control box:
      horizontal → steering, vertical → throttle (up = accelerate).

    The simulation always runs in slow mode (sleep between steps).
    Hold SPACE to record frames; release SPACE to practice without recording.

    Returns
    -------
    observations : float32 ndarray, shape (N, obs_dim)
    actions      : float32 ndarray, shape (N, 2)
    """
    obs, _ = env.reset()
    obs_dim = obs.shape[0]

    dot_x = float(_BOX_INNER / 2)   # updated each frame from mouse position
    dot_y = float(_BOX_INNER / 2)
    recording = False

    obs_list: list = []
    act_list: list = []
    step = 0

    if verbose:
        print("  Mouse controls the dot in the bottom-right box.")
        print("  Hold SPACE to record.  Release SPACE to practice without recording.")
        print("  ESC to end episode.")

    try:
        while True:
            render_frame(env, dot_x, dot_y, recording,
                         n_frames=len(obs_list), step=step)

            quit_flag, mouse_x, mouse_y, space_held = process_events()
            if quit_flag:
                break

            recording = space_held
            dot_x, dot_y = mouse_to_dot(mouse_x, mouse_y)
            action = dot_to_action(dot_x, dot_y)

            if recording:
                obs_list.append(obs.copy())
                act_list.append(action.copy())

            if step % 100 == 0:
                tag = "REC" if recording else "---"
                print(f"    [{tag}]  step={step}  recorded={len(obs_list)}", flush=True)

            obs, _reward, terminated, truncated, _info = env.step(action)
            step += 1

            time.sleep(slow_delay)

            if terminated or truncated:
                break

    except (KeyboardInterrupt, SystemExit):
        pass

    if verbose:
        print(f"  Episode ended — {step} steps driven, {len(obs_list)} frames recorded.")

    if not obs_list:
        return (np.zeros((0, obs_dim), dtype=np.float32),
                np.zeros((0, 2),      dtype=np.float32))

    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32))


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_demos(path: Path, observations: np.ndarray, actions: np.ndarray):
    """Write (observations, actions) to *path*, appending if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = np.load(path)
        observations = np.concatenate([existing["observations"], observations], axis=0)
        actions      = np.concatenate([existing["actions"],      actions],      axis=0)

    np.savez(path, observations=observations, actions=actions)
    print(f"  Saved {len(observations)} total frames → {path}")


def prompt_save_or_discard(observations: np.ndarray, actions: np.ndarray,
                           output_path: Path) -> bool:
    """Print a summary and ask the user whether to keep the data."""
    n = len(observations)
    if n == 0:
        print("  No frames recorded — nothing to save.")
        return False

    print()
    print("=" * 52)
    print(f"  Recorded {n} (observation, action) pairs")
    print(f"  Observations : {observations.shape}  dtype={observations.dtype}")
    print(f"  Actions      : {actions.shape}  dtype={actions.dtype}")
    print(f"  Output file  : {output_path}")
    print("=" * 52)

    while True:
        choice = input("  Save this episode? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            save_demos(output_path, observations, actions)
            return True
        if choice in ("n", "no"):
            print("  Data discarded.")
            return False
        print("  Please enter y or n.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect human driving demonstrations for behavior cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map",
        default=None,
        choices=list(RACING_MAPS.keys()),
        help="Track to race on (default: random).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Output .npz file path.  New data is appended if the file already "
            "exists.  Defaults to demos/demo_<timestamp>.npz."
        ),
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        metavar="N",
        help="Total number of agents including the human-controlled ego agent.",
    )
    parser.add_argument(
        "--opponent",
        default="still",
        choices=["aggressive", "random", "still"],
        help="Policy used by the opponent agent(s).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes to collect in this session.",
    )
    parser.add_argument(
        "--slow-delay",
        type=float,
        default=_SLOW_DELAY,
        metavar="SEC",
        help="Seconds to sleep between steps in slow/practice mode (no SPACE held).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.map is None:
        args.map = np.random.choice(list(RACING_MAPS.keys()))

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("demos") / f"demo_{timestamp}.npz"
    else:
        output_path = Path(args.output)

    print()
    print("=" * 60)
    print("  Behavior-Cloning Data Collection")
    print(f"  Map        : {args.map}")
    print(f"  Agents     : {args.num_agents}  (opponent: {args.opponent})")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Output     : {output_path}")
    print(f"  Slow delay : {args.slow_delay}s/step when not recording")
    print("=" * 60)
    print()
    print("  Controls")
    print("  --------")
    print("  Mouse         move cursor into the top-right control box to steer")
    print("                  horizontal = steering, vertical = throttle")
    print("  SPACE (hold)  record frames")
    print("  (no SPACE)    practice — nothing recorded")
    print("  ESC           end episode early")
    print()

    restore_map = set_racing_map(args.map)

    try:
        env = RacingEnv(
            num_agents=args.num_agents,
            opponent_policy=args.opponent,
            extra_config={"idle_done": False, "out_of_road_done": False},
        )
    except Exception as exc:
        print(f"[ERROR] Could not create environment: {exc}")
        restore_map()
        sys.exit(1)

    saved_total = 0
    try:
        for ep_idx in range(args.episodes):
            if args.episodes > 1:
                print(f"\n--- Episode {ep_idx + 1} / {args.episodes} ---")
            else:
                print()

            observations, actions = collect_episode(
                env, slow_delay=args.slow_delay
            )
            saved = prompt_save_or_discard(observations, actions, output_path)
            if saved:
                saved_total += len(observations)

            if ep_idx < args.episodes - 1:
                try:
                    input("\n  Press Enter to start the next episode (Ctrl+C to quit)…")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()
        restore_map()

    print()
    if saved_total:
        print(f"Session complete. {saved_total} frames saved to {output_path}")
    else:
        print("Session complete. No data was saved.")
    print()


if __name__ == "__main__":
    main()
