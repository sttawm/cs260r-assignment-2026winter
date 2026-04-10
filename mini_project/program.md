# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants (`MAX_TRAIN_DURATION_SECONDS`), evaluation harness (`evaluate()`), and baseline opponent loading. Do not modify.
   - `train.py` — the file you modify. RL algorithm, model architecture, hyperparameters, training loop, device selection.
4. **Verify agent exists**: Check that `agents/example_agent/model.pt` exists. This is the baseline opponent used during evaluation.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU/MPS. The training script runs for a **fixed time budget of 2 hours** (wall clock training time, excluding startup and evaluation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation (`evaluate()`), baseline opponent (example_agent), and training constants (time budget, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: maximize `speed_score` against example_agent across all 7 maps.** Since the time budget is fixed, you don't need to worry about training time — it's always 2 hours. Everything is fair game: change the RL algorithm entirely, the model architecture, the optimizer, the hyperparameters, the network size. The only constraint is that the code runs without crashing and finishes within the time budget.

`speed_score` is defined as:

```
completion  = sum(distance_covered_i) / sum(track_length_i)   # distance traveled / total possible distance
velocity    = sum(distance_covered_i) / sum(steps_i)          # total distance / total time steps
speed_score = completion × velocity
```

across all 21 races (7 maps × 3 races). Higher is better. `distance_covered_i = route_completion_i × track_length_i` in meters (from `navigation.total_length`), so longer tracks are weighted proportionally. An unfinished race is penalised twice: it reduces both completion and velocity.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny speed_score improvement that adds 20 lines of hacky code? Probably not worth it. A tiny speed_score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
========================================================================
                     POST-TRAINING EVALUATION
  7 maps · 3 races/map · 21 total · opponent: example_agent
  Model parameters: 264,194
========================================================================
  Map              Wins   Win%   Route%   Finish Steps
------------------------------------------------------------------------
  circuit          1/3    33%    72.4%         941
  hairpin          0/3     0%    45.1%           —
  oval             2/3    67%    88.3%         823
  chicane          1/3    33%    65.7%        1104
  technical        1/3    33%    61.2%        1247
  mountain         0/3     0%    38.9%           —
  street           0/3     0%    29.4%           —
========================================================================

  speed_score:           0.000312
  win_rate:              5/21   (23.8%)
  avg_route_completion:  57.3%
  avg_steps_to_finish:   1029   (5 of 21 races finished)
  num_params:            264,194
========================================================================
```

Note that the script is configured to always stop after 2 hours. You can extract the key metric from the log file:

```
grep "^  speed_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	speed_score	win_rate	status	description
```

1. git commit hash (short, 7 chars)
2. speed_score (e.g. 0.000312) — use 0.000000 for crashes
3. win_rate achieved (e.g. 0.238095) — use 0.000000 for crashes; kept for reference
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	speed_score	win_rate	status	description
a1b2c3d	0.000312	0.238095	keep	baseline
b2c3d4e	0.000389	0.333333	keep	increase LR to 3e-3
c3d4e5f	0.000271	0.190476	discard	switch to ReLU activation
d4e5f6g	0.000000	0.000000	crash	512x512 network (OOM on MPS)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results:
   ```
   grep "^  speed_score:\|^  win_rate:" run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git — it must survive `git reset` when discarding experiments)
8. If `speed_score` improved (higher), you "advance" the branch, keeping the git commit
9. If equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment takes exactly 2 hours (the `TimeLimitCallback` enforces this). Budget ~15 minutes extra for startup and post-training evaluation. If a run exceeds 2.5 hours total, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Ideas
Here are a few ideas to try, and some information about the problem.

The bottleneck is simulation (not GPU / training). So, it makes sense to try to make use of multiple cores for simulation. Moreover, methods that are sample efficient are probably going to do better.

One of the trickier problems is the issue of compounding errors and error recovery. Methods that do well for learning from errors (like the car spinning around) may be interesting.

Some amount of reward-shaping should probably be attempted. However, be cautious, because trying to over-optimize the reward may result in unexpected learnings. Definitely value simplificty, here.

Try different learning algorithms (i.e. PPO, SAC, TD3, etc.)

If you find it useful, start from a pre-trained model.

Consider changing the discount factor, or even annealing it. Empirically it seems like 0.95 led to faster learning than 0.99

There's some training data that you can use for behavior-cloning. Then, you can use RL algorithms for fine-tuning on top of that. It's in the demos/ folder.

You can try producing multiple actions at a time, to help with temporal consistency of actions.

You can also try receding horizon control.

Architectures that may be interesting to try:
* CNN
* LSTM
* Transformers
