# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b mini_project/autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants (`MAX_TRAIN_DURATION_SECONDS`), evaluation harness (`evaluate()`), and baseline opponent loading. Do not modify.
   - `train.py` — the file you modify. RL algorithm, model architecture, hyperparameters, training loop, device selection.
4. **Verify agent exists**: Check that `agents/example_agent/model.pt` exists. This is the baseline opponent used during evaluation.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU/MPS. The training script runs for up to a **fixed time budget of 2 hours** (wall clock training time, excluding startup and evaluation). Training may stop earlier if `speed_score` has not improved for 3 consecutive 20-minute evaluations (60 min without progress). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation (`evaluate()`), baseline opponent (example_agent), and training constants (time budget, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: maximize `win_rate` against example_agent across all 7 maps.** Since the time budget is fixed, you don't need to worry about training time — it's always 2 hours (`MAX_TRAIN_DURATION_SECONDS` in `prepare.py`). Everything is fair game: change the RL algorithm entirely, the model architecture, the optimizer, the hyperparameters, the network size. The only constraint is that the code runs without crashing and finishes within the time budget.

`win_rate` is the fraction of races (out of 126 total: 7 maps × 18 races) where the ego agent finishes before the opponent. Higher is better.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny win_rate improvement that adds 20 lines of hacky code? Probably not worth it. A tiny win_rate improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
========================================================================
                     POST-TRAINING EVALUATION
  7 maps · 18 races/map · 126 total · opponent: example_agent
  Model parameters: 264,194
========================================================================
  Map              Wins    Win%   Route%   Finish Steps
------------------------------------------------------------------------
  circuit         12/18    67%    72.4%         941
  hairpin          4/18    22%    45.1%           —
  oval            18/18   100%    88.3%         823
  chicane          6/18    33%    65.7%        1104
  technical        8/18    44%    61.2%        1247
  mountain         2/18    11%    38.9%           —
  street           2/18    11%    29.4%           —
========================================================================

  win_rate:              52/126   (41.3%)
  avg_route_completion:  57.3%
  avg_steps_to_finish:   1029   (52 of 126 races finished)
  num_params:            264,194
========================================================================
```

The script evaluates every 20 minutes and stops early if there is no improvement for 3 consecutive evaluations. You can extract the per-checkpoint win rates from the log file:

```
grep "^\[.*min\] win_rate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row. Columns are `commit`, one column per 20-minute checkpoint, `status`, and `description`:

```
commit	20min	40min	60min	80min	100min	120min	status	description
```

1. git commit hash (short, 7 chars)
2–7. `win_rate` (0.0–1.0) at each 20-minute evaluation checkpoint. Use `—` if training stopped before that point, `unknown` if the checkpoint ran but the metric wasn't recorded, `0.0000` for a crash.
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	20min	40min	60min	80min	100min	120min	status	description
a1b2c3d	0.3810	0.4286	0.4762	0.4524	0.4603	0.4841	keep	baseline
b2c3d4e	0.4127	0.5079	0.5238	—	—	—	keep	increase LR to 3e-3 (early stop at 60min)
c3d4e5f	0.3175	0.3651	0.3810	0.3968	0.3810	0.3651	discard	switch to ReLU
d4e5f6g	0.0000	—	—	—	—	—	crash	512x512 network (OOM on MPS)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `mini_project/autoresearch/apr9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the per-checkpoint win rates:
   ```
   grep "^\[.*min\] win_rate:" run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv — fill in each 20-min column from the grep output above, using `—` for any checkpoint not reached. (NOTE: do not commit results.tsv — it must survive `git reset` when discarding experiments)
8. If the best checkpoint `win_rate` improved (higher) over the previous best, you "advance" the branch, keeping the git commit
9. If equal or worse at every checkpoint, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment runs for up to 2 hours (`TimeLimitCallback`) but may stop earlier via early stopping. Budget ~30 minutes extra for startup and post-training evaluation. If a run exceeds 2.5 hours total, kill it and treat it as a failure (discard and revert).

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

You can also try self_play, or playing against recent good agents, or against the baseline.

We can also try using natural gradients and calculating the fisher information matrix. 

Architectures that may be interesting to try:
* CNN
* LSTM
* Transformers
