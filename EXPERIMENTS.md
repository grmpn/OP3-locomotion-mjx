# OP3 experiment decision log

This log records choices, evidence, outcomes, and rejected alternatives for the
walking-policy project. It will be the source for the final experiment summary.
New entries should retain earlier decisions and explain what new evidence changed.
Full configurations, metrics, trajectories, and checkpoints remain under `outputs/`.
Those generated artifacts are local and ignored by Git; this log is committed.

The entries through D010 were backfilled on **2026-11-07 at 00:20 UTC** from run
metadata, evaluation reports, commit history, and the work performed in this
thread. Exact run starts below come from `run.json`. Setup decisions without a
recorded event timestamp are marked as backfilled rather than assigned one.

## Run index

| Run | Start UTC | Initialization and settings | Status or evidence |
| --- | --- | --- | --- |
| `smoke` | 2026-11-06, before 23:54 | Initial newest-package dependency set | Failed before PPO updates: removed JAX replication API |
| `smoke-compatible` | 2026-11-06 23:54:54 | Seed 0; 32 envs; batch 16 × 2; 640 control steps | PPO update, export/reload, rendering, and viewer checks passed; not a walking policy |
| `baseline` | 2026-11-06 23:58:58 | Seed 0; 512 envs; batch 128 × 4; uniform combined commands plus 20% standing; 30M requested steps | Intentionally stopped; last saved checkpoint 16,056,320; selected evaluated checkpoint 15,052,800 for refinement |
| `balanced` | 2026-11-07 00:17:06 | Weights/normalization from baseline 15,052,800; seed 0; same PPO/physics/rewards; balanced command modes; 15M requested additional steps | Finished at 15,052,800 additional steps in 826.8 s; selected 12,042,240 after all-direction and transition validation; final checkpoint rejected for left-turn regression |

Run settings are in `outputs/<run>/config.json`, provenance in `run.json`, and
training/evaluation aggregates in `metrics.jsonl`. The baseline start revision is
`49ddbaf`; balanced refinement starts from clean revision `b8e1e52`. Refinement
resets optimizer, rollout RNG, and step count: it is **not a full-state resume**.
The policy lineage at refinement step N has 15,052,800 + N training transitions;
the unused tail of the baseline run is additional compute, not inherited training.

## D001 Physics implementation and collision scope

**Backfilled setup decision.** Use the pinned Menagerie revision
`98a70c569c4c1b6da467a20012f766ee1f8dc521` and unchanged `scene.xml` floor.
The first MJX-JAX full-mesh step compilation produced coplanar-mesh warnings and
was still compiling after several minutes with substantial host memory use. It
was deliberately stopped; this was a performance decision, not evidence that the
model could not simulate correctly.

Use MJX's official Warp backend for NVIDIA GPU physics. Following the user's
explicit authorization, disable head/arm contact participation while retaining
all masses, inertias, joints, visuals, and actuators. Keep torso/leg collisions
and the four original foot boxes. No collision mesh was decimated. No physics
randomization, reset perturbation, terrain change, or terrain curriculum was added.
The same model preparation is used by training and native inference.

The RTX 3070 Laptop GPU has 8 GB and SM 8.6; driver 591.59 reports CUDA driver
13.1. CUDA 13 wheels were selected after checking the JAX installation guide.
Disable JAX preallocation and default its memory fraction to 0.35 to leave space
for Warp and the desktop. Model download checksums and upstream license are kept
under `assets/robotis_op3/`; the reproducible download script is committed.

## D002 Dependency compatibility and export loading

**Backfilled setup decision.** The first PPO attempt with JAX/jaxlib 0.12.0.1
failed because Brax 0.14.2 still calls `jax.device_put_replicated`. Pin JAX/jaxlib
0.8.2, Flax 0.12.2, and Orbax 0.11.33 with MuJoCo/MJX 4.0.0 and Brax 0.14.2.
`uv.lock` records the complete resolved stack, including Warp 1.18.0. The locked
GPU smoke run subsequently passed; the PPO algorithm itself was not modified.

Brax's generic policy-config loader then raised `KeyError: None` for optional
kernel initializers. Keep Brax/Orbax parameter and normalization serialization,
but reconstruct the shared network factory from the saved typed configuration.
An export/reload check later reproduced deterministic actions exactly, including
normalization (`outputs/provenance-check/`).

## D003 Small baseline control and reward design

**Backfilled initial implementation, committed 2026-11-06 23:58:38 UTC.**
Control the 12 leg joints with bounded position offsets of ±0.5 rad around a fixed
standing pose. Hold head/arm targets at nominal. Use 2 ms physics, 20 ms control,
and ten internal physics steps; Brax action repeat is one. Observations contain
projected gravity, local base velocities, all joint positions/velocities, command,
and previous action. Actor and critic are separate 128 × 128 × 128 MLPs.

Use ordinary PPO and standard tracking/upright/regularization rewards. Planar
tracking has weight 3 and width 0.15 m/s; yaw tracking has weight 1 and width
0.4 rad/s. Terms are scaled by control dt once. Full weights and conventions are
in the run config and README. Start without foot-air-time, clearance, or slip
shaping; only add these if rollout evidence requires them.

Command bounds are `[0.25, 0.15, 0.8]` for body-frame forward/lateral velocity
(m/s) and yaw rate (rad/s). The initial sampler uses uniform components and a
20% exactly-zero command, updated every three seconds. Physical resets remain
identical across seeds; only commands and PPO exploration are stochastic.

## D004 Integration gates before longer training

**Backfilled checks completed before the baseline run.** Fixed nominal targets
stand for two seconds, with base height approximately 0.26824 m after settling.
The one-second small-action native/MJX comparison had maximum qpos difference
`5.60e-6` and qvel difference about `5.22e-5` in the first locked-stack check.
Frame conversion, finite reset/step, command/action reset, and fall-versus-timeout
checks passed. The fall test was corrected to use an upside-down pose above the
floor: its earlier inside-floor fixture could be launched by penetration forces.

GPU-only enforcement, a 640-step PPO update, export reload, deterministic native
inference, video rendering, and viewer startup passed. Smoke evaluation correctly
showed standing rather than command tracking. Evidence:
`outputs/checks.json`, `outputs/smoke-compatible/`, `outputs/reload-smoke/`.

## D005 Continue the baseline instead of adding gait incentives early

**Backfilled evaluation decisions, approximately 00:01–00:10 UTC on 2026-11-07.**
At 1M–5M baseline steps, the policy was upright but mostly stationary; at 5M it
sometimes unloaded one foot without cycling. Do not equate survival or high PPO
return with walking. Continue the baseline while checking a fixed command suite.

By 7,024,640 steps the policy developed repeated lift/landing without any extra
gait rewards. For a 0.15 m/s forward command it averaged about 0.183 m/s over five
seconds, but yaw averaged about −0.648 rad/s despite a zero-yaw command. Backward
and leftward motion improved; rightward and in-place turn commands remained weak.

**Decision:** retain the original reward design. The emergence of steps is
counter-evidence to adding air-time shaping at this point. Continue training and
focus assessment on yaw, individual motion directions, and command changes.
Evidence: `outputs/baseline-eval-1m/`, `-2m/`, `-5m/`, `-7m/`, and the inspected
forward video/frames in `outputs/baseline-video-7m/`.

## D006 Measure gait and slip rather than infer them from return

**2026-11-07 00:05–00:13 UTC, commits `20c0191` and `92c2eb1`.** Add evaluation-only
foot diagnostics: sole clearance from the original boxes, normal-force support
(>1 N), swing fraction, and touchdown counts. Exclude initial settling from gait
summaries. These do not change the environment's training observations or rewards.

Foot-body speed while supported also includes rolling, so it overestimates slip.
Add force-weighted tangential contact-point velocity using MuJoCo contact
Jacobians. In the rendered 7M forward rollout, left foot-body speed while
supported was about 0.190 m/s, but contact-point slip was about 0.027 m/s; right
contact slip was also about 0.027 m/s. This supports actual stepping with some
slip, not pure sliding, but does not establish adequate command tracking.
Evidence: `outputs/gait-diagnostics-check/`, `outputs/baseline-slip-7m/`.

## D007 Preserve provenance and validate the actual keyboard path

**2026-11-07 00:07–00:10 UTC, commit `dfaf667`.** Actual X11 key events sent only
to a newly launched OP3 window verified W/S, A/D, Q/E, Space, R, and Escape.
Window titles reflected commands and Escape exited cleanly. Direct callback
checks verified clamping at trained bounds. Native policy/physics inference
measured about 180 control steps/s, excluding rendering, versus a required 50 Hz.
Evidence: `outputs/viewer-keyboard-check.json`, `outputs/viewer-keyboard.log`.

Snapshot provenance once at run start and reuse it in subsequent exports, so
editing the repository during training cannot mislabel the policy's source.
The baseline predates this fix: use its **run.json** as authoritative start
provenance rather than per-export Git metadata. Balanced refinement includes the
fix. Exact deterministic policy/normalization export roundtrip also passed.

## D008 Dedicated command coverage is the remaining training gap

**2026-11-07 00:10–00:16 UTC.** At 10M steps the command-change rollout could walk
forward and turn while moving. At 14M, a −0.48 rad/s in-place command produced
about −0.412 rad/s, but a +0.48 rad/s command settled near +0.010 rad/s. At
15,052,800 steps the left turn again settled near zero (+0.0044 rad/s after the
first second), whereas the right turn settled near −0.398 rad/s. All fixed-suite
cases survived, including the 25-second command-change sequence.

The 15M policy's five-second mean velocities were:

| Command `[vx, vy, yaw]` | Measured `[vx, vy, yaw]` |
| --- | --- |
| `[0, 0, 0]` | `[0.0019, -0.0021, 0.0906]` |
| `[0.15, 0, 0]` | `[0.1449, -0.0096, -0.0295]` |
| `[-0.15, 0, 0]` | `[-0.1296, -0.0035, 0.0323]` |
| `[0, 0.09, 0]` | `[-0.0078, 0.0684, -0.0620]` |
| `[0, -0.09, 0]` | `[0.0212, -0.0817, 0.0018]` |
| `[0, 0, 0.48]` | `[0.0027, -0.0030, 0.0971]` |
| `[0, 0, -0.48]` | `[0.0150, 0.0006, -0.3943]` |

The positive-turn mean includes its initial transient and therefore overstates
sustained turning. Evidence: `outputs/baseline-eval-10m/`,
`outputs/baseline-turns-14m/`, `outputs/baseline-eval-15m/`.

**Hypothesis:** mostly combined training commands provide insufficient direct
coverage of starting an in-place turn from rest. This is a plausible diagnosis,
not a controlled causal finding. Address command coverage before adding gait
incentives or changing robot dynamics.

## D009 Warm-start balanced command refinement

**2026-11-07 00:15:52 UTC (`b8e1e52`), run started 00:17:06 UTC.** Add a `balanced`
command sampler: 20% standing, 20% forward/backward-only, 20% lateral-only, 20%
yaw-only, and 20% combined motion. Signed nonzero components retain the original
uniform bounds. A seeded 4,096-command check found counts 813, 794, 810, 820,
and 859 for those modes; both signs and bounds passed. Nominal pose/masses were
unchanged. Warm-start validation accepts only this sampling change and continues
to reject changed action meaning.

Stop the baseline after its latest completed save (16,056,320), and initialize
refinement from the fully evaluated **15,052,800** export. Keep physics, rewards,
network, action mapping, timing, and command bounds unchanged. This is a new
standard PPO run with fresh optimizer/RNG state, not a training-state resume.
The deliberate stop is recorded in `outputs/baseline/stopped.json`.

```sh
uv run op3-train --config outputs/baseline/config.json --output outputs/balanced \
  --initialize-from outputs/baseline/checkpoints/000015052800 \
  --command-sampling balanced --num-timesteps 15000000 --num-evals 16
```

Do not compare baseline and balanced training returns directly: their command
distributions differ. Compare the unchanged deterministic command suite instead.

## D010 First refinement assessment and logging correction

**Recorded 2026-11-07 00:20 UTC.** After 1,003,520 additional steps all fixed-suite
cases still survived. Forward mean was 0.1438 m/s for 0.15 requested; lateral
means were +0.0790 and −0.0722 m/s for ±0.09. The moving-turn segment settled near
0.4456 rad/s for 0.48 requested. The left in-place turn remains weak: its settled
rate is only +0.0436 rad/s. Backward motion still has measurable slip (left-foot
contact slip about 0.0463 m/s) and undesired positive yaw. Continue refinement;
these results do not justify declaring the project complete.

Evidence: `outputs/balanced-eval-1m/report.json` and its trajectories. No new gait
reward has been added. Before selecting a final policy, recheck longer standing,
both turn directions, all translations, command changes, slip/clearance, native
versus MJX behavior, and rendered rollouts.

The training wrapper's `episode/reward` field was an unused zero because Brax
aggregates actual return as `episode/sum_reward`. Remove that misleading field
for future runs; evaluation's `eval/episode_reward` remains valid. The focused
GPU wrapper check passed, including actual reward-sum aggregation, fall/timeouts,
and native/MJX parity (`outputs/checks-final-wrapper.json`). The running balanced
process was already loaded before this logging-only correction.

## D011 Five million refinement steps

**2026-11-07 00:23 UTC.** Evaluate refinement checkpoint `000005017600` on the
unchanged suite (`outputs/balanced-eval-5m/report.json`). All cases survived.
Forward/backward means were +0.146/−0.131 m/s for ±0.15; lateral means were
+0.067/−0.088 m/s for ±0.09. The left in-place turn now has repeated touchdowns
and a settled rate of +0.165 rad/s (up from +0.0044 before refinement), but still
undershoots +0.48 substantially. Right turn settles near −0.340 for −0.48.
The continuous change sequence has planar RMSE 0.0228 m/s and yaw RMSE
0.1183 rad/s, including transitions.

**Decision:** continue the planned refinement without changing rewards or
physics. Left-turn behavior is improving on the fixed test, so do not add gait
shaping or declare the run stuck. The comparison is still a single training
lineage with additional optimization, not an isolated command-sampler ablation.
Final acceptance remains unproven until sustained turning and longer rollouts
are checked.

## D012 Sustained left turns in the longer test

**2026-11-07 00:29 UTC.** Test refinement checkpoint `000009031680` for ten seconds
each at zero command and both in-place turn directions. All survived. Left turn
settles at +0.457 rad/s for +0.48 requested, with repeated foot lift/landing and
mean contact slip about 0.0016/0.0107 m/s for the two feet. Standing settled planar
velocity is approximately `[0.0000, -0.0008]` m/s and yaw +0.0016 rad/s. Right
turn still undershoots: settled −0.285 rad/s for −0.48 requested.

**Decision:** retain the current rewards and physics while the planned refinement
finishes. Left-turn behavior is no longer stationary; evaluate the final
checkpoints across all directions and longer transitions before selecting the
demo policy. No reward-weight or gait-shaping change is justified by this result
alone. Evidence: `outputs/balanced-turns-9m/report.json`.

## D013 Final checkpoint regression and selection

**2026-11-07 00:33 UTC.** Balanced refinement completed 15,052,800 additional
steps in 826.8 seconds. The final checkpoint survives the seven independent
ten-second native cases and tracks translations and right yaw, but left yaw
regressed: settled +0.0188 rad/s for +0.48 requested, with no touchdowns after
settling. Right yaw settles at −0.446 rad/s for −0.48. This invalidates using the
last export automatically as the demo policy, despite good aggregate return.
Evidence: `outputs/balanced-final-eval/` (continuous changes still being evaluated
when this decision was written).

**Decision:** evaluate saved refinement checkpoints 9,031,680 and 12,042,240 on
the same complete ten-second suite. The former already demonstrated sustained
left turns. Select using directional tracking, survival, gait evidence and
continuous transitions; do not select using aggregate PPO return alone. Do not
change rewards or launch another training run before evaluating these existing
candidates. The single-run comparisons are checkpoint selection, not independent
replication or proof of the sampler's causal effect.

## D014 Choose the twelve-million-step refinement checkpoint

**2026-11-07 00:35 UTC.** All eight cases survive the complete ten-second suite
for both candidates (50 seconds continuous for `changes`). Checkpoint 12,042,240
has settled left/right yaw +0.429/−0.404 rad/s for ±0.48, versus +0.457/−0.285
at checkpoint 9,031,680. Its settled forward/backward speeds are +0.150/−0.144 m/s
for ±0.15, and lateral speeds +0.084/−0.076 m/s for ±0.09. The continuous change
sequence has planar/yaw RMSE 0.017/0.088 (m/s, rad/s), including transients.

**Decision:** select `outputs/balanced/checkpoints/000012042240` for final demo
validation. This retains both turn directions with smaller worst-direction bias
than the earlier candidate and avoids the final checkpoint's stationary left
turn. The selected lineage contains 15,052,800 baseline plus 12,042,240 refinement
transitions (27,095,040 total); the actual runs performed additional unselected
updates. Evidence: `outputs/balanced-candidate-9m/report.json` and
`outputs/balanced-candidate-12m/report.json`.

Now check rendered gait, GPU MJX behavior, and reload the packaged export in the
keyboard viewer. These checks must pass before calling the deliverable complete.
No reward, physical-model, or command-range change was made during selection.

## D015 Command limits and keyboard behavior

**2026-11-07 00:37 UTC.** Probe each signed command limit for eight seconds and
actual keyboard yaw increments ±0.4 rad/s. All cases survive, as does a continuous
40-second forward/backward/left-yaw/right-yaw/stop sequence. Settled speeds at
limits are +0.240/−0.243 m/s forward/backward (requested ±0.25), +0.136/−0.127 m/s
lateral (requested ±0.15), and +0.751/−0.698 rad/s yaw (requested ±0.8).
The +0.4 rad/s left command settles near +0.208; −0.4 right settles near −0.353.
Backward motion at the maximum command has unwanted yaw around −0.195 rad/s.
Evidence: `outputs/demo-command-limits/report.json`; each case's exact commands
and trajectories are saved there. These are individual-axis probes and one
reversal sequence, not exhaustive coverage of simultaneous command extremes.

**Decision:** retain the trained bounds and describe imperfect low-rate turning
and backward yaw drift. Do not hide this by narrowing the displayed command range.
The packaged policy at `outputs/op3-policy/` loads successfully in the real GLFW
viewer for five seconds and exits cleanly (`outputs/demo-viewer.log`). The
previous actual-key-event integration check covers every key, reset, stop and
exit; the UI path is unchanged. Package the selected export and its lineage as
a local archive after remaining rendering and parity checks finish.

## D016 Native/MJX comparison and rendered gait

**2026-11-07 00:38 UTC.** The selected policy survives ten-second GPU MJX forward
and both in-place turns, plus the full 50-second continuous change sequence.
Settled forward speed is 0.149 m/s; left/right yaw rates are +0.422/−0.426 rad/s.
The change sequence has planar/yaw RMSE 0.0174/0.0914 (m/s, rad/s), close to
native MuJoCo's 0.0171/0.0878. Evidence: `outputs/demo-mjx/report.json` and
`outputs/balanced-candidate-12m/report.json`.

One-second replay of identical trained actions measures maximum native/MJX qpos
component differences of 0.00171 (forward), 0.0167 (left yaw), and 0.00157 (right
yaw); qvel maxima are 0.0906, 0.417, and 0.0335 respectively. CPU/GPU actions on
identical observations differ by at most 3.88e-7. The larger contact-rich state
differences mean these backends are not trajectory-identical; the small-action
smoke check's tighter errors must not be presented as full-policy equivalence.
Timing, model preparation, observations, action transform and normalization are
shared, and the closed-loop behavior agrees. Evidence: `outputs/demo-parity.json`.

Rendered forward and both turn rollouts show upright motion with foot lifting
and changing orientation. Native ten-second forward diagnostics record 43
landings per foot after settling, 8.1/6.6 mm maximum sole clearance, and
0.0032/0.0090 m/s mean contact-point slip. The gait is rapid and short-stepped,
with some slip; it is not a smooth humanlike gait. Standing has small foot
unloading and yaw drift. Forward/turn videos are complete; the continuous
command-change video is still rendering as this entry is written.

**Decision:** accept native MuJoCo for the interactive demo with the documented
numerical and tracking limits. Keep the selected export and finish the video,
artifact packaging, documentation review and commit. No additional training or
reward shaping is justified for the requested simple baseline.

## D017 Interactive rendering performance

**2026-11-07 00:41 UTC.** All requested videos finished, including the 15-second
continuous command-change clip, and sampled frames were inspected. The packaged
policy and archive reload successfully. Before finalizing, measure actual viewer
throughput: this display uses `llvmpipe` software OpenGL and processes only 11
control steps in 5.71 wall seconds (0.22 simulated seconds). This is too slow for
a useful interactive demo despite successful startup and key handling.
Evidence: `outputs/demo-viewer-performance.json`.

**Decision:** benchmark disabling shadows/reflections in the viewer. This changes
only rendering, not meshes, collision masks, physics, observations, policy or
control dt. Keep the full original scene/floor. Do not claim real-time playback
from GPU training throughput or native inference speed alone.

## D018 Final viewer fix and completed deliverable

**2026-11-07 00:43 UTC.** Disabling shadows/reflections alone improves the software
viewer from 1.93 to 9.18 steps/s, but still couples simulation speed to rendering.
Use elapsed-wall-time accumulation to perform fixed 20 ms policy updates between
frames. Cap catch-up at 250 ms to keep a paused or overloaded window responsive.
Reset clears accumulated time; falls still freeze until reset. Preserve all
physics substeps, observations, action transformations, model geometry and policy.
The renderer changes lighting effects only; the scene floor remains unchanged.

The final real-window check advances 9.98 simulated seconds during the ten-second
viewer loop; total measured wall time including rendering setup is 10.67 s.
It renders about 9.56 frames/s on llvmpipe, so motion is near real time but visually
less smooth than GPU rendering. Evidence: `outputs/demo-viewer-final.json` and
the earlier `demo-viewer-performance.json`/`demo-viewer-fast-check.json` comparisons.
Lint and format checks pass after the change. No extra PPO training was needed.

All four final videos finished and decode with the expected duration and 25 fps:
three seconds each for forward, left turn and right turn; fifteen seconds for
the continuous stand/forward/forward+turn/lateral/stop sequence. Sampled frames
show upright stepping, turning and stopping. Reports and videos are under
`outputs/demo-video/`. The selected export, normalization, settings, mappings,
lineage and evaluation reports are packaged in `outputs/op3-policy.zip` and can
be loaded from `outputs/op3-policy/`. Generated artifacts remain outside Git.

## End-of-training experiment summary

The requested simple baseline now has a saved policy and runnable keyboard demo.
The selected policy inherits **27,095,040 transitions**: 15,052,800 initial plus
12,042,240 refinement steps. Both training stages use seed 0, standard Brax PPO,
512 environments, fixed physical resets and the same rewards/physics. The
refinement changes command sampling and warm-starts weights/normalization;
optimizer/RNG state is restarted. It is not an exact training resume.

| Experiment | Observation | Resulting decision |
| --- | --- | --- |
| Full-mesh MJX-JAX compilation | Excessive compilation cost for this machine | Use official MJX Warp; disable only authorized head/arm collision participation |
| Newest JAX/Brax combination | Removed JAX replication API prevented PPO | Pin a compatible locked stack and verify GPU updates/export reload |
| Uniform-command baseline | Translation learned; sustained left in-place yaw remained near zero | Refine with balanced standing, axial and combined command modes |
| Refinement 9.03M / 12.04M / 15.05M | Settled left/right yaw +0.457/−0.285, +0.429/−0.404, +0.019/−0.446 rad/s | Choose 12.04M for both turn directions; reject latest despite good aggregate return |
| Selected policy validation | All native ten-second cases and 50-second changes survive; GPU MJX behavior is comparable | Package selected policy for native interactive inference |
| Software-rendered viewer | Rendering originally slowed simulation to ~2 control steps/s | Disable costly effects and decouple fixed control updates from frame rate |

The selected native policy settles near +0.150/−0.144 m/s for forward/backward
commands ±0.15, +0.084/−0.076 m/s for lateral ±0.09, and +0.429/−0.404 rad/s for
yaw ±0.48. Its continuous changes have planar/yaw RMSE 0.0171 m/s / 0.0878 rad/s,
including transitions. Forward movement has repeated foot lift/landing and low
but nonzero contact slip, not merely sliding or standing under a motion command.
No foot-air-time, clearance or slip reward, gait generator, terrain curriculum,
physics randomization or mesh decimation was added.

Limitations remain: short rapid steps, yaw drift, asymmetric small turns, some
slip, and small foot unloading while standing. A +0.4 yaw request undershoots
near +0.208 rad/s. High-speed backward motion introduces unwanted yaw. Individual
command limits and one reversal sequence passed eight-second segments; simultaneous
extreme commands and long-duration robustness are not exhaustively established.
These are deterministic, fixed-start default-floor simulation demonstrations,
not physical-robot transfer evidence. Native and MJX contact trajectories differ
numerically even when overall behavior agrees. Software display rendering is
approximately ten fps despite near-real-time simulation.

The sampler comparison is one lineage with more optimization and no independent
ablation. It supports the practical checkpoint choice, not a causal claim that
balancing alone produced the improvement. Longer training was not monotonically
better; directional held-command tests exposed a regression hidden by average
return. Future work should start from the saved evidence and those limitations,
not from a claim that all command tracking is solved.

## D019 Contact-slip penalty and collision hull reduction

**2026-11-07, user-requested experiment.** Implemented in `e28b286`. The earlier
baseline's backward slip and the user's request motivate a small additional
penalty. Use the actual foot/floor contact-point tangential velocities, including
rotation, rather than foot-body-origin speed. MuJoCo contact sensors provide four
fixed contact slots for each of the four original foot boxes. Within each foot,
normal-force-weight the squared tangential speed; sum the two foot averages,
excluding feet carrying at most 1 N. Weight −5 and multiply by control timestep
once. Penalize slip during standing too, without penalizing airborne feet. These
reward-only sensors do not change observations, dynamics or actuator mappings.
Old exports/configs load with zero slip weight, preserving their objective.

Also expose `collision_maxhullvert`, using MuJoCo's compiler to reduce collision
hulls in memory. Original source assets, visual meshes, explicit inertias, foot
boxes, floor, contact masks, friction, actuator settings and control timing stay
unchanged. Hulls approximate mesh contact surfaces and the cap is saved in each
export; native inference and MJX use the same preparation. No new randomization,
terrain, gait generator, or custom learning algorithm was introduced.

Four runs start independently from the selected `outputs/op3-policy` export.
All use seed 0, balanced commands, 512 environments, batch 128 × 4, unroll 20,
32 evaluation environments and three evaluations. Each requests 1M transitions
and completes 1,003,520. Other settings match the selected policy. Warm starts
restore actor, critic and normalization; optimizer, RNG and step count restart.

| Run | Start UTC | Slip weight | Hull cap |
| --- | --- | ---: | ---: |
| `slip-hull-full` | 20:19:24 | −5 | −1 (original) |
| `slip-hull-16` | 20:22:04 | −5 | 16 |
| `slip-hull-8` | 20:24:38 | −5 | 8 |
| `slip-control-zero` | 20:27:01 | 0 | −1 |

The first trial ran the implementation before its commit; the later runs record
`e28b286`. Run provenance flags the workspace as dirty (including pre-existing
untracked `.obsidian/`). Generated outputs and that unrelated directory were not
committed. Each run has resolved settings, versions, mapping, reward metrics,
checkpoints and a completion record under `outputs/<run>/`.

### Collision timing

| Hull cap | Total vertices in 15 active collision hulls | Trainer elapsed | Median steady transitions/s |
| --- | ---: | ---: | ---: |
| Original | 1,973 | 122.6 s | 25,423 |
| 16 | 240 | 124.4 s | 25,852 |
| 8 | 120 | 126.8 s | 25,174 |

Elapsed time comes from `finished.json` and includes PPO compilation, evaluations
and exports, starting after environment construction. Steady throughput is the
median `episode/sps`, excluding the first interval after initialization and the
mid-run evaluation (steps 51,200 and 512,000). The 16-vertex cap changes this
median by +1.7%; the 8-vertex cap changes it by −1.0%. Neither demonstrates a
training-time decrease. These are single runs on the same RTX 3070 Laptop GPU,
not a repeated performance study; small differences are within observed timing
variation. Most walking contacts use the unchanged foot boxes, so reducing mesh
vertices need not reduce the main contact-solving work. This is an interpretation,
not a measured bottleneck profile. Keep the original hulls as the default.

### Policy behavior

All four final exports were reloaded and evaluated in native MuJoCo with seed
123, seven independent 10 s commands and the continuous 50 s change sequence.
All survived. Reports and trajectories are in `outputs/<run>-eval/`.

The following means give equal weight to the eight cases; slip additionally
averages both feet's supported contact-slip speeds. RMSE includes transients.
Slip is the existing diagnostic in m/s, not the squared training reward.

| Policy | Mean slip (m/s) | Mean planar RMSE (m/s) | Mean yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: |
| Existing selected policy | 0.005541 | 0.019933 | 0.122825 |
| Extra 1M, zero slip weight | 0.005399 | 0.018398 | 0.121489 |
| Extra 1M, slip −5, original hulls | 0.004697 | 0.020265 | 0.111091 |
| Extra 1M, slip −5, cap 16 | 0.004746 | 0.017926 | 0.113663 |
| Extra 1M, slip −5, cap 8 | 0.004823 | 0.020333 | 0.144029 |

Against the matched zero-weight control, the original-hull slip trial reduces
mean slip by 13.0% and backward slip by 34.9% (0.01266 → 0.00824 m/s).
Planar RMSE worsens by 10.2%, while mean yaw RMSE improves by 8.6%. Improvements
are not uniform: standing and leftward slip increase. Its settled forward speed
is 0.138 m/s for a 0.15 request. This is useful evidence for trying the penalty,
not a claim of uniformly better locomotion or a multi-seed ablation.

The reduced-hull trials have directional regressions: cap 16 turns right at
−0.322 rad/s for −0.48 requested; cap 8 turns left at +0.091 for +0.48 requested.
The selected policy's corresponding values are −0.404 and +0.429. These outcomes
do not isolate hull geometry from divergence during training, but they rule out
promoting these particular exports. Keep `outputs/op3-policy/` as the selected
demo. The slip trial remains runnable with
`uv run op3-play outputs/slip-hull-full`; fresh runs default to slip weight −5,
while the original recipe explicitly uses zero to reproduce the old baseline.

## D020 Hold-to-ramp keyboard commands and validation

**2026-11-07.** Track GLFW press/release state and ignore OS repeat events. At each
20 ms control tick, approach the held direction's trained bound over 0.5 s, or
return from the bound to zero over 0.35 s when released. Opposite held keys cancel
that axis. Space, R, and focus loss clear commands and held keys immediately;
release/repress resumes input after a clear. W/S, A/D, Q/E and Escape retain their
directional/exit meanings. The title and overlay show the changing command.

`scripts/check_slip_controls.py` passed all six ramp/release directions, repeat
independence, opposing keys, reset, and focus loss. It also checks contact slip
against an independent native Jacobian calculation with translation and rotation,
zero airborne penalty, and invariance of masses/inertias, joint/actuator settings,
foot boxes and floor under hull caps 16 and 8.
Legacy mappings without a hull-cap field still load; a different inference hull
cap is rejected.

GPU model/reset/step/wrapper checks passed for original and 8-vertex hulls:
`outputs/slip-checks.json` and `outputs/slip-hull-8-parity.json`. In the one-second
small-action native/MJX comparison, maximum qpos discrepancy was 5.60e−6 and qvel
discrepancy 6.42e−5. These are limited parity checks, not a claim that long contact
trajectories coincide. All four actual PPO jobs completed with finite metrics,
zero reported buffer overflow and reloadable policy exports.
An additional same-state translating/rotating-foot check compared recomputed GPU
contact sensors against native MuJoCo. The largest slip-term discrepancy was
4.7e−7 (m/s)² (`outputs/slip-sensor-parity.json`).

Actual X11 press/release events sent only to a newly launched viewer verified all
six directions reaching the trained bound and returning to zero after release.
Intermediate title snapshots on subsequent keys show gradual ramps (the first
W snapshot preceded its first displayed update). R and Escape also worked, with
clean exit. Evidence is in
`outputs/held-viewer.json` and `outputs/held-viewer-check.log`. The checks used the
existing selected policy, so it benefits from the new controls without retraining.

Rendered the original-hull slip trial's forward, backward and left-turn cases
(3 s each), plus a 15 s standing/forward/turn/lateral/stop sequence. All survived.
Sampled frames were inspected alongside the trajectories: the robot remains
upright, changes heading and retains its short, rapid, low-clearance steps.
Videos, frame sheets and the report are in `outputs/slip-trial-video/`. Rendering
24 simulated seconds took 317.5 s with this display setup; it was performed after
the PPO timing trials and is not included in their timing comparison.

## D021 Air time without a prescribed rhythm

**2026-11-07.** Following the Wolfgang reward comparison, add a small touchdown
reward without its phase reward or swing-duration offset/cap. At each 2 ms physics
step, sum normal floor force per foot; >1 N means support. Once a foot has seen
support during a motion command, accumulate unsupported seconds and pay that
duration once on its next supported sample. Sum events across the control
interval, multiply by weight +2 and the common 0.02 s reward scaling. A 0.1 s
completed swing pays 0.004. There is no minimum or target duration, cadence, phase,
height target, alternation bonus, or reward per step count. Equal total completed
unsupported time gives equal raw reward, although PPO discounting still favors
earlier payment.

Motion means any command exceeds 5% of its trained bound, including pure yaw.
Standing clears timers and eligibility; resets do too. The initial landing cannot
earn a bonus, and physical failure suppresses it. An unfinished swing earns no
terminal bonus. This rewards unloading, which need not imply useful foot lift;
measure clearance, simultaneous loss of support, slip and tracking as well.

Matched trials start from the same `outputs/slip-hull-full` checkpoint
`000001003520`, with original hulls and slip weight −5. `outputs/air-time-2` uses
air-time weight +2; `outputs/air-time-control-0` uses zero. Each requests 1M
transitions, seed 0, 512 environments and three PPO evaluations, retaining the
source's balanced commands and other hyperparameters. Actor, critic and
normalization are restored; optimizer/RNG/step count restart. Implementation is
commit `622ada9`. Configuration details are in `CONFIG.md`; the selected demo
remains separate from these experimental exports.

Both runs completed 1,003,520 transitions on the RTX 3070 Laptop GPU. Training
plus its evaluations took 123.4 s for +2 and approximately 123 s for zero. Final
PPO evaluation episodes all reached 500 steps; logged metrics were finite, with
zero buffer overflow. The +2 run's mean air-time contribution was 0.2481 per
episode, versus 29.0957 linear-tracking and 9.1952 yaw-tracking reward. These
sampled training evaluations are separate from the fixed-command comparison.

Focused JIT checks cover initial landing exclusion, one-time touchdown payment,
independent feet, sub-control-step events, standing transitions, equal total air
time across different step counts, diagnostic partial-swing exclusion, legacy
config loading, and air time with slip disabled. GPU reset/step checks verify
history clearing on truncation/failure and no positive reward on failure. The
one-second native/MJX small-action comparison gave maximum qpos discrepancy
5.48e−6 and qvel discrepancy 9.93e−5 (`outputs/air-time-checks.json`). Ruff and
format checks passed. Reward history is absent from observations, so the saved
policy input/action mapping and inference timing remain unchanged.

### Matched command-suite results

Both exports survived all eight native cases: seven independent 10 s commands
and a 50 s standing/forward/turn/lateral/stop sequence. Tracking/slip entries below
average cases equally, including standing and changes. Air-time/clearance entries
average the two feet across the six independent nonzero-command cases, after
discarding the first second. Durations use complete swings at 50 Hz, not the
reward's 500 Hz sampling; clearance is the average of each foot/case maximum.

| Metric | Source slip policy | Extra 1M, air weight 0 | Extra 1M, air weight +2 |
| --- | ---: | ---: | ---: |
| Planar velocity RMSE, m/s | 0.02027 | 0.01722 | 0.01807 |
| Yaw-rate RMSE, rad/s | 0.11109 | 0.09177 | 0.09825 |
| Supported contact slip speed, m/s | 0.004697 | 0.004105 | 0.004385 |
| Mean completed unsupported duration, ms | 79.7 | 85.3 | 81.1 |
| Unsupported fraction per foot | 41.0% | 44.6% | 40.5% |
| Mean peak sole clearance, mm | 5.89 | 5.81 | 5.18 |
| Both feet unsupported | 0.259% | 0.037% | 0.111% |

The +2 trial does not establish a useful air-time improvement at this weight and
training budget. Relative to the matched control, planar RMSE is 4.9% higher,
yaw RMSE 7.1% higher and slip 6.8% higher; mean completed air time is 4.9% shorter.
Forward settled speed is 0.1465 m/s for a 0.15 request, versus 0.1502 for control.
Left-turn tracking improves (0.4763 versus 0.4363 rad/s for +0.48), but right-turn
tracking worsens (−0.3578 versus −0.4606 for −0.48). The +2 standing case still
has small foot motions despite receiving no air-time reward at zero command.

Inspected saved trajectories and representative forward/backward/left-turn
frames, selected at maximum left-foot clearance. The robot remains upright with
low foot lift; these samples do not show a transition to long steps. All measured
completed swings in these suites are at most 0.18 s for +2 and 0.20 s for control.
The +2 case's brief simultaneous loss of support occurs during backward motion
(0.67% of settled samples), rather than persistent flight. This is a single-seed
comparison; stochastic training/numerical divergence prevents attributing every
directional change to the reward.

Reports and trajectories are in `outputs/air-time-2-eval/` and
`outputs/air-time-control-0-eval/`; the combined table is
`outputs/air-time-comparison.json` and inspected frames are
`outputs/air-time-rollout-frames.jpg`. Keep `outputs/op3-policy` as the selected
demo. The requested air-time term is available with +2 as the new-run default;
existing saved configs retain their original weights. Try this experimental
export with `uv run op3-play outputs/air-time-2`. No rhythm or clearance target
was added to obtain these results.

## D022 Stronger posture regularization while standing

**2026-11-07.** The requested "zero action input" means exactly zero requested
`[vx, vy, yaw_rate]`, not zero policy joint actions. Add a separately logged
`standing_posture` term, equal to summed squared leg-joint deviation from nominal
only at that command. Test weight −0.45 on top of the existing −0.05 posture
weight: total −0.5 standing (10×), unchanged −0.05 moving. Common 0.02 s reward
scaling applies once. The new term defaults to zero for fresh and old configs.

`outputs/standing-posture-10x` and shared control `outputs/standing-lift-control`
both start from `outputs/air-time-2` checkpoint `000001003520`. Each requests 1M
transitions with three evaluations, seed 0, 512 training/32 evaluation worlds,
original hulls, balanced commands, slip −5 and air time +2. Actor, critic and
normalization are restored; optimizer/RNG/step count restart. The control changes
neither posture nor reset distribution. Implementation: `cf18745`.

After 1,003,520 transitions, standing summed squared joint error is 0.54590 rad²
versus 0.57637 for the matched control (5.3% lower). The equivalent per-joint RMS
angle is 12.22° versus 12.56°. After the command-change sequence returns to zero,
settled joint RMS is 12.34° versus 13.24° (6.8% lower). This is a modest improvement
toward neutral, not neutral-pose standing. The original source's standing error
was 0.60916 rad²; comparing only against it would overstate the effect.

All eight standard cases survive, but the tradeoffs are mixed: mean planar RMSE
is 2.3% higher, yaw RMSE 3.9% higher and contact slip 21.1% higher than control.
Forward speed is 0.1502 m/s for a 0.15 request. Right yaw improves to −0.4702
versus −0.3572 rad/s for −0.48, while left yaw weakens to 0.3757 versus 0.4135
for +0.48. Keep this an experimental export, with the additional penalty default
still zero. Inspect with `uv run op3-play outputs/standing-posture-10x`.

## D023 Randomly selected raised-foot starts

**2026-11-07.** In the separate `outputs/lift-reset-25` experiment, retain all
source reward settings and sample 75% standing, 12.5% left raised, 12.5% right
raised. This explicitly requested exception to fixed standing resets does not
randomize masses, friction, forces, sensors or terrain. Selection uses a seeded
stream derived separately from commands and PPO exploration. It changes only
the initial state, with no imposed gait clock or repeated swing trajectory.

Mirror two fixed poses: add 0.45 rad knee flexion, half the corresponding
hip/ankle pitch offsets, and 0.15 rad hip/ankle roll toward the supporting side.
Resolve base height from the original sole boxes: raised clearance 20.094 mm,
supporting clearance 0.500 mm. Both poses stay within the policy's ±0.5 rad
target range and have zero initial velocity, upright base orientation and no
penetrating contacts. Actuator targets initially match the pose; nominal action
and observation references remain the original standing configuration. Timers
start unarmed, so the initial raised-foot landing cannot earn an air-time bonus.

The source, budget, PPO settings and shared control match D022; its additional
standing penalty stays zero. Both requested features remain opt-in. Evaluation
and play default to standing for a fair locomotion comparison; `--reset-pose
left` / `right` additionally test recovery. A raised start exposes recovery and
single-foot support, but does not guarantee the policy learns to initiate lifts.

Focused checks passed on CPU/GPU. Across 2,048 seeded choices, counts were
1,532 standing, 255 left, 261 right, reproducibly. Both held poses remained
upright for one native second. Native/MJX half-second zero-action rollouts from
left/right raised starts had maximum qpos discrepancies 5.05e−7 / 8.11e−7 and
qvel discrepancies 2.24e−5 / 6.65e−5. Batched reset/timeout and observation
consistency checks passed. Selection uses `mjx.Data.where` to preserve shared
Warp contact buffers; generic JAX conditional selection is unsuitable here.
Existing default-reset/fall, air-time, slip, mapping and keyboard checks also
passed. Evidence: `outputs/standing-reset-checks-{cpu,gpu}.json` and
`outputs/standing-reset-baseline-checks.json`.

### Matched results and recovery checks

All three PPO jobs completed 1,003,520 transitions with finite logged metrics,
zero reported buffer overflow and 500-step final evaluation episodes. Resolved
configs differ only in `reward.standing_posture` for D022 or
`lift_reset_probability` for D023. Each exported policy then survived all eight
standard native cases from standing: seven 10 s commands and a 50 s command-change
sequence. Tracking/slip means below weight those cases equally. Air-time and
clearance means cover the two feet in six independent nonzero-command cases,
excluding the first second; clearance averages the per-foot/case maximum.
Air-time diagnostics sample at 50 Hz and omit incomplete swings.

| Metric | Matched control | Standing posture 10× | Raised resets 25% |
| --- | ---: | ---: | ---: |
| Standing posture error, rad² | 0.57637 | 0.54590 | 0.49856 |
| Standing joint RMS error, degrees | 12.56 | 12.22 | 11.68 |
| Settled stop joint RMS error, degrees | 13.24 | 12.34 | 13.44 |
| Planar velocity RMSE, m/s | 0.01727 | 0.01767 | 0.01760 |
| Yaw-rate RMSE, rad/s | 0.09678 | 0.10053 | 0.08157 |
| Supported contact slip, m/s | 0.004293 | 0.005197 | 0.004923 |
| Mean completed air time, ms | 79.5 | 79.9 | 86.2 |
| Mean peak sole clearance, mm | 4.93 | 4.69 | 4.32 |

Raised resets increase measured air time by 8.4% but reduce peak clearance by
12.4% against the matched control. Yaw RMSE improves 15.7%, while planar RMSE and
slip worsen 1.9% and 14.7%. Standing posture also improves, but its settled stop
posture does not. The additional reset exposure does not establish higher foot
lifting in this budget. It should not be interpreted as solving the low-clearance
gait because it produces longer unsupported intervals.

Both the control and raised-reset policy additionally survived all eight forced
recovery cases each: left/right raised starts under stand, forward, left yaw and
right yaw, each for 10 s. Thus successful recovery alone is not evidence of a new
capability from the raised-reset training. Inspected initial reset frames and
standing/forward rollout frames alongside the trajectories: both initial 2 cm
lifts are visible, while subsequent walking retains low clearance. No cadence,
phase or foot-height reward was added. These are single-seed continuations;
directional changes may also reflect stochastic training/numerical divergence.

Reports/traces are in `outputs/{standing-posture-10x,lift-reset-25,standing-lift-control}-eval/`.
Raised-start reports use the additional `-left-eval` / `-right-eval` suffixes for
the latter two runs. Combined metrics: `outputs/standing-lift-comparison.json`.
Inspected images: `outputs/standing-lift-rollout-frames.jpg` and
`outputs/standing-lift-reset-poses.jpg`. The selected demo remains
`outputs/op3-policy`; both new options remain disabled by default. Inspect the
raised-start export with `uv run op3-play outputs/lift-reset-25 --reset-pose left`.

## D024 Full training with stronger upright reward and air time

**2026-11-07.** Start a fresh 30M-transition run rather than another continuation
of the earlier policy. Increase upright weight from +0.3 to +1.0, retain the
original −0.05 posture weight with no extra standing penalty, and enable the
new +2 air-time reward from the start. Retain slip −5, original collision hulls,
and fixed nominal standing resets. No cadence, foot-height target or reset mix
is introduced. This tests the combined recipe; it is not an isolated ablation
of upright weight.

The complete typed configuration is `configs/upright_air_time.json`. Use seed 0,
balanced commands, 512 training/32 evaluation environments, 16 evaluations,
128 × 4 minibatches, unroll length 20 and the existing PPO/network settings.
Actor, critic and observation normalization start fresh; no checkpoint is loaded.
Run directory: `outputs/upright-air-time-full`. Check fixed-command behavior and
foot clearance as well as training return before selecting an export.

The run started at 21:52:48 UTC on the RTX 3070 Laptop GPU using commit
`ed11acc`; `run.json` records no initialization checkpoint. The 2,007,040 and
4,014,080 exports survived short standing/forward/left-yaw/right-yaw checks but
barely moved. At 10,035,200 transitions, all eight standard 10 s/segment cases
survived, yet settled forward speed was only 0.00169 m/s for 0.15 requested and
left/right yaw was 0.0407 / 0.00066 rad/s for ±0.48 requested. This was still
largely a standing solution despite a sampled training return near 40.
Evidence: `outputs/upright-full-{2m,4m,10m}-eval/`. At 16,056,320 transitions,
forward speed reached 0.1108 m/s, but the command-change case fell after 32.04 s
and several other directions barely moved. At 20,070,400, all cases survived:
forward reached 0.1178 m/s and left yaw 0.3320 rad/s; backward, rightward and
right-yaw responses remained weak. Those reports use the `16m` / `20m` suffixes.

### Completed full run

Completed **30,105,600 transitions** in 1,946.8 s (32.4 min trainer wall time,
including its evaluations). Final export:
`outputs/upright-air-time-full/checkpoints/000030105600`. Config roundtrip and
comparison against the committed recipe passed. All logged metrics were finite
with zero reported physics-buffer overflow. Final sampled PPO evaluation length
was 495.8/500 steps, with 2/32 episodes reporting physical termination; success
must be judged separately from these aggregate training metrics.

The 28,098,560 and final exports both survived the full native suite: seven
independent 10 s commands and a 50 s command-change sequence. Tracking and slip
means below weight the eight cases equally. Body tilt is the angle of body up
from world up, averaged by case after its first second. Foot statistics cover
six independent moving cases after the first second; peak clearance is the mean
of each foot/case maximum. Durations omit incomplete swings and use the 50 Hz
trajectory, rather than the reward's 500 Hz contacts.

| Metric | Existing selected demo | 28M fresh run | Final 30M fresh run |
| --- | ---: | ---: | ---: |
| Planar velocity RMSE, m/s | 0.01993 | 0.04050 | 0.03216 |
| Yaw-rate RMSE, rad/s | 0.12283 | 0.23000 | 0.26930 |
| Mean body tilt, degrees | 5.36 | 6.79 | 6.77 |
| Standing leg-joint RMS error, degrees | 10.88 | 9.48 | 11.26 |
| Supported contact slip, m/s | 0.005541 | 0.009004 | 0.006235 |
| Mean peak sole clearance, mm | 6.52 | 8.49 | 6.84 |
| Mean completed air time, ms | 77.5 | 147.0 | 94.1 |
| Foot/case pairs with completed swings, of 12 | 12 | 12 | 9 |
| Both feet unsupported | 0.259% | 5.259% | 5.556% |

The final policy's settled forward/backward speeds are +0.1272/−0.1344 m/s for
±0.15 requests, with unwanted yaw of −0.1393/+0.1911 rad/s. Left/right lateral
speeds are +0.0756/−0.0637 for ±0.09 requests. Left yaw reaches +0.4390 for +0.48,
but right yaw is only −0.00346 for −0.48. The earlier 28M checkpoint turns right
at −0.2848, while its lateral tracking is worse; final reward or final step count
alone cannot identify the best behavior in each direction.

Longer air time is not evidence of a clean walking gait. During backward motion,
both feet are unsupported for 19.3% of settled samples, with both soles more than
0.5 mm clear for 17.3%: this includes actual flight. During rightward motion the
left foot is unsupported for 99.8% of the settled samples, and during right yaw
neither foot completes a swing. Those three missing foot/case duration statistics
are excluded from the 94.1 ms mean. Inspected rendered frames and trajectories
confirm low-clearance stepping mixed with hopping/one-foot behavior. Images are
in `outputs/upright-full-rollout-frames.jpg`.

The same final eight-case suite also survived in MJX. Its mean planar/yaw RMSE
was 0.03521 m/s / 0.25255 rad/s; settled right yaw was −0.00371 rad/s, confirming
the failed turn in training physics too. Backward motion had both feet unsupported
in 20.9% of settled samples. Native and MJX trajectories are not identical
(left yaw was +0.4390 versus +0.5022), but they show the same major limitations.
Final reports and trajectories are `outputs/upright-full-30m-eval/` and
`outputs/upright-full-30m-mjx/`; the earlier export uses `upright-full-28m-eval/`.
Combined metrics and per-case tilt measurements are saved in
`outputs/upright-full-comparison.json`.

This combined recipe did not improve overall command tracking or mean uprightness
over the selected demo. It combines fresh initialization, balanced commands and
the current foot rewards with stronger upright weight, so this single-seed run
does not isolate which choice caused the poorer result. Keep `outputs/op3-policy`
as the selected demo and the new run as an experimental export. Reproduce with
`uv run op3-train --config configs/upright_air_time.json --output OUTPUT`, or inspect
with `uv run op3-play outputs/upright-air-time-full`.

## D025 Asymmetric critic from an earlier original checkpoint

**2026-11-07.** The user requested a critic with two values each for foot height,
time since contact and binary contact state, retaining the actor and the latest
experiment's rewards (upright +1, slip −5, air time +2). Initialize from the
original baseline's 7,024,640-transition export, before its selected 15M export
and balanced refinement. This changes initialization as well as critic inputs
relative to D024; it is not an isolated critic ablation.

Use `configs/asymmetric_critic.json`, matching D024's 30M-transition budget,
seed 0, balanced command distribution, 512 environments, 32 evaluation
environments and 16 evaluations. Default floor, original hulls, fixed standing
resets, actor architecture, action mapping, PPO and reward settings are unchanged.
The original actor parameters and running statistics carry over exactly. The
critic's first layer gains six zero rows; its initial function is preserved.
New scaled critic features start with zero-mean/unit-variance normalization
priors at the original shared sample count. Optimizer, RNG and step count restart.

The critic's unsupported-time history is independent of command and reward
eligibility, so it remains informative during standing. Both observation
branches and the new timer reset together on termination/truncation. Play loads
only the actor inputs and never needs privileged foot signals.

Run: `outputs/asymmetric-critic`. Per the user's updated instructions in
`AGENTS.md`, report final trainer mean total episode reward alongside previous
runs and provide the play command; leave behavior evaluation to the user.

Validation before the run: Ruff lint/format checks passed. The focused CPU and
GPU checks passed actor parameter/statistic preservation, critic-only input
isolation, initial value-function preservation (float32 matrix products), export
reload, native actor-only inference, sole-height geometry, command-independent
timers, same-action reward parity and batched timeout/reset parity. Logs:
`outputs/asymmetric-checks-{cpu,gpu}.log`. GPU reward comparisons allow float32
roundoff; no reward formulas or weights were changed by the critic feature.

### Interrupted run and continuation

**2026-11-08 00:55 UTC.** At the user's request, continue the interrupted
asymmetric-critic run. The original run stopped at 2,969,600 logged transitions;
its latest complete export is 2,007,040. The 962,560 subsequent transitions were
not saved. `outputs/asymmetric-critic/stopped.json` records the interruption.
Only policy/critic weights and observation normalization were exported, so the
continuation necessarily restarts optimizer, RNG and local step count.

Launch `outputs/asymmetric-critic-continued` from that exact export and the
original saved configuration. Request the remaining 27,992,960 transitions with
15 evaluations (initial plus 14 epochs), retaining the original 2,007,040-step
evaluation/export cadence. Brax rounds this to 28,098,560 new transitions and
30,105,600 total saved asymmetric-critic transitions. The original baseline
warm-start lineage is unchanged; no rewards, networks, physics or commands change.

The detached GPU process writes `outputs/asymmetric-critic-continued.log` and
the usual run artifacts. Launch metadata is in
`outputs/asymmetric-critic-continued-launch.json`. Completion requires its
`finished.json`; report the final `eval/episode_reward` and the new play command
when complete. No post-training behavior suite is requested.

The continuation finished at 28,098,560 new transitions (30,105,600 saved
asymmetric-critic transitions including its source). Final trainer mean episode
return was 48.6393, versus 43.1005 for `upright-air-time-full` under identical
reward weights, and 40.3249 for `balanced` under its older reward definition.
The subsequent viewer diagnostic showed roughly 7° torso tilt at zero command;
policy-derived up matched true world up. This was a short display check.

## D026 Stronger upright reward and contact-gated sole-area shaping

**2026-11-08 UTC.** The user requested increased upright reward, a reward for
maximum foot surface area on the floor only while touching, and a full training
run. Use `configs/upright_contact_area.json`: upright 1→3 and optional sole-area
weight +0.5, retaining all other rewards and the asymmetric critic. Start from
`outputs/asymmetric-critic-continued/checkpoints/000028098560` and run 30M
additional transitions, seed 0, 512 environments, 32 eval environments and 16
evaluations. Optimizer/RNG/local step count restart. Output:
`outputs/upright-contact-area`.

MuJoCo's rigid-foot contacts are discrete, so the term is explicitly a smooth
geometric area estimate, not a physical contact-patch measurement (see the
[MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)). Integrate
`exp(-(height / 0.002)²)` over the union of the two original sole boxes using
80 area-weighted samples per foot. Gate each foot on >1 N actual floor load;
average over loaded feet to avoid a separate two-feet-down bonus. No contact
means zero; deep penetration and sideways/upside-down soles cannot earn a high
score. The default terrain and robot dynamics remain unchanged.

The combined peak planar/yaw tracking weight (4) exceeds upright plus sole-area
weight (3.5). This still needs manual behavior assessment: changing two terms
together is not an ablation, and larger total returns are expected from the
reward changes alone. Keep the separately selected demo unchanged. Compare
trainer final mean total episode returns with reward definitions identified;
provide the play command and leave post-training behavior evaluation to the user.

CPU checks passed union-area accounting (0.00962 m² per foot), contact-force
gating, zero airborne score, flat-over-tilted ordering, one/two-foot normalization,
penetration handling, unchanged model/action mappings and legacy weight=0.
Nominal loaded-standing area fraction was 0.999992.

The same checks passed on GPU, including finite batched MJX reset/step outputs,
no contact-buffer overflow and once-only area reward scaling. A one-ulp GPU
summation difference motivated clamping the normalized area fraction to [0, 1].

A 10,240-transition PPO smoke run with the full 512-environment batch completed
and exported a policy (`outputs/contact-area-ppo-smoke`). Reload preserved the
new reward settings and mapping, and one native inference step produced finite
outputs. Its post-update mean episode return was 68.6532 under the new rewards;
the full experiment still starts from the original latest completed policy,
not the smoke update. Ruff lint/format and diff checks passed.

### Result and user-requested early stop

The user subsequently requested opening the latest checkpoint if reward was
still plateaued after the next checkpoint. At that point, the 22,077,440-step
evaluation was 71.5435. The next evaluation at 24,084,480 scored 71.2868, so it
did not improve. Stop the trainer and open that latest saved export in the
viewer with gravity diagnostics. This applies the user's stopping condition;
one checkpoint difference is not a statistical convergence claim.

The original 30M budget was not completed. The last logged training step before
SIGTERM was 24,678,400, but the latest complete policy remains 24,084,480.
`outputs/upright-contact-area/stopped.json` records the stop and policy-export
limitation. The final saved evaluation occurred after 1,243.7 seconds of trainer
time. No post-training behavior suite was run.

| Run / evaluation | Trainer mean total episode return | Reward definition |
| --- | ---: | --- |
| New run, initial policy | 68.6602 | Upright 3, area 0.5, slip −5, air time 2 |
| New run, latest saved 24,084,480 | 71.2868 | Same new rewards |
| Previous asymmetric continuation, final | 48.6393 | Upright 1, area 0, slip −5, air time 2 |
| Earlier upright/air-time full run, final | 43.1005 | Upright 1, area 0, slip −5, air time 2 |
| Balanced refinement, final | 40.3249 | Upright 0.3, area/slip/air time 0 |

The same-objective gain from initialization is 2.6266. The final area contribution
was 3.9380 versus 0.2266 initially, while upright contribution was 29.6388 versus
29.8012 initially. Thus the logs support a larger sole-area score but do not
establish improved torso alignment. Across the trainer's sampled command episodes,
final mean planar/yaw errors were 0.01631 m/s and 0.09568 rad/s, with mean episode
length 500 steps. Metrics stayed finite with no buffer overflow. Larger returns
than older runs also reflect changed rewards. Manual assessment remains with
the user. Relaunch: `uv run op3-play outputs/upright-contact-area`.

## D027 Train the latest reward configuration from scratch for 40M steps

**2026-11-08 UTC.** The user requested a fresh policy for 40 million steps using
the latest weights. Interpret weights as the latest reward coefficients:
upright +3, contact area +0.5, slip −5 and air time +2, with every other reward
coefficient/width unchanged. Actor, critic, observation normalization and
optimizer initialize from scratch; no source policy is loaded.

Recipe: `configs/upright_contact_area_scratch_40m.json`. Output:
`outputs/upright-contact-area-scratch-40m`. Seed 0, 512 training environments,
32 evaluation environments, 64/70-input asymmetric actor/critic, balanced
commands, fixed standing resets and unchanged model/physics/control settings.
Request 40,000,000 transitions and 21 evaluations to retain the preceding
2,007,040-step evaluation/export cadence. Expected full-batch total: 40,140,800.
This new run's budget supersedes the previous experiment's completed early-stop
instruction. Report the trainer's final average episode return and play command;
leave post-training behavior assessment to the user.

### Completed 40M run

Finished all 40,140,800 transitions in 2,425.3 seconds (40.4 minutes) on the
RTX 3070 Laptop GPU. `run.json` confirms `initialize_from="None"`, and the saved
configuration matches the recipe. Final export:
`outputs/upright-contact-area-scratch-40m/checkpoints/000040140800`.

| Run / evaluation | Trainer mean total episode return | Reward comparison |
| --- | ---: | --- |
| Scratch 40M, initial untrained policy | 63.1253 | Latest weights |
| Scratch 40M, final | 71.7568 | Latest weights |
| Sole-area warm start, latest saved 24M | 71.2868 | Identical weights |
| Earlier asymmetric continuation, final | 48.6393 | Upright 1, area 0; different rewards |
| Earlier upright/air-time full run, final | 43.1005 | Upright 1, area 0; different rewards |

The final trainer evaluation averaged 500 control steps per episode, with mean
per-step planar velocity error 0.01897 m/s and yaw error 0.09900 rad/s. Weighted
upright and sole-area episode contributions were 29.8614 and 4.1913. All logged
metrics were finite with no contact-buffer overflow. The final return exceeds
the preceding warm-start export by 0.4700 under the same reward definition; this
single-run trainer comparison does not establish better gait or standing posture.
No post-training behavior suite was run.

Completion is recorded in the run's `finished.json`; comparison data is in
`outputs/scratch-40m-comparison.json`. Manual launch:
`uv run op3-play outputs/upright-contact-area-scratch-40m`.

## D028 Two-second cumulative squared-torque symmetry, scratch 30M

**2026-11-08 UTC.** The user reports asymmetric walking and requests a penalty
for the left/right difference in cumulative torque squared over two seconds,
then 30M steps from scratch. Compare each of the six mirrored leg-joint pairs:
sum the absolute differences of their sliding squared-torque integrals. This
allows alternating effort in time while preventing cancellation across joints.
Use actual unit-gear actuator torques sampled at 500 Hz; update the two-second
window and pay its penalty at 50 Hz. See `CONFIG.md` for units and reset rules.

Choose weight −0.1 for this first trial (default disabled). This costs 0.1 reward
per second for each 1 (Nm)² s of summed integral imbalance. It is an experiment
coefficient, not a tuned result. Retain the global effort penalty −0.002 so
adding unnecessary torque to the less active leg still has a cost. The new term
is not command-gated, so it also applies to standing and turns. Matching torque
usage alone does not establish symmetric foot placement or improved walking.

Recipe: `configs/torque_symmetry_scratch_30m.json`. Output:
`outputs/torque-symmetry-scratch-30m`. All other rewards/settings match D027:
upright +3, area +0.5, slip −5, air time +2, asymmetric 64/70-input actor/critic,
balanced commands, fixed standing resets, seed 0, 512 training environments and
32 evaluation environments. No source checkpoint; parameters, normalization,
optimizer and RNG start fresh. Request 30,000,000 transitions and 16 evaluations,
rounding to 30,105,600 transitions. Total returns now include a new negative term
and are not directly comparable with the preceding 71.7568 final mean return.
Report trainer evaluation results and leave behavioral assessment to the user.

Validation passed on CPU and GPU: independent 305-step rolling-window oracle,
alternating effort cancellation, per-joint isolation, exact expiry, configuration
validation, named unit-gear mapping, physics-substep integration, once-only reward
scaling, standing/command-change history and batched fall/timeout reset. The GPU
log is `outputs/torque-symmetry-checks-gpu.log`. Ruff lint/format and diff checks
passed. A fresh 10,240-transition GPU PPO smoke run completed and exported
`outputs/torque-symmetry-ppo-smoke`; reload preserved settings/mapping and ten
native CPU inference steps were finite. The sandboxed reload stalled, so it was
stopped and rerun successfully outside the restricted runtime. This smoke policy
is not used to initialize the full run.

### Completed 30M run

Finished 30,105,600 transitions in 1,905.3 seconds (31.8 minutes) on the RTX 3070
Laptop GPU. Final export:
`outputs/torque-symmetry-scratch-30m/checkpoints/000030105600`.
The saved run configuration exactly matches the recipe, `initialize_from="None"`
confirms fresh initialization, and `finished.json` records completion. Final
export configuration/mapping and step metadata match the run. All logged
metrics are finite, with no physics buffer overflow. Final training throughput
was approximately 16.9k transitions/s.

| Run / evaluation | Trainer mean total episode return | Reward comparison |
| --- | ---: | --- |
| Torque symmetry, initial untrained policy | 62.6075 | New penalty −0.1, two-second window |
| Torque symmetry, final 30,105,600 | 34.2463 | Same new rewards |
| Previous scratch 40M, final | 71.7568 | No torque-symmetry penalty; otherwise identical rewards |
| Sole-area warm start, latest saved 24M | 71.2868 | No torque-symmetry penalty; otherwise identical rewards |
| Asymmetric continuation, final | 48.6393 | No torque penalty; upright 1, area 0; different rewards |

The new penalty contributed −27.8922 to the final mean episode return, compared
with −0.5178 for the initial untrained policy. Final episodes averaged 500
control steps; mean per-step planar/yaw errors were 0.06158 m/s and 0.20261 rad/s.
The preceding 40M scratch run's corresponding errors were 0.01897 m/s and
0.09900 rad/s. The new run's final return excluding only its torque penalty was
62.1385; that arithmetic is not a retraining or evaluation under the old objective.

Optimization was uneven: mean return fell to 19.3953 at 20M, recovered to
35.4517 at 26M, regressed to 25.9054 at 28M, then finished at 34.2463. This run
does not show improved command tracking in the trainer metrics. The additional
negative term changes total-return comparisons, and torque-usage symmetry does
not by itself establish gait symmetry. Keep the selected demo unchanged.
No post-training behavior suite was run; leave gait assessment to the user.

Comparison data: `outputs/torque-symmetry-30m-comparison.json`. Manual launch:
`uv run op3-play outputs/torque-symmetry-scratch-30m`.

## D029 Torque-symmetry warm start from the 40M scratch policy, additional 20M

**2026-11-08 UTC.** The user requested training with the torque penalty starting
from the previous 40M scratch run for 20M steps. Initialize from the exact final
export `outputs/upright-contact-area-scratch-40m/checkpoints/000040140800`.
Restore actor, critic and observation normalization; optimizer, RNG, environment
histories and local step count restart. This tests adaptation of the previously
trained policy to the new objective, instead of learning it from scratch.

Recipe: `configs/torque_symmetry_warm_start_20m.json`. Output:
`outputs/torque-symmetry-warm-start-20m`. Preserve D028's reward formula/weights,
including the −0.1 torque-symmetry weight and two-second sliding window, and all
other environment/network/PPO settings. Request 20,000,000 additional transitions
with 11 evaluations (initial plus ten epochs), rounding to 20,070,400 additional
transitions. Seed 0, 512 training/32 evaluation environments; 2,007,040-step
evaluation/export cadence. The source lineage totals 60,211,200 saved transitions.

Compare the final trainer mean episode return against D028's 34.2463 under
identical rewards. The source 40M run's 71.7568 excludes the new torque penalty;
the initial warm-start evaluation will measure it under the new objective.
Report the final reward and play command; leave post-training gait assessment
to the user. No application-code or reward changes are needed for this run.
