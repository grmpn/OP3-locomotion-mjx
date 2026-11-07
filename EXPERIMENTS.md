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
