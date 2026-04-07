# GigaLearnCPP — Wiki

Full reference for everything in the project: rewards, obs builders, config options, CLI flags, training setup, and more.

---

## Table of Contents

1. [How It Works (Overview)](#how-it-works)
2. [ExampleMain.cpp Walkthrough](#examplemaincpp-walkthrough)
3. [Command-Line Flags](#command-line-flags)
4. [LearnerConfig Reference](#learnerconfig-reference)
5. [PPOLearnerConfig Reference](#ppolearnerconfig-reference)
6. [Model Architecture](#model-architecture)
7. [Rewards](#rewards)
8. [ZeroSumReward](#zerosumreward)
9. [Observation Builders](#observation-builders)
10. [State Setters](#state-setters)
11. [Terminal Conditions](#terminal-conditions)
12. [Transfer Learning](#transfer-learning)
13. [Metrics & Wandb](#metrics--wandb)
14. [Skill Tracker](#skill-tracker)
15. [Writing Custom Rewards](#writing-custom-rewards)
16. [Writing Custom Obs Builders](#writing-custom-obs-builders)

---

## How It Works

GigaLearnCPP runs many parallel Rocket League game simulations using [RocketSim](https://github.com/ZealanL/RocketSim) (no game client needed). Each simulation is a self-contained environment. The agent interacts with these environments by:

1. Receiving an **observation** (the current game state encoded as a float vector)
2. Choosing an **action** (one of the discrete actions from the action parser)
3. Receiving a **reward** (based on the configured reward functions)
4. The PPO algorithm uses collected experience to update the neural network

This loop repeats across hundreds of parallel games simultaneously, generating training data fast.

**Key components:**
- `RocketSim` — physics simulation (no Rocket League needed)
- `RLGymCPP` — reward functions, obs builders, state setters, terminal conditions
- `GigaLearnCPP` — PPO learner, neural network, checkpoint management
- `src/ExampleMain.cpp` — your configuration entrypoint

---

## ExampleMain.cpp Walkthrough

This is the file you edit to configure your training run. It has three main parts:

### `EnvCreateFunc(int index)`

Called once per environment (game instance) at startup. Returns everything that environment needs:

```cpp
EnvCreateResult result = {};
result.actionParser = new DefaultAction();    // How actions are parsed
result.obsBuilder   = new AdvancedObsPadded(); // How obs are built
result.stateSetter  = new KickoffState();       // How episodes start
result.terminalConditions = { ... };            // When episodes end
result.rewards      = { ... };                  // What the agent is rewarded for
result.arena        = arena;                    // The RocketSim arena
```

The `index` parameter lets you vary game modes per environment (e.g. cycle 1v1/2v2/3v3 with `index % 3`).

### `StepCallback(Learner*, states, report)`

Called every iteration. Use it to log custom metrics:

```cpp
report.AddAvg("Player/In Air Ratio", !player.isOnGround);
report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
```

`report.AddAvg` accumulates a rolling average that gets sent to wandb if metrics are enabled.

### `main()`

Parses CLI args, builds the `LearnerConfig`, creates the `Learner`, and calls `learner->Start()`.

---

## Command-Line Flags

All flags are parsed in `main()` from `argv`. None are required — all have defaults.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gpu` | bool | false | Force CUDA GPU. Fails if GPU unsupported by this LibTorch build. |
| `--cpu` | bool | false | Force CPU inference and training. |
| `--no-load` | bool | false | Start fresh — ignore any existing checkpoint in the checkpoint folder. |
| `--checkpoint <path>` | string | `"checkpoints"` | Path to load/save checkpoints from. (this is optional is will auto save to where it should if this arg is not added) |
| `--num-games <n>` | int | 512 | Number of parallel game instances. More = more data per iteration but more RAM/CPU. |
| `--render` | bool | false | Enable rendering. **Severely slows training** — only for demos/debugging. |
| `--render-timescale <f>` | float | 8.0 | Speed multiplier for rendering. 1.0 = real time, 8.0 = 8x speed. |
| `--send-metrics` | bool | false | Send metrics to the Python wandb receiver. Requires Python + wandb. |
| `--no-send-metrics` | bool | — | Disable metrics even if default is true. |
| `--add-rewards` | bool | true | Include per-reward breakdowns in metrics. |
| `--no-add-rewards` | bool | — | Disable reward metrics. |
| `--tl <path>` | string | — | Path to teacher checkpoint for transfer learning. |
| `--transfer-learn <path>` | string | — | Alias for `--tl`. |

**Examples:**

```bat
# Fast CPU run, fresh start
GigaLearnBot.exe --cpu --no-load --num-games 256

# GPU with metrics
GigaLearnBot.exe --gpu --send-metrics 

# Transfer learn from a saved checkpoint
GigaLearnBot.exe --tl checkpoints\685272918 --gpu
```

---

## LearnerConfig Reference

Defined in `GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h`.

| Field | Default | Description |
|---|---|---|
| `numGames` | 300 | Number of parallel game environments. Set via `--num-games`. |
| `tickSkip` | 8 | Physics ticks per action step (6 = ~15 actions/sec, 8 = ~11 actions/sec). |
| `actionDelay` | 7 | Number of ticks between action selection and execution. Usually `tickSkip - 1`. |
| `renderMode` | false | Enable rendering. |
| `renderTimeScale` | 1.0 | Rendering speed multiplier. |
| `checkpointFolder` | `"checkpoints"` | Where to save/load model checkpoints. |
| `tsPerSave` | 1,000,000 | Save a checkpoint every N timesteps. |
| `randomSeed` | -1 | RNG seed. -1 = use current time. |
| `checkpointsToKeep` | 8 | Max checkpoints to keep before deleting oldest. -1 = keep all. |
| `deviceType` | AUTO | `CPU`, `GPU_CUDA`, or `AUTO` (uses GPU if available). |
| `standardizeObs` | false | Normalize obs values using Welford running stats. Rarely helps. |
| `standardizeReturns` | true | Normalize returns for the critic. Leave this on. |
| `addRewardsToMetrics` | true | Log individual reward breakdowns. |
| `sendMetrics` | true | Send metrics to Python wandb receiver. |
| `metricsProjectName` | `"gigalearncpp"` | Wandb project name. |
| `metricsGroupName` | `"unnamed-runs"` | Wandb group name. |
| `metricsRunName` | `"gigalearncpp-run"` | Wandb run name. |
| `savePolicyVersions` | false | Save older policy snapshots for self-play. |
| `tsPerVersion` | 25,000,000 | How often to save a policy version snapshot. |
| `trainAgainstOldVersions` | false | Occasionally pit the agent against older versions of itself. |
| `trainAgainstOldChance` | 0.15 | Probability (0–1) of using an old version in a given iteration. |

---

## PPOLearnerConfig Reference

Nested under `cfg.ppo` in `LearnerConfig`. Defined in `GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h`.

| Field | Default | Description |
|---|---|---|
| `tsPerItr` | 50,000 | Timesteps collected per PPO iteration. More = more stable but slower iterations. |
| `batchSize` | 50,000 | Batch size for PPO update. Usually equal to `tsPerItr`. |
| `miniBatchSize` | 0 | Mini-batch size. 0 = use full batch. Smaller = less VRAM needed. |
| `epochs` | 2 | Number of PPO epochs per iteration (passes over the batch). |
| `policyLR` | 3e-4 | Policy network learning rate. |
| `criticLR` | 3e-4 | Critic network learning rate. |
| `entropyScale` | 0.018 | Scale of the entropy bonus. Higher = more exploration. |
| `maskEntropy` | false | If true, entropy is calculated only over valid (non-masked) actions. |
| `clipRange` | 0.2 | PPO clipping parameter. Prevents large policy updates. |
| `policyTemperature` | 1.0 | Softmax temperature. < 1.0 = more confident/peaked, > 1.0 = more uniform. |
| `gaeLambda` | 0.95 | GAE lambda. Higher = less bias, more variance. Typical range: 0.9–0.99. |
| `gaeGamma` | 0.99 | Discount factor. Higher = longer time horizon. |
| `rewardClipRange` | 10.0 | Clip normalized rewards to this range. 0 = disabled. |
| `useHalfPrecision` | false | Use FP16 for inference. Faster on GPU, not useful on CPU. |
| `deterministic` | false | Always pick highest-probability action. Good for eval, breaks training. |
| `maxEpisodeDuration` | 120 | Max episode length in seconds before forced reset. |
| `overbatching` | true | Use remaining experience > batchSize if it's < 2x batchSize. Prevents waste. |
| `useGuidingPolicy` | false | KL-penalize against a fixed "guiding" policy to bias behavior. |
| `guidingPolicyPath` | `"guiding_policy/"` | Path to guiding policy models. |
| `guidingStrength` | 0.03 | Strength of the guiding policy penalty. |

### Recommended settings from ExampleMain.cpp

```cpp
cfg.tickSkip = 8;
cfg.actionDelay = 7;  // tickSkip - 1
cfg.numGames = 512;

cfg.ppo.tsPerItr = 196608;   // ~6 * 32768
cfg.ppo.batchSize = 196608;
cfg.ppo.miniBatchSize = 65536; // ~3 mini-batches
cfg.ppo.epochs = 1;

cfg.ppo.gaeGamma = 0.998f;
cfg.ppo.gaeLambda = 0.958f;
cfg.ppo.policyLR = 1e-4f;
cfg.ppo.criticLR = 1e-4f;

cfg.ppo.entropyScale = 0.015f;
cfg.ppo.maskEntropy = true;
cfg.ppo.policyTemperature = 0.9f;
```

---

## Model Architecture

The neural network has three parts, each configured with a `PartialModelConfig`:

| Part | Field | Role |
|---|---|---|
| Shared head | `cfg.ppo.sharedHead` | Processes obs before splitting to policy/critic |
| Policy | `cfg.ppo.policy` | Outputs action probabilities |
| Critic | `cfg.ppo.critic` | Outputs value estimate |

**Layer sizes:**
```cpp
cfg.ppo.sharedHead.layerSizes = { 1024, 1024, 1024, 1024 };
cfg.ppo.policy.layerSizes     = { 1024, 1024, 1024, 1024 };
cfg.ppo.critic.layerSizes     = { 1024, 1024, 1024, 1024 };
```

**Activation functions** (`ModelActivationType`):

| Value | Description |
|---|---|
| `RELU` | Standard ReLU. Fast, can die on negative inputs. |
| `LEAKY_RELU` | ReLU but negative inputs give small gradient. Generally better. |
| `SIGMOID` | Squashes to (0,1). Rarely used in hidden layers. |
| `TANH` | Squashes to (-1,1). Can be useful for some architectures. |

**Optimizers** (`ModelOptimType`):

| Value | Description |
|---|---|
| `ADAM` | Standard Adam. Default and recommended. |
| `ADAMW` | Adam with weight decay. Can help with overfitting. |
| `ADAGRAD` | Adaptive gradient. Less common for RL. |
| `RMSPROP` | Classic RL optimizer. Alternative to Adam. |
| `MAGSGD` | Momentum SGD variant. |

**Layer norm:** Set `addLayerNorm = true` to add layer normalization after each hidden layer. Strongly recommended for deep networks (4+ layers).

---

## Rewards

Rewards are defined in `GigaLearnCPP/RLGymCPP/src/RLGymCPP/Rewards/CommonRewards.h`.

Each reward implements `GetReward(player, state, isFinal)` and returns a float.

Rewards are added to `EnvCreateFunc` as `WeightedReward` pairs:

```cpp
std::vector<WeightedReward> rewards = {
    { new SomeReward(), 1.0f },          // weight = 1.0
    { new ZeroSumReward(new SomeOtherReward(), 1.f, 0.7f), 5.0f },
};
```

The weight scales the reward contribution. A weight of `5.0f` makes that reward 5x as impactful as a weight of `1.0f`.

### All built-in rewards

| Class | Output Range | Description |
|---|---|---|
| `GoalReward(concedeScale)` | `[concedeScale, 1]` | +1 when your team scores, `concedeScale` (default -1) when opponent scores. Already zero-sum. |
| `VelocityReward(isNegative)` | `[-1, 1]` | Reward proportional to car speed. `isNegative=true` penalizes speed. |
| `VelocityBallToGoalReward(ownGoal)` | `[-1, 1]` | Dot product of ball velocity towards the opponent's goal. `ownGoal=true` rewards ball towards own goal. |
| `VelocityPlayerToBallReward` | `[-1, 1]` | Dot product of player velocity towards the ball. Encourages chasing the ball. |
| `FaceBallReward` | `[-1, 1]` | Dot product of player forward direction towards the ball. Encourages facing the ball. |
| `TouchBallReward` | `{0, 1}` | +1 every step the player touches the ball. |
| `TouchAccelReward` | `[0, 1]` | Reward for accelerating the ball on touch. Capped at 110 KPH ball speed. |
| `StrongTouchReward(minKPH, maxKPH)` | `[0, 1]` | Reward for hitting the ball hard. Scales linearly from `minKPH` to `maxKPH`. Default 20–90 KPH. |
| `SpeedReward` | `[0, 1]` | Raw car speed normalized by max car speed. |
| `AirReward` | `{0, 1}` | +1 every step the player is in the air. Encourages aerial play. |
| `WavedashReward` | `{0, 1}` | +1 when the player executes a wavedash (flip landing on ground). |
| `PickupBoostReward` | `[0, ~1]` | Reward proportional to boost gained this step (sqrt-scaled). |
| `SaveBoostReward(exponent)` | `[0, 1]` | Reward for having boost. `exponent=0.5` rewards having any boost more than having full boost. |
| `PlayerGoalReward` | `{0, 1}` | +1 to the player who last touched the ball when a goal is scored. |
| `AssistReward` | `{0, 1}` | +1 for an assist. |
| `ShotReward` | `{0, 1}` | +1 for a shot on goal. |
| `SaveReward` | `{0, 1}` | +1 for a save. |
| `BumpReward` | `{0, 1}` | +1 for bumping an opponent. |
| `BumpedPenalty` | `{0, -1}` | -1 for being bumped. |
| `DemoReward` | `{0, 1}` | +1 for demolishing an opponent. |
| `DemoedPenalty` | `{0, -1}` | -1 for being demolished. |

### Choosing reward weights

Weights are relative to each other. A rough starting point:

- High-frequency rewards (velocity, facing) → low weight (0.1–5.0)
- Medium-frequency rewards (ball touch) → medium weight (5.0–25.0)
- Sparse rewards (goals) → high weight (100–1000)

Too high a weight on sparse rewards early in training can cause instability. Start conservative.

---

## ZeroSumReward

Wraps another reward to make it zero-sum and team-distributed.

```cpp
new ZeroSumReward(Reward* child, float teamSpirit, float opponentScale)
```

**Formula per player:**
```
reward = ownReward * (1 - teamSpirit)
       + avgTeamReward * teamSpirit
       - avgOpponentReward * opponentScale
```

| Parameter | Description |
|---|---|
| `child` | The underlying reward function to wrap. |
| `teamSpirit` | 0 = purely individual, 1 = purely team average. For 1v1 use 0 or 1 (no teammates). |
| `opponentScale` | Scale of opponent punishment. 1.0 = true zero-sum. 0.7 = softer punishment. |

**Example:**
```cpp
// Ball-to-goal reward: fully team-shared, 70% opponent punishment
{ new ZeroSumReward(new VelocityBallToGoalReward(), 1.f, 0.7f), 25.f }
```

---

## Observation Builders

Obs builders convert the game state into a float vector fed into the neural network.

### `DefaultObs`
Bare-bones obs. Small vector. Good for simple experiments.

### `AdvancedObs`
More complete obs including:
- Ball position, velocity, angular velocity
- Player position, velocity, rotation, angular velocity, boost, on-ground flag, has-jump flag, has-flip flag
- Previous action
- Inversion for orange team (obs are always from the perspective of blue)

**Normalization coefficients:**
- Position: `/ 5000`
- Velocity: `/ 2300`
- Angular velocity: `/ 3`

### `AdvancedObsPadded` (recommended)
Same as `AdvancedObs` but pads teammate/opponent slots to a fixed size:
- Up to 2 teammates (zero-padded if fewer)
- Up to 3 opponents (zero-padded if fewer)
- Boost pad states (active = 1.0, inactive = 1/(1 + timer))

This means the obs size is always constant regardless of player count, so one model can handle 1v1, 2v2, and 3v3.

**Feature size per player:** 29 floats

### `DefaultObsPadded`
DefaultObs with similar padding approach.

### Choosing an obs builder

- For 1v1 only: `AdvancedObs` is fine
- For multi-mode training (1v1/2v2/3v3): use `AdvancedObsPadded` (consistent obs size)
- For custom obs: extend `ObsBuilder` (see [Writing Custom Obs Builders](#writing-custom-obs-builders))

---

## State Setters

State setters control how episodes are initialized (reset state).

### `KickoffState`
Places cars and ball in standard kickoff positions. Randomizes which kickoff variant. This is the default and most natural starting state for learning Rocket League.

### `FuzzedKickoffState`
Kickoff positions with small random perturbations. Adds variety to prevent the agent overfitting to exact kickoff positions.

### `RandomState`
Fully random initial state — random ball position/velocity, random car positions/velocities/boost. Useful for:
- Learning recovery skills
- Aerial training
- Generalization

### `CombinedState`
Randomly picks from multiple state setters each episode. Example:

```cpp
result.stateSetter = new CombinedState({
    { new KickoffState(), 0.7f },   // 70% kickoff
    { new RandomState(),  0.3f },   // 30% random
});
```

Weights are relative (normalized internally).

---

## Terminal Conditions

Terminal conditions decide when an episode ends and resets.

### `GoalScoreCondition`
Terminates the episode when a goal is scored. Essentially mandatory — without it the agent won't experience scoring/conceding.

```cpp
new GoalScoreCondition()
```

### `NoTouchCondition(maxTime)`
Terminates after `maxTime` seconds with no ball touch by any player. Prevents games from stalling indefinitely.

```cpp
new NoTouchCondition(30)  // Reset after 30s no touch
```

`IsTruncation()` returns true — meaning this is a timeout, not a natural episode end. The value bootstrap is handled differently for truncations.

### Typical setup

```cpp
std::vector<TerminalCondition*> terminalConditions = {
    new NoTouchCondition(30),
    new GoalScoreCondition()
};
```

---

## Transfer Learning

Transfer learning lets you initialize a new model's weights from a pre-trained checkpoint, using KL-divergence loss to match the teacher's behavior.

```cpp
TransferLearnConfig tlConfig = {};
tlConfig.makeOldObsFn = []() { return new AdvancedObsPadded(); };
tlConfig.makeOldActFn = []() { return new DefaultAction(); };

// Old model architecture (must match the checkpoint exactly)
const std::vector<int> oldLayers = { 1024, 1024, 1024, 1024, 512 };
tlConfig.oldSharedHeadConfig.layerSizes = oldLayers;
tlConfig.oldSharedHeadConfig.activationType = ModelActivationType::RELU;
tlConfig.oldSharedHeadConfig.addLayerNorm = true;
tlConfig.oldSharedHeadConfig.addOutputLayer = false;
tlConfig.oldPolicyConfig.layerSizes = oldLayers;
tlConfig.oldPolicyConfig.activationType = ModelActivationType::RELU;
tlConfig.oldPolicyConfig.addLayerNorm = true;

tlConfig.oldModelsPath = "path/to/teacher/checkpoint";
tlConfig.lr = 4e-4f;
tlConfig.batchSize = 32768;
tlConfig.epochs = 5;
tlConfig.useKLDiv = true;
tlConfig.lossScale = 500.f;

learner->StartTransferLearn(tlConfig);
```

**When to use:**
- Your current obs/architecture differs from an existing well-trained bot
- You want to bootstrap from a strong bot without running full RL from scratch
- Faster than training from random init, especially for complex behaviors

**Trigger via CLI:**
```bat
GigaLearnBot.exe --tl checkpoints\685272918
```

The code auto-resolves to the highest-numbered subdirectory inside the given path.

---

## Metrics & Wandb

Metrics are sent to a Python `metric_receiver.py` script via IPC, which logs them to [wandb](https://wandb.ai).

### Setup

1. Install wandb: `pip install wandb`
2. Log in: `wandb login`
3. Run with `--send-metrics`

### Metric names

The `StepCallback` in `ExampleMain.cpp` logs these automatically:

| Metric | Description |
|---|---|
| `Player/In Air Ratio` | Fraction of steps the player is airborne |
| `Player/Ball Touch Ratio` | Fraction of steps with a ball touch |
| `Player/Boost` | Average boost amount (0–100) |
| `Player/Has Flip Reset Ratio` | Fraction of steps player has flip reset |
| `Player/Is Flipping Ratio` | Fraction of steps player is flipping |
| `Player/Aerial Touch Height` | Average ball height on aerial touches |
| `Player/Goal/Shot/Save/Assist/Demo Ratio` | Event rates |
| `Player/Speed` | Average car speed |
| `Player/Speed Towards Ball` | Average speed component towards ball |
| `Game/Goal Speed` | Ball speed at goal events |

### Adding custom metrics

```cpp
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
    for (auto& state : states) {
        report.AddAvg("MyMetric/SomeThing", someValue);
    }
}
```

`report.AddAvg` accumulates values and reports the average per iteration.

---

## Skill Tracker

The skill tracker periodically runs evaluation games between the current policy and previous checkpoints to produce an Elo-like rating.

```cpp
cfg.skillTracker.enabled = true;
cfg.skillTracker.numArenas = 16;      // Parallel eval arenas (don't exceed CPU threads)
cfg.skillTracker.simTime = 45.f;      // Seconds per eval game per arena
cfg.skillTracker.updateInterval = 14; // Iterations between eval runs
cfg.skillTracker.ratingInc = 5.f;     // Elo increment per goal
cfg.skillTracker.deterministic = false; // Use stochastic policy for eval
```

**Disabled by default** because it uses extra CPU threads. Enable once you have a saved checkpoint to compare against.

---

## Writing Custom Rewards

Extend `Reward` and implement `GetReward`:

```cpp
#include <RLGymCPP/Rewards/Reward.h>

class MyReward : public RLGC::Reward {
public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Return a float. Typical range: 0 to 1, or -1 to 1.
        // player.pos, player.vel, player.boost, player.isOnGround, player.ballTouchedStep, etc.
        // state.ball.pos, state.ball.vel
        // state.prev -> previous game state (may be null on first step)
        return 0.f;
    }
};
```

**Useful player fields:**

| Field | Type | Description |
|---|---|---|
| `player.pos` | `Vec` | Car position |
| `player.vel` | `Vec` | Car velocity |
| `player.rotMat.forward` | `Vec` | Car forward direction |
| `player.angVel` | `Vec` | Car angular velocity |
| `player.boost` | `float` | Boost amount (0–100) |
| `player.isOnGround` | `bool` | Car is on ground |
| `player.isFlipping` | `bool` | Car is currently flipping |
| `player.isDemoed` | `bool` | Car is demolished |
| `player.ballTouchedStep` | `bool` | Car touched ball this step |
| `player.team` | `Team` | `Team::BLUE` or `Team::ORANGE` |
| `player.eventState.goal` | `bool` | Goal scored by this player this step |
| `player.eventState.save` | `bool` | Save by this player this step |
| `player.HasFlipReset()` | `bool` | Player currently has a flip reset |
| `player.prev` | `Player*` | Previous step state (null on first step) |

**Useful ball fields:**

| Field | Type | Description |
|---|---|---|
| `state.ball.pos` | `Vec` | Ball position |
| `state.ball.vel` | `Vec` | Ball velocity |
| `state.ball.angVel` | `Vec` | Ball angular velocity |
| `state.goalScored` | `bool` | A goal was scored this step |

**Useful constants** (`CommonValues`):

| Constant | Value |
|---|---|
| `CAR_MAX_SPEED` | ~2300 uu/s |
| `BALL_MAX_SPEED` | ~6000 uu/s |
| `ORANGE_GOAL_BACK` | Orange goal position |
| `BLUE_GOAL_BACK` | Blue goal position |

---

## Writing Custom Obs Builders

Extend `ObsBuilder` and implement `BuildObs`:

```cpp
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

class MyObs : public RLGC::ObsBuilder {
public:
    virtual RLGC::FList BuildObs(const Player& player, const GameState& state) override {
        FList obs = {};

        bool inv = (player.team == Team::ORANGE); // invert for orange

        // Add ball data
        obs += state.ball.pos * (1.f / 5000.f);
        obs += state.ball.vel * (1.f / 2300.f);

        // Add player data
        obs += player.pos * (1.f / 5000.f);
        obs += player.vel * (1.f / 2300.f);
        obs += (float)player.boost / 100.f;
        obs += (float)player.isOnGround;

        return obs;
    }
};
```

`FList` is a `std::vector<float>` with `+=` overloaded for `Vec`, `float`, etc.

The obs size must be **consistent** across all calls — the network input size is fixed at startup. If you have variable player counts, use padding (see `AdvancedObsPadded` for reference).
