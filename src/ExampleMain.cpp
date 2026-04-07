#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// Find collision_meshes folder: try cwd, then parent dirs. Prefer absolute path so it works from build/.
static std::string FindCollisionMeshesPath() {
	const char* candidates[] = { "collision_meshes", "../collision_meshes", "../../collision_meshes" };
	for (const char* sub : candidates) {
		fs::path p = fs::current_path() / sub;
		fs::path soccar = p / "soccar";
		if (fs::exists(p) && fs::is_directory(p) && fs::exists(soccar) && fs::is_directory(soccar)) {
			try {
				return fs::absolute(p).string();
			} catch (...) {}
			return p.string();
		}
	}
	return "collision_meshes"; // fallback (Learner will init; may fail if missing)
}

using namespace GGL;
using namespace RLGC;

static bool ParseBoolArg(int argc, char* argv[], const char* flag, bool defaultValue) {
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], flag) == 0) return true;
		std::string arg = argv[i];
		if (arg == std::string("--no-") + (flag + 2)) return false;
	}
	return defaultValue;
}
static int ParseIntArg(int argc, char* argv[], const char* flag, int defaultValue) {
	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], flag) == 0) {
			int v = std::atoi(argv[i + 1]);
			return v > 0 ? v : defaultValue;
		}
	}
	return defaultValue;
}
static float ParseFloatArg(int argc, char* argv[], const char* flag, float defaultValue) {
	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], flag) == 0)
			return (float)std::atof(argv[i + 1]);
	}
	return defaultValue;
}
static std::string ParseStrArg(int argc, char* argv[], const char* flag, const char* defaultValue) {
	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], flag) == 0)
			return argv[i + 1];
	}
	return defaultValue ? std::string(defaultValue) : std::string();
}

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {
		// --- Ball mechanics ---
		{ new ZeroSumReward(new StrongTouchReward(20, 90), 0.f, 0.7f), 5.0f },
		{ new ZeroSumReward(new RLGC::VelocityBallToGoalReward(), 1.f, 0.7f), 25.f },
		{ new RLGC::FaceBallReward(), 0.1f },
		{ new RLGC::AirReward(), 0.12f },
		{ new RLGC::TouchBallReward(), 5.f },
		{ new RLGC::VelocityPlayerToBallReward(), 5.f },

	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(30),
		new GoalScoreCondition()
	};


	int combo = index % 3;
    GameMode gameMode = GameMode::SOCCAR;
    int playersPerTeam = 1;  // change to 2 for 2v2 and 3 for 3v3 etc......
	//int playersPerTeam = combo + 1; // put this if you want it to cycle 1v1 2v2 and 3v3

    auto arena = Arena::Create(gameMode);
    for (int i = 0; i < playersPerTeam; i++) {
    arena->AddCar(Team::BLUE);
    arena->AddCar(Team::ORANGE);
}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObsPadded();  // TL from 60B (AdvancedObsPadded) onto CustomObs
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = arena;

	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	bool doExpensiveMetrics = (rand() % 8) == 0;
	const int maxPlayers = 2000;
	int count = 0;
	bool done = false;

	for (auto& state : states) {
		if (done) break;
		for (auto& player : state.players) {
			if (count++ >= maxPlayers) { done = true; break; }
			report.AddAvg("Player/In Air Ratio", !player.isOnGround);
			report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
			report.AddAvg("Player/Demoed Ratio", player.isDemoed);
			report.AddAvg("Player/Boost", player.boost);

			bool hasFlipReset = player.HasFlipReset();
			bool gotFlipReset = player.GotFlipReset();
			bool hasFlipOrJump = player.HasFlipOrJump();
			report.AddAvg("Player/Has Flip Reset Ratio", hasFlipReset);
			report.AddAvg("Player/Got Flip Reset Ratio", gotFlipReset);
			report.AddAvg("Player/Has Flip Or Jump Ratio", hasFlipOrJump);
			report.AddAvg("Player/Is Flipping Ratio", player.isFlipping);
			if (player.ballTouchedStep && !player.isOnGround)
				report.AddAvg("Player/Aerial Touch Height", state.ball.pos.z);

			report.AddAvg("Player/Goal Ratio", player.eventState.goal);
			report.AddAvg("Player/Assist Ratio", player.eventState.assist);
			report.AddAvg("Player/Shot Ratio", player.eventState.shot);
			report.AddAvg("Player/Save Ratio", player.eventState.save);
			report.AddAvg("Player/Bump Ratio", player.eventState.bump);
			report.AddAvg("Player/Bumped Ratio", player.eventState.bumped);
			report.AddAvg("Player/Demo Ratio", player.eventState.demo);

			if (doExpensiveMetrics) {
				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0.f, player.vel.Dot(dirToBall)));
				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes (auto-find: cwd, ../, ../../ so server and PC both work)
	std::string meshPath = FindCollisionMeshesPath();
	RocketSim::Init(meshPath);

	// Make configuration for the learner
	LearnerConfig cfg = {};

	// Default: load/save from local checkpoints (GigaLearnCPP-Leak-Ref\build\Release\checkpoints). Use --checkpoint <path> to override, --no-load to start fresh.
	std::string checkpointPath = ParseStrArg(argc, argv, "--checkpoint", "");
	bool skipLoad = ParseBoolArg(argc, argv, "--no-load", false);
	if (skipLoad)
		cfg.checkpointFolder = "checkpoints";
	else
		cfg.checkpointFolder = checkpointPath.empty() ? "checkpoints" : checkpointPath;

	// Default CPU so the app runs on all machines (e.g. RTX 50 / sm_120 not in prebuilt LibTorch yet).
	// Use --gpu to force CUDA if your GPU is supported by this LibTorch build.
	bool useGpu = ParseBoolArg(argc, argv, "--gpu", false);
	bool useCpu = ParseBoolArg(argc, argv, "--cpu", false);
	cfg.deviceType = (useGpu && !useCpu) ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;

	cfg.ppo.useHalfPrecision = false;  // Keep FP32; set true for faster rollout inference if desired

	cfg.tickSkip = 6;
	cfg.actionDelay = cfg.tickSkip - 1;

	cfg.numGames = 512;
	cfg.randomSeed = -1;

	// Balance that gave ~18k overall before:
	// - 196608 steps per iteration
	// - batchSize = tsPerItr
	// - miniBatchSize = 65536 (3 minibatches)
	// - 1 epoch
	int tsPerItr = 196608;  // 6 * 32768
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 65536;   // Fits ~8GB VRAM
	cfg.ppo.epochs = 1;

	// Base GGL: raw entropy -sum(p*log(p)) normalized by log(num_valid_actions) so reported entropy ~0.6.
	const bool useZealanEntropy = true;
	if (useZealanEntropy) {
		cfg.ppo.entropyScale = 0.015f;
		cfg.ppo.maskEntropy = true;   // normalize by valid action count = actual natural entropy, matches base GGL ~0.6
	}
	cfg.ppo.policyTemperature = 0.9f;  // slightly peakier softmax → lower natural entropy (~0.6)
	cfg.ppo.gaeGamma = 0.998f;
	cfg.ppo.gaeLambda = 0.958f;

	cfg.ppo.policyLR = 1e-4f;
	cfg.ppo.criticLR = 1e-4f;

	// Set to true to use alternate entropy (mask-based normalization) instead of Zealan's.
	bool useShitEntropy = false;
	if (useShitEntropy) {
		cfg.ppo.maskEntropy = true;
	}

	cfg.ppo.sharedHead.layerSizes = { 1024, 1024, 1024, 1024, 512 };
	cfg.ppo.policy.layerSizes = { 1024, 1024, 1024, 1024, 512 };
	cfg.ppo.critic.layerSizes = { 1024, 1024, 1024, 1024, 512 };

	//cfg.ppo.sharedHead.layerSizes = { 1024, 1024, 1024, 1024, 512 };
//	cfg.ppo.policy.layerSizes = { 1024, 1024, 1024, 1024, 512 };
//	cfg.ppo.critic.layerSizes = { 1024, 1024, 1024, 1024, 512 };

	auto optim = ModelOptimType::ADAM;  // rocket-learn / SB3 default
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	//auto activation = ModelActivationType::RELU;
	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

	cfg.skillTracker.enabled = false;
	cfg.skillTracker.numArenas = 16;
	cfg.skillTracker.simTime = 45.f;
	cfg.skillTracker.updateInterval = 14;  // Less frequent = more SPS

	cfg.addRewardsToMetrics = ParseBoolArg(argc, argv, "--add-rewards", true);

	int numGamesOverride = ParseIntArg(argc, argv, "--num-games", 0);
	if (numGamesOverride > 0)
		cfg.numGames = numGamesOverride;

	// Disable metrics by default on server (avoids python_scripts / wandb issues).
	// You can re-enable with --send-metrics true if python_scripts is set up.
	cfg.sendMetrics = ParseBoolArg(argc, argv, "--send-metrics", false);
	cfg.renderMode = ParseBoolArg(argc, argv, "--render", false);
	cfg.renderTimeScale = ParseFloatArg(argc, argv, "--render-timescale", 8.0f);

	std::string tlPath = ParseStrArg(argc, argv, "--tl", "");
	if (tlPath.empty())
		tlPath = ParseStrArg(argc, argv, "--transfer-learn", "");
	if (tlPath.empty()) {
		// Check if --tl/--transfer-learn flag present without path → use default (works from repo root or build/Release)
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "--tl") == 0 || strcmp(argv[i], "--transfer-learn") == 0) {
				tlPath = "checkpoints/685272918";
				break;
			}
		}
	}
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	if (!tlPath.empty()) {
		// Transfer learn: teacher = checkpoint trained with CustomObs, student = current CustomObs + leaky_relu
		TransferLearnConfig tlConfig = {};
		tlConfig.makeOldObsFn = []() { return new AdvancedObsPadded(); };
		tlConfig.makeOldActFn = []() { return new DefaultAction(); };
		// Old model (teacher): same arch as saved checkpoint (4x1024 + 512)
		const std::vector<int> oldLayers = { 1024, 1024, 1024, 1024, 512 };
		tlConfig.oldSharedHeadConfig.layerSizes = oldLayers;
		tlConfig.oldSharedHeadConfig.activationType = ModelActivationType::RELU;
		tlConfig.oldSharedHeadConfig.addLayerNorm = true;
		tlConfig.oldSharedHeadConfig.addOutputLayer = false;
		tlConfig.oldPolicyConfig.layerSizes = oldLayers;
		tlConfig.oldPolicyConfig.activationType = ModelActivationType::RELU;
		tlConfig.oldPolicyConfig.addLayerNorm = true;
		tlConfig.oldModelsPath = tlPath;
		tlConfig.lr = 4e-4f;
		tlConfig.batchSize = 32768;  // Lower for 8GB VRAM (teacher + student + batch); was 200000
		tlConfig.epochs = 5;
		tlConfig.useKLDiv = true;
		tlConfig.lossScale = 500.f;

		// Resolve to latest checkpoint if path is a folder with numbered subdirs
		if (std::filesystem::exists(tlPath) && std::filesystem::is_directory(tlPath)) {
			int64_t highest = -1;
			for (auto& entry : std::filesystem::directory_iterator(tlPath)) {
				if (!entry.is_directory()) continue;
				std::string name = entry.path().filename().string();
				bool allDigits = true;
				for (char c : name) { if (!isdigit(c)) { allDigits = false; break; } }
				if (allDigits && !name.empty()) {
					int64_t n = std::stoll(name);
					highest = std::max(highest, n);
				}
			}
			if (highest >= 0) {
				std::filesystem::path loadFolder = std::filesystem::path(tlPath) / std::to_string(highest);
				std::filesystem::path nestedFolder = loadFolder / std::to_string(highest);
				if (std::filesystem::exists(nestedFolder / "POLICY.lt") && !std::filesystem::exists(loadFolder / "POLICY.lt"))
					loadFolder = nestedFolder;
				tlConfig.oldModelsPath = loadFolder;
			}
		}

		learner->StartTransferLearn(tlConfig);
	} else {
		try {
			learner->Start();
		} catch (const std::exception& e) {
			std::string msg = e.what();
			const bool noKernelImage = (msg.find("no kernel image") != std::string::npos) ||
				(msg.find("cudaErrorNoKernelImageForDevice") != std::string::npos);
			if (cfg.deviceType == LearnerDeviceType::GPU_CUDA && noKernelImage) {
				std::cerr << "GPU not supported by this LibTorch build (no kernel image for this device)." << std::endl;
				std::cerr << "  Run with --cpu to use CPU:  ./build/GigaLearnBot --cpu" << std::endl;
				std::cerr << "  For RTX 50 / Blackwell: build LibTorch from source with the correct TORCH_CUDA_ARCH_LIST." << std::endl;
				delete learner;
				return EXIT_FAILURE;
			}
			throw;
		}
	}

	return EXIT_SUCCESS;
}

