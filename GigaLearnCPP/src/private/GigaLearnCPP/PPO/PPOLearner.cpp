#include "PPOLearner.h"

#include <cmath>
#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	result = result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
	// Sanitize so multinomial never sees nan/inf/<0 (avoids CUDA device-side assert and training stop)
	result = torch::where(result.isfinite(), result, torch::full_like(result, ACTION_MIN_PROB));
	auto rowSum = result.sum(-1, true).clamp_min(1e-10f);
	result = result / rowSum;
	return result;
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

	// Defensive: ensure valid probs before multinomial (avoids device assert if upstream produced nan)
	constexpr float EPS = 1e-8f;
	probs = probs.clamp_min(EPS);
	probs = torch::where(probs.isfinite(), probs, torch::full_like(probs, 1.0f / probs.size(-1)));
	probs = probs / probs.sum(-1, true).clamp_min(1e-10f);

	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		auto logProb = torch::log(probs).gather(-1, action);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = logProb.flatten();
	}
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy (use clamp_min to avoid log(0) -> -inf)
	auto logP = probs.clamp_min(1e-10f).log();
	auto entropy = -(logP * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy; avoid division by zero (log(1)=0)
		auto maskSum = actionMasks.to(torch::kFloat32).sum(-1).clamp_min(1);
		auto denom = maskSum.log().clamp_min(1e-10f);
		entropy = entropy / denom;
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip;

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();

	bool useGpuAccum = device.is_cuda();
	torch::Tensor tEntropySum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	int64_t tEntropyCount = 0;
	torch::Tensor tPolicyLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor tCriticLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor tDivergenceSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor tClipSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor tGuidingLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	int64_t tCount = 0;

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	constexpr float ACTION_DISABLED_LOGIT = -1e10f;
	constexpr float ACTION_MIN_PROB = 1e-11f;

	for (int epoch = 0; epoch < config.epochs; epoch++) {
		models.ZeroGrad();

		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchActionMasks = batch.actionMasks;
			auto batchTargetValues = batch.targetValues;
			auto batchAdvantages = batch.advantages;
			const int64_t curBatchSize = batch.states.size(0);

			auto fnRunMinibatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)curBatchSize;

				auto acts = batchActs.slice(0, start, stop).to(device, true, true).to(torch::kLong);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				auto actionMasks = batchActionMasks.slice(0, start, stop).to(device, true, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);

				// Single shared_head forward per minibatch (policy + critic reuse) for faster consumption
				torch::Tensor features = models["shared_head"] ? models["shared_head"]->Forward(obs, false) : obs;
				auto vals = models["critic"]->Forward(features, false).flatten().view_as(targetValues);

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					auto actionMasksBool = actionMasks.to(torch::kBool);
					auto logits = models["policy"]->Forward(features, false) / config.policyTemperature;
					probs = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasksBool.logical_not(), -1)
						.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
					// Sanitize so entropy/log_probs never see nan/inf (avoids NaN in report and gradients)
					probs = torch::where(probs.isfinite(), probs, torch::full_like(probs, ACTION_MIN_PROB));
					auto rowSum = probs.sum(-1, true).clamp_min(1e-10f);
					probs = probs / rowSum;
					logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
					entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);

					if (useGpuAccum) {
						int64_t mbs = stop - start;
						tEntropySum += entropy.detach().to(torch::kFloat32) * static_cast<float>(mbs);
						tEntropyCount += mbs;
					} else {
						float curEntropy = entropy.detach().cpu().item<float>();
						avgEntropy += curEntropy;
					}

					logProbs = logProbs.view_as(oldProbs);

					ratio = exp(logProbs - oldProbs);
					if (!useGpuAccum)
						avgRatio += ratio.mean().detach().cpu().item<float>();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					if (useGpuAccum) {
						tPolicyLossSum += policyLoss.detach();
					} else {
						float curPolicyLoss = policyLoss.detach().cpu().item<float>();
						avgPolicyLoss += curPolicyLoss;
						avgRelEntropyLoss += (entropy.detach().cpu().item<float>() * config.entropyScale) / curPolicyLoss;
					}

					ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
						}
						auto guidingLoss = (guidingProbs - probs).abs().mean();
						if (useGpuAccum)
							tGuidingLossSum += guidingLoss.detach();
						else
							avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
						guidingLoss = guidingLoss * config.guidingStrength;
						ppoLoss = ppoLoss + guidingLoss;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					if (useGpuAccum)
						tCriticLossSum += criticLoss.detach();
					else
						avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}

				if (trainPolicy) {
					{
						RG_NO_GRAD;
						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						if (useGpuAccum)
							tDivergenceSum += klTensor.mean().detach();
						else
							avgDivergence += klTensor.mean().detach().cpu().item<float>();

						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						if (useGpuAccum)
							tClipSum += clipFraction.detach();
						else
							avgClip += clipFraction.cpu().item<float>();
					}
				}
				if (useGpuAccum) tCount++;

				if (trainPolicy && trainCritic) {
					auto combinedLoss = ppoLoss + criticLoss;
					combinedLoss.backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward();
					if (trainCritic)
						criticLoss.backward();
				}
			};

			if (device.is_cpu()) {
				fnRunMinibatch(0, (int)curBatchSize);
			} else {
				for (int64_t mbs = 0; mbs < curBatchSize; mbs += config.miniBatchSize) {
					int stop = (int)RS_MIN(mbs + config.miniBatchSize, curBatchSize);
					fnRunMinibatch((int)mbs, stop);
				}
			}

			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			models.StepOptims();
		}
	}

	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();
	// Use -1 as sentinel when non-finite so we can tell "no update" vs "NaN/Inf"
	if (!std::isfinite(policyUpdateMagnitude)) policyUpdateMagnitude = -1.f;
	if (!std::isfinite(criticUpdateMagnitude)) criticUpdateMagnitude = -1.f;

	if (useGpuAccum && tCount > 0) {
		float reportEntropy = (tEntropyCount > 0) ? (tEntropySum / tEntropyCount).cpu().item<float>() : 0.f;
		if (!std::isfinite(reportEntropy)) reportEntropy = 0.f;
		report["Policy Entropy"] = reportEntropy;
		float klDiv = (tDivergenceSum / tCount).cpu().item<float>();
		report["Mean KL Divergence"] = std::isfinite(klDiv) ? klDiv : 0.f;
		if (!isFirstIteration) {
			float plLoss = (tPolicyLossSum / tCount).cpu().item<float>();
			float cLoss = (tCriticLossSum / tCount).cpu().item<float>();
			report["Policy Loss"] = std::isfinite(plLoss) ? plLoss : 0.f;
			report["Critic Loss"] = std::isfinite(cLoss) ? cLoss : 0.f;
			if (config.useGuidingPolicy)
				report["Guiding Loss"] = (tGuidingLossSum / tCount).cpu().item<float>();
			report["SB3 Clip Fraction"] = (tClipSum / tCount).cpu().item<float>();
			report["Policy Update Magnitude"] = policyUpdateMagnitude;
			report["Critic Update Magnitude"] = criticUpdateMagnitude;
		}
	} else {
		float reportEntropy = avgEntropy.Get();
		if (!std::isfinite(reportEntropy)) reportEntropy = 0.f;
		report["Policy Entropy"] = reportEntropy;
		report["Mean KL Divergence"] = avgDivergence.Get();
		if (!isFirstIteration) {
			report["Policy Loss"] = avgPolicyLoss.Get();
			report["Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
			report["Critic Loss"] = avgCriticLoss.Get();
			if (config.useGuidingPolicy)
				report["Guiding Loss"] = avgGuidingLoss.Get();
			report["SB3 Clip Fraction"] = avgClip.Get();
			report["Policy Update Magnitude"] = policyUpdateMagnitude;
			report["Critic Update Magnitude"] = criticUpdateMagnitude;
		}
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	torch::Tensor oldProbs;
	{ // No grad for old model inference
		RG_NO_GRAD;
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

		// Non-summative KL div	loss
		torch::Tensor transferLearnLoss;
		if (tlConfig.useKLDiv) {
			transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
		} else {
			transferLearnLoss = (oldProbs - newProbs).abs();
		}
		transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
		transferLearnLoss = transferLearnLoss.mean();
		transferLearnLoss *= tlConfig.lossScale;

		if (i == 0) {
			RG_NO_GRAD;
			torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
			report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
			report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

			float tlEntropy = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
			report["Policy Entropy"] = std::isfinite(tlEntropy) ? tlEntropy : 0.f;
		}

		transferLearnLoss.backward();

		models.StepOptims();
	}

	auto policyAfter = models["policy"]->CopyParams();
	float tlUpdateMag = (policyBefore - policyAfter).norm().item<float>();
	report["Policy Update Magnitude"] = std::isfinite(tlUpdateMag) ? tlUpdateMag : 0.f;
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}