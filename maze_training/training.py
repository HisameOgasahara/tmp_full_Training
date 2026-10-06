"""Train each stage independently, including environment-interactive GRPO."""

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import build_maps, sample_world_batch, build_expert_examples
from .environment import step_environment
from .evaluation import predict_action_probabilities
from .runtime import read_config, seed_runtime, choose_device, mixed_precision, create_model, load_model, save_checkpoint, write_json

PREDECESSOR = {"sft": "pretrain", "grpo": "sft"}


def normalize_group_rewards(rewards, epsilon):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, correction=0)
    advantages = (rewards - mean) / std.clamp_min(epsilon)
    return advantages, std.squeeze(1) > epsilon


def optimize_loss(loss, model, optimizer, scaler, gradient_clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("학습 손실이 유한하지 않습니다. 학습률과 보상 설정을 확인하세요.")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return float(gradient_norm)


@torch.no_grad()
def collect_rollouts(model, encoder, maps, config, rng):
    settings, rules = config["grpo"], config["environment"]
    group_size, prompts = settings["group_size"], settings["prompts"]
    selected = [maps[int(rng.integers(len(maps)))] for _ in range(prompts)]
    mazes = [maze for maze in selected for _ in range(group_size)]
    noise = rng.random((prompts, rules["max_steps"]))
    positions = [maze.start for maze in mazes]
    returns = np.zeros(len(mazes), dtype=np.float32)
    successes = np.zeros(len(mazes), dtype=bool)
    active = list(range(len(mazes)))
    queries, actions, log_probabilities, episode_indices = [], [], [], []
    model.eval()
    for step in range(rules["max_steps"]):
        if not active:
            break
        probabilities = predict_action_probabilities(model, encoder, [mazes[i] for i in active], [positions[i] for i in active], settings["temperature"])
        sampled = torch.multinomial(probabilities, 1).squeeze(-1)
        old_log_probs = probabilities.gather(-1, sampled[:, None]).clamp_min(1e-10).log().squeeze(-1)
        next_active = []
        for offset, index in enumerate(active):
            action = int(sampled[offset])
            queries.append(encoder.encode_query(mazes[index], positions[index]))
            actions.append(action)
            log_probabilities.append(float(old_log_probs[offset]))
            episode_indices.append(index)
            # Common random numbers within a group reduce outcome noise in reward comparisons.
            outcome = step_environment(mazes[index], positions[index], action, rules, noise[index // group_size, step])
            positions[index] = outcome.position
            returns[index] += outcome.reward
            successes[index] = outcome.success
            if not outcome.done:
                next_active.append(index)
        active = next_active
    return {
        "queries": np.asarray(queries), "actions": np.asarray(actions),
        "old_log_probabilities": np.asarray(log_probabilities, dtype=np.float32),
        "episode_indices": np.asarray(episode_indices),
        "returns": returns.reshape(prompts, group_size), "successes": successes,
    }


def train_grpo_step(model, reference, encoder, maps, config, rng, optimizer, scaler):
    device, settings = next(model.parameters()).device, config["grpo"]
    rollout = collect_rollouts(model, encoder, maps, config, rng)
    returns = torch.as_tensor(rollout["returns"], device=device)
    advantages, useful = normalize_group_rewards(returns, settings["advantage_epsilon"])
    episode_ids = torch.as_tensor(rollout["episode_indices"], device=device)
    useful_steps = useful[episode_ids // settings["group_size"]]
    indices = torch.where(useful_steps)[0]
    metrics = {
        "mean_return": float(returns.mean()), "sampled_success_rate": float(rollout["successes"].mean()),
        "useful_group_fraction": float(useful.float().mean()), "transitions": len(episode_ids),
        "loss": 0.0, "kl": 0.0, "clip_fraction": 0.0,
    }
    if not len(indices):
        return metrics
    queries = torch.as_tensor(rollout["queries"], device=device)
    actions = torch.as_tensor(rollout["actions"], device=device)
    old_log_probabilities = torch.as_tensor(rollout["old_log_probabilities"], device=device)
    per_step_advantage = advantages.flatten()[episode_ids]
    lengths = torch.bincount(episode_ids, minlength=returns.numel()).float()
    # Each trajectory contributes its average token loss, rather than weighting long failures more.
    weights = lengths[episode_ids].reciprocal()
    model.train()
    totals = []
    for _ in range(settings["epochs"]):
        permutation = indices[torch.randperm(len(indices), device=device)]
        optimizer.zero_grad(set_to_none=True)
        weighted_loss, weighted_kl, clipped_count = 0.0, 0.0, 0.0
        for start in range(0, len(permutation), settings["microbatch"]):
            batch = permutation[start:start + settings["microbatch"]]
            with mixed_precision(device):
                logits = model(queries[batch])[:, encoder.action_slice].float() / settings["temperature"]
            log_probs = logits.log_softmax(-1)
            selected = log_probs.gather(-1, actions[batch, None]).squeeze(-1)
            with torch.no_grad(), mixed_precision(device):
                reference_log_probs = (reference(queries[batch])[:, encoder.action_slice].float() / settings["temperature"]).log_softmax(-1)
            # Exact categorical KL over four actions, not a noisy single-action estimator.
            kl = (log_probs.exp() * (log_probs - reference_log_probs)).sum(-1)
            ratio = (selected - old_log_probabilities[batch]).exp()
            advantage = per_step_advantage[batch]
            clipped = ratio.clamp(1 - settings["clip_epsilon"], 1 + settings["clip_epsilon"])
            surrogate = -torch.minimum(ratio * advantage, clipped * advantage)
            loss = ((surrogate + settings["kl_coefficient"] * kl) * weights[batch]).sum() / (useful.sum() * settings["group_size"])
            if not torch.isfinite(loss):
                raise FloatingPointError("GRPO 손실이 유한하지 않습니다.")
            scaler.scale(loss).backward()
            weighted_loss += float(loss.detach())
            weighted_kl += float((kl.detach() * weights[batch]).sum()) / int(useful.sum() * settings["group_size"])
            clipped_count += float((ratio.detach() != clipped.detach()).sum())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        totals.append((weighted_loss, weighted_kl, clipped_count / len(indices)))
    metrics.update(dict(zip(("loss", "kl", "clip_fraction"), np.mean(totals, axis=0).tolist())))
    return metrics


def run_stage(stage, config_path="configs/t4.json", output_dir="runs/t4", steps=None, initialize_from=None, resume=False):
    if stage not in ("pretrain", "sft", "grpo"):
        raise ValueError("stage는 pretrain, sft, grpo 중 하나입니다.")
    config = read_config(config_path)
    seed_runtime(config["seed"])
    device = choose_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{stage}.pt"
    start_step, prior_logs = 0, []
    payload = None
    if resume:
        model, encoder, payload = load_model(target, device)
        if payload["stage"] != stage or payload["config"] != config:
            raise ValueError("이어 학습은 같은 단계와 설정 파일로 실행하세요.")
        start_step = payload["step"]
        prior_logs = payload.get("history", [])
    elif initialize_from is not None or stage in PREDECESSOR:
        source = Path(initialize_from) if initialize_from else output_dir / f"{PREDECESSOR[stage]}.pt"
        if not source.exists():
            raise FileNotFoundError(f"이전 단계 모델이 없습니다: {source}. 이전 학습 셀을 먼저 실행하세요.")
        model, encoder, source_payload = load_model(source, device)
        if source_payload["config"] != config:
            raise ValueError("단계 간에는 같은 설정을 사용하세요. smoke와 t4 체크포인트는 호환되지 않습니다.")
    else:
        model, encoder = create_model(config, device)
    settings = config[stage]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config["seed"] + {"pretrain": 10, "sft": 20, "grpo": 30}[stage])
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    maps = build_maps(config, "train")
    reference = None
    if stage == "grpo":
        reference = copy.deepcopy(model).eval()
        if payload is not None:
            reference.load_state_dict(payload["reference"])
        reference.requires_grad_(False)
    expert_inputs, expert_targets = build_expert_examples(maps, encoder) if stage == "sft" else (None, None)
    total_steps = settings["steps"] if steps is None else steps
    if total_steps <= start_step:
        print(f"이미 {start_step}단계까지 저장되어 있습니다.")
        return str(target)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    print(f"{stage}: {parameter_count:,} parameters | {gpu_name} | {start_step} → {total_steps} updates", flush=True)
    history = prior_logs
    started = time.monotonic()

    def persist(step):
        extra = {
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        if reference is not None:
            extra["reference"] = {key: value.detach().cpu().clone() for key, value in reference.state_dict().items()}
        save_checkpoint(target, model, config, stage, step, **extra)
        write_json(output_dir / f"{stage}_history.json", history)

    completed = start_step
    try:
        for step in range(start_step + 1, total_steps + 1):
            if stage == "grpo":
                metrics = train_grpo_step(model, reference, encoder, maps, config, rng, optimizer, scaler)
            else:
                model.train()
                if stage == "pretrain":
                    inputs, labels, _ = sample_world_batch(maps, encoder, config["environment"], settings["batch_size"], rng)
                    output_slice = encoder.position_slice
                else:
                    selected = rng.integers(len(expert_inputs), size=settings["batch_size"])
                    inputs, labels = expert_inputs[selected], expert_targets[selected]
                    output_slice = encoder.action_slice
                tokens = torch.as_tensor(inputs, device=device)
                targets = torch.as_tensor(labels, device=device)
                with mixed_precision(device):
                    logits = model(tokens)[:, output_slice]
                    loss = F.cross_entropy(logits.float(), targets)
                gradient_norm = optimize_loss(loss, model, optimizer, scaler, config["training"]["gradient_clip"])
                metrics = {"loss": float(loss.detach()), "accuracy": float((logits.argmax(-1) == targets).float().mean()), "gradient_norm": gradient_norm}
            completed = step
            history.append({"step": step, **metrics})
            if step == 1 or step % config["training"]["log_every"] == 0 or step == total_steps:
                print(f"[{stage} {step}/{total_steps}] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()) + f" elapsed={time.monotonic() - started:.1f}s", flush=True)
            if step % config["training"]["checkpoint_every"] == 0:
                persist(step)
        persist(completed)
    except KeyboardInterrupt:
        # Do not label a partially applied GRPO update as a completed training step.
        if target.exists():
            print(f"중단했습니다. 마지막 완전한 체크포인트에서 resume=True로 재개하세요: {target}", flush=True)
        else:
            print("첫 체크포인트 저장 전에 중단했습니다. 단계를 다시 시작하세요.", flush=True)
        raise
    model.eval()
    print(f"저장 완료: {target}", flush=True)
    return str(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pretrain", "sft", "grpo"))
    parser.add_argument("--config", default="configs/t4.json")
    parser.add_argument("--output-dir", default="runs/t4")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run_stage(arguments.stage, arguments.config, arguments.output_dir, arguments.steps, arguments.initialize_from, arguments.resume)


if __name__ == "__main__":
    main()
