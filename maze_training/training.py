"""Train each stage independently, including actor-critic PPO with complete-episode GAE."""

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import build_maps, sample_world_batch, build_expert_examples
from .environment import step_environment, compute_goal_distances, move_nominal
from .policy import build_action_masks, mask_action_logits
from .evaluation import predict_action_probabilities
from .runtime import read_config, seed_runtime, choose_device, mixed_precision, create_model, load_model, save_checkpoint, write_json

PREDECESSOR = {"sft": "pretrain", "ppo": "sft"}


def optimize_loss(loss, model, optimizer, scaler, gradient_clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("학습 손실이 유한하지 않습니다. 학습률을 확인하세요.")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return float(gradient_norm)


def compute_gae(rewards, values, gamma, gae_lambda):
    """Complete episodes: the goal and finite-horizon timeout both have zero continuation."""
    advantages = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    next_value = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * gae_lambda * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def shortest_action_loss(logits, targets):
    """Any action that decreases BFS distance is a valid supervised answer."""
    log_probs = logits.float().log_softmax(-1)
    return -log_probs.masked_fill(~targets, -1e9).logsumexp(-1).mean()


@torch.no_grad()
def refresh_correction_weights(model, encoder, maps, queries, targets, settings, rng):
    lookup = {tuple(query):index for index,query in enumerate(queries)}
    weights = np.ones(len(queries),dtype=np.float64)
    selected = [maps[int(rng.integers(len(maps)))] for _ in range(settings["correction_maps"])]
    positions = [maze.start for maze in selected]
    active = list(range(len(selected)))
    model.eval()
    for _ in range(max(maze.size**2 for maze in selected)):
        if not active:
            break
        probs = predict_action_probabilities(model,encoder,[selected[i] for i in active],[positions[i] for i in active])
        actions = probs.argmax(-1).cpu().tolist()
        remaining = []
        for i,action in zip(active,actions):
            index = lookup[tuple(encoder.encode_query(selected[i],positions[i]))]
            if not targets[index,action]:
                weights[index] = settings["correction_weight"]
            # Only geometry is needed here; move_nominal does not consult costs.
            positions[i],_ = move_nominal(selected[i],positions[i],action)
            if positions[i] != selected[i].goal:
                remaining.append(i)
        active = remaining
    return weights


@torch.no_grad()
def collect_rollouts(model, encoder, maps, config, rng, reference):
    settings,rules = config["ppo"],config["environment"]
    selected = [maps[int(rng.integers(len(maps)))] for _ in range(settings["num_envs"])]
    distances = [compute_goal_distances(maze) for maze in selected]
    keys = ("queries","actions","masks","log_probs","reference_log_probs","rewards")
    trajectories = [{**{key:[] for key in keys},"cost":0.0,"success":False,"raw_return":0.0} for _ in selected]
    positions = [maze.start for maze in selected]
    active = list(range(len(selected)))
    device = next(model.parameters()).device
    model.eval()
    for step in range(rules["max_steps"]):
        if not active:
            break
        inputs = [encoder.encode_query(selected[i],positions[i]) for i in active]
        masks = build_action_masks([selected[i] for i in active],[positions[i] for i in active])
        tokens = torch.tensor(inputs,device=device)
        with mixed_precision(device):
            logits = model(tokens)[:,encoder.action_slice]
            reference_logits = reference(tokens)[:,encoder.action_slice]
        distribution = torch.distributions.Categorical(logits=mask_action_logits(logits,masks))
        reference_log_probs = mask_action_logits(reference_logits,masks).log_softmax(-1)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        remaining = []
        for offset,index in enumerate(active):
            action = int(actions[offset])
            before = positions[index]
            outcome = step_environment(selected[index],before,action,rules)
            terminal = outcome.done or step+1==rules["max_steps"]
            before_phi = -settings["potential_scale"]*distances[index][before]
            after_phi = 0.0 if terminal else -settings["potential_scale"]*distances[index][outcome.position]
            training_reward = outcome.reward+settings["gamma"]*after_phi-before_phi
            episode = trajectories[index]
            for key,value in (("queries",inputs[offset]),("actions",action),("masks",masks[offset]),("log_probs",float(log_probs[offset])),("reference_log_probs",reference_log_probs[offset].cpu().tolist()),("rewards",training_reward)):
                episode[key].append(value)
            episode["cost"] += outcome.cost
            episode["raw_return"] += outcome.reward
            episode["success"] = outcome.success
            positions[index] = outcome.position
            if not terminal:
                remaining.append(index)
        active = remaining
    combined = {key:[] for key in keys}
    lengths = []
    for episode in trajectories:
        lengths.append(len(episode["actions"]))
        for key in keys:
            combined[key].extend(episode[key])
    tensors = {key:torch.as_tensor(np.asarray(value),device=device,dtype=torch.long if key in ("queries","actions") else torch.bool if key=="masks" else torch.float32) for key,value in combined.items()}
    metrics = {"mean_return":float(np.mean([e["raw_return"] for e in trajectories])),"sampled_success_rate":float(np.mean([e["success"] for e in trajectories])),"mean_cost":float(np.mean([e["cost"] for e in trajectories])),"transitions":len(tensors["actions"])}
    return tensors,lengths,metrics


@torch.no_grad()
def calculate_rollout_targets(model,rollout,lengths,settings):
    values=[]
    for start in range(0,len(rollout["actions"]),settings["minibatch"]):
        with mixed_precision(rollout["queries"].device):
            _,_,predicted=model(rollout["queries"][start:start+settings["minibatch"]],with_auxiliary=True)
        values.append(predicted.float())
    values=torch.cat(values)
    advantages,returns=[],[]
    offset=0
    for length in lengths:
        advantage,target=compute_gae(rollout["rewards"][offset:offset+length],values[offset:offset+length],settings["gamma"],settings["gae_lambda"])
        advantages.append(advantage);returns.append(target);offset+=length
    return torch.cat(advantages),torch.cat(returns)


def warm_value_head(model,rollout,lengths,settings,optimizer,scaler,gradient_clip):
    targets=[]
    offset=0
    for length in lengths:
        rewards=rollout["rewards"][offset:offset+length]
        _,returns=compute_gae(rewards,torch.zeros_like(rewards),settings["gamma"],1.0)
        targets.append(returns);offset+=length
    targets=torch.cat(targets)
    for _ in range(settings["value_warmup_epochs"]):
        indices=torch.randperm(len(targets),device=targets.device)
        for start in range(0,len(indices),settings["minibatch"]):
            batch=indices[start:start+settings["minibatch"]]
            with mixed_precision(targets.device):
                _,_,values=model(rollout["queries"][batch],with_auxiliary=True)
            optimize_loss(F.smooth_l1_loss(values.float(),targets[batch]),model,optimizer,scaler,gradient_clip)


def train_ppo_step(model,reference,encoder,maps,config,rng,optimizer,scaler,warmup=False):
    settings=config["ppo"]
    rollout,lengths,metrics=collect_rollouts(model,encoder,maps,config,rng,reference)
    if warmup:
        warm_value_head(model,rollout,lengths,settings,optimizer,scaler,config["training"]["gradient_clip"])
    advantages,returns=calculate_rollout_targets(model,rollout,lengths,settings)
    advantages=(advantages-advantages.mean())/advantages.std(correction=0).clamp_min(1e-8)
    model.train()
    device=next(model.parameters()).device
    totals=[]
    stopped=False
    for _ in range(settings["epochs"]):
        indices=torch.randperm(len(advantages),device=device)
        for start in range(0,len(indices),settings["minibatch"]):
            batch=indices[start:start+settings["minibatch"]]
            with mixed_precision(device):
                logits,_,values=model(rollout["queries"][batch],with_auxiliary=True)
            distribution=torch.distributions.Categorical(logits=mask_action_logits(logits[:,encoder.action_slice],rollout["masks"][batch]))
            log_probs=distribution.log_prob(rollout["actions"][batch])
            log_ratio=log_probs-rollout["log_probs"][batch]
            ratio=log_ratio.exp()
            approx_kl=((ratio-1)-log_ratio).mean()
            if float(approx_kl.detach())>settings["target_kl"]:
                stopped=True;break
            clipped=ratio.clamp(1-settings["clip_epsilon"],1+settings["clip_epsilon"])
            policy_loss=-torch.minimum(ratio*advantages[batch],clipped*advantages[batch]).mean()
            value_loss=F.smooth_l1_loss(values.float(),returns[batch])
            entropy=distribution.entropy().mean()
            reference_kl=(distribution.probs*(distribution.logits-rollout["reference_log_probs"][batch])).sum(-1).mean()
            loss=policy_loss+settings["value_coefficient"]*value_loss-settings["entropy_coefficient"]*entropy+settings["reference_kl_coefficient"]*reference_kl
            optimize_loss(loss,model,optimizer,scaler,config["training"]["gradient_clip"])
            totals.append((float(loss.detach()),float(value_loss.detach()),float(entropy.detach()),float(approx_kl.detach()),float(reference_kl.detach())))
        if stopped:break
    averages=np.mean(totals,axis=0).tolist() if totals else [0.0]*5
    metrics.update(dict(zip(("loss","value_loss","entropy","approx_kl","reference_kl"),averages)))
    metrics["early_stop"]=stopped
    return metrics


def run_stage(stage, config_path="configs/t4.json", output_dir="runs/weighted_maze_v2/t4", steps=None, initialize_from=None, resume=False):
    if stage not in ("pretrain", "sft", "ppo"):
        raise ValueError("stage는 pretrain, sft, ppo 중 하나입니다.")
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
            raise ValueError("단계 간에는 같은 설정을 사용하세요.")
    else:
        model, encoder = create_model(config, device)
    settings = config[stage]
    parameters = model.parameters()
    if stage == "ppo":
        parameters = [{"params":[p for name,p in model.named_parameters() if not name.startswith("predict_value.")],"lr":settings["learning_rate"]}, {"params":model.predict_value.parameters(),"lr":settings["critic_learning_rate"]}]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config["seed"] + {"pretrain": 10, "sft": 20, "ppo": 30}[stage])
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    maps = build_maps(config, "train")
    expert_inputs, expert_targets = build_expert_examples(maps, encoder) if stage == "sft" else (None, None)
    expert_masks = encoder.decode_action_masks(expert_inputs) if stage == "sft" else None
    sampling_weights = payload["sampling_weights"].numpy() if payload is not None and "sampling_weights" in payload else None
    if stage == "sft" and sampling_weights is None:
        sampling_weights = np.ones(len(expert_inputs),dtype=np.float64)
    reference = copy.deepcopy(model).eval() if stage == "ppo" else None
    if reference is not None:
        if payload is not None:
            reference.load_state_dict(payload["reference"])
        reference.requires_grad_(False)
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
            extra["reference"] = {key:value.detach().cpu().clone() for key,value in reference.state_dict().items()}
        if stage == "sft":
            extra["sampling_weights"] = torch.as_tensor(sampling_weights)
        save_checkpoint(target, model, config, stage, step, **extra)
        write_json(output_dir / f"{stage}_history.json", history)

    completed = start_step
    try:
        for step in range(start_step + 1, total_steps + 1):
            if stage == "ppo":
                metrics = train_ppo_step(model, reference, encoder, maps, config, rng, optimizer, scaler, warmup=step == 1)
            else:
                model.train()
                if stage == "pretrain":
                    inputs, labels, costs = sample_world_batch(maps, encoder, config["environment"], settings["batch_size"], rng)
                    output_slice = encoder.position_slice
                else:
                    if step == 1 or step % settings["correction_every"] == 0:
                        sampling_weights = refresh_correction_weights(model,encoder,maps,expert_inputs,expert_targets,settings,rng)
                    selected = rng.choice(len(expert_inputs), size=settings["batch_size"],p=sampling_weights/sampling_weights.sum())
                    inputs, labels = expert_inputs[selected], expert_targets[selected]
                    output_slice = encoder.action_slice
                tokens = torch.as_tensor(inputs, device=device)
                targets = torch.as_tensor(labels, device=device, dtype=torch.bool if stage == "sft" else torch.long)
                with mixed_precision(device):
                    if stage == "pretrain":
                        all_logits, predicted_cost, _ = model(tokens, with_auxiliary=True)
                        logits = all_logits[:, output_slice]
                        cost_targets = torch.as_tensor(costs, device=device)
                        cost_loss = F.smooth_l1_loss(predicted_cost.float(), cost_targets)
                        loss = F.cross_entropy(logits.float(), targets) + settings["cost_loss_coefficient"] * cost_loss
                    else:
                        logits = mask_action_logits(model(tokens)[:,output_slice],expert_masks[selected])
                        loss = shortest_action_loss(logits,targets)
                gradient_norm = optimize_loss(loss, model, optimizer, scaler, config["training"]["gradient_clip"])
                correct = targets.gather(1,logits.argmax(-1)[:,None]).squeeze(1) if stage == "sft" else logits.argmax(-1) == targets
                metrics = {"loss":float(loss.detach()),"accuracy":float(correct.float().mean()),"gradient_norm":gradient_norm}
            completed = step
            history.append({"step": step, **metrics})
            if step == 1 or step % config["training"]["log_every"] == 0 or step == total_steps:
                print(f"[{stage} {step}/{total_steps}] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()) + f" elapsed={time.monotonic() - started:.1f}s", flush=True)
            if step % config["training"]["checkpoint_every"] == 0:
                persist(step)
        persist(completed)
    except KeyboardInterrupt:
        # Do not label a partially applied PPO update as a completed training step.
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
    parser.add_argument("stage", choices=("pretrain", "sft", "ppo"))
    parser.add_argument("--config", default="configs/t4.json")
    parser.add_argument("--output-dir", default="runs/weighted_maze_v2/t4")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run_stage(arguments.stage, arguments.config, arguments.output_dir, arguments.steps, arguments.initialize_from, arguments.resume)


if __name__ == "__main__":
    main()
