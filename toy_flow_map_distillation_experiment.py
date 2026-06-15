#!/usr/bin/env python3

"""Toy direct training for a Flow Map two-time denoiser.

This trains a two-time denoiser delta_theta(x_s, s, t) directly from an
autoregressive teacher. The objective mirrors the PSD/semigroup objective from
Flow Map Language Models, but the diagonal denoiser target is supplied directly
from the AR teacher:

1. ``exact``:
   use the exact enumerated AR-induced denoiser D_t.
2. ``smc``:
   use an SMC approximation to the same denoiser.

The off-diagonal target is the stop-gradient semigroup teacher

    gamma * delta_{s,u}(x_s) + (1 - gamma) * delta_{u,t}(X_{s,u}(x_s)).

This avoids first training a separate one-time FLM denoiser.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from toy_flm_distillation_experiment import (
    ExactTeacherFlow,
    SyntheticTeacher,
    distillation_kl,
    get_device,
    make_generator,
    sample_noised_batch,
    smc_token_posteriors,
    summarize_generated_samples,
)


TARGET_DISPLAY_NAMES = {
    "exact": "Flow Map direct (exact denoiser)",
    "smc": "Flow Map direct (SMC denoiser)",
}


@dataclass
class Config:
    vocab_size: int = 4
    seq_len: int = 6
    hidden_dim: int = 256
    train_steps: int = 2000
    batch_size: int = 256
    eval_batch_size: int = 1024
    eval_sequences: int = 8192
    ode_steps: int = 8
    t_min: float = 0.02
    t_max: float = 0.98
    t_final: float = 0.995
    min_interval: float = 1e-3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    log_every: int = 200
    seed: int = 0
    diagonal_target: str = "exact"
    diagonal_weight: float = 1.0
    semigroup_weight: float = 1.0
    smc_particles: int = 64
    smc_resample_threshold: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagonal-target", type=str, choices=sorted(TARGET_DISPLAY_NAMES), default="exact")
    parser.add_argument("--diagonal-weight", type=float, default=1.0)
    parser.add_argument("--semigroup-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-sequences", type=int, default=8192)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smc-particles", type=int, default=64)
    parser.add_argument("--smc-resample-threshold", type=float, default=0.5)
    parser.add_argument("--run-name", type=str, default="", help="Logical run name used for saved JSON paths.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="If set, save JSON to <results-root>/<run-name>/result_seed<seed>.json unless --save-json is set.",
    )
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path | None:
    if args.save_json:
        return Path(args.save_json)
    if not args.results_root:
        return None
    run_name = args.run_name or f"flowmap_{args.diagonal_target}"
    return Path(args.results_root) / run_name / f"result_seed{args.seed}.json"


class TwoTimeDenoiserMLP(torch.nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = seq_len * vocab_size + 10
        output_dim = seq_len * vocab_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def time_features(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        two_pi_s = 2.0 * math.pi * s
        four_pi_s = 4.0 * math.pi * s
        two_pi_t = 2.0 * math.pi * t
        four_pi_t = 4.0 * math.pi * t
        return torch.stack(
            [
                s,
                torch.sin(two_pi_s),
                torch.cos(two_pi_s),
                torch.sin(four_pi_s),
                torch.cos(four_pi_s),
                t,
                torch.sin(two_pi_t),
                torch.cos(two_pi_t),
                torch.sin(four_pi_t),
                torch.cos(four_pi_t),
            ],
            dim=-1,
        )

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x_s.reshape(x_s.size(0), -1), self.time_features(s, t)], dim=-1)
        logits = self.net(features)
        return logits.view(x_s.size(0), self.seq_len, self.vocab_size)

    def probs(self, x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x_s, s, t), dim=-1)


def flow_map_update(x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, delta_st: torch.Tensor) -> torch.Tensor:
    s_view = s[:, None, None]
    t_view = t[:, None, None]
    denom = torch.clamp(1.0 - s_view, min=1e-6)
    return ((1.0 - t_view) / denom) * x_s + ((t_view - s_view) / denom) * delta_st


def sample_intervals(config: Config, batch_size: int, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = max(config.t_max - config.min_interval, 0.0)
    s = max_start * torch.rand(batch_size, device=device, generator=generator)
    max_gap = torch.clamp(config.t_max - s, min=config.min_interval)
    gap = config.min_interval + (max_gap - config.min_interval) * torch.rand(batch_size, device=device, generator=generator)
    t = torch.clamp(s + gap, max=config.t_max)
    u = s + (t - s) * torch.rand(batch_size, device=device, generator=generator)
    return s, u, t


def noised_at_time(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1


@torch.no_grad()
def diagonal_targets(
    diagonal_target: str,
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    x_t: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if diagonal_target == "exact":
        return exact_flow.token_posteriors(x_t, t)
    if diagonal_target == "smc":
        return smc_token_posteriors(teacher, config, x_t, t, generator)
    raise ValueError(f"Unknown diagonal target: {diagonal_target}")


def compute_flow_map_loss(
    model: TwoTimeDenoiserMLP,
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    sequences, x0, x1, x_diag, t_diag = sample_noised_batch(teacher, config, config.batch_size, generator)

    diag_target = diagonal_targets(
        config.diagonal_target,
        teacher,
        exact_flow,
        config,
        x_diag,
        t_diag,
        generator,
    )
    diag_probs = model.probs(x_diag, t_diag, t_diag)
    diag_loss = distillation_kl(diag_probs, diag_target)

    s, u, t = sample_intervals(config, config.batch_size, teacher.device, generator)
    x_s = noised_at_time(x0, x1, s)
    pred_st = model.probs(x_s, s, t)

    with torch.no_grad():
        delta_su = model.probs(x_s, s, u)
        x_su = flow_map_update(x_s, s, u, delta_su)
        delta_ut = model.probs(x_su, u, t)
        gamma = ((1.0 - t) * (u - s)) / torch.clamp((1.0 - u) * (t - s), min=1e-8)
        semigroup_target = gamma[:, None, None] * delta_su + (1.0 - gamma)[:, None, None] * delta_ut

    semigroup_loss = distillation_kl(pred_st, semigroup_target)
    total_loss = config.diagonal_weight * diag_loss + config.semigroup_weight * semigroup_loss
    diagnostics = {
        "diag_kl_loss": float(diag_loss.item()),
        "semigroup_kl_loss": float(semigroup_loss.item()),
        "total_loss": float(total_loss.item()),
        "diagonal_weight": float(config.diagonal_weight),
        "semigroup_weight": float(config.semigroup_weight),
    }
    return total_loss, diagnostics


@torch.no_grad()
def evaluate_model(
    model: TwoTimeDenoiserMLP,
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    generator: torch.Generator,
) -> dict[str, float]:
    sequences, _, _, x_t, t = sample_noised_batch(teacher, config, config.eval_batch_size, generator)
    exact_targets = exact_flow.token_posteriors(x_t, t)
    logits = model.forward(x_t, t, t)
    probs = torch.softmax(logits, dim=-1)

    hard_ce = F.cross_entropy(logits.reshape(-1, config.vocab_size), sequences.reshape(-1)).item()
    exact_kl = distillation_kl(probs, exact_targets).item()
    token_acc = (probs.argmax(dim=-1) == sequences).float().mean().item()

    scale = torch.clamp((1.0 - t)[:, None, None], min=1e-4)
    exact_velocity = (exact_targets - x_t) / scale
    student_velocity = (probs - x_t) / scale
    exact_velocity_mse = torch.mean((exact_velocity - student_velocity) ** 2).item()
    return {
        "hard_ce": hard_ce,
        "exact_kl": exact_kl,
        "token_acc": token_acc,
        "exact_velocity_mse": exact_velocity_mse,
    }


@torch.no_grad()
def generate_from_flow_map(
    model: TwoTimeDenoiserMLP,
    config: Config,
    num_sequences: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = config.eval_batch_size
    outputs = []
    dt = config.t_final / config.ode_steps
    device = next(model.parameters()).device

    for start in range(0, num_sequences, batch_size):
        current_batch = min(batch_size, num_sequences - start)
        x = torch.randn(
            current_batch,
            config.seq_len,
            config.vocab_size,
            device=device,
            generator=generator,
        )

        for step in range(config.ode_steps):
            s_value = step * dt
            t_value = (step + 1) * dt
            s = torch.full((current_batch,), s_value, device=device)
            t = torch.full((current_batch,), t_value, device=device)
            delta = model.probs(x, s, t)
            x = flow_map_update(x, s, t, delta)

        final_t = torch.full((current_batch,), config.t_final, device=device)
        final_probs = model.probs(x, final_t, final_t)
        tokens = torch.multinomial(final_probs.reshape(-1, config.vocab_size), 1, generator=generator)
        outputs.append(tokens.view(current_batch, config.seq_len))

    return torch.cat(outputs, dim=0)


def train_model(
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    generator: torch.Generator,
) -> tuple[TwoTimeDenoiserMLP, list[dict[str, float]]]:
    model = TwoTimeDenoiserMLP(config.seq_len, config.vocab_size, config.hidden_dim).to(teacher.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for step in range(1, config.train_steps + 1):
        loss, loss_parts = compute_flow_map_loss(model, teacher, exact_flow, config, generator)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            metrics = evaluate_model(model, teacher, exact_flow, config, generator)
            metrics["train_loss"] = float(loss.item())
            metrics["step"] = float(step)
            metrics.update(loss_parts)
            history.append(metrics)
            print(
                f"[flowmap-{config.diagonal_target:5s}] step={step:4d} "
                f"train_loss={loss.item():.4f} "
                f"diag_kl={loss_parts['diag_kl_loss']:.4f} "
                f"semi_kl={loss_parts['semigroup_kl_loss']:.4f} "
                f"hard_ce={metrics['hard_ce']:.4f} "
                f"exact_kl={metrics['exact_kl']:.4f} "
                f"token_acc={metrics['token_acc']:.4f}"
            )

    return model, history


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    config = Config(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_sequences=args.eval_sequences,
        ode_steps=args.ode_steps,
        hidden_dim=args.hidden_dim,
        log_every=args.log_every,
        seed=args.seed,
        diagonal_target=args.diagonal_target,
        diagonal_weight=args.diagonal_weight,
        semigroup_weight=args.semigroup_weight,
        smc_particles=args.smc_particles,
        smc_resample_threshold=args.smc_resample_threshold,
    )
    output_path = resolve_output_path(args)
    torch.manual_seed(config.seed)

    train_generator = make_generator(device, config.seed)
    eval_generator = make_generator(device, config.seed + 1)
    sample_generator = make_generator(device, config.seed + 2)

    print(f"Using device: {device}")
    print(f"Config: {config}")
    if output_path is not None:
        print(f"Results will be saved to: {output_path}")

    teacher = SyntheticTeacher(config.vocab_size, device)
    exact_flow = ExactTeacherFlow(teacher, config.seq_len)
    print(f"Enumerated exact teacher distribution over {len(exact_flow.exact_distribution)} sequences.")

    start_time = time.time()
    model, history = train_model(teacher, exact_flow, config, train_generator)
    generated = generate_from_flow_map(model, config, config.eval_sequences, sample_generator)
    local_metrics = evaluate_model(model, teacher, exact_flow, config, eval_generator)
    generation_metrics = summarize_generated_samples(
        generated,
        exact_flow.exact_distribution,
        exact_flow.exact_token_marginals,
    )
    duration = time.time() - start_time

    summary_metrics = {}
    summary_metrics.update(local_metrics)
    summary_metrics.update(generation_metrics)

    results = {
        "kind": "flow_map",
        "diagonal_target": config.diagonal_target,
        "display_name": TARGET_DISPLAY_NAMES[config.diagonal_target],
        "run_name": args.run_name or f"flowmap_{config.diagonal_target}",
        "config": asdict(config),
        "local_metrics": local_metrics,
        "generation_metrics": generation_metrics,
        "summary_metrics": summary_metrics,
        "history": history,
        "duration_sec": duration,
    }

    print()
    print("Final metrics")
    print("-------------")
    print(
        f"{TARGET_DISPLAY_NAMES[config.diagonal_target]}: "
        f"hard_ce={local_metrics['hard_ce']:.4f}, "
        f"exact_kl={local_metrics['exact_kl']:.4f}, "
        f"token_acc={local_metrics['token_acc']:.4f}, "
        f"seq_ce={generation_metrics['sequence_cross_entropy']:.4f}, "
        f"seq_kl={generation_metrics['sequence_kl_emp_to_teacher']:.4f}, "
        f"seq_tv={generation_metrics['sequence_tv_to_teacher']:.4f}"
    )
    print(f"Total runtime: {duration:.1f}s")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
