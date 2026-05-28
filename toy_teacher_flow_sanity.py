#!/usr/bin/env python3

"""Sanity check for exact flow generation induced by an AR teacher.

This script uses the same synthetic autoregressive teacher as
`toy_flm_distillation_experiment.py`, but does not train a student model.
Instead, it:

1. Enumerates the exact teacher sequence distribution.
2. Builds the exact Gaussian-mixture denoiser and vector field induced by that
   discrete sequence distribution.
3. Generates sequences in two ways:
   - direct autoregressive sampling from the teacher,
   - solving the exact probability-flow ODE and decoding at the end.
4. Compares both samplers against the exact teacher distribution.

The purpose is a sanity check: if the Gaussian-mixture flow implementation is
correct, then its generation metrics should be close to the AR baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from toy_flm_distillation_experiment import SyntheticTeacher, get_device


@dataclass
class Config:
    vocab_size: int = 4
    seq_len: int = 6
    eval_sequences: int = 8192
    eval_batch_size: int = 256
    ode_steps: int = 192
    t_final: float = 0.995
    seed: int = 0
    integrator: str = "rk4"
    decode: str = "joint"
    sampler: str = "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-sequences", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--ode-steps", type=int, default=192)
    parser.add_argument("--t-final", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--integrator", type=str, choices=["euler", "rk4"], default="rk4")
    parser.add_argument("--decode", type=str, choices=["joint", "token", "argmax"], default="joint")
    parser.add_argument("--sampler", type=str, choices=["ar", "flow", "both"], default="both")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--results-root", type=str, default="")
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path | None:
    if args.save_json:
        return Path(args.save_json)
    if not args.results_root:
        return None
    run_name = args.run_name or {
        "ar": "01_true_ar",
        "flow": "02_true_ode",
        "both": "00_true_both",
    }[args.sampler]
    return Path(args.results_root) / run_name / f"result_seed{args.seed}.json"


def empirical_distribution(sequences: torch.Tensor) -> dict[tuple[int, ...], float]:
    counts: dict[tuple[int, ...], int] = {}
    total = sequences.size(0)
    for row in sequences.tolist():
        key = tuple(int(x) for x in row)
        counts[key] = counts.get(key, 0) + 1
    return {key: value / total for key, value in counts.items()}


def total_variation(
    p_exact: dict[tuple[int, ...], float],
    p_empirical: dict[tuple[int, ...], float],
) -> float:
    support = set(p_exact) | set(p_empirical)
    return 0.5 * sum(abs(p_exact.get(seq, 0.0) - p_empirical.get(seq, 0.0)) for seq in support)


class ExactTeacherFlow:
    """Exact Gaussian-mixture teacher induced by the AR sequence law."""

    def __init__(self, teacher: SyntheticTeacher, seq_len: int) -> None:
        self.teacher = teacher
        self.seq_len = seq_len
        self.vocab_size = teacher.vocab_size

        exact_distribution = teacher.exact_sequence_distribution(seq_len)
        seq_list = list(exact_distribution.keys())
        prob_list = list(exact_distribution.values())

        self.exact_distribution = exact_distribution
        self.sequence_tensor = torch.tensor(seq_list, dtype=torch.long, device=teacher.device)
        self.sequence_probs = torch.tensor(prob_list, dtype=torch.float32, device=teacher.device)
        self.log_sequence_probs = torch.log(torch.clamp(self.sequence_probs, min=1e-30))
        self.endpoint_onehot = teacher.eye[self.sequence_tensor]
        self.endpoint_flat = self.endpoint_onehot.reshape(self.endpoint_onehot.size(0), -1)

        self.exact_entropy = float(
            -(self.sequence_probs * self.log_sequence_probs).sum().item()
        )
        self.exact_token_marginals = torch.einsum(
            "m,mlk->lk", self.sequence_probs, self.endpoint_onehot
        )

    def _component_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x_t.reshape(x_t.size(0), -1)
        sigma2 = torch.clamp((1.0 - t) ** 2, min=1e-8)
        beta = t / sigma2
        alignment = x_flat @ self.endpoint_flat.T
        return self.log_sequence_probs[None, :] + beta[:, None] * alignment

    @torch.no_grad()
    def posterior_over_sequences(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._component_logits(x_t, t), dim=-1)

    @torch.no_grad()
    def token_posteriors(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        posterior = self.posterior_over_sequences(x_t, t)
        token_flat = posterior @ self.endpoint_flat
        return token_flat.view(x_t.size(0), self.seq_len, self.vocab_size)

    @torch.no_grad()
    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        denoiser = self.token_posteriors(x_t, t)
        scale = torch.clamp((1.0 - t)[:, None, None], min=1e-4)
        return (denoiser - x_t) / scale

    @torch.no_grad()
    def decode(self, x_t: torch.Tensor, t: torch.Tensor, generator: torch.Generator, mode: str) -> torch.Tensor:
        if mode == "joint":
            posterior = self.posterior_over_sequences(x_t, t)
            indices = torch.multinomial(posterior, num_samples=1, generator=generator).squeeze(-1)
            return self.sequence_tensor[indices]

        token_probs = self.token_posteriors(x_t, t)
        if mode == "token":
            tokens = torch.multinomial(
                token_probs.reshape(-1, self.vocab_size), num_samples=1, generator=generator
            )
            return tokens.view(x_t.size(0), self.seq_len)

        return token_probs.argmax(dim=-1)


@torch.no_grad()
def rk4_step(flow: ExactTeacherFlow, x: torch.Tensor, t_value: float, dt: float) -> torch.Tensor:
    batch_size = x.size(0)
    device = x.device

    t1 = torch.full((batch_size,), t_value, device=device)
    k1 = flow.velocity(x, t1)

    t2 = torch.full((batch_size,), t_value + 0.5 * dt, device=device)
    k2 = flow.velocity(x + 0.5 * dt * k1, t2)
    k3 = flow.velocity(x + 0.5 * dt * k2, t2)

    t4 = torch.full((batch_size,), t_value + dt, device=device)
    k4 = flow.velocity(x + dt * k3, t4)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@torch.no_grad()
def sample_sequences_via_flow(
    flow: ExactTeacherFlow,
    config: Config,
    num_sequences: int,
    generator: torch.Generator,
) -> torch.Tensor:
    outputs = []
    dt = config.t_final / config.ode_steps
    device = flow.teacher.device

    for start in range(0, num_sequences, config.eval_batch_size):
        current_batch = min(config.eval_batch_size, num_sequences - start)
        x = torch.randn(
            current_batch,
            config.seq_len,
            config.vocab_size,
            device=device,
            generator=generator,
        )

        for step in range(config.ode_steps):
            t_value = step * dt
            if config.integrator == "rk4":
                x = rk4_step(flow, x, t_value, dt)
            else:
                t = torch.full((current_batch,), t_value, device=device)
                x = x + dt * flow.velocity(x, t)

        final_t = torch.full((current_batch,), config.t_final, device=device)
        outputs.append(flow.decode(x, final_t, generator, config.decode))

    return torch.cat(outputs, dim=0)


def token_marginals_from_sequences(
    sequences: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(sequences, num_classes=vocab_size).float()
    return onehot.mean(dim=0)


def sequence_cross_entropy_from_samples(
    sequences: torch.Tensor,
    exact_distribution: dict[tuple[int, ...], float],
) -> float:
    total = 0.0
    for row in sequences.tolist():
        total -= math.log(exact_distribution[tuple(int(x) for x in row)])
    return total / len(sequences)


def empirical_kl_to_exact(
    p_empirical: dict[tuple[int, ...], float],
    p_exact: dict[tuple[int, ...], float],
) -> float:
    total = 0.0
    for seq, p_emp in p_empirical.items():
        total += p_emp * (math.log(p_emp) - math.log(p_exact[seq]))
    return total


def summarize_samples(
    label: str,
    sequences: torch.Tensor,
    exact_distribution: dict[tuple[int, ...], float],
    exact_token_marginals: torch.Tensor,
) -> dict[str, float]:
    empirical = empirical_distribution(sequences)
    token_marginals = token_marginals_from_sequences(sequences, exact_token_marginals.size(-1))
    token_tv_by_position = 0.5 * torch.sum(torch.abs(token_marginals - exact_token_marginals), dim=-1)
    sequence_cross_entropy = sequence_cross_entropy_from_samples(sequences, exact_distribution)
    token_cross_entropy = sequence_cross_entropy / sequences.size(1)

    metrics = {
        "sequence_cross_entropy": sequence_cross_entropy,
        "token_cross_entropy": token_cross_entropy,
        "token_perplexity": math.exp(token_cross_entropy),
        "sequence_kl_emp_to_teacher": empirical_kl_to_exact(empirical, exact_distribution),
        "sequence_tv_to_teacher": total_variation(exact_distribution, empirical),
        "token_tv_mean": float(token_tv_by_position.mean().item()),
        "token_tv_max": float(token_tv_by_position.max().item()),
    }

    print(
        f"{label:12s} | "
        f"seq_CE={metrics['sequence_cross_entropy']:.4f} "
        f"tok_CE={metrics['token_cross_entropy']:.4f} "
        f"tok_PPL={metrics['token_perplexity']:.4f} "
        f"seq_KL={metrics['sequence_kl_emp_to_teacher']:.4f} "
        f"seq_TV={metrics['sequence_tv_to_teacher']:.4f} "
        f"tok_TV_mean={metrics['token_tv_mean']:.4f} "
        f"tok_TV_max={metrics['token_tv_max']:.4f}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    config = Config(
        eval_sequences=args.eval_sequences,
        eval_batch_size=args.eval_batch_size,
        ode_steps=args.ode_steps,
        t_final=args.t_final,
        seed=args.seed,
        integrator=args.integrator,
        decode=args.decode,
        sampler=args.sampler,
    )
    output_path = resolve_output_path(args)

    torch.manual_seed(config.seed)
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(config.seed)

    print(f"Using device: {device}")
    print(f"Config: {config}")
    if output_path is not None:
        print(f"Results will be saved to: {output_path}")

    teacher = SyntheticTeacher(config.vocab_size, device)
    flow = ExactTeacherFlow(teacher, config.seq_len)

    start_time = time.time()
    ar_samples = None
    flow_samples = None
    if config.sampler in {"ar", "both"}:
        ar_samples = teacher.sample_sequences(config.eval_sequences, config.seq_len, generator)
    if config.sampler in {"flow", "both"}:
        flow_samples = sample_sequences_via_flow(flow, config, config.eval_sequences, generator)
    duration = time.time() - start_time

    teacher_entropy_per_token = flow.exact_entropy / config.seq_len
    print(f"Exact teacher support size: {len(flow.exact_distribution)}")
    print(
        f"Teacher entropy: H[p_AR]={flow.exact_entropy:.4f} nats/seq "
        f"({teacher_entropy_per_token:.4f} nats/token)"
    )
    print()
    print("Comparison against exact teacher distribution")
    print("-------------------------------------------")
    ar_metrics = None
    flow_metrics = None
    pairwise_tv = None

    if ar_samples is not None:
        ar_metrics = summarize_samples(
            "AR sample",
            ar_samples,
            flow.exact_distribution,
            flow.exact_token_marginals,
        )
    if flow_samples is not None:
        flow_metrics = summarize_samples(
            "Flow ODE",
            flow_samples,
            flow.exact_distribution,
            flow.exact_token_marginals,
        )
    if ar_samples is not None and flow_samples is not None:
        ar_empirical = empirical_distribution(ar_samples)
        flow_empirical = empirical_distribution(flow_samples)
        pairwise_tv = total_variation(ar_empirical, flow_empirical)

    print()
    if pairwise_tv is not None:
        print(f"Empirical TV between AR samples and flow samples: {pairwise_tv:.4f}")
    print(f"Total runtime: {duration:.1f}s")

    if config.sampler == "ar":
        summary_metrics = ar_metrics
        display_name = "True AR sampling"
    elif config.sampler == "flow":
        summary_metrics = flow_metrics
        display_name = "True ODE sampling"
    else:
        summary_metrics = {
            "ar_sequence_tv_to_teacher": ar_metrics["sequence_tv_to_teacher"] if ar_metrics is not None else None,
            "flow_sequence_tv_to_teacher": flow_metrics["sequence_tv_to_teacher"] if flow_metrics is not None else None,
            "ar_vs_flow_empirical_tv": pairwise_tv,
        }
        display_name = "AR and exact flow sanity check"

    results = {
        "kind": "oracle",
        "sampler": config.sampler,
        "display_name": display_name,
        "run_name": args.run_name or config.sampler,
        "config": asdict(config),
        "teacher_entropy_per_sequence": flow.exact_entropy,
        "teacher_entropy_per_token": teacher_entropy_per_token,
        "ar_metrics": ar_metrics,
        "flow_metrics": flow_metrics,
        "ar_vs_flow_empirical_tv": pairwise_tv,
        "summary_metrics": summary_metrics,
        "duration_sec": duration,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
