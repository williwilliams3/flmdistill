#!/usr/bin/env python3

"""Toy experiment for AR-to-FLM distillation under multiple training targets.

Supported student objectives:

1. ``ce``:
   standard cross-entropy on clean sampled tokens.
2. ``kl_exact``:
   KL to the exact denoiser induced by the full teacher sequence distribution.
3. ``ce_kl_exact``:
   mixed objective ``CE + lambda * KL_exact``.
4. ``kl_smc``:
   KL to a particle-based SMC approximation of the exact denoiser.

The script evaluates both local denoiser metrics and generation metrics against
the exact teacher sequence distribution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


MODE_DISPLAY_NAMES = {
    "ce": "CE only",
    "kl_exact": "KL to exact denoiser",
    "ce_kl_exact": "CE + exact-denoiser KL",
    "kl_smc": "KL to SMC denoiser",
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
    ode_steps: int = 192
    t_min: float = 0.02
    t_max: float = 0.98
    t_final: float = 0.995
    lr: float = 2e-3
    weight_decay: float = 1e-4
    log_every: int = 200
    seed: int = 0
    mode: str = "ce"
    lambda_kl: float = 1.0
    smc_particles: int = 64
    smc_resample_threshold: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(MODE_DISPLAY_NAMES),
        default="ce",
        help="Training objective for the student model.",
    )
    parser.add_argument("--lambda-kl", type=float, default=1.0, help="Weight in CE + lambda * KL_exact.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-sequences", type=int, default=8192)
    parser.add_argument("--ode-steps", type=int, default=192)
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
        help="If set, save JSON to <results-root>/<run-name or mode>/result_seed<seed>.json unless --save-json is set.",
    )
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(seed)
    return generator


def resolve_output_path(args: argparse.Namespace) -> Path | None:
    if args.save_json:
        return Path(args.save_json)
    if not args.results_root:
        return None
    run_name = args.run_name or args.mode
    return Path(args.results_root) / run_name / f"result_seed{args.seed}.json"


class StudentMLP(torch.nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = seq_len * vocab_size + 5
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

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        two_pi_t = 2.0 * math.pi * t
        four_pi_t = 4.0 * math.pi * t
        return torch.stack(
            [t, torch.sin(two_pi_t), torch.cos(two_pi_t), torch.sin(four_pi_t), torch.cos(four_pi_t)],
            dim=-1,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x_t.reshape(x_t.size(0), -1), self.time_features(t)], dim=-1)
        logits = self.net(features)
        return logits.view(x_t.size(0), self.seq_len, self.vocab_size)

    def probs(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x_t, t), dim=-1)


class SyntheticTeacher:
    """A tiny autoregressive teacher with clear contextual structure."""

    def __init__(self, vocab_size: int, device: torch.device) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.eye = torch.eye(vocab_size, device=device)
        self.start_probs = torch.tensor([0.52, 0.18, 0.18, 0.12], device=device)

    def next_probs_from_prefix(self, sequences: torch.Tensor, pos: int) -> torch.Tensor:
        batch = sequences.size(0)
        if pos == 0:
            return self.start_probs.expand(batch, -1)

        prev = sequences[:, pos - 1]
        prev2 = sequences[:, pos - 2] if pos >= 2 else None

        logits = torch.full((batch, self.vocab_size), -0.35, device=self.device)
        logits[torch.arange(batch, device=self.device), prev] += 1.65

        if pos % 2 == 1:
            logits[torch.arange(batch, device=self.device), (prev + 1) % self.vocab_size] += 0.95
        else:
            logits[torch.arange(batch, device=self.device), (prev + 2) % self.vocab_size] += 0.95

        if prev2 is not None:
            logits[torch.arange(batch, device=self.device), prev2] += 0.45

        return torch.softmax(logits, dim=-1)

    def sample_sequences(self, batch_size: int, seq_len: int, generator: torch.Generator) -> torch.Tensor:
        sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        for pos in range(seq_len):
            probs = self.next_probs_from_prefix(sequences, pos)
            sequences[:, pos] = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return sequences

    def exact_sequence_distribution(self, seq_len: int) -> dict[tuple[int, ...], float]:
        distribution: dict[tuple[int, ...], float] = {}
        for seq in itertools.product(range(self.vocab_size), repeat=seq_len):
            prob = float(self.start_probs[seq[0]].item())
            for pos in range(1, seq_len):
                prefix = torch.tensor(seq[:pos], device=self.device).view(1, -1)
                probs = self.next_probs_from_prefix(prefix, pos)[0]
                prob *= float(probs[seq[pos]].item())
            distribution[tuple(seq)] = prob
        return distribution

class ExactTeacherFlow:
    """Exact Gaussian-mixture teacher induced by the full AR sequence law."""

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
        self.exact_entropy = float(-(self.sequence_probs * self.log_sequence_probs).sum().item())
        self.exact_token_marginals = torch.einsum("m,mlk->lk", self.sequence_probs, self.endpoint_onehot)

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
def smc_particles_and_weights(
    teacher: SyntheticTeacher,
    config: Config,
    x_t: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, vocab_size = x_t.shape
    particles = torch.zeros(
        batch_size,
        config.smc_particles,
        seq_len,
        dtype=torch.long,
        device=teacher.device,
    )
    log_weights = torch.zeros(batch_size, config.smc_particles, device=teacher.device)

    means = t[:, None, None] * teacher.eye[None, :, :]
    sigma2 = torch.clamp((1.0 - t) ** 2, min=1e-8)
    diffs = x_t[:, :, None, :] - means[:, None, :, :]
    log_likelihood = -0.5 * torch.sum(diffs * diffs, dim=-1) / sigma2[:, None, None]

    for pos in range(seq_len):
        if pos == 0:
            prior = teacher.start_probs.expand(batch_size * config.smc_particles, -1)
        else:
            prefixes = particles[:, :, :pos].reshape(batch_size * config.smc_particles, pos)
            prior = teacher.next_probs_from_prefix(prefixes, pos)

        prior = torch.clamp(prior, min=1e-12).view(batch_size, config.smc_particles, vocab_size)
        proposal_logits = torch.log(prior) + log_likelihood[:, pos, None, :]
        proposal_probs = torch.softmax(proposal_logits, dim=-1)

        sampled = torch.multinomial(
            proposal_probs.reshape(batch_size * config.smc_particles, vocab_size),
            num_samples=1,
            generator=generator,
        ).view(batch_size, config.smc_particles)
        particles[:, :, pos] = sampled

        log_weights = log_weights + torch.logsumexp(proposal_logits, dim=-1)
        if pos == seq_len - 1:
            continue

        normalized = torch.softmax(log_weights, dim=-1)
        ess = 1.0 / torch.sum(normalized * normalized, dim=-1)
        resample_rows = torch.where(ess < config.smc_resample_threshold * config.smc_particles)[0]
        if resample_rows.numel() == 0:
            continue

        ancestor_idx = torch.multinomial(
            normalized[resample_rows],
            num_samples=config.smc_particles,
            replacement=True,
            generator=generator,
        )
        gathered_particles = torch.gather(
            particles[resample_rows],
            dim=1,
            index=ancestor_idx[:, :, None].expand(-1, -1, seq_len),
        )
        particles[resample_rows] = gathered_particles
        log_weights[resample_rows] = 0.0

    normalized = torch.softmax(log_weights, dim=-1)
    return particles, normalized


@torch.no_grad()
def smc_token_posteriors(
    teacher: SyntheticTeacher,
    config: Config,
    x_t: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    particles, normalized = smc_particles_and_weights(teacher, config, x_t, t, generator)
    particle_onehots = F.one_hot(particles, num_classes=teacher.vocab_size).float()
    return torch.sum(normalized[:, :, None, None] * particle_onehots, dim=1)


def sample_noised_batch(
    teacher: SyntheticTeacher,
    config: Config,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = teacher.sample_sequences(batch_size, config.seq_len, generator)
    x1 = teacher.eye[sequences]
    x0 = torch.randn(batch_size, config.seq_len, config.vocab_size, device=teacher.device, generator=generator)
    t = config.t_min + (config.t_max - config.t_min) * torch.rand(batch_size, device=teacher.device, generator=generator)
    x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
    return sequences, x0, x1, x_t, t


def distillation_kl(student_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        targets
        * (
            torch.log(torch.clamp(targets, min=1e-12))
            - torch.log(torch.clamp(student_probs, min=1e-12))
        ),
        dim=-1,
    ).mean()


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


def token_marginals_from_sequences(
    sequences: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    onehot = F.one_hot(sequences, num_classes=vocab_size).float()
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


def summarize_generated_samples(
    sequences: torch.Tensor,
    exact_distribution: dict[tuple[int, ...], float],
    exact_token_marginals: torch.Tensor,
) -> dict[str, float]:
    empirical = empirical_distribution(sequences)
    token_marginals = token_marginals_from_sequences(sequences, exact_token_marginals.size(-1))
    token_tv_by_position = 0.5 * torch.sum(torch.abs(token_marginals - exact_token_marginals), dim=-1)
    sequence_cross_entropy = sequence_cross_entropy_from_samples(sequences, exact_distribution)
    token_cross_entropy = sequence_cross_entropy / sequences.size(1)

    return {
        "sequence_cross_entropy": sequence_cross_entropy,
        "token_cross_entropy": token_cross_entropy,
        "token_perplexity": math.exp(token_cross_entropy),
        "sequence_kl_emp_to_teacher": empirical_kl_to_exact(empirical, exact_distribution),
        "sequence_tv_to_teacher": total_variation(exact_distribution, empirical),
        "token_tv_mean": float(token_tv_by_position.mean().item()),
        "token_tv_max": float(token_tv_by_position.max().item()),
    }


@torch.no_grad()
def evaluate_model(
    model: StudentMLP,
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    generator: torch.Generator,
) -> dict[str, float]:
    sequences, _, _, x_t, t = sample_noised_batch(teacher, config, config.eval_batch_size, generator)
    exact_targets = exact_flow.token_posteriors(x_t, t)
    probs = model.probs(x_t, t)
    logits = model.forward(x_t, t)

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
def generate_from_student(
    model: StudentMLP,
    config: Config,
    num_sequences: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = config.eval_batch_size
    outputs = []
    dt = config.t_final / config.ode_steps

    for start in range(0, num_sequences, batch_size):
        current_batch = min(batch_size, num_sequences - start)
        x = torch.randn(
            current_batch,
            config.seq_len,
            config.vocab_size,
            device=next(model.parameters()).device,
            generator=generator,
        )

        for step in range(config.ode_steps):
            t_value = step * dt
            t = torch.full((current_batch,), t_value, device=x.device)
            probs = model.probs(x, t)
            velocity = (probs - x) / max(1.0 - t_value, 1e-4)
            x = x + dt * velocity

        final_t = torch.full((current_batch,), config.t_final, device=x.device)
        final_probs = model.probs(x, final_t)
        tokens = torch.multinomial(final_probs.reshape(-1, config.vocab_size), 1, generator=generator)
        outputs.append(tokens.view(current_batch, config.seq_len))

    return torch.cat(outputs, dim=0)


def compute_training_loss(
    mode: str,
    config: Config,
    logits: torch.Tensor,
    probs: torch.Tensor,
    sequences: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    lambda_kl: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostics: dict[str, float] = {}
    ce_loss = F.cross_entropy(logits.reshape(-1, teacher.vocab_size), sequences.reshape(-1))
    diagnostics["ce_loss"] = float(ce_loss.item())

    exact_kl_loss: torch.Tensor | None = None
    if mode in {"kl_exact", "ce_kl_exact"}:
        with torch.no_grad():
            exact_targets = exact_flow.token_posteriors(x_t, t)
        exact_kl_loss = distillation_kl(probs, exact_targets)
        diagnostics["exact_kl_loss"] = float(exact_kl_loss.item())

    if mode == "kl_smc":
        with torch.no_grad():
            smc_targets = smc_token_posteriors(teacher, config, x_t, t, generator)
        smc_kl_loss = distillation_kl(probs, smc_targets)
        diagnostics["smc_kl_loss"] = float(smc_kl_loss.item())
        return smc_kl_loss, diagnostics

    if mode == "ce":
        return ce_loss, diagnostics
    if mode == "kl_exact":
        assert exact_kl_loss is not None
        return exact_kl_loss, diagnostics
    if mode == "ce_kl_exact":
        assert exact_kl_loss is not None
        total_loss = ce_loss + lambda_kl * exact_kl_loss
        diagnostics["lambda_kl"] = float(lambda_kl)
        diagnostics["total_loss"] = float(total_loss.item())
        return total_loss, diagnostics

    raise ValueError(f"Unknown mode: {mode}")


def train_student(
    teacher: SyntheticTeacher,
    exact_flow: ExactTeacherFlow,
    config: Config,
    generator: torch.Generator,
) -> tuple[StudentMLP, list[dict[str, float]]]:
    model = StudentMLP(config.seq_len, config.vocab_size, config.hidden_dim).to(teacher.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for step in range(1, config.train_steps + 1):
        sequences, _, _, x_t, t = sample_noised_batch(teacher, config, config.batch_size, generator)
        logits = model.forward(x_t, t)
        probs = torch.softmax(logits, dim=-1)
        loss, loss_parts = compute_training_loss(
            config.mode,
            config,
            logits,
            probs,
            sequences,
            x_t,
            t,
            teacher,
            exact_flow,
            config.lambda_kl,
            generator,
        )

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
                f"[{config.mode:17s}] step={step:4d} "
                f"train_loss={loss.item():.4f} "
                f"hard_ce={metrics['hard_ce']:.4f} "
                f"exact_kl={metrics['exact_kl']:.4f} "
                f"token_acc={metrics['token_acc']:.4f} "
                f"exact_vel_mse={metrics['exact_velocity_mse']:.4f}"
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
        mode=args.mode,
        lambda_kl=args.lambda_kl,
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
    model, history = train_student(teacher, exact_flow, config, train_generator)
    generated = generate_from_student(model, config, config.eval_sequences, sample_generator)
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
        "kind": "student",
        "mode": config.mode,
        "display_name": MODE_DISPLAY_NAMES[config.mode],
        "run_name": args.run_name or config.mode,
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
        f"{MODE_DISPLAY_NAMES[config.mode]}: "
        f"hard_ce={local_metrics['hard_ce']:.4f}, "
        f"exact_kl={local_metrics['exact_kl']:.4f}, "
        f"token_acc={local_metrics['token_acc']:.4f}, "
        f"exact_vel_mse={local_metrics['exact_velocity_mse']:.4f}, "
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
