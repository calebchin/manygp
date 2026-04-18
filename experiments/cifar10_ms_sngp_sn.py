"""
CIFAR-10 Multi-Similarity Loss + SNGP with spectrally-normalized WRN-28-10 backbone.

Ablation of cifar10_ms_sngp.py: replaces the plain WRN-28-10 encoder
(CifarResNetEncoder, no spectral norm) with the spectrally-normalized
WideResNet28SNGPBackbone used in the standard SNGP baseline.

Why this ablation?
  The current MS-SNGP uses a plain backbone (no SN). The original SNGP paper
  argues that SN enforces bi-Lipschitz smoothness, which aligns distance in
  feature space with the GP's distance-based uncertainty estimates. This
  experiment tests whether adding SN to the backbone improves calibration
  and OOD detection when MS Loss is also applied.

Resume support
  Pass --auto-resume to automatically find the latest checkpoint in the seed
  directory and resume the existing W&B run.  Alternatively pass
  --resume-from <path.pt> and optionally --wandb-resume-id <run_id>.

Usage:
    # Fresh training
    python experiments/cifar10_ms_sngp_sn.py \\
        --config configs/experiment_april4_ms_sngp_sn.yaml --seed 0

    # Auto-resume a crashed job
    python experiments/cifar10_ms_sngp_sn.py \\
        --config configs/experiment_april4_ms_sngp_sn.yaml --seed 1 \\
        --run-name ms_sngp_sn_seed1 --auto-resume
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.sngp import SNGPResNetClassifier, laplace_predictive_probs
from src.training.contrastive import MultiSimilarityLoss
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper
from src.utils.model_summary import print_model_summary


# ── Resume helpers ─────────────────────────────────────────────────────────────

def find_wandb_run_id(entity: str, project: str, run_name: str) -> "str | None":
    """Return the W&B internal run ID for a run with the given display name."""
    try:
        import wandb as _wandb
        api = _wandb.Api()
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_name},
            per_page=10,
        )
        for r in runs:
            if r.name == run_name:
                print(f"  Found W&B run '{run_name}' → id={r.id}  state={r.state}")
                return r.id
    except Exception as exc:
        print(f"  W&B run lookup failed: {exc}")
    return None


def find_latest_checkpoint(seed_dir: Path) -> "Path | None":
    """Scan seed_dir/**/*.pt for the epoch checkpoint with the highest epoch number."""
    import re
    best_epoch, best_path = -1, None
    for p in seed_dir.glob("**/*_epoch*_*.pt"):
        if "best_model" in p.name:
            continue
        m = re.search(r"_epoch(\d+)_", p.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_path = epoch, p
    return best_path


def load_existing_checkpoints(
    ckpt_dir: Path,
    stem: str,
    metric_name: str,
    lower_is_better: bool,
    top_k: int,
) -> list:
    """Populate saved_checkpoints list from epoch checkpoint files already on disk."""
    import re
    pattern = re.compile(
        rf"{re.escape(stem)}_epoch(\d+)_{re.escape(metric_name)}([0-9.]+)\.pt$"
    )
    found = []
    for f in ckpt_dir.glob(f"{stem}_epoch*_{metric_name}*.pt"):
        m = pattern.match(f.name)
        if m:
            found.append({"epoch": int(m.group(1)), "metric": float(m.group(2)), "path": f})
    found.sort(key=lambda x: x["metric"], reverse=not lower_is_better)
    return found[:top_k]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def resolve_timestamped_checkpoint_path(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p.parent / ts / p.name)


def update_topk_checkpoints(
    saved_checkpoints, top_k, checkpoint_path, state,
    metric_name, metric_value, epoch, lower_is_better=False,
):
    if top_k <= 0:
        return
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    candidate = p.parent / f"{p.stem}_epoch{epoch:03d}_{metric_name}{metric_value:.4f}{p.suffix or '.pt'}"
    if len(saved_checkpoints) < top_k:
        torch.save(state, candidate)
        saved_checkpoints.append({"metric": metric_value, "path": candidate, "epoch": epoch})
    else:
        worst = (max if lower_is_better else min)(
            saved_checkpoints, key=lambda x: (x["metric"], -x["epoch"])
        )
        if (lower_is_better and metric_value >= worst["metric"]) or \
           (not lower_is_better and metric_value <= worst["metric"]):
            return
        torch.save(state, candidate)
        saved_checkpoints.append({"metric": metric_value, "path": candidate, "epoch": epoch})
        worst["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst)
    saved_checkpoints.sort(key=lambda x: x["metric"], reverse=not lower_is_better)


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, ms_loss_fn, ce_loss_fn, device, epoch,
    ms_weight, ce_weight, run=None, log_every_steps=None, global_step=0,
):
    model.train()
    running_total = running_ms = running_ce = 0.0
    total_correct = total_examples = 0

    progress = tqdm(loader, desc=f"MS-SNGP-SN Epoch {epoch}", leave=False)
    for views, labels in progress:
        labels = labels.to(device, non_blocking=True)
        B, V, C, H, W = views.shape
        views = views.to(device, non_blocking=True).view(B * V, C, H, W)
        ce_labels = labels.repeat_interleave(V)

        optimizer.zero_grad(set_to_none=True)
        # SNGPResNetClassifier.forward() supports return_features=True
        logits, gp_inputs = model(views, update_precision=True, return_features=True)
        ms_loss = ms_loss_fn(gp_inputs, ce_labels)
        ce_loss = ce_loss_fn(logits, ce_labels)
        total_loss = ms_weight * ms_loss + ce_weight * ce_loss
        total_loss.backward()
        optimizer.step()

        global_step += 1
        running_total += total_loss.item()
        running_ms    += ms_loss.item()
        running_ce    += ce_loss.item()
        total_correct   += (logits.argmax(dim=1) == ce_labels).sum().item()
        total_examples  += ce_labels.size(0)
        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            ms=f"{ms_loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
        )
        if run is not None and log_every_steps and global_step % log_every_steps == 0:
            run.log({
                "train/step_total_loss": total_loss.item(),
                "train/step_ms_loss":    ms_loss.item(),
                "train/step_ce_loss":    ce_loss.item(),
                "train/global_step":     global_step,
                "train/epoch":           epoch,
                "train/lr_step":         optimizer.param_groups[0]["lr"],
            })

    return {
        "total_loss": running_total / len(loader),
        "ms_loss":    running_ms    / len(loader),
        "ce_loss":    running_ce    / len(loader),
        "accuracy":   total_correct / total_examples,
    }, global_step


@torch.no_grad()
def evaluate(model, loader, device, num_mc_samples=10):
    model.eval()
    running_loss = total_correct = total_examples = 0
    total_nll = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, variances = model(images, return_cov=True)
        probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        running_loss  += (-log_probs.gather(1, labels.unsqueeze(1)).mean()).item()
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll     += (-log_probs.gather(1, labels.unsqueeze(1)).sum()).item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t  = torch.cat(all_probs)
    all_labels_t = torch.cat(all_labels)
    return {
        "loss":     running_loss  / len(loader),
        "accuracy": total_correct / total_examples,
        "nll":      total_nll     / total_examples,
        "ece":      _classification_ece(all_probs_t, all_labels_t),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = cfg.get("training", {}).get("seed")
    if seed is not None:
        import random, numpy as np
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    # ── Resume: load checkpoint early (CPU) so we know start_epoch ──────────────
    train_cfg = cfg["training"]
    resume_from = train_cfg.get("resume_from")
    resume_ckpt_data = None
    start_epoch = 0
    if resume_from:
        print(f"[resume] Loading checkpoint: {resume_from}")
        resume_ckpt_data = torch.load(resume_from, map_location="cpu", weights_only=False)
        start_epoch = resume_ckpt_data.get("epoch", 0)
        print(f"[resume] Will resume from epoch {start_epoch}")

    # ── W&B ─────────────────────────────────────────────────────────────────────
    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        resume_id = wandb_cfg.get("resume_id")
        if resume_id:
            run = wandb.init(
                project=wandb_cfg.get("project", "sngp"),
                entity=wandb_cfg.get("entity") or "sta414manygp",
                id=resume_id,
                resume="must",
            )
            print(f"[resume] Resumed W&B run: {run.url}")
        else:
            run = wandb.init(
                project=wandb_cfg.get("project", "sngp"),
                entity=wandb_cfg.get("entity") or "sta414manygp",
                name=wandb_cfg.get("run_name") or None,
                config=cfg,
            )

    # ── Data ────────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_loader, _, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        get_cifar10_supcon_loaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            smoke_test=smoke_test,
        )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # ── Model ───────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dim=model_cfg["hidden_dim"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
        kernel_type=model_cfg.get("kernel_type", "legacy"),
        input_normalization=model_cfg.get("input_normalization", None),
        kernel_scale=model_cfg.get("kernel_scale", 1.0),
        length_scale=model_cfg.get("length_scale", 1.0),
    ).to(device)
    print_model_summary(model)

    # ── Optimizer ────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    if resume_ckpt_data is not None:
        model.load_state_dict(resume_ckpt_data["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt_data["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", train_cfg["lr"])

    # ── Scheduler ────────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1 if smoke_test else train_cfg["epochs"],
        last_epoch=start_epoch if start_epoch > 0 else -1,
    )
    if resume_ckpt_data is not None and "scheduler_state_dict" in resume_ckpt_data:
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state_dict"])

    # ── Loss functions ──────────────────────────────────────────────────────────
    ms_loss_fn = MultiSimilarityLoss(
        alpha=train_cfg.get("ms_alpha", 2.0),
        beta=train_cfg.get("ms_beta",  50.0),
        base=train_cfg.get("ms_base",  0.5),
        eps=train_cfg.get("ms_eps",    0.1),
    )
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    ms_weight  = train_cfg.get("ms_loss_weight", 1.0)
    ce_weight  = train_cfg.get("ce_loss_weight",  1.0)
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval   = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps")
    num_mc_samples  = train_cfg.get("num_mc_samples", 10)

    # ── Checkpoint setup ────────────────────────────────────────────────────────
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    resolved_ckpt   = resolve_timestamped_checkpoint_path(checkpoint_path) if checkpoint_path else None
    top_k            = output_cfg.get("top_k", 5)
    checkpoint_metric = output_cfg.get("checkpoint_metric", "val_accuracy")
    lower_is_better   = checkpoint_metric in ("val_ece", "val_loss")
    saved_checkpoints: list = []
    global_step = 0
    best_val_acc = -1.0

    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_ckpt

    # ── Apply resume state (checkpoint discovery + counters) ─────────────────────
    if resume_ckpt_data is not None:
        global_step = start_epoch * len(train_loader)
        stored_resolved = (
            resume_ckpt_data.get("config", {})
            .get("output", {})
            .get("resolved_checkpoint_path")
        )
        if stored_resolved and checkpoint_path:
            resolved_ckpt = stored_resolved
            runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_ckpt
        if resolved_ckpt:
            ckpt_dir  = Path(resolved_ckpt).parent
            ckpt_stem = Path(resolved_ckpt).stem
            metric_label = checkpoint_metric.replace("val_", "")
            saved_checkpoints = load_existing_checkpoints(
                ckpt_dir, ckpt_stem, metric_label, lower_is_better, top_k
            )
            if saved_checkpoints:
                best_val_acc = (
                    saved_checkpoints[0]["metric"]
                    if not lower_is_better
                    else saved_checkpoints[-1]["metric"]
                )
        print(
            f"[resume] Model/optimizer/scheduler loaded. "
            f"Continuing epoch {start_epoch + 1}/{num_epochs}. "
            f"Pre-loaded {len(saved_checkpoints)} existing checkpoints."
        )

    model.reset_precision_matrix()

    # ── Training loop ────────────────────────────────────────────────────────────
    epoch_progress = tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_metrics, global_step = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            ms_loss_fn=ms_loss_fn, ce_loss_fn=ce_loss_fn, device=device,
            epoch=epoch, ms_weight=ms_weight, ce_weight=ce_weight,
            run=run, log_every_steps=log_every_steps, global_step=global_step,
        )
        scheduler.step()

        val_metrics = None
        if epoch % eval_interval == 0 or epoch == num_epochs:
            val_metrics = evaluate(model, val_loader, device, num_mc_samples)
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {train_metrics['total_loss']:.4f} | "
                f"MS: {train_metrics['ms_loss']:.4f} | "
                f"CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"Val ECE: {val_metrics['ece']:.4f}"
            )
            epoch_progress.set_postfix(val_acc=f"{val_metrics['accuracy']*100:.2f}%")
        else:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {train_metrics['total_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}%"
            )

        if run is not None:
            log_data = {
                "train/total_loss": train_metrics["total_loss"],
                "train/ms_loss":    train_metrics["ms_loss"],
                "train/ce_loss":    train_metrics["ce_loss"],
                "train/accuracy":   train_metrics["accuracy"],
                "train/lr":         optimizer.param_groups[0]["lr"],
                "train/epoch":      epoch,
            }
            if val_metrics:
                log_data.update({
                    "val/loss":     val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/nll":      val_metrics["nll"],
                    "val/ece":      val_metrics["ece"],
                })
            run.log(log_data)

        if val_metrics is None:
            continue
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
        if resolved_ckpt:
            _metric_map = {
                "val_accuracy": val_metrics["accuracy"],
                "val_ece":      val_metrics["ece"],
                "val_loss":     val_metrics["loss"],
            }
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints, top_k=top_k,
                checkpoint_path=resolved_ckpt, state={
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics":       train_metrics,
                    "val_metrics":         val_metrics,
                    "config":              runtime_cfg,
                },
                metric_name=checkpoint_metric.replace("val_", ""),
                metric_value=_metric_map.get(checkpoint_metric, val_metrics["accuracy"]),
                epoch=epoch, lower_is_better=lower_is_better,
            )

    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    # ── Post-training eval ────────────────────────────────────────────────────────
    if saved_checkpoints:
        import shutil
        best_ckpt = torch.load(saved_checkpoints[0]["path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_model_path = saved_checkpoints[0]["path"].parent / "best_model.pt"
        shutil.copy2(saved_checkpoints[0]["path"], best_model_path)
        print(f"Best model saved to: {best_model_path}")

        print("\nEvaluating best checkpoint on held-out test set...")
        test_metrics = evaluate(model, test_loader, device, num_mc_samples)
        print(
            f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
            f"Test NLL: {test_metrics['nll']:.4f} | "
            f"Test ECE: {test_metrics['ece']:.4f}"
        )
        if run is not None:
            run.log({
                "test/accuracy": test_metrics["accuracy"],
                "test/nll":      test_metrics["nll"],
                "test/ece":      test_metrics["ece"],
                "test/loss":     test_metrics["loss"],
            })

        if not smoke_test and cfg.get("ood", {}).get("enabled", True):
            print("\nRunning OOD + CIFAR-C evaluation...")
            from src.training.post_training_eval import run_full_ood_eval
            wrapper = ModelWrapper(model=model, has_cov=True,
                                   num_mc_samples=num_mc_samples, model_type="sngp_augmented")
            id_logits, id_probs, _, _ = collect_logits_and_probs(
                wrapper, test_loader, device, num_mc_samples)
            run_full_ood_eval(
                model=model, has_cov=True, id_logits=id_logits, id_probs=id_probs,
                cfg=cfg, device=device, run=run,
                num_mc_samples=num_mc_samples, model_type="sngp_augmented",
            )

    if run is not None:
        if best_val_acc >= 0.0:
            run.log({"best/val_accuracy": best_val_acc})
        if saved_checkpoints:
            run.log({f"best/{checkpoint_metric}": saved_checkpoints[0]["metric"]})
            import wandb
            artifact = wandb.Artifact("cifar10_ms_sngp_sn_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]),
                              name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 MS Loss + SNGP + SN backbone")
    parser.add_argument("--config",          required=True)
    parser.add_argument("--seed",            type=int, default=None)
    parser.add_argument("--run-name",        type=str, default=None, dest="run_name")
    parser.add_argument("--resume-from",     type=str, default=None, dest="resume_from",
                        help="Path to a checkpoint .pt file to resume training from")
    parser.add_argument("--wandb-resume-id", type=str, default=None, dest="wandb_resume_id",
                        help="W&B run ID to resume into (used with --resume-from)")
    parser.add_argument("--auto-resume",     action="store_true", dest="auto_resume",
                        help="Auto-detect latest checkpoint and W&B run to resume")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
        if cfg.get("output", {}).get("checkpoint_path"):
            p = Path(cfg["output"]["checkpoint_path"])
            cfg["output"]["checkpoint_path"] = str(p.parent / f"seed{args.seed}" / p.name)
    if args.run_name:
        cfg.setdefault("wandb", {})["run_name"] = args.run_name

    # ── Resume handling ──────────────────────────────────────────────────────────
    if args.auto_resume:
        seed_dir = Path(cfg["output"]["checkpoint_path"]).parent
        latest_ckpt = find_latest_checkpoint(seed_dir)
        if latest_ckpt:
            cfg.setdefault("training", {})["resume_from"] = str(latest_ckpt)
            print(f"[auto-resume] Found checkpoint: {latest_ckpt}")
        else:
            print(f"[auto-resume] No checkpoint found in {seed_dir} — starting fresh")
        _wc = cfg.get("wandb", {})
        run_id = find_wandb_run_id(
            entity=_wc.get("entity", "sta414manygp"),
            project=_wc.get("project", "april_4_experiments"),
            run_name=_wc.get("run_name", ""),
        )
        if run_id:
            cfg.setdefault("wandb", {})["resume_id"] = run_id
            print(f"[auto-resume] W&B run id → {run_id}")
        else:
            print("[auto-resume] No matching W&B run — will create a new one")
    elif args.resume_from:
        cfg.setdefault("training", {})["resume_from"] = args.resume_from
        if args.wandb_resume_id:
            cfg.setdefault("wandb", {})["resume_id"] = args.wandb_resume_id

    main(cfg)
