"""
CIFAR-10 OOD detection and corruption robustness evaluation.

Loads a trained CIFAR-10 checkpoint (any model type) and evaluates:
  1. In-distribution accuracy (CIFAR-10 test set)
  2. OOD detection vs SVHN (easy) and CIFAR-100 (hard)
     - Dempster-Shafer uncertainty score
     - Max-probability baseline (1 - p_max)
     - Metrics: AUROC, AUPR
  3. CIFAR-10-C: accuracy, ECE, MCE across 15 corruptions × 5 severities
  4. Inference latency (ms/image) for each evaluation

Usage:
    python experiments/cifar10_ood_eval.py --config configs/cifar10_ood_eval.yaml
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.data.cifar100 import get_cifar100_loaders
from src.data.cifarc import CORRUPTIONS, get_cifarc_loader, maybe_download_cifarc
from src.data.svhn import get_svhn_loader
from src.training.ood_evaluate import (
    collect_logits_and_probs,
    evaluate_cifarc_split,
    evaluate_ood_split,
)
from src.utils.model_loader import load_model_from_checkpoint


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "sngp"),
            entity=wandb_cfg.get("entity") or "sta414manygp",
            name=wandb_cfg.get("run_name") or None,
            config=cfg,
        )

    model_cfg = cfg["model"]
    checkpoint_path = model_cfg["checkpoint_path"]
    num_mc_samples = model_cfg.get("num_mc_samples", 10)

    model_wrapper = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_mc_samples=num_mc_samples,
    )

    data_cfg = cfg["data"]
    data_root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)
    ood_cfg = cfg.get("ood", {})

    # -------------------------------------------------------------------------
    # 1. In-distribution: CIFAR-10 test set
    # -------------------------------------------------------------------------
    print("\n=== In-distribution: CIFAR-10 test ===")
    _, id_loader, _, _ = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    id_logits, id_probs, id_labels, id_latency = collect_logits_and_probs(
        model_wrapper, id_loader, device, num_mc_samples=num_mc_samples
    )
    id_accuracy = (id_probs.argmax(dim=-1) == id_labels).float().mean().item()
    print(f"  Accuracy:            {id_accuracy * 100:.2f}%")
    print(f"  Latency:             {id_latency:.3f} ms/image")

    results = {
        "model_type": model_wrapper.model_type,
        "checkpoint_path": checkpoint_path,
        "id": {
            "accuracy": id_accuracy,
            "latency_ms_per_image": id_latency,
        },
        "ood": {},
        "cifarc": {},
    }

    if run is not None:
        run.log({"id/accuracy": id_accuracy, "id/latency_ms_per_image": id_latency})

    # -------------------------------------------------------------------------
    # 2. OOD detection: SVHN (easy)
    # -------------------------------------------------------------------------
    print("\n=== OOD detection: SVHN (easy) ===")
    svhn_loader = get_svhn_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        id_normalization="cifar10",
    )
    ood_logits, ood_probs, _, ood_latency = collect_logits_and_probs(
        model_wrapper, svhn_loader, device, num_mc_samples=num_mc_samples
    )
    svhn_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs)
    svhn_metrics["latency_ms_per_image"] = ood_latency
    results["ood"]["svhn"] = svhn_metrics
    print(f"  DS  AUROC: {svhn_metrics['dempster_shafer']['auroc']:.4f}  "
          f"AUPR: {svhn_metrics['dempster_shafer']['aupr']:.4f}")
    print(f"  p_max AUROC: {svhn_metrics['max_prob']['auroc']:.4f}  "
          f"AUPR: {svhn_metrics['max_prob']['aupr']:.4f}")
    print(f"  Latency: {ood_latency:.3f} ms/image")

    if run is not None:
        run.log({
            "ood/svhn/ds_auroc": svhn_metrics["dempster_shafer"]["auroc"],
            "ood/svhn/ds_aupr": svhn_metrics["dempster_shafer"]["aupr"],
            "ood/svhn/mp_auroc": svhn_metrics["max_prob"]["auroc"],
            "ood/svhn/mp_aupr": svhn_metrics["max_prob"]["aupr"],
            "ood/svhn/latency_ms_per_image": ood_latency,
        })

    # -------------------------------------------------------------------------
    # 3. OOD detection: CIFAR-100 (hard)
    # -------------------------------------------------------------------------
    print("\n=== OOD detection: CIFAR-100 (hard) ===")
    _, cifar100_ood_loader, _, _ = get_cifar100_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    ood_logits, ood_probs, _, ood_latency = collect_logits_and_probs(
        model_wrapper, cifar100_ood_loader, device, num_mc_samples=num_mc_samples
    )
    cifar100_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs)
    cifar100_metrics["latency_ms_per_image"] = ood_latency
    results["ood"]["cifar100"] = cifar100_metrics
    print(f"  DS  AUROC: {cifar100_metrics['dempster_shafer']['auroc']:.4f}  "
          f"AUPR: {cifar100_metrics['dempster_shafer']['aupr']:.4f}")
    print(f"  p_max AUROC: {cifar100_metrics['max_prob']['auroc']:.4f}  "
          f"AUPR: {cifar100_metrics['max_prob']['aupr']:.4f}")
    print(f"  Latency: {ood_latency:.3f} ms/image")

    if run is not None:
        run.log({
            "ood/cifar100/ds_auroc": cifar100_metrics["dempster_shafer"]["auroc"],
            "ood/cifar100/ds_aupr": cifar100_metrics["dempster_shafer"]["aupr"],
            "ood/cifar100/mp_auroc": cifar100_metrics["max_prob"]["auroc"],
            "ood/cifar100/mp_aupr": cifar100_metrics["max_prob"]["aupr"],
            "ood/cifar100/latency_ms_per_image": ood_latency,
        })

    # -------------------------------------------------------------------------
    # 4. CIFAR-10-C: corruption robustness
    # -------------------------------------------------------------------------
    print("\n=== CIFAR-10-C: corruption robustness ===")
    cifarc_root = ood_cfg.get("cifarc_root", data_root)
    cifarc_dir = maybe_download_cifarc(cifarc_root, "cifar10")

    corruptions_to_eval = ood_cfg.get("corruptions") or CORRUPTIONS
    severities_to_eval = ood_cfg.get("severities") or [1, 2, 3, 4, 5]

    csv_records = []

    corruption_progress = tqdm(corruptions_to_eval, desc="Corruptions")
    for corruption in corruption_progress:
        results["cifarc"][corruption] = {}
        for severity in severities_to_eval:
            cifarc_loader, cifarc_labels = get_cifarc_loader(
                cifarc_dir=cifarc_dir,
                corruption=corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=num_workers,
                id_normalization="cifar10",
            )
            _, probs, _, latency = collect_logits_and_probs(
                model_wrapper, cifarc_loader, device, num_mc_samples=num_mc_samples
            )
            split_metrics = evaluate_cifarc_split(probs, cifarc_labels)
            split_metrics["latency_ms_per_image"] = latency
            results["cifarc"][corruption][str(severity)] = split_metrics

            csv_records.append({
                "corruption": corruption,
                "severity": severity,
                "accuracy": split_metrics["accuracy"],
                "ece": split_metrics["ece"],
                "mce": split_metrics["mce"],
                "latency_ms_per_image": latency,
            })

            if run is not None:
                run.log({
                    f"cifarc/{corruption}/severity_{severity}/accuracy": split_metrics["accuracy"],
                    f"cifarc/{corruption}/severity_{severity}/ece": split_metrics["ece"],
                    f"cifarc/{corruption}/severity_{severity}/mce": split_metrics["mce"],
                })

    # -------------------------------------------------------------------------
    # 5. Serialize results
    # -------------------------------------------------------------------------
    output_cfg = cfg.get("output", {})
    results_path = output_cfg.get("results_path", "./results/cifar10_ood_eval.json")
    csv_path = output_cfg.get("csv_path", "./results/cifar10_ood_cifarc.csv")

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")

    if csv_records:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["corruption", "severity", "accuracy", "ece", "mce", "latency_ms_per_image"]
            )
            writer.writeheader()
            writer.writerows(csv_records)
        print(f"CIFAR-C CSV written to {csv_path}")

    if run is not None:
        import wandb

        artifact = wandb.Artifact("cifar10_ood_eval_results", type="evaluation")
        artifact.add_file(results_path, name="results.json")
        if csv_records:
            artifact.add_file(csv_path, name="cifarc.csv")
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 OOD evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
