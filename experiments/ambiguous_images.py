"""
experiments/ambiguous_images.py

Compare SNGP uncertainty across model checkpoints on ambiguous real-world images.

T = 1 (no trajectory).  For each image × each model checkpoint, computes:
  f_c        raw GP mean logit at the declared true class c
  σ²_c       GP posterior variance at class c
  p̂_c_det   deterministic Laplace: softmax(f / √(1 + π/8·σ²))_c
  p̂_c_mc    Monte Carlo Laplace: mean_S softmax(f̃)_c,  f̃~N(f, diag(σ²))

Images are loaded from URLs or local paths, centre-cropped and resized to
32×32, and normalised with CIFAR-10 statistics before being fed to each model.

W&B layout — 1 run = full comparison:
  image_NNN/
    photo        — actual image (PIL) + caption: description | true=<class>
    f_c          — bar chart: model name → f_c
    sigma_sq_c   — bar chart: model name → σ²_c
    prob_det_c   — bar chart: model name → p̂_c deterministic
    prob_mc_c    — bar chart: model name → p̂_c MC (S=50)

Manifest YAML format (configs/ambiguous_images_manifest.yaml):
  images:
    - url: "https://..."          # or local path under `path:`
      true_label: 5               # CIFAR-10 class index
      description: "Shiba Inu — Dog vs Cat"
      class_pair: [5, 3]          # [true, confusable]  (informational only)

  models:
    - name: "SNGP"
      experiment: sngp
      config: configs/experiment_april2_sngp.yaml
      checkpoint: checkpoints_april2/sngp/seed0/.../best_model.pt

Usage:
    python experiments/ambiguous_images.py \\
        --manifest configs/ambiguous_images_manifest.yaml \\
        --run-name  ambiguous_comparison_v1 \\
        --num-mc    50
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CIFAR-10 normalisation constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Image loading & preprocessing ─────────────────────────────────────────────

def _fetch_pil(source: str) -> PILImage.Image:
    """Load PIL image from a URL or local path."""
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(source, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return PILImage.open(io.BytesIO(data)).convert("RGB")
    return PILImage.open(source).convert("RGB")


def load_and_preprocess(source: str) -> tuple[PILImage.Image, torch.Tensor]:
    """
    Returns:
        pil_orig  — original PIL image (for W&B display, before normalisation)
        tensor    — float32 tensor of shape (1, 3, 32, 32), CIFAR-10 normalised
    """
    from torchvision import transforms

    pil = _fetch_pil(source)

    preprocess = transforms.Compose([
        transforms.Resize(36),          # slightly larger then centre-crop to 32
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    tensor = preprocess(pil).unsqueeze(0)   # (1, 3, 32, 32)

    # Keep a display-size version (upscaled back so it's readable in W&B)
    pil_display = pil.resize((128, 128), PILImage.LANCZOS)
    return pil_display, tensor


# ── Model construction (mirrors uncertainty_trajectory.py) ────────────────────

def build_model(exp_type: str, cfg: dict, device: torch.device,
                num_inducing_override: int | None = None):
    model_cfg = cfg["model"]
    num_inducing = num_inducing_override or model_cfg["num_inducing"]

    if exp_type in ("hybrid_sngp", "ms_sngp_sn", "sngp", "sngp_augmented",
                    "hybrid_sngp_rff4096"):
        from src.models.sngp import SNGPResNetClassifier
        return SNGPResNetClassifier(
            num_classes     = model_cfg["num_classes"],
            widen_factor    = model_cfg.get("widen_factor", 10),
            hidden_dim      = model_cfg["hidden_dim"],
            spec_norm_bound = model_cfg["spec_norm_bound"],
            num_inducing    = num_inducing,
            ridge_penalty   = model_cfg["ridge_penalty"],
            feature_scale   = model_cfg["feature_scale"],
            gp_cov_momentum = model_cfg["gp_cov_momentum"],
            normalize_input = model_cfg["normalize_input"],
        ).to(device)

    elif exp_type == "ms_sngp_no_skip":
        from src.models.sngp import WRNNoSkipSupConSNGPClassifier
        return WRNNoSkipSupConSNGPClassifier(
            num_classes         = model_cfg["num_classes"],
            widen_factor        = model_cfg.get("widen_factor", 10),
            hidden_dim          = model_cfg["hidden_dim"],
            spec_norm_bound     = model_cfg["spec_norm_bound"],
            num_inducing        = num_inducing,
            ridge_penalty       = model_cfg["ridge_penalty"],
            feature_scale       = model_cfg["feature_scale"],
            gp_cov_momentum     = model_cfg["gp_cov_momentum"],
            normalize_input     = model_cfg["normalize_input"],
            kernel_type         = model_cfg["kernel_type"],
            input_normalization = model_cfg["input_normalization"],
            kernel_scale        = model_cfg["kernel_scale"],
            length_scale        = model_cfg["length_scale"],
        ).to(device)

    elif exp_type in ("ms_sngp", "supcon_sngp"):
        from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier
        return CifarResNetSupConSNGPClassifier(
            embedding_dim       = model_cfg["embedding_dim"],
            num_classes         = model_cfg["num_classes"],
            widen_factor        = model_cfg.get("widen_factor", 10),
            hidden_dims         = model_cfg.get("hidden_dims", []),
            dropout_rate        = model_cfg.get("dropout_rate", 0.0),
            num_inducing        = model_cfg["num_inducing"],
            ridge_penalty       = model_cfg["ridge_penalty"],
            feature_scale       = model_cfg["feature_scale"],
            gp_cov_momentum     = model_cfg["gp_cov_momentum"],
            normalize_input     = model_cfg.get("normalize_input", False),
            kernel_type         = model_cfg.get("kernel_type", "legacy"),
            input_normalization = model_cfg.get("input_normalization", None),
            kernel_scale        = model_cfg.get("kernel_scale", 1.0),
            length_scale        = model_cfg.get("length_scale", 1.0),
        ).to(device)

    else:
        raise ValueError(f"Unknown experiment type: {exp_type!r}")


# ── Forward pass ──────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_sngp(
    model,
    x: torch.Tensor,        # (1, C, H, W)
    device: torch.device,
    num_mc: int = 50,
) -> dict[str, float]:
    """
    Returns a dict with scalar values (indexed at true class by the caller):
        logits     shape (1, num_classes)
        variances  shape (1, num_classes)
        probs_mc   shape (1, num_classes)
        probs_det  shape (1, num_classes)
    All on CPU.
    """
    from src.models.sngp import laplace_predictive_probs

    x = x.to(device)
    logits, variances = model(x, return_cov=True)

    probs_mc = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc)

    scale     = torch.sqrt(1.0 + (math.pi / 8.0) * variances)
    probs_det = (logits / scale).softmax(dim=-1)

    return {
        "logits":    logits.cpu(),     # (1, C)
        "variances": variances.cpu(),  # (1, C)
        "probs_mc":  probs_mc.cpu(),   # (1, C)
        "probs_det": probs_det.cpu(),  # (1, C)
    }


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_image_section(
    run,
    img_idx: int,
    pil_img: PILImage.Image,
    true_label: int,
    description: str,
    class_pair: list[int],
    model_names: list[str],
    results: list[dict[str, float]],   # one dict per model, keys: f_c σ²_c p̂_det p̂_mc
) -> None:
    """
    Log one image section: the image + 4 bar charts (one per metric).
    Bar charts have model name on x-axis.
    """
    import wandb

    prefix   = f"image_{img_idx:03d}"
    cls_name = CIFAR10_CLASSES[true_label]
    pair_str = " vs ".join(CIFAR10_CLASSES[c] for c in class_pair)

    caption = f"{description} | true={cls_name} | pair: {pair_str}"

    # Combined data table (one row per model)
    table = wandb.Table(
        columns=["model", "f_c", "sigma_sq_c", "prob_det_c", "prob_mc_c"],
        data=[
            [model_names[i],
             results[i]["f_c"],
             results[i]["sigma_sq_c"],
             results[i]["prob_det_c"],
             results[i]["prob_mc_c"]]
            for i in range(len(model_names))
        ],
    )

    def bar(col, title):
        mini = wandb.Table(
            columns=["model", col],
            data=[[model_names[i], results[i][col]] for i in range(len(model_names))],
        )
        return wandb.plot.bar(mini, "model", col, title=title)

    # Full MC softmax table: 8 rows (models) × 10 columns (classes)
    mc_table = wandb.Table(
        columns=["model"] + CIFAR10_CLASSES,
        data=[
            [model_names[i]] + results[i]["prob_mc_all"]
            for i in range(len(model_names))
        ],
    )

    run.log({
        f"{prefix}/photo": wandb.Image(pil_img, caption=caption),
        f"{prefix}/prob_mc_all_classes": mc_table,
        f"{prefix}/f_c": bar(
            "f_c",
            f"[{description}]  f_c(x*)  — GP mean logit at true class ({cls_name})",
        ),
        f"{prefix}/sigma_sq_c": bar(
            "sigma_sq_c",
            f"[{description}]  σ²_c(x*)  — GP posterior variance at true class ({cls_name})",
        ),
        f"{prefix}/prob_det_c": bar(
            "prob_det_c",
            f"[{description}]  p̂_c deterministic  softmax(f/√(1+π/8·σ²))  ({cls_name})",
        ),
        f"{prefix}/prob_mc_c": bar(
            "prob_mc_c",
            f"[{description}]  p̂_c MC (S={results[0].get('num_mc', 50)})  ({cls_name})",
        ),
        f"{prefix}/data": table,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SNGP uncertainty on ambiguous images across checkpoints"
    )
    parser.add_argument("--manifest", required=True,
                        help="YAML manifest with images + models list")
    parser.add_argument("--run-name",  default=None, dest="run_name")
    parser.add_argument("--num-mc",    type=int, default=50, dest="num_mc")
    parser.add_argument("--wandb-project", default="manygp_ambiguous",
                        dest="wandb_project")
    parser.add_argument("--wandb-entity",  default="sta414manygp",
                        dest="wandb_entity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    image_specs = manifest["images"]   # list of {url/path, true_label, description, class_pair}
    model_specs = manifest["models"]   # list of {name, experiment, config, checkpoint}

    N = len(image_specs)
    M = len(model_specs)
    print(f"Images: {N}   Models: {M}")

    # ── Pre-load all images ───────────────────────────────────────────────────
    print("\nLoading images ...")
    loaded_images = []
    for i, spec in enumerate(image_specs):
        src = spec.get("url") or spec.get("path")
        print(f"  [{i}] {spec['description']}  ← {src[:60]}...")
        pil, tensor = load_and_preprocess(src)
        loaded_images.append((pil, tensor))
    print("  Done.")

    # ── W&B init ─────────────────────────────────────────────────────────────
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name or "ambiguous_images",
        config=dict(
            N_images=N,
            M_models=M,
            num_mc=args.num_mc,
            model_names=[s["name"] for s in model_specs],
            image_descriptions=[s["description"] for s in image_specs],
        ),
        tags=["ambiguous_images"],
    )
    print(f"\nW&B: {run.url}\n")

    # ── Per-image results: results[n][m] = {f_c, σ²_c, p̂_det_c, p̂_mc_c} ──
    all_results: list[list[dict]] = [[None] * M for _ in range(N)]

    # ── Iterate over models ───────────────────────────────────────────────────
    for m, mspec in enumerate(model_specs):
        model_name = mspec["name"]
        print(f"[{m+1}/{M}] Loading model: {model_name}")

        with open(mspec["config"]) as f:
            cfg = yaml.safe_load(f)

        # Resolve checkpoint path — supports glob wildcards for timestamp dirs
        ckpt_pattern = mspec["checkpoint"]
        if "*" in ckpt_pattern or "?" in ckpt_pattern:
            import glob as _glob
            matches = sorted(_glob.glob(ckpt_pattern, recursive=True))
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint found matching pattern: {ckpt_pattern}"
                )
            ckpt_path = matches[-1]   # take the most recently modified
            print(f"   resolved → {ckpt_path}")
        else:
            ckpt_path = ckpt_pattern

        model = build_model(mspec["experiment"], cfg, device,
                            num_inducing_override=mspec.get("num_inducing_override"))
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch   = ckpt.get("epoch", "?")
        val_acc = ckpt.get("val_accuracy", ckpt.get("val_acc", "?"))
        print(f"   epoch={epoch}  val_acc={val_acc}")

        for n, (spec, (pil_img, tensor)) in enumerate(zip(image_specs, loaded_images)):
            true_label = int(spec["true_label"])
            out = forward_sngp(model, tensor, device, num_mc=args.num_mc)

            c = true_label
            all_results[n][m] = {
                "f_c":          out["logits"][0, c].item(),
                "sigma_sq_c":   out["variances"][0, c].item(),
                "prob_det_c":   out["probs_det"][0, c].item(),
                "prob_mc_c":    out["probs_mc"][0, c].item(),
                "prob_mc_all":  out["probs_mc"][0].tolist(),   # all 10 classes
                "num_mc":       args.num_mc,
            }

        # Free GPU memory before next model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Log to W&B ───────────────────────────────────────────────────────────
    print("\nLogging to W&B ...")
    model_names = [s["name"] for s in model_specs]

    for n, spec in enumerate(image_specs):
        pil_img, _ = loaded_images[n]
        log_image_section(
            run         = run,
            img_idx     = n,
            pil_img     = pil_img,
            true_label  = int(spec["true_label"]),
            description = spec["description"],
            class_pair  = spec.get("class_pair", [int(spec["true_label"])]),
            model_names = model_names,
            results     = all_results[n],
        )
        print(f"  image_{n:03d} logged.")

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
