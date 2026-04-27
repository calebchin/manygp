from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.data.cifar10 import get_cifar10_eval_transform
from src.data.cifar100 import get_cifar100_loaders
from src.data.svhn import get_svhn_loader
from src.models.sngp import SNGPResNetClassifier, laplace_predictive_probs


@dataclass(frozen=True)
class SplitEmbeddings:
    embeddings: np.ndarray
    labels: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True)
class EmbeddingCollection:
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_epoch: int
    checkpoint_val_accuracy: float
    num_mc_samples: int
    svhn_split: str
    id_normalization: str
    classes: np.ndarray
    cifar10_classes: np.ndarray
    cifar100_classes: np.ndarray
    svhn_classes: np.ndarray
    train: SplitEmbeddings
    test: SplitEmbeddings
    svhn: SplitEmbeddings
    cifar100_test: SplitEmbeddings
    cifar10_test_accuracy: float
    cifar10_test_nll: float


def find_repo_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {cwd}")


def resolve_data_root(repo_root: Path, data_root: str) -> Path:
    candidate = Path(data_root)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def checkpoint_cache_key(checkpoint_path: Path) -> str:
    digest = hashlib.sha1(str(checkpoint_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{checkpoint_path.stem}_{digest}"


def load_checkpoint_and_config(
    checkpoint_path: Path,
    fallback_config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config")
    if cfg is None:
        with fallback_config_path.open() as handle:
            cfg = yaml.safe_load(handle)
    return checkpoint, cfg


def build_sngp_model(cfg: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> SNGPResNetClassifier:
    model_cfg = cfg["model"]
    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        width=model_cfg["width"],
        hidden_dim=model_cfg["hidden_dim"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_eval_loaders(
    repo_root: Path,
    cfg: dict[str, Any],
    num_workers_override: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    data_cfg = cfg["data"]
    data_root = resolve_data_root(repo_root, data_cfg["root"])
    batch_size = data_cfg.get("batch_size", 256)
    num_workers = data_cfg.get("num_workers", 0) if num_workers_override is None else num_workers_override
    svhn_split = cfg.get("ood", {}).get("svhn_split", "test")
    id_normalization = cfg.get("ood", {}).get("id_normalization", "cifar10")

    cifar10_eval_transform = get_cifar10_eval_transform()
    cifar10_train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=cifar10_eval_transform,
    )
    cifar10_test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=cifar10_eval_transform,
    )
    cifar10_train_loader = DataLoader(
        cifar10_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_supports_pin_memory(),
    )
    cifar10_test_loader = DataLoader(
        cifar10_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_supports_pin_memory(),
    )

    svhn_loader = get_svhn_loader(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        id_normalization=id_normalization,
        split=svhn_split,
    )
    _, cifar100_test_loader, _, cifar100_test_dataset = get_cifar100_loaders(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        smoke_test=False,
    )

    loaders = {
        "cifar10_train": cifar10_train_loader,
        "cifar10_test": cifar10_test_loader,
        "svhn": svhn_loader,
        "cifar100_test": cifar100_test_loader,
    }
    metadata = {
        "svhn_split": svhn_split,
        "id_normalization": id_normalization,
        "cifar10_classes": np.array(cifar10_test_dataset.classes),
        "cifar100_classes": np.array(cifar100_test_dataset.classes),
        "svhn_classes": np.array([str(i) for i in range(10)]),
    }
    return loaders, metadata


def device_supports_pin_memory() -> bool:
    return torch.cuda.is_available()


@torch.no_grad()
def compute_sngp_logits_and_variances(
    model: SNGPResNetClassifier,
    batch_embeddings: torch.Tensor,
    precision_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    classifier = model.classifier
    phi = classifier._random_features(batch_embeddings)
    logits = classifier.beta(phi)
    variances = torch.einsum("bi,kij,bj->bk", phi, precision_inv, phi)
    variances = classifier.ridge_penalty * variances
    return logits, variances


@torch.no_grad()
def collect_embeddings(
    model: SNGPResNetClassifier,
    loader: DataLoader,
    device: torch.device,
    num_mc_samples: int = 10,
    split_name: str | None = None,
) -> dict[str, np.ndarray]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    precision_inv = torch.linalg.pinv(model.classifier.precision_matrix)

    batch_iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Batches:{split_name}" if split_name is not None else "Batches",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for images, batch_labels in batch_iterator:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        batch_embeddings = model.encode(images)
        batch_logits, batch_variances = compute_sngp_logits_and_variances(
            model=model,
            batch_embeddings=batch_embeddings,
            precision_inv=precision_inv,
        )
        batch_probs = laplace_predictive_probs(
            batch_logits,
            batch_variances,
            num_mc_samples=num_mc_samples,
        )

        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels.cpu())
        logits.append(batch_logits.cpu())
        variances.append(batch_variances.cpu())
        probs.append(batch_probs.cpu())
        preds.append(batch_probs.argmax(dim=1).cpu())

    return {
        "embeddings": torch.cat(embeddings, dim=0).numpy(),
        "labels": torch.cat(labels, dim=0).numpy(),
        "logits": torch.cat(logits, dim=0).numpy(),
        "variances": torch.cat(variances, dim=0).numpy(),
        "probs": torch.cat(probs, dim=0).numpy(),
        "preds": torch.cat(preds, dim=0).numpy(),
    }


def extract_embeddings_for_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    fallback_config_path: str | Path | None = None,
    device: torch.device | None = None,
    force: bool = False,
    num_workers_override: int | None = None,
) -> Path:
    repo_root = find_repo_root()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    fallback_config_path = (
        Path(fallback_config_path).resolve()
        if fallback_config_path is not None
        else (repo_root / "configs" / "cifar10_sngp.yaml").resolve()
    )

    cache_key = checkpoint_cache_key(checkpoint_path)
    output_path = embedding_dir / f"sngp_embeddings_{cache_key}.npz"
    if output_path.exists() and not force:
        return output_path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, cfg = load_checkpoint_and_config(checkpoint_path, fallback_config_path, device)
    model = build_sngp_model(cfg, checkpoint, device)

    loaders, loader_meta = make_eval_loaders(repo_root, cfg, num_workers_override=num_workers_override)
    num_mc_samples = cfg.get("training", {}).get("num_mc_samples", 10)

    split_outputs: dict[str, dict[str, np.ndarray]] = {}
    split_items = tqdm(
        loaders.items(),
        total=len(loaders),
        desc=f"Splits:{checkpoint_path.stem}",
        unit="split",
        leave=False,
        dynamic_ncols=True,
    )
    for split_name, loader in split_items:
        split_outputs[split_name] = collect_embeddings(
            model=model,
            loader=loader,
            device=device,
            num_mc_samples=num_mc_samples,
            split_name=split_name,
        )

    cifar10_test_probs = split_outputs["cifar10_test"]["probs"]
    cifar10_test_labels = split_outputs["cifar10_test"]["labels"]
    cifar10_test_preds = split_outputs["cifar10_test"]["preds"]
    cifar10_test_accuracy = float(np.mean(cifar10_test_preds == cifar10_test_labels))
    cifar10_test_nll = float(
        -np.mean(
            np.log(
                np.clip(
                    cifar10_test_probs[np.arange(len(cifar10_test_labels)), cifar10_test_labels],
                    1e-12,
                    1.0,
                )
            )
        )
    )

    np.savez_compressed(
        output_path,
        train_embeddings=split_outputs["cifar10_train"]["embeddings"],
        train_labels=split_outputs["cifar10_train"]["labels"],
        train_logits=split_outputs["cifar10_train"]["logits"],
        train_preds=split_outputs["cifar10_train"]["preds"],
        train_probs=split_outputs["cifar10_train"]["probs"],
        train_variances=split_outputs["cifar10_train"]["variances"],
        test_embeddings=split_outputs["cifar10_test"]["embeddings"],
        test_labels=split_outputs["cifar10_test"]["labels"],
        test_logits=split_outputs["cifar10_test"]["logits"],
        test_preds=split_outputs["cifar10_test"]["preds"],
        test_probs=split_outputs["cifar10_test"]["probs"],
        test_variances=split_outputs["cifar10_test"]["variances"],
        svhn_embeddings=split_outputs["svhn"]["embeddings"],
        svhn_labels=split_outputs["svhn"]["labels"],
        svhn_logits=split_outputs["svhn"]["logits"],
        svhn_preds=split_outputs["svhn"]["preds"],
        svhn_probs=split_outputs["svhn"]["probs"],
        svhn_variances=split_outputs["svhn"]["variances"],
        cifar100_test_embeddings=split_outputs["cifar100_test"]["embeddings"],
        cifar100_test_labels=split_outputs["cifar100_test"]["labels"],
        cifar100_test_logits=split_outputs["cifar100_test"]["logits"],
        cifar100_test_preds=split_outputs["cifar100_test"]["preds"],
        cifar100_test_probs=split_outputs["cifar100_test"]["probs"],
        cifar100_test_variances=split_outputs["cifar100_test"]["variances"],
        classes=loader_meta["cifar10_classes"],
        cifar10_classes=loader_meta["cifar10_classes"],
        cifar100_classes=loader_meta["cifar100_classes"],
        svhn_classes=loader_meta["svhn_classes"],
        checkpoint_id=checkpoint_path.stem,
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=checkpoint.get("epoch", -1),
        checkpoint_val_accuracy=checkpoint.get("val_accuracy", np.nan),
        num_mc_samples=num_mc_samples,
        svhn_split=loader_meta["svhn_split"],
        id_normalization=loader_meta["id_normalization"],
        cifar10_test_accuracy=cifar10_test_accuracy,
        cifar10_test_nll=cifar10_test_nll,
    )
    return output_path


def split_embeddings_from_npz(data: np.lib.npyio.NpzFile, prefix: str) -> SplitEmbeddings:
    return SplitEmbeddings(
        embeddings=data[f"{prefix}_embeddings"].astype(np.float64),
        labels=data[f"{prefix}_labels"].astype(np.int64),
        logits=data[f"{prefix}_logits"].astype(np.float64),
        probs=data[f"{prefix}_probs"].astype(np.float64),
        preds=data[f"{prefix}_preds"].astype(np.int64),
        variances=data[f"{prefix}_variances"].astype(np.float64),
    )


def load_embedding_collection(embedding_path: str | Path) -> EmbeddingCollection:
    data = np.load(Path(embedding_path), allow_pickle=True)
    return EmbeddingCollection(
        checkpoint_id=str(data["checkpoint_id"]),
        checkpoint_path=str(data["checkpoint_path"]),
        checkpoint_epoch=int(data["checkpoint_epoch"]),
        checkpoint_val_accuracy=float(data["checkpoint_val_accuracy"]),
        num_mc_samples=int(data["num_mc_samples"]),
        svhn_split=str(data["svhn_split"]),
        id_normalization=str(data["id_normalization"]),
        classes=np.array(data["classes"]),
        cifar10_classes=np.array(data["cifar10_classes"]),
        cifar100_classes=np.array(data["cifar100_classes"]),
        svhn_classes=np.array(data["svhn_classes"]),
        train=split_embeddings_from_npz(data, "train"),
        test=split_embeddings_from_npz(data, "test"),
        svhn=split_embeddings_from_npz(data, "svhn"),
        cifar100_test=split_embeddings_from_npz(data, "cifar100_test"),
        cifar10_test_accuracy=float(data["cifar10_test_accuracy"]),
        cifar10_test_nll=float(data["cifar10_test_nll"]),
    )


def fit_label_assigned_gmm(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    covariance_reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dim = x.shape[1]
    means = np.zeros((num_classes, dim), dtype=np.float64)
    covariances = np.zeros((num_classes, dim, dim), dtype=np.float64)
    priors = np.zeros(num_classes, dtype=np.float64)

    global_variance = np.var(x, axis=0).mean()
    reg = covariance_reg * max(global_variance, 1e-12)

    for cls in range(num_classes):
        class_x = x[y == cls]
        if len(class_x) == 0:
            raise ValueError(f"class {cls} has no samples")
        priors[cls] = len(class_x) / len(x)
        means[cls] = class_x.mean(axis=0)
        centered = class_x - means[cls]
        cov = centered.T @ centered / max(len(class_x) - 1, 1)
        covariances[cls] = cov + reg * np.eye(dim)

    return priors, means, covariances, reg


def fit_em_gmm(
    x: np.ndarray,
    num_components: int,
    covariance_reg: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, GaussianMixture]:
    global_variance = np.var(x, axis=0).mean()
    reg = covariance_reg * max(global_variance, 1e-12)

    model = GaussianMixture(
        n_components=num_components,
        covariance_type="full",
        reg_covar=reg,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        init_params="kmeans",
    )
    model.fit(x)
    return (
        model.weights_.astype(np.float64),
        model.means_.astype(np.float64),
        model.covariances_.astype(np.float64),
        reg,
        model,
    )


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def gaussian_logpdf_by_class(x: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    num_classes, dim = means.shape
    logpdf = np.empty((x.shape[0], num_classes), dtype=np.float64)
    log_2pi = dim * np.log(2.0 * np.pi)

    for cls in range(num_classes):
        chol = np.linalg.cholesky(covariances[cls])
        diff = (x - means[cls]).T
        solved = np.linalg.solve(chol, diff)
        mahalanobis = np.sum(solved * solved, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        logpdf[:, cls] = -0.5 * (log_2pi + logdet + mahalanobis)

    return logpdf


def gmm_probabilities(x: np.ndarray, priors: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> dict[str, np.ndarray]:
    log_likelihood = gaussian_logpdf_by_class(x, means, covariances)
    log_joint = np.log(priors)[None, :] + log_likelihood
    log_px = logsumexp(log_joint, axis=1)
    posterior = np.exp(log_joint - log_px[:, None])
    return {
        "log_likelihood": log_likelihood,
        "log_joint": log_joint,
        "log_px": log_px,
        "posterior": posterior,
    }


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def classification_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def dempster_shafer_uncertainty(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits)
    return logits.shape[1] / (np.sum(exp_logits, axis=1) + logits.shape[1])


def max_prob_uncertainty(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probs, axis=1)


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    y_true = np.concatenate(
        [
            np.zeros(len(id_scores), dtype=np.int64),
            np.ones(len(ood_scores), dtype=np.int64),
        ]
    )
    y_score = np.concatenate([id_scores, ood_scores])
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
    }


def evaluate_posterior_predictions(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "posterior_accuracy": float(np.mean(preds == labels)),
        "posterior_ece": classification_ece(probs, labels),
        "posterior_nll": negative_log_likelihood(probs, labels),
    }


def flatten_metric_dict(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in metrics.items()}


def evaluate_checkpoint_from_embeddings(
    embeddings: EmbeddingCollection,
    covariance_reg: float = 1e-3,
    em_max_iter: int = 100,
    em_tol: float = 1e-3,
    em_random_state: int = 42,
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "checkpoint_id": embeddings.checkpoint_id,
        "checkpoint_path": embeddings.checkpoint_path,
        "checkpoint_epoch": embeddings.checkpoint_epoch,
        "checkpoint_val_accuracy": embeddings.checkpoint_val_accuracy,
        "num_mc_samples": embeddings.num_mc_samples,
        "svhn_split": embeddings.svhn_split,
        "id_normalization": embeddings.id_normalization,
        "cifar10_test_accuracy": embeddings.cifar10_test_accuracy,
        "cifar10_test_nll": embeddings.cifar10_test_nll,
        "num_classes": int(len(embeddings.classes)),
        "train_examples": int(len(embeddings.train.labels)),
        "test_examples": int(len(embeddings.test.labels)),
        "svhn_examples": int(len(embeddings.svhn.labels)),
        "cifar100_test_examples": int(len(embeddings.cifar100_test.labels)),
    }

    num_classes = len(embeddings.classes)

    train_fit_priors, train_fit_means, train_fit_covariances, train_fit_covariance_reg = fit_label_assigned_gmm(
        embeddings.train.embeddings,
        embeddings.train.labels,
        num_classes=num_classes,
        covariance_reg=covariance_reg,
    )
    train_fit_train_gmm = gmm_probabilities(
        embeddings.train.embeddings,
        train_fit_priors,
        train_fit_means,
        train_fit_covariances,
    )
    train_fit_test_gmm = gmm_probabilities(
        embeddings.test.embeddings,
        train_fit_priors,
        train_fit_means,
        train_fit_covariances,
    )

    rows["train_fit_covariance_reg"] = float(train_fit_covariance_reg)
    rows.update(
        flatten_metric_dict(
            "train_fit_train",
            evaluate_posterior_predictions(train_fit_train_gmm["posterior"], embeddings.train.labels),
        )
    )
    rows.update(
        flatten_metric_dict(
            "train_fit_test",
            evaluate_posterior_predictions(train_fit_test_gmm["posterior"], embeddings.test.labels),
        )
    )

    em_priors, em_means, em_covariances, em_covariance_reg, em_model = fit_em_gmm(
        embeddings.train.embeddings,
        num_components=num_classes,
        covariance_reg=covariance_reg,
        max_iter=em_max_iter,
        tol=em_tol,
        random_state=em_random_state,
    )
    em_train_gmm = gmm_probabilities(
        embeddings.train.embeddings,
        em_priors,
        em_means,
        em_covariances,
    )
    em_test_gmm = gmm_probabilities(
        embeddings.test.embeddings,
        em_priors,
        em_means,
        em_covariances,
    )
    rows["em_covariance_reg"] = float(em_covariance_reg)
    rows["em_converged"] = bool(em_model.converged_)
    rows["em_n_iter"] = int(em_model.n_iter_)
    rows["em_lower_bound"] = float(em_model.lower_bound_)
    rows["em_train_log_px_min"] = float(em_train_gmm["log_px"].min())
    rows["em_train_log_px_max"] = float(em_train_gmm["log_px"].max())
    rows["em_test_log_px_min"] = float(em_test_gmm["log_px"].min())
    rows["em_test_log_px_max"] = float(em_test_gmm["log_px"].max())

    test_fit_priors, test_fit_means, test_fit_covariances, test_fit_covariance_reg = fit_label_assigned_gmm(
        embeddings.test.embeddings,
        embeddings.test.labels,
        num_classes=num_classes,
        covariance_reg=covariance_reg,
    )
    test_fit_test_gmm = gmm_probabilities(
        embeddings.test.embeddings,
        test_fit_priors,
        test_fit_means,
        test_fit_covariances,
    )
    test_fit_train_gmm = gmm_probabilities(
        embeddings.train.embeddings,
        test_fit_priors,
        test_fit_means,
        test_fit_covariances,
    )
    rows["test_fit_covariance_reg"] = float(test_fit_covariance_reg)
    rows["test_fit_test_posterior_accuracy"] = float(
        np.mean(test_fit_test_gmm["posterior"].argmax(axis=1) == embeddings.test.labels)
    )
    rows["test_fit_train_posterior_accuracy"] = float(
        np.mean(test_fit_train_gmm["posterior"].argmax(axis=1) == embeddings.train.labels)
    )

    ood_specs = {
        "svhn": embeddings.svhn,
        "cifar100": embeddings.cifar100_test,
    }
    for dataset_name, dataset_split in ood_specs.items():
        label_ood_gmm = gmm_probabilities(
            dataset_split.embeddings,
            train_fit_priors,
            train_fit_means,
            train_fit_covariances,
        )
        label_metrics = compute_ood_metrics(-train_fit_test_gmm["log_px"], -label_ood_gmm["log_px"])
        rows.update(flatten_metric_dict(f"{dataset_name}_label_gmm", label_metrics))
        label_mp_metrics = compute_ood_metrics(
            max_prob_uncertainty(train_fit_test_gmm["posterior"]),
            max_prob_uncertainty(label_ood_gmm["posterior"]),
        )
        rows.update(flatten_metric_dict(f"{dataset_name}_label_gmm_mp", label_mp_metrics))

        em_ood_gmm = gmm_probabilities(
            dataset_split.embeddings,
            em_priors,
            em_means,
            em_covariances,
        )
        em_metrics = compute_ood_metrics(-em_test_gmm["log_px"], -em_ood_gmm["log_px"])
        rows.update(flatten_metric_dict(f"{dataset_name}_em_gmm", em_metrics))
        em_mp_metrics = compute_ood_metrics(
            max_prob_uncertainty(em_test_gmm["posterior"]),
            max_prob_uncertainty(em_ood_gmm["posterior"]),
        )
        rows.update(flatten_metric_dict(f"{dataset_name}_em_gmm_mp", em_mp_metrics))

        ds_metrics = compute_ood_metrics(
            dempster_shafer_uncertainty(embeddings.test.logits),
            dempster_shafer_uncertainty(dataset_split.logits),
        )
        rows.update(flatten_metric_dict(f"{dataset_name}_dempster_shafer", ds_metrics))

        mp_metrics = compute_ood_metrics(
            max_prob_uncertainty(embeddings.test.probs),
            max_prob_uncertainty(dataset_split.probs),
        )
        rows.update(flatten_metric_dict(f"{dataset_name}_max_prob", mp_metrics))
        rows.update(flatten_metric_dict(f"{dataset_name}_mp", mp_metrics))

    return rows


def evaluate_checkpoint_path(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    fallback_config_path: str | Path | None = None,
    device: torch.device | None = None,
    force_embeddings: bool = False,
    num_workers_override: int | None = None,
    covariance_reg: float = 1e-3,
    em_max_iter: int = 100,
    em_tol: float = 1e-3,
    em_random_state: int = 42,
) -> tuple[dict[str, Any], Path]:
    embedding_path = extract_embeddings_for_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        fallback_config_path=fallback_config_path,
        device=device,
        force=force_embeddings,
        num_workers_override=num_workers_override,
    )
    embeddings = load_embedding_collection(embedding_path)
    row = evaluate_checkpoint_from_embeddings(
        embeddings,
        covariance_reg=covariance_reg,
        em_max_iter=em_max_iter,
        em_tol=em_tol,
        em_random_state=em_random_state,
    )
    row["embedding_path"] = str(embedding_path)
    return row, embedding_path


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    metric_suffixes = ("accuracy", "ece", "nll", "auroc", "aupr")
    numeric_keys: list[str] = []
    sample = rows[0]
    for key, value in sample.items():
        if not key.endswith(metric_suffixes):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            numeric_keys.append(key)

    summary_rows: list[dict[str, Any]] = []
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        if len(values) > 1:
            std = float(np.std(values, ddof=1))
        else:
            std = 0.0
        summary_rows.append(
            {
                "metric": key,
                "mean": float(np.mean(values)),
                "std": std,
                "num_checkpoints": len(values),
            }
        )
    return summary_rows


def write_csv_rows(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
