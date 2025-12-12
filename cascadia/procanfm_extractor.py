"""
Embedding extraction utilities for Cascadia -> ProCanFM integration.

Example
-------
>>> extractor = CascadiaEmbeddingExtractor("models/cascadia_astral_tuned.ckpt", device="cpu")
>>> dummy = torch.randn(2, 10, 4)
>>> embs = extractor.extract_batch_embeddings(dummy, pooling="cls")
>>> embs.shape
(2, extractor.embedding_dim)
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .model import AugmentedSpec2Pep

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _choose_device(device: str) -> torch.device:
    """Resolve a safe torch.device."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_spectra(batch: Any) -> torch.Tensor:
    """Extract spectra tensor from a batch produced by Cascadia dataloaders."""
    if isinstance(batch, dict):
        for key in ("spectra", "spectrum", "spectra_tensor"):
            if key in batch:
                return batch[key]
        # Fallback to first value if dict but no known key
        return next(iter(batch.values()))
    if isinstance(batch, (list, tuple)):
        return batch[0]
    raise TypeError(f"Unsupported batch type for spectra extraction: {type(batch)}")


class CascadiaEmbeddingExtractor:
    """Extract and aggregate Cascadia encoder embeddings for ProCanFM."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        model_kwargs: dict | None = None,
    ) -> None:
        self.device = _choose_device(device)
        self.model = self._load_model(checkpoint_path, model_kwargs or {})
        self.model.to(self.device)
        self.model.eval()
        # AugmentedPeakEncoder stores d_model; this matches encoder output dim.
        self.embedding_dim = self.model.peak_encoder.d_model

    def _load_model(
        self,
        checkpoint_path: str,
        model_kwargs: dict,
    ) -> AugmentedSpec2Pep:
        """Load Cascadia model from a Lightning checkpoint."""
        model = AugmentedSpec2Pep.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            **model_kwargs,
        )
        logger.info("Loaded Cascadia checkpoint from %s", checkpoint_path)
        return model

    @torch.no_grad()
    def extract_batch_embeddings(
        self,
        spectra: torch.Tensor,
        pooling: str = "cls",
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of spectra.

        Parameters
        ----------
        spectra : torch.Tensor
            Batch of spectra as produced by Cascadia dataloaders.
        pooling : str
            Pooling method passed to `AugmentedSpec2Pep.get_spectrum_embeddings`.

        Returns
        -------
        np.ndarray
            Array of shape (batch, embedding_dim).
        """
        spectra = spectra.to(self.device)
        embeddings = self.model.get_spectrum_embeddings(spectra, pooling=pooling)
        return embeddings.cpu().numpy()

    def extract_from_dataloader(
        self,
        dataloader: Iterable,
        pooling: str = "cls",
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract embeddings for all spectra in a dataloader.

        Returns
        -------
        embeddings : np.ndarray
            Shape (n_spectra, embedding_dim).
        spectrum_ids : list[str]
            Per-spectrum identifiers (fallback to incremental ids).
        sample_ids : list[str]
            Per-spectrum sample ids (fallback to 'sample_<idx>').
        """
        all_embeddings: List[np.ndarray] = []
        all_spectrum_ids: List[str] = []
        all_sample_ids: List[str] = []

        spectrum_counter = 0
        for batch in tqdm(dataloader, desc="Extracting Cascadia embeddings"):
            spectra = _pick_spectra(batch)

            if isinstance(batch, dict):
                spectrum_ids = batch.get("spectrum_id") or batch.get("spectrum_ids")
                sample_ids = batch.get("sample_id") or batch.get("sample_ids")
            else:
                spectrum_ids = None
                sample_ids = None

            if spectrum_ids is None:
                spectrum_ids = [
                    f"spectrum_{spectrum_counter + i}"
                    for i in range(len(spectra))
                ]
            if sample_ids is None:
                sample_ids = [
                    f"sample_{spectrum_counter + i}"
                    for i in range(len(spectra))
                ]

            spectrum_counter += len(spectra)

            embeddings = self.extract_batch_embeddings(spectra, pooling=pooling)

            all_embeddings.append(embeddings)
            all_spectrum_ids.extend([str(sid) for sid in spectrum_ids])
            all_sample_ids.extend([str(sid) for sid in sample_ids])

        stacked = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, self.embedding_dim))
        logger.info(
            "Extracted %d spectrum embeddings (dim=%d)",
            len(stacked),
            self.embedding_dim,
        )
        return stacked, all_spectrum_ids, all_sample_ids

    @staticmethod
    def aggregate_to_sample_level(
        spectrum_embeddings: np.ndarray,
        sample_ids: List[str],
        method: str = "mean",
    ) -> dict[str, np.ndarray]:
        """
        Aggregate spectrum-level embeddings to sample-level.

        Parameters
        ----------
        spectrum_embeddings : np.ndarray
            Shape (n_spectra, embedding_dim).
        sample_ids : list[str]
            Sample identifier for each spectrum.
        method : str
            Aggregation method: mean, max, or weighted_mean.
        """
        sample_groups: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        for emb, sid in zip(spectrum_embeddings, sample_ids):
            sample_groups[sid].append(emb)

        sample_embeddings: dict[str, np.ndarray] = {}
        for sid, embs in sample_groups.items():
            embs_arr = np.stack(embs, axis=0)
            if method == "mean":
                sample_embeddings[sid] = np.mean(embs_arr, axis=0)
            elif method == "max":
                sample_embeddings[sid] = np.max(embs_arr, axis=0)
            elif method == "weighted_mean":
                weights = np.linalg.norm(embs_arr, axis=1, keepdims=True)
                weights = weights / (weights.sum() + 1e-8)
                sample_embeddings[sid] = (embs_arr * weights).sum(axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        return sample_embeddings

    def extract_and_save(
        self,
        dataloader: Iterable,
        output_path: str,
        pooling: str = "cls",
        aggregation: str = "mean",
    ) -> Path:
        """
        Full pipeline: extract, aggregate, and write to HDF5.
        """
        spectrum_embs, spectrum_ids, sample_ids = self.extract_from_dataloader(
            dataloader,
            pooling=pooling,
        )

        sample_embeddings = self.aggregate_to_sample_level(
            spectrum_embs,
            sample_ids,
            method=aggregation,
        )

        sample_ids_sorted = sorted(sample_embeddings.keys())
        sample_emb_array = np.stack(
            [sample_embeddings[sid] for sid in sample_ids_sorted],
            axis=0,
        ) if sample_embeddings else np.empty((0, self.embedding_dim))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            f.create_dataset("sample_embeddings", data=sample_emb_array.astype(np.float32))
            f.create_dataset("sample_ids", data=np.array(sample_ids_sorted, dtype="S"))

            f.create_dataset("spectrum_embeddings", data=spectrum_embs.astype(np.float32))
            f.create_dataset("spectrum_ids", data=np.array(spectrum_ids, dtype="S"))
            f.create_dataset("spectrum_sample_ids", data=np.array(sample_ids, dtype="S"))

            f.attrs["embedding_dim"] = self.embedding_dim
            f.attrs["n_samples"] = len(sample_ids_sorted)
            f.attrs["n_spectra"] = len(spectrum_ids)
            f.attrs["pooling_method"] = pooling
            f.attrs["aggregation_method"] = aggregation
            f.attrs["model_type"] = "Cascadia"

        logger.info(
            "Saved embeddings to %s (samples=%d, spectra=%d)",
            output_path,
            len(sample_ids_sorted),
            len(spectrum_ids),
        )
        return output_path


def main() -> None:
    """CLI entry point for embedding extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Cascadia encoder embeddings for ProCanFM.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to Cascadia checkpoint (.ckpt).")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument(
        "--pooling",
        default="cls",
        choices=["cls", "mean", "max"],
        help="Spectrum-level pooling strategy.",
    )
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "max", "weighted_mean"],
        help="Sample-level aggregation strategy.",
    )
    parser.add_argument("--device", default="cuda", help="Device for extraction (cuda or cpu).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Placeholder for future dataloader creation.",
    )
    parser.add_argument(
        "--data",
        required=False,
        help="Optional data path; dataloader creation is left to the user.",
    )
    args = parser.parse_args()

    extractor = CascadiaEmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    raise NotImplementedError(
        "Dataloader construction is project-specific. "
        "Create a dataloader that yields batches compatible with Cascadia "
        "and pass it to `extract_and_save`."
    )


if __name__ == "__main__":
    main()

