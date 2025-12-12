from .depthcharge.data.spectrum_datasets import AnnotatedSpectrumDataset
from .depthcharge.data.preprocessing import scale_to_unit_norm, scale_intensity
from .depthcharge.tokenizers import PeptideTokenizer
import torch
import numpy as np
import pytorch_lightning as pl
import os
import sys
import argparse
from lightning.pytorch import loggers as pl_loggers
from .depthcharge.utils import *
from .model import AugmentedSpec2Pep
from .augment import *
from datetime import datetime
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")
import h5py


def get_mzml_files(input_path):
    """Return list of .mzML files from input path (file or directory)."""
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".mzml":
        return [path]
    elif path.is_dir():
        return sorted(path.glob("*.mzML")) + sorted(path.glob("*.mzml"))
    else:
        raise ValueError(
            f"Invalid input: {input_path} (must be .mzML file or directory)"
        )


def get_output_path(mzml_file, output_dir):
    """Return output h5 path for a given mzML file."""
    return Path(output_dir) / f"{Path(mzml_file).stem}_embeddings.h5"


def get_pending_files(mzml_files, output_dir, resume=True):
    """Filter files to process, skipping already completed ones if resume=True."""
    if not resume:
        return mzml_files
    pending = []
    for f in mzml_files:
        out_path = get_output_path(f, output_dir)
        if not out_path.exists():
            pending.append(f)
        else:
            print(f"Skipping (already exists): {f.name} -> {out_path.name}")
    return pending


def process_single_file(
    spectrum_file,
    output_path,
    model,
    tokenizer,
    batch_size,
    max_charge,
    save_embeddings,
    aggregation_method="mean",
):
    """Process a single mzML file and save embeddings.

    Args:
        spectrum_file: Path to the mzML file.
        output_path: Path to save the output HDF5 file.
        model: Loaded Cascadia model.
        tokenizer: Peptide tokenizer.
        batch_size: Batch size for inference.
        max_charge: Maximum precursor charge state.
        save_embeddings: Whether to save embeddings.
        aggregation_method: Method for aggregating spectrum embeddings to sample-level.
            Options: 'mean', 'max', 'weighted_mean', 'none' (saves spectrum-level).
    """
    temp_path = os.getcwd() + "/cascadia_" + datetime.now().strftime("%m-%d-%H:%M:%S")
    os.mkdir(temp_path)
    train_index_filename = temp_path + "/index.hdf5"

    print("Augmenting spectra from:", spectrum_file)
    asf_file, isolation_window_size, cycle_time = augment_spectra(
        str(spectrum_file), temp_path, max_charge=max_charge
    )

    train_dataset = AnnotatedSpectrumDataset(
        tokenizer,
        asf_file,
        index_path=train_index_filename,
        preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm],
    )
    train_loader = train_dataset.loader(
        batch_size=batch_size, num_workers=4, pin_memory=True
    )

    if os.path.exists(asf_file):
        os.remove(asf_file)

    print("Running inference on augmented spectra from:", spectrum_file)

    if save_embeddings:
        print("Extracting encoder embeddings...")
        model_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(model_device)
        all_embs = []
        spectrum_ids = []
        with torch.no_grad():
            for batch in train_loader:
                spectra = batch[0].to(model_device)
                emb = model.get_spectrum_embeddings(spectra, pooling="cls")
                all_embs.append(emb.cpu().numpy())
                # generate simple sequential IDs
                start_idx = len(spectrum_ids)
                spectrum_ids.extend(
                    [f"spectrum_{start_idx + i}" for i in range(len(emb))]
                )
        embeddings = (
            np.concatenate(all_embs, axis=0).astype(np.float32)
            if all_embs
            else np.empty((0, model.peak_encoder.d_model), dtype=np.float32)
        )

        n_spectra = len(embeddings)
        embedding_dim = (
            embeddings.shape[1] if embeddings.size else model.peak_encoder.d_model
        )

        if aggregation_method != "none" and embeddings.size > 0:
            # Aggregate to sample-level embedding
            if aggregation_method == "mean":
                sample_embedding = embeddings.mean(axis=0, keepdims=True)
            elif aggregation_method == "max":
                sample_embedding = embeddings.max(axis=0, keepdims=True)
            elif aggregation_method == "weighted_mean":
                # Weight by L2 norm - larger norm = more informative spectrum
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                weights = norms / (norms.sum() + 1e-8)
                sample_embedding = (embeddings * weights).sum(axis=0, keepdims=True)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            # Save sample-level embedding
            sample_id = Path(spectrum_file).stem
            with h5py.File(output_path, "w") as f:
                f.create_dataset("embeddings", data=sample_embedding, dtype="float32")
                f.create_dataset("sample_ids", data=np.array([sample_id], dtype="S"))
                f.attrs["n_spectra"] = n_spectra
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["aggregation_method"] = aggregation_method
                f.attrs["level"] = "sample"
                f.attrs["pooling_method"] = "cls"
                f.attrs["model_type"] = "Cascadia"
            print(
                f"Saved sample-level embedding to: {output_path} "
                f"(aggregated {n_spectra} spectra using '{aggregation_method}')"
            )
        else:
            # Save spectrum-level embeddings (original behavior)
            with h5py.File(output_path, "w") as f:
                f.create_dataset(
                    "spectrum_embeddings", data=embeddings, dtype="float32"
                )
                dt = h5py.special_dtype(vlen=str)
                ids_ds = f.create_dataset(
                    "spectrum_ids", (len(spectrum_ids),), dtype=dt
                )
                for i, sid in enumerate(spectrum_ids):
                    ids_ds[i] = sid
                f.attrs["n_spectra"] = n_spectra
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["aggregation_method"] = "none"
                f.attrs["level"] = "spectrum"
                f.attrs["pooling_method"] = "cls"
                f.attrs["model_type"] = "Cascadia"
            print(
                f"Saved spectrum-level embeddings to: {output_path} with shape {embeddings.shape}"
            )

    os.remove(train_index_filename)
    os.rmdir(temp_path)


def sequence():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="A single .mzML file or a directory containing .mzML files",
    )
    parser.add_argument(
        "model", type=str, help="A path to a trained Cascadia model checkpoint."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="cascadia_embeddings",
        help="Output directory for embeddings (created if not exists).",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode (reprocess all files).",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "-t",
        "--score_threshold",
        type=float,
        default=0.8,
        help="Score threshold for Cascadia predictions.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Number of spectra to include in a batch.",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=2,
        help="Number of adjacent scans to use when constructing each augmented spectrum.",
    )
    parser.add_argument(
        "-c",
        "--max_charge",
        type=int,
        default=4,
        help="Maximum precursor charge state to consider",
    )
    parser.add_argument(
        "-p",
        "--modifications",
        type=str,
        default="mskb",
        help="A path to a json file containing a list of the PTMs to consider.",
    )
    parser.add_argument(
        "--no_save_embeddings",
        dest="save_embeddings",
        action="store_false",
        help="Disable saving encoder embeddings. Enabled by default.",
    )
    parser.set_defaults(save_embeddings=True)
    parser.add_argument(
        "-a",
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "weighted_mean", "none"],
        help="Aggregation method for sample-level embedding. 'none' saves spectrum-level.",
    )

    args = parser.parse_args(args=sys.argv[2:])

    input_path = args.input_path
    model_ckpt_path = args.model
    output_dir = Path(args.output_dir)
    batch_size = args.batch_size
    max_charge = args.max_charge
    mods = args.modifications
    save_embeddings = args.save_embeddings
    resume = args.resume
    aggregation_method = args.aggregation

    # Get files to process
    mzml_files = get_mzml_files(input_path)
    print(f"Found {len(mzml_files)} .mzML file(s)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for pending files (resume mode)
    pending_files = get_pending_files(mzml_files, output_dir, resume)
    print(
        f"Files to process: {len(pending_files)} (skipped {len(mzml_files) - len(pending_files)})"
    )

    if not pending_files:
        print("All files already processed. Use --no_resume to reprocess.")
        return parser

    # Initialize tokenizer
    if mods == "mskb":
        tokenizer = PeptideTokenizer.from_massivekb(
            reverse=False, replace_isoleucine_with_leucine=True
        )
    else:
        with open(mods, "r") as f:
            proforma = json.load(f)
        tokenizer = PeptideTokenizer.from_proforma(
            proforma, reverse=False, replace_isoleucine_with_leucine=True
        )

    # Load model once (outside loop)
    print("Loading model from:", model_ckpt_path)
    model = AugmentedSpec2Pep.load_from_checkpoint(
        model_ckpt_path,
        d_model=512,
        n_layers=9,
        n_head=8,
        dim_feedforward=1024,
        dropout=0,
        rt_width=2,
        tokenizer=tokenizer,
        max_charge=10,
    )

    if torch.cuda.is_available():
        print("GPU found")
    else:
        print("No GPU found - running inference on cpu")

    # Process each file
    for i, mzml_file in enumerate(pending_files, 1):
        print(f"\n[{i}/{len(pending_files)}] Processing: {mzml_file.name}")
        output_path = get_output_path(mzml_file, output_dir)
        process_single_file(
            mzml_file,
            output_path,
            model,
            tokenizer,
            batch_size,
            max_charge,
            save_embeddings,
            aggregation_method,
        )

    print(
        f"\nCompleted processing {len(pending_files)} file(s). Outputs saved to: {output_dir}"
    )
    return parser


def train():
    print("train", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("train_spectrum_file")
    parser.add_argument("val_spectrum_file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help="A path to a Cascadia model checkpoint to fine tune.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Number of spectra to include in a batch.",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=2,
        help="Number of adjacent scans to use when constructing each augmented spectrum.",
    )
    parser.add_argument(
        "-c",
        "--max_charge",
        type=int,
        default=4,
        help="Maximum precursor charge state to consider.",
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for model training.",
    )
    parser.add_argument(
        "-p",
        "--modifications",
        type=str,
        default="mskb",
        help="A path to a json file containing a list of the PTMs to consider.",
    )

    args = parser.parse_args(args=sys.argv[2:])

    train_spectrum_file = args.train_spectrum_file
    val_spectrum_file = args.val_spectrum_file
    model_ckpt_path = args.model
    batch_size = args.batch_size
    augmentation_width = args.width
    max_charge = args.max_charge
    max_epochs = args.max_epochs
    lr = args.learning_rate
    mods = args.modifications

    if torch.cuda.is_available():
        device = "gpu"
        print("GPU found")
    else:
        device = "cpu"
        print("No GPU Available - training on CPU will be extremely slow!")

    print("Training on spectra from:", train_spectrum_file)
    print("Validating on spectra from:", val_spectrum_file)

    ckpt_path = os.getcwd() + "/checkpoint_" + datetime.now().strftime("%m-%d-%H:%M:%S")
    os.mkdir(ckpt_path)
    train_index_filename = ckpt_path + "/train_gpu.hdf5"
    val_index_filename = ckpt_path + "/val_gpu.hdf5"

    if os.path.exists(train_index_filename):
        os.remove(train_index_filename)
    if os.path.exists(val_index_filename):
        os.remove(val_index_filename)

    if mods == "mskb":
        tokenizer = PeptideTokenizer.from_massivekb(
            reverse=False, replace_isoleucine_with_leucine=True
        )
    else:
        with open(mods, "r") as f:
            proforma = json.load(f)
        tokenizer = PeptideTokenizer.from_proforma(
            proforma, reverse=False, replace_isoleucine_with_leucine=True
        )

    if ".hdf5" in train_spectrum_file:
        train_dataset = AnnotatedSpectrumDataset(
            tokenizer,
            index_path=train_spectrum_file,
            preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm],
        )
        val_dataset = AnnotatedSpectrumDataset(
            tokenizer,
            index_path=val_spectrum_file,
            preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm],
        )
    else:
        train_dataset = AnnotatedSpectrumDataset(
            tokenizer,
            train_spectrum_file,
            index_path=train_index_filename,
            preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm],
        )
        val_dataset = AnnotatedSpectrumDataset(
            tokenizer,
            val_spectrum_file,
            index_path=val_index_filename,
            preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm],
        )

    train_loader = train_dataset.loader(
        batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True
    )
    val_loader = val_dataset.loader(
        batch_size=batch_size, num_workers=10, pin_memory=True
    )

    if model_ckpt_path is None:
        print("Training model from scratch")
        model = AugmentedSpec2Pep(
            d_model=512,
            n_layers=9,
            n_head=8,
            dim_feedforward=1024,
            dropout=0,
            rt_width=augmentation_width,
            tokenizer=tokenizer,
            max_charge=max_charge,
            lr=lr,
        )

    else:
        print("Loading model from checkpoint:", model_ckpt_path)
        model = AugmentedSpec2Pep.load_from_checkpoint(
            model_ckpt_path,
            d_model=512,
            n_layers=9,
            n_head=8,
            dim_feedforward=1024,
            dropout=0,
            rt_width=2,
            tokenizer=tokenizer,
            max_charge=max_charge,
            lr=lr,
        )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_path,
        filename="Cascadia-{epoch}-{step}",
        monitor="Val Pep. Acc.",
        mode="max",
        save_top_k=2,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
        log_every_n_steps=10000,
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        callbacks=[ckpt_callback],
        accelerator=device,
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


def main():
    print("running customised main")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["sequence", "train"], help="Which command to call"
    )
    args = parser.parse_args(args=sys.argv[1:2])
    mode = args.mode
    if mode == "sequence":
        sequence()
    elif mode == "train":
        train()


if __name__ == "__main__":
    main()
