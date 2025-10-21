"""
Compute cosine self-similarity matrices for audio files using Mel spectrogram
frames and visualize the results with imshow in time order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aiff",
    ".aif",
}

DEFAULT_AUDIO_LOCATIONS = [Path("audios") / "SelfSimilarity_export"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine self-similarity matrices from Mel spectrogram frames "
            "and visualize them over time."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_AUDIO_LOCATIONS],
        help=(
            "Audio files or directories containing audio files. "
            "Directories are searched recursively for supported formats. "
            "Defaults to audios/SelfSimilarity_export."
        ),
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sampling rate for loading audio (default: %(default)s).",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        dest="n_fft",
        default=2048,
        help="FFT window size for the Mel spectrogram (default: %(default)s).",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for the Mel spectrogram (default: %(default)s).",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of Mel bands for the spectrogram (default: %(default)s).",
    )
    parser.add_argument(
        "--savefig",
        type=Path,
        help=(
            "Optional base filename for saving plots. "
            "If multiple files are processed, an index and stem are appended."
        ),
    )
    return parser.parse_args()


def resolve_audio_paths(inputs: Sequence[str | Path]) -> List[Path]:
    audio_files: List[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    audio_files.append(candidate)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(path)
        else:
            raise FileNotFoundError(f"No audio files found at {path}")

    # Deduplicate while preserving order
    deduped_files = list(dict.fromkeys(audio_files))
    if not deduped_files:
        raise ValueError("Need at least one audio file to compute similarity.")
    return deduped_files


def load_mel_spectrogram(
    path: Path,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> tuple[np.ndarray, int]:
    y, effective_sr = librosa.load(path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=effective_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    return mel.astype(np.float32, copy=False), effective_sr


def compute_self_similarity(
    mel: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    mel_db = librosa.power_to_db(mel, ref=np.max)
    frame_features = mel_db.T  # frames x mel bands
    similarity = cosine_similarity(frame_features)
    frame_edges = librosa.frames_to_time(
        np.arange(similarity.shape[0] + 1), sr=sr, hop_length=hop_length
    )
    return similarity, frame_edges


def plot_similarity_matrix(
    matrix: np.ndarray,
    frame_edges: np.ndarray,
    label: str,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [frame_edges[0], frame_edges[-1], frame_edges[0], frame_edges[-1]]
    im = ax.imshow(matrix, origin="lower", cmap="magma", extent=extent, aspect="auto")
    ax.set_title(f"Cosine Self-Similarity – {label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Time (s)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Cosine similarity")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved similarity plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    audio_files = resolve_audio_paths(args.paths)
    print("Loaded audio files:")
    for idx, path in enumerate(audio_files):
        print(f"  [{idx}] {path}")

    np.set_printoptions(precision=3, suppress=True)
    for idx, path in enumerate(audio_files):
        mel, effective_sr = load_mel_spectrogram(
            path,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
        )
        similarity, frame_edges = compute_self_similarity(
            mel,
            sr=effective_sr,
            hop_length=args.hop_length,
        )

        print(f"\nCosine self-similarity matrix for {path.name}:")
        print(similarity)

        save_path = None
        if args.savefig:
            base = args.savefig
            suffix = base.suffix or ".png"
            if len(audio_files) == 1:
                save_path = base if base.suffix else base.with_suffix(".png")
            else:
                stem = base.stem if base.suffix else base.name
                target_dir = base.parent
                save_name = f"{stem}_{idx}_{path.stem}{suffix}"
                save_path = target_dir / save_name

        plot_similarity_matrix(similarity, frame_edges, path.stem, save_path=save_path)


if __name__ == "__main__":
    main()
