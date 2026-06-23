"""Extract audio embeddings from a video file, one embedding per second.

Supported encoders:
  ast    - Audio Spectrogram Transformer (AudioSet, speech/music/ambient)
  pann   - PANN CNN14 (Google, AudioSet pretrain, fast baseline)
  clap   - CLAP music+speech (laion/larger_clap_music_and_speech)
  w2v2   - wav2vec2-base (speech-focused features)
  hubert - HuBERT-base (self-supervised speech, complements AST)
  mert   - MERT-v1-95M (music-specialized, 24 kHz internal resample)
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL_REGISTRY = {
    "ast":    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "pann":   "CNN14",   # downloaded automatically by panns-inference
    "clap":   "laion/larger_clap_music_and_speech",
    "w2v2":   "facebook/wav2vec2-base-960h",
    "hubert": "facebook/hubert-base-ls960",
    "mert":   "m-a-p/MERT-v1-95M",
}

WINDOW_SEC = 2.0
HOP_SEC = 1.0
TARGET_SR = 16_000


# ── audio I/O ─────────────────────────────────────────────────────────────────

def extract_wav(src: Path, tmp_dir: str) -> Path:
    """Use ffmpeg to extract mono 16 kHz wav from any container."""
    out = Path(tmp_dir) / (src.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(TARGET_SR), str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def load_waveform(path: Path) -> torch.Tensor:
    # ffmpeg already output mono 16 kHz wav, so no resampling needed
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr}"
    return torch.from_numpy(data)  # (T,)


# ── encoder wrappers ──────────────────────────────────────────────────────────

def build_encoder(encoder: str, device: str):
    if encoder == "ast":
        from transformers import ASTFeatureExtractor, ASTModel
        fe = ASTFeatureExtractor.from_pretrained(MODEL_REGISTRY["ast"])
        model = ASTModel.from_pretrained(MODEL_REGISTRY["ast"]).to(device).eval()

        def encode(chunk: np.ndarray) -> np.ndarray:
            inputs = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                return model(**inputs).pooler_output.squeeze(0).cpu().numpy()  # (768,)

    elif encoder == "pann":
        from panns_inference import AudioTagging
        model = AudioTagging(checkpoint_path=None, device=device)

        def encode(chunk: np.ndarray) -> np.ndarray:
            # panns expects (batch, samples), returns clipwise_output + embedding
            x = chunk[np.newaxis, :]  # (1, T)
            with torch.no_grad():
                _, emb = model.inference(x)
            return emb.squeeze(0)  # (2048,)

    elif encoder == "clap":
        from transformers import ClapModel, ClapProcessor
        processor = ClapProcessor.from_pretrained(MODEL_REGISTRY["clap"])
        model = ClapModel.from_pretrained(MODEL_REGISTRY["clap"]).to(device).eval()

        def encode(chunk: np.ndarray) -> np.ndarray:
            inputs = processor(audios=chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                return model.get_audio_features(**inputs).squeeze(0).cpu().numpy()  # (512,)

    elif encoder == "w2v2":
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(MODEL_REGISTRY["w2v2"])
        model = Wav2Vec2Model.from_pretrained(MODEL_REGISTRY["w2v2"]).to(device).eval()

        def encode(chunk: np.ndarray) -> np.ndarray:
            inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state  # (1, T, 768)
                return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

    elif encoder == "hubert":
        from transformers import AutoFeatureExtractor, AutoModel
        fe = AutoFeatureExtractor.from_pretrained(MODEL_REGISTRY["hubert"])
        model = AutoModel.from_pretrained(MODEL_REGISTRY["hubert"]).to(device).eval()

        def encode(chunk: np.ndarray) -> np.ndarray:
            inputs = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state  # (1, T, 768)
                return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

    elif encoder == "mert":
        import librosa
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        # MERT is trained at 24 kHz; we resample on the fly per-chunk.
        fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_REGISTRY["mert"], trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_REGISTRY["mert"], trust_remote_code=True).to(device).eval()
        mert_sr = fe.sampling_rate

        def encode(chunk: np.ndarray) -> np.ndarray:
            chunk24 = librosa.resample(chunk, orig_sr=TARGET_SR, target_sr=mert_sr)
            inputs = fe(chunk24, sampling_rate=mert_sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state  # (1, T, 768)
                return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

    else:
        raise ValueError(f"Unknown encoder: {encoder}. Choose from {list(MODEL_REGISTRY)}")

    return encode


# ── main extraction ───────────────────────────────────────────────────────────

def extract_embeddings(video_path: Path, output_path: Path, encoder: str, device: str) -> None:
    print(f"[{encoder}] Loading audio from {video_path.name}...")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = extract_wav(video_path, tmp)
        waveform = load_waveform(wav_path)

    total_sec = len(waveform) / TARGET_SR
    print(f"  Duration : {total_sec:.1f}s")

    encode = build_encoder(encoder, device)

    window_samples = int(WINDOW_SEC * TARGET_SR)
    hop_samples = int(HOP_SEC * TARGET_SR)
    starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    print(f"  Windows  : {len(starts)}  device={device}")

    embeddings, timestamps = [], []
    for i, start in enumerate(starts):
        chunk = waveform[start : start + window_samples].numpy()
        embeddings.append(encode(chunk))
        timestamps.append(start / TARGET_SR)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(starts)}")

    embeddings = np.stack(embeddings)   # (N, D)
    timestamps = np.array(timestamps)   # (N,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=embeddings, timestamps=timestamps, encoder=encoder)
    print(f"  Saved {embeddings.shape} → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio embeddings from a video file")
    parser.add_argument("video", type=Path)
    parser.add_argument("--encoder", choices=list(MODEL_REGISTRY), default="ast")
    parser.add_argument("--output", type=Path, default=None)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    out = args.output or Path("features/audio") / f"{args.video.stem}_{args.encoder}.npz"
    extract_embeddings(args.video, out, encoder=args.encoder, device=args.device)


if __name__ == "__main__":
    main()
