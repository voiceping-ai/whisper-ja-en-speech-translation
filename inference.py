"""
Standalone inference script for whisper-ja-en-speech-translation.

Translates audio files between English and Japanese using a distilled Whisper model.
Uses forced_decoder_ids to control translation direction.

Usage:
    python inference.py <audio_file> --direction <en2ja|ja2en> [--device DEVICE]

Examples:
    python inference.py audio_en.wav --direction en2ja
    python inference.py audio_ja.wav --direction ja2en
    python inference.py audio.wav --direction en2ja --device cuda:0
"""

import argparse

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "voiceping-ai/whisper-ja-en-speech-translation"

DIRECTIONS = {
    "en2ja": {"language": "en", "label": "EN -> JA"},
    "ja2en": {"language": "ja", "label": "JA -> EN"},
}


def translate(audio_path: str, direction: str, device: str, model_id: str):
    cfg = DIRECTIONS[direction]

    print(f"Model     : {model_id}")
    print(f"Direction : {cfg['label']}")
    print(f"Audio     : {audio_path}")
    print(f"Device    : {device}")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg["language"], task="translate"
    )

    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Duration  : {len(audio) / sr:.1f}s")

    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"\nResult    : {text}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Whisper JA-EN Speech Translation")
    parser.add_argument("audio", help="Path to audio file (16kHz mono recommended)")
    parser.add_argument(
        "--direction",
        required=True,
        choices=list(DIRECTIONS.keys()),
        help="Translation direction: en2ja or ja2en",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    translate(args.audio, args.direction, args.device, args.model_id)


if __name__ == "__main__":
    main()
