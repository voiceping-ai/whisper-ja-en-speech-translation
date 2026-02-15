"""
Gradio demo for Whisper JA-EN Speech Translation.

Bidirectional speech translation between Japanese and English.
- EN audio -> JA text
- JA audio -> EN text
"""

import gradio as gr
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "voiceping-ai/whisper-ja-en-speech-translation"

DIRECTIONS = {
    "English -> Japanese": {"language": "en", "label": "EN -> JA"},
    "Japanese -> English": {"language": "ja", "label": "JA -> EN"},
}

# Load model once at startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
model.eval()


def translate(audio, direction):
    if audio is None:
        return "Please provide an audio input."

    cfg = DIRECTIONS[direction]

    # Gradio returns (sample_rate, numpy_array)
    sr, audio_data = audio

    # Convert to float32 and normalize if integer type
    if audio_data.dtype in (np.int16, np.int32):
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # Convert stereo to mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg["language"], task="translate"
    )

    input_features = processor(
        audio_data, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    model.config.forced_decoder_ids = forced_decoder_ids

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], label="Audio Input"),
        gr.Radio(
            choices=list(DIRECTIONS.keys()),
            value="English -> Japanese",
            label="Translation Direction",
        ),
    ],
    outputs=gr.Textbox(label="Translation", lines=3),
    title="Whisper JA-EN Speech Translation",
    description=(
        "Bidirectional speech translation between Japanese and English "
        "using a distilled Whisper model.\n\n"
        "- **English -> Japanese**: Speak/upload English audio to get Japanese text\n"
        "- **Japanese -> English**: Speak/upload Japanese audio to get English text\n\n"
        "Set the direction to match the **source audio language**."
    ),
    article=(
        "This model uses `forced_decoder_ids` to control translation direction. "
        "See the [model card](https://huggingface.co/voiceping-ai/"
        "whisper-ja-en-speech-translation) for more details."
    ),
    examples=[
        ["example_en.wav", "English -> Japanese"],
        ["example_ja.wav", "Japanese -> English"],
    ],
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(show_error=True)
