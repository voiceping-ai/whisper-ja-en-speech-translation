---
title: Whisper JA-EN Speech Translation
emoji: "\U0001F30F"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# Whisper JA-EN Speech Translation

Bidirectional speech translation between Japanese and English using a distilled [Whisper](https://huggingface.co/openai/whisper-large-v2) model.

- **EN audio -> JA text**
- **JA audio -> EN text**

Model: [voiceping-ai/whisper-ja-en-speech-translation](https://huggingface.co/voiceping-ai/whisper-ja-en-speech-translation)

## How It Works

The model uses `forced_decoder_ids` to control the translation direction.
Set `language` to the **source audio language** and `task="translate"`.

> **Note:** Do not use the pipeline `task="translate"` parameter directly — Whisper's translate task
> always targets English. Use `forced_decoder_ids` with `model.generate()` for reliable
> bidirectional translation.

## Installation

```bash
pip install torch transformers librosa gradio
```

## Inference

### EN audio -> JA text

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("voiceping-ai/whisper-ja-en-speech-translation")
model = WhisperForConditionalGeneration.from_pretrained("voiceping-ai/whisper-ja-en-speech-translation")

# Load audio (16kHz mono)
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)

input_features = processor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features

# EN audio -> JA text
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="translate"
)

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
    )

print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

### JA audio -> EN text

```python
# JA audio -> EN text
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja", task="translate"
)

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
    )

print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

## Standalone Inference Script

See [`inference.py`](inference.py) for a complete standalone script that handles audio file input, device selection, and both translation directions.

```bash
# EN audio -> JA text
python inference.py audio_en.wav --direction en2ja

# JA audio -> EN text
python inference.py audio_ja.wav --direction ja2en

# Use GPU
python inference.py audio.wav --direction en2ja --device cuda:0
```

## Example Predictions

Predictions on [FLEURS](https://huggingface.co/datasets/google/fleurs) test set samples.

### EN -> JA

| Source (EN audio) | Prediction (JA text) |
|---|---|
| however due to the slow communication channels styles in the west could lag behind by 25 to 30 year | しかし、通信の速度が遅いため、西洋では二十五年から三十年ほど遅れをとることがあります。 |
| all nouns alongside the word sie for you always begin with a capital letter even in the middle of a sentence | 世界中の言葉によれば、すべての言葉は、文の途中でも、たとえ一文の途中でも、常に大文字で始まるべきだとされています。 |
| the cabbage juice changes color depending on how acidic or basic alkaline the chemical is | 化学物質の酸性やアルカリ性の程度によって、キャベツのジュースの色が変わります。 |
| many people don't think about them as dinosaurs because they have feathers and can fly | 多くの人々は、恐竜とは思わない。なぜなら、恐竜には羽があり、飛ぶことができるからです。 |
| the hospital has followed protocol for infection control including separating the patient from others to prevent possible infection of others | この病院は、他の病気の感染を防ぐために、患者を他の病気から分離するような感染のプロトコルを実施しています。 |

### JA -> EN

| Source (JA audio) | Prediction (EN text) |
|---|---|
| バルセロナの公用語はカタルーニャ語とスペイン語です 約半数がカタルーニャ語を好み 大多数がカタルーニャ語を理解し ほぼ全員がスペイン語を知っています | The official languages used in Barcelona are Catalan and Spanish. Approximately half of the people prefer Catalan, while the majority of them prefer Catalan. Almost everyone knows Spanish. |
| 群島や湖では 必ずしもヨットは必要ありません | On the islands or lakes, yachts are not necessary at all. |
| パリジャンは 自己中心的で横柄で失礼な人が多いと言われています | It is said to be a place where many people are treated unfairly, with a sense of self-centeredness. |
| メインステージの音楽が終わっても フェスティバルには夜遅くまで演奏を流し続けるセクションがあるかもしれないことを覚えておいてください | Even after the main stage is over, there may still be sections where the music continues to be played. Please keep this in mind. |
| 香港の最高の景色を見るには 島から出て九龍のウォーターフロントに向かいましょう | To enjoy the best scenery in Hong Kong, let's leave the island and head towards the waterfront of Kureon. |

## License

Apache 2.0
