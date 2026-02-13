# Whisper JA-EN Speech Translation

Bidirectional speech translation between Japanese and English using a distilled [Whisper](https://huggingface.co/openai/whisper-large-v2) model.

- **EN audio -> JA text**
- **JA audio -> EN text**

Model: [do-not-use-this-account-token/whisper-ja-en-speech-translation](https://huggingface.co/do-not-use-this-account-token/whisper-ja-en-speech-translation)

## Installation

```bash
pip install torch transformers librosa
```

## Inference

Use `forced_decoder_ids` to control the translation direction.
Set `language` to the **source audio language** and `task="translate"`.

> **Note:** Do not use the pipeline `task="translate"` parameter directly — Whisper's translate task
> always targets English. Use `forced_decoder_ids` with `model.generate()` for reliable
> bidirectional translation.

### EN audio -> JA text

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("do-not-use-this-account-token/whisper-ja-en-speech-translation")
model = WhisperForConditionalGeneration.from_pretrained("do-not-use-this-account-token/whisper-ja-en-speech-translation")

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
| to the north and within easy reach is the romantic and fascinating town of sintra... | 北北部には、Cintraという、とてもロマンチックで魅力的な町があります。この町は、Biron主が記録した豊かな歴史を持つ外国人にとって有名な場所です。 |
| the cabbage juice changes color depending on how acidic or basic alkaline the chemical is | 化学物質の酸性やアルカリ性の程度によって、キャベツのジュースの色が変わります。 |
| many people don't think about them as dinosaurs because they have feathers and can fly | 多くの人々は、恐竜とは思わない。なぜなら、恐竜には羽があり、飛ぶことができるからです。 |
| the hospital has followed protocol for infection control including separating the patient from others to prevent possible infection of others | この病院は、他の病気の感染を防ぐために、患者を他の病気から分離するような感染のプロトコルを実施しています。 |
| the northern marianas emergency management office said that there were no damages reported in the nation | メルトロニア州北部の救急管理局によると、この国で報告されているような損害は一切ありませんでした。 |
| twentieth century research has shown that there are two pools of genetic variation hidden and expressed | 二十世紀の研究によると、遺伝的な変化は隠されており、表現されているものです。 |
| the aspect ratio of this format dividing by twelve to obtain the simplest whole-number ratio is therefore said to be 3:2 | この形式のアスペクト比を十二で割ることで、最も単純な整数比が得られます。したがって、自由に二にすることができます。 |
| as light pollution in their heyday was not the kind of problem it is today they are usually located in cities or at campuses easier to reach than those built in modern times | 彼らの曜日は、今日のような問題ではありません。彼らは通常、都市部やキャンパスに位置しています。現代の汚染物件よりも、都市部での生活が楽になります。 |

### JA -> EN

| Source (JA audio) | Prediction (EN text) |
|---|---|
| インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう | If you search for "Internet" related to hostile environmental conditions, it's likely that your location will be revealed. |
| また 北側に行くなら世界的に有名なマリア像の聖地であるファティマの聖母の聖域神社を訪れましょう | Also, if you go to the northern side, you'll visit the sacred site of the world's most famous Marya. This is the sacred place for the Virgin Mary. |
| バルセロナの公用語はカタルーニャ語とスペイン語です 約半数がカタルーニャ語を好み 大多数がカタルーニャ語を理解し ほぼ全員がスペイン語を知っています | The official languages used in Barcelona are Catalan and Spanish. Approximately half of the people prefer Catalan, while the majority of them prefer Catalan. Almost everyone knows Spanish. |
| その長い顎には70本以上の鋭い歯が並び 口蓋には別の歯列があり つまりここを通ったら逃げ道はないということになります | In that long jaw, there are more than 70mm tool-sized blades. There's also a separate area outside the mouth. In other words, there's no way to get through this area. |
| ロスビー数が小さいほど磁気反転に関して星の活性が低下するわけです | The smaller the Rosbyte, the less active the star is in the next step. |
| 群島や湖では 必ずしもヨットは必要ありません | On the islands or lakes, yachts are not necessary at all. |
| パリジャンは 自己中心的で横柄で失礼な人が多いと言われています | It is said to be a place where many people are treated unfairly, with a sense of self-centeredness. |
| メインステージの音楽が終わっても フェスティバルには夜遅くまで演奏を流し続けるセクションがあるかもしれないことを覚えておいてください | Even after the main stage is over, there may still be sections where the music continues to be played. Please keep this in mind. |
| 参列者の数が多すぎて 全員がサンピエトロ広場での葬儀に参加することは不可能でした | There were too many people in the third row. It was impossible for everyone to participate in the funeral at a 3-meter square. |
| 香港の最高の景色を見るには 島から出て九龍のウォーターフロントに向かいましょう | To enjoy the best scenery in Hong Kong, let's leave the island and head towards the waterfront of Kureon. |

## License

Apache 2.0
