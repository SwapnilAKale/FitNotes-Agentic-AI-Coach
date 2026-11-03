# Third-party notices: dictation

Dictation (`src/dictation/`) uses the components below. Each keeps its own licence. The speech models are not stored in this repository: `python scripts/fetch_speech_models.py` (or the Docker build) downloads them into `models/speech/` (sources and SHA-256 checksums in `models.json`), and each keeps its licence files in its folder.

## Speech models

| Component | Use | Licence | Source |
|---|---|---|---|
| Parakeet TDT 0.6B v2, int8 (sherpa-onnx export) | Speech recognition | CC BY 4.0 (NVIDIA). Attribution, model card and licence are in the model folder. | Original model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2. Conversion: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models. No fine-tuning. |
| Silero VAD v6.2.3 (`silero_vad.onnx`) | Finding where speech starts and ends, run by sherpa-onnx | MIT, Copyright (c) 2020-present Silero Team. The licence text is in `models/speech/silero-vad/LICENSE`. | https://github.com/snakers4/silero-vad (tag v6.2.3, commit 5cd7945) |

CC BY 4.0 asks for attribution where the model is used: the README's Acknowledgements section credits it.

## Libraries

| Component | Version | Licence | Source |
|---|---|---|---|
| sherpa-onnx (Python package) | 1.13.7 | Apache 2.0 | https://github.com/k2-fsa/sherpa-onnx |
| ONNX Runtime (bundled in sherpa-onnx) | — | MIT | https://github.com/microsoft/onnxruntime |

## Code

The recognition pipeline is a Python port of confab-stt, the dictation backend taken from Confab at commit `5eeebb3` (MIT, Copyright (c) 2026 Swapnil Kale). `frontend/vendor/confab-stt.js` is confab-stt's browser client, with an opt-in AnalyserNode for the live waveform.

## Test data

- `tests/fixtures/speech-sample.wav` comes from Confab, where it was generated with Confab's bundled Kokoro voice (Apache 2.0) for tests.
