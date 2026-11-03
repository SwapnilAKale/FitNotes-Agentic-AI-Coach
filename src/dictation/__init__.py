"""
Dictation for the chat's mic button: speech recognition on this server.

NVIDIA Parakeet TDT 0.6B v2 (int8) for the words and Silero VAD for where
speech is, both run with sherpa-onnx; models in models/speech/ (or SPEECH_MODELS),
downloaded by scripts/fetch_speech_models.py.

  service.py     the /stt WebSocket: protocol, limits, one recording per slot
  engine.py      starts the worker process on demand and stops it when idle
  worker.py      the process that holds the model
  recognizer.py  Parakeet, and one recording's pipeline
  turns.py       speech detection, turns, overlap trimming

Ported to Python from confab-stt (the dictation backend taken from Confab at
commit 5eeebb3, MIT, Swapnil Kale), keeping the settings Confab's dictation
uses. Third-party notices: THIRD_PARTY_NOTICES.md in this folder.
"""
