# CyberCrusaderVTT — Live Transcription

Real-time microphone transcription with a Rich terminal UI, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Captures audio from your microphone in 6-second chunks, transcribes them locally using Whisper, and displays a live scrolling transcript. Press `Ctrl+C` to stop and save the transcript to a file.

## Requirements

- Python 3.12+
- A working microphone

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py                         # small model, English, auto-named output file
python main.py --model tiny            # faster, less accurate
python main.py --model medium          # slower, more accurate
python main.py --language auto         # auto-detect spoken language
python main.py --output my_notes.txt   # custom output filename
python main.py --list-devices          # show available audio input devices
python main.py --device 2              # use a specific device index
```

### Model sizes

| Model    | Speed   | Accuracy |
|----------|---------|----------|
| tiny     | fastest | lowest   |
| base     | fast    | low      |
| small    | default | good     |
| medium   | slow    | better   |
| large-v3 | slowest | best     |

Models are downloaded automatically on first use via Hugging Face.

## Output

Transcripts are saved as plain text files named `transcript_YYYYMMDD_HHMMSS.txt` (or your custom name) when you press `Ctrl+C`.
