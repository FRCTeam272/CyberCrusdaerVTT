"""
Real-time microphone transcription with a Rich terminal UI.

Usage:
    python main.py                        # small model, English, auto-named output file
    python main.py --model tiny           # faster, less accurate
    python main.py --model medium         # slower, more accurate
    python main.py --language auto        # auto-detect spoken language
    python main.py --output my_notes.txt  # custom output filename
    python main.py --list-devices         # show available audio input devices
    python main.py --device 2            # use a specific device index
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from rich.console import Console
from rich.live import Live

from audio_capture import AudioCapture
from transcription import Transcriber
import ui

console = Console()


def list_devices() -> None:
    console.print("\n[bold cyan]Available audio input devices:[/bold cyan]\n")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            marker = "[green]>[/green]" if i == sd.default.device[0] else " "
            console.print(
                f"  {marker} [{i:2d}] [cyan]{device['name']}[/cyan]"
                f"  ({device['max_input_channels']} ch, {int(device['default_samplerate'])} Hz)"
            )
    console.print()


def save_transcript(entries: list, path: Path) -> None:
    if not entries:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPT\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for entry in entries:
            f.write(f"[{entry.timestamp}] {entry.text}\n")
    console.print(f"\n[green]Transcript saved to[/green] [cyan]{path}[/cyan]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code spoken in the audio (default: en). Use 'auto' to detect automatically.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename for transcript (default: transcript_YYYYMMDD_HHMMSS.txt)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    output_path = Path(
        args.output
        if args.output
        else f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    if args.device is not None:
        sd.default.device[0] = args.device

    # Shared state
    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    transcript_store: list = []
    transcript_lock = threading.Lock()

    # Load model before entering Live UI so download progress is visible
    console.print(f"\n[cyan]Loading Whisper model [bold]{args.model}[/bold]...[/cyan]")
    language = None if args.language == "auto" else args.language
    transcriber = Transcriber(
        audio_queue=audio_queue,
        stop_event=stop_event,
        transcript_store=transcript_store,
        transcript_lock=transcript_lock,
        model_size=args.model,
        language=language,
    )

    audio_capture = AudioCapture(audio_queue=audio_queue, stop_event=stop_event)

    # Start transcription worker thread
    trans_thread = threading.Thread(target=transcriber.run, daemon=True, name="transcriber")
    trans_thread.start()

    # Start audio capture
    audio_capture.start()

    # Run Rich Live UI on main thread
    try:
        with Live(
            ui.render(transcript_store, transcript_lock, transcriber.get_status(), args.model),
            refresh_per_second=4,
            screen=True,
            console=console,
        ) as live:
            while not stop_event.is_set():
                live.update(
                    ui.render(transcript_store, transcript_lock, transcriber.get_status(), args.model)
                )
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        # Signal shutdown and wait for threads to finish
        stop_event.set()
        audio_capture.stop()
        trans_thread.join(timeout=10)

        with transcript_lock:
            final_entries = list(transcript_store)

        save_transcript(final_entries, output_path)


if __name__ == "__main__":
    main()
