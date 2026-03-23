import queue
import threading
from datetime import datetime

import numpy as np
from faster_whisper import WhisperModel


class TranscriptEntry:
    def __init__(self, timestamp: str, text: str):
        self.timestamp = timestamp
        self.text = text


class Transcriber:
    """
    Reads audio chunks from audio_queue, transcribes them with faster-whisper,
    and appends TranscriptEntry objects to transcript_store under transcript_lock.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        transcript_store: list,
        transcript_lock: threading.Lock,
        model_size: str = "small",
        device: str = "cpu",
        language: str = "en",
    ):
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.transcript_store = transcript_store
        self.transcript_lock = transcript_lock
        self.language = language
        self.status = "Loading model..."
        self.status_lock = threading.Lock()
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        with self.status_lock:
            self.status = "Listening"

    def get_status(self) -> str:
        with self.status_lock:
            return self.status

    def _set_status(self, status: str) -> None:
        with self.status_lock:
            self.status = status

    def _build_initial_prompt(self) -> str:
        """Return the last few transcript entries as context for Whisper."""
        with self.transcript_lock:
            recent = self.transcript_store[-3:]
        return " ".join(e.text for e in recent)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=self._build_initial_prompt() or None,
            condition_on_previous_text=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def run(self) -> None:
        """Worker loop — call this in a dedicated thread."""
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._set_status("Transcribing...")
            text = self._transcribe_chunk(audio)

            if text:
                entry = TranscriptEntry(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    text=text,
                )
                with self.transcript_lock:
                    self.transcript_store.append(entry)

            self._set_status("Listening")

        # Drain remaining items after stop is signalled
        while True:
            try:
                audio = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            self._set_status("Transcribing...")
            text = self._transcribe_chunk(audio)
            if text:
                entry = TranscriptEntry(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    text=text,
                )
                with self.transcript_lock:
                    self.transcript_store.append(entry)

        self._set_status("Stopped")
