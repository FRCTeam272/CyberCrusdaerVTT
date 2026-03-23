import threading
import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_SECONDS = 6    # seconds of audio per transcription chunk — longer = more Whisper context
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class AudioCapture:
    """Captures audio from the default microphone and puts numpy arrays onto a queue."""

    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if self.stop_event.is_set():
            return
        chunk = indata[:, 0].copy()  # take first channel → mono float32
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)

        if self._buffer_samples >= CHUNK_SAMPLES:
            audio = np.concatenate(self._buffer)
            self.audio_queue.put(audio)
            self._buffer = []
            self._buffer_samples = 0

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush any remaining buffered audio
        if self._buffer_samples > 0:
            audio = np.concatenate(self._buffer)
            self.audio_queue.put(audio)
            self._buffer = []
            self._buffer_samples = 0
