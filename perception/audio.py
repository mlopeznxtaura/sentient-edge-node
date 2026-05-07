"""
perception/audio.py — Microphone capture with VAD (Cluster 01: Sentient Edge Node)

Captures audio from mic via PortAudio, detects voice activity,
buffers speech segments, and yields raw audio chunks for downstream
transcription or audio analysis.

SDKs: PortAudio (pyaudio)
"""
import time
import json
import logging
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Generator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1024
FORMAT_INT16 = 8  # pyaudio.paInt16


@dataclass
class AudioChunk:
    chunk_id: int
    timestamp_ms: float
    sample_rate: int
    channels: int
    frames: int
    is_speech: bool
    rms_db: float
    raw: bytes = b""  # excluded from JSON output


@dataclass
class SpeechSegment:
    segment_id: int
    start_ms: float
    end_ms: float
    duration_ms: float
    sample_rate: int
    audio_bytes: bytes


class SimpleVAD:
    """
    Energy-based voice activity detection.
    No ML model required — works well for clean mic input.
    """

    def __init__(self, threshold_db: float = -40.0, hangover_chunks: int = 10):
        self._threshold_db = threshold_db
        self._hangover = hangover_chunks
        self._hangover_counter = 0
        self._active = False

    def process(self, rms_db: float) -> bool:
        if rms_db >= self._threshold_db:
            self._active = True
            self._hangover_counter = self._hangover
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
        else:
            self._active = False
        return self._active


class AudioPerception:
    """
    Continuous microphone capture with VAD.
    Yields AudioChunk for every frame, and SpeechSegment when a
    complete speech utterance finishes.
    """

    def __init__(self, config: dict):
        self._cfg = config.get("audio", {})
        self._sample_rate = self._cfg.get("sample_rate", SAMPLE_RATE)
        self._chunk_frames = self._cfg.get("chunk_frames", CHUNK_FRAMES)
        self._device_index = self._cfg.get("device_index", None)
        self._vad_threshold_db = self._cfg.get("vad_threshold_db", -40.0)
        self._pa = None
        self._stream = None
        self._chunk_id = 0
        self._segment_id = 0
        self._vad = SimpleVAD(threshold_db=self._vad_threshold_db)
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._running = False

    def load(self) -> None:
        import pyaudio
        self._pa = pyaudio.PyAudio()
        logger.info("PortAudio initialized. Devices: %d", self._pa.get_device_count())
        if self._device_index is None:
            info = self._pa.get_default_input_device_info()
            logger.info("Default mic: %s", info["name"])

    def _rms_db(self, raw: bytes) -> float:
        import struct, math
        if not raw:
            return -100.0
        samples = struct.unpack(f"{len(raw)//2}h", raw)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        if rms == 0:
            return -100.0
        return 20 * math.log10(rms / 32768.0)

    def _callback(self, in_data, frame_count, time_info, status):
        self._q.put_nowait(in_data)
        return (None, 0)  # paContinue

    def open_stream(self) -> None:
        import pyaudio
        self._stream = self._pa.open(
            format=FORMAT_INT16,
            channels=CHANNELS,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._chunk_frames,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self._running = True
        logger.info("Audio stream started at %dHz", self._sample_rate)

    def stream_chunks(self) -> Generator[AudioChunk, None, None]:
        """Yield AudioChunk for every mic frame."""
        if self._stream is None:
            self.open_stream()
        try:
            while self._running:
                try:
                    raw = self._q.get(timeout=1.0)
                except queue.Empty:
                    continue
                rms_db = self._rms_db(raw)
                is_speech = self._vad.process(rms_db)
                self._chunk_id += 1
                yield AudioChunk(
                    chunk_id=self._chunk_id,
                    timestamp_ms=time.time() * 1000,
                    sample_rate=self._sample_rate,
                    channels=CHANNELS,
                    frames=self._chunk_frames,
                    is_speech=is_speech,
                    rms_db=round(rms_db, 2),
                    raw=raw,
                )
        finally:
            self.release()

    def stream_segments(self) -> Generator[SpeechSegment, None, None]:
        """
        Buffer chunks into complete speech segments.
        Yields a SpeechSegment when VAD transitions from active to inactive.
        """
        buffer = []
        in_speech = False
        start_ms = 0.0

        for chunk in self.stream_chunks():
            if chunk.is_speech and not in_speech:
                in_speech = True
                start_ms = chunk.timestamp_ms
                buffer = [chunk.raw]
            elif chunk.is_speech and in_speech:
                buffer.append(chunk.raw)
            elif not chunk.is_speech and in_speech:
                in_speech = False
                end_ms = chunk.timestamp_ms
                self._segment_id += 1
                yield SpeechSegment(
                    segment_id=self._segment_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=round(end_ms - start_ms, 1),
                    sample_rate=self._sample_rate,
                    audio_bytes=b"".join(buffer),
                )
                buffer = []

    def release(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        logger.info("Audio stream released")

    @staticmethod
    def list_devices() -> list:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({"index": i, "name": info["name"], "sample_rate": int(info["defaultSampleRate"])})
        pa.terminate()
        return devices


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--mode", choices=["chunks", "segments"], default="chunks")
    args = parser.parse_args()

    if args.list_devices:
        print(json.dumps(AudioPerception.list_devices(), indent=2))
        exit(0)

    with open(args.config) as f:
        cfg = json.load(f)

    audio = AudioPerception(cfg)
    audio.load()

    if args.mode == "chunks":
        for chunk in audio.stream_chunks():
            c = asdict(chunk)
            del c["raw"]
            print(json.dumps(c))
    else:
        for seg in audio.stream_segments():
            print(json.dumps({
                "segment_id": seg.segment_id,
                "start_ms": seg.start_ms,
                "duration_ms": seg.duration_ms,
                "bytes": len(seg.audio_bytes),
            }))
