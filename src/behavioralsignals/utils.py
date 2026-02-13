import time
from typing import Iterator

from pydub import AudioSegment
from pydub.utils import make_chunks

from .streams import AudioStream


class AudioFileStream(AudioStream):
    """File-backed :class:`behavioralsignals.streams.AudioStream`.

    Loads an audio file, converts it to 16-bit mono, and yields raw PCM chunks.

    Args:
        file_path: Path to the audio file.
        chunk_size: Chunk size in seconds. Default is 0.25s.
        realtime_pacing: If True, sleeps between chunks to simulate real-time streaming.
        pacing_factor: Speed multiplier when realtime_pacing is enabled.
            - 1.0 -> real-time
            - 2.0 -> 2x faster
            - 0.5 -> 2x slower
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: float = 0.25,
        realtime_pacing: bool = False,
        pacing_factor: float = 1.0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if pacing_factor <= 0:
            raise ValueError("pacing_factor must be > 0")

        snd = AudioSegment.from_file(file_path)
        snd = snd.set_sample_width(2)
        snd = snd.set_channels(1)

        self._snd = snd
        self._chunk_ms = int(chunk_size * 1000)
        self._realtime_pacing = realtime_pacing
        self._pacing_factor = pacing_factor

    @property
    def sample_rate(self) -> int:
        return int(self._snd.frame_rate)

    def __iter__(self) -> Iterator[bytes]:
        if not self._realtime_pacing:
            for chunk in make_chunks(self._snd, self._chunk_ms):
                yield chunk.raw_data
            return

        start = time.monotonic()
        stream_time = 0.0

        for chunk in make_chunks(self._snd, self._chunk_ms):
            yield chunk.raw_data

            stream_time += (len(chunk) / 1000.0) / self._pacing_factor
            target_time = start + stream_time
            sleep_for = target_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)


def make_audio_stream(
    file_path: str,
    chunk_size: float = 0.25,
    realtime_pacing: bool = False,
    pacing_factor: float = 1.0,
) -> AudioStream:
    """Factory for creating an :class:`behavioralsignals.streams.AudioStream` from a file."""

    return AudioFileStream(
        file_path=file_path,
        chunk_size=chunk_size,
        realtime_pacing=realtime_pacing,
        pacing_factor=pacing_factor,
    )

