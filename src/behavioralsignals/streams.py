from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator
from dataclasses import dataclass


class AudioStream(ABC):
    """Audio stream interface for the gRPC streaming APIs.

    Implementations must be iterable and yield raw audio chunks (bytes).
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate (Hz) of yielded audio."""

    @abstractmethod
    def __iter__(self) -> Iterator[bytes]:
        raise NotImplementedError


@dataclass(frozen=True)
class IterableAudioStream(AudioStream):
    """Adapter for using an existing iterable of bytes as an AudioStream."""

    iterable: Iterable[bytes]
    sample_rate: int

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.iterable)
