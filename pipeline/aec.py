"""Acoustic echo cancellation for recordings made without headphones.

The call comes out of the laptop speakers and back into the mic, delayed and coloured by the
room. We have the exact signal that was played (the loopback track), so the echo can be
predicted and subtracted: an adaptive filter learns the speaker -> room -> mic response.

Algorithm: partitioned-block frequency-domain NLMS (overlap-save, constrained gradient),
the standard building block of software AEC (speex, WebRTC AEC3 use variants of it).
  block B = 256 samples (16 ms), filter length P*B = 16*256 = 4096 taps (256 ms: delay plus
  room tail), step size MU, per-bin power normalisation.
Adaptation is frozen while you talk over the call (double talk), otherwise your own voice
would teach the filter to cancel you: a block is double talk when the mic carries clearly
more than the echo the filter currently predicts.

Offline and chunked: state carries across calls to `process`, so an hour streams through in
fixed memory. Runs ~20x faster than real time on a laptop CPU.
"""

from __future__ import annotations

import numpy as np

B = 256
P = 16
MU = 0.5
EPS = 1e-10


class EchoCanceller:
    def __init__(self, coupling_db: float | None = None):
        self.w = np.zeros((P, B + 1), np.complex128)          # filter, per partition
        self.xf = np.zeros((P, B + 1), np.complex128)         # reference spectra history
        self.x_prev = np.zeros(B)
        self.power = np.full(B + 1, 1e-6)                     # smoothed reference power
        self.rest_d = np.zeros(0)
        self.rest_x = np.zeros(0)
        # double-talk threshold: mic energy vs predicted echo energy (+ the expected coupling
        # before the filter has converged)
        self.coupling = 10 ** ((coupling_db if coupling_db is not None else -10.0) / 20)
        self.blocks = 0
        self.adapted = 0

    def _block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(np.concatenate([self.x_prev, x]))
        self.x_prev = x
        self.xf = np.roll(self.xf, 1, axis=0)
        self.xf[0] = X
        Y = np.sum(self.w * self.xf, axis=0)
        y = np.fft.irfft(Y)[B:]
        e = d - y
        self.blocks += 1
        self.power = 0.9 * self.power + 0.1 * np.abs(X) ** 2
        ex, ed, ey, ee = (float(np.dot(v, v)) for v in (x, d, y, e))
        playing = ex > B * 1e-7
        # double talk: the mic has much more than the echo (predicted, or expected while the
        # filter is still learning), so the extra energy is a voice in the room
        expected = max(ey, (self.coupling ** 2) * ex)
        double_talk = ed > 4.0 * expected and ee > 0.5 * ed
        if playing and not double_talk:
            E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
            G = MU * np.conj(self.xf) * E / (P * self.power + EPS)
            g = np.fft.irfft(G, axis=1)
            g[:, B:] = 0.0                                     # constrain to linear conv.
            self.w += np.fft.rfft(g, axis=1)
            self.adapted += 1
        return e

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Echo-cancelled mic for this chunk (same length; the last <B samples are delayed
        to the next call, and `flush` returns them)."""
        d = np.concatenate([self.rest_d, mic.astype(np.float64)])
        x = np.concatenate([self.rest_x, ref.astype(np.float64)])
        n = (len(d) // B) * B
        out = [self._block(d[i:i + B], x[i:i + B]) for i in range(0, n, B)]
        self.rest_d, self.rest_x = d[n:], x[n:]
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

    def flush(self) -> np.ndarray:
        """Remaining samples (passed through: less than one block)."""
        tail, self.rest_d, self.rest_x = self.rest_d, np.zeros(0), np.zeros(0)
        return tail.astype(np.float32)
