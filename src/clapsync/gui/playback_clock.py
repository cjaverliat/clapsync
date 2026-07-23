"""Smoothly track a coarse master clock without periodic jumps.

The audio played-clock (``AudioEngine.clock_s``) is the master for playback, but
it updates on a coarse ~30 Hz timer and its device time runs at a slightly
different rate than the CPU perf-counter the visuals interpolate on. Two things
consume it — the video decode loop (picture) and the GUI render loop (playhead +
c3d preview) — and both must advance smoothly at 60 Hz between the master's
updates while staying locked to what is audible.

The previous approach extrapolated on a local perf-counter and *hard-snapped*
back to the master whenever the two drifted past a threshold. Because the two
clocks run at slightly different rates the estimate drifts until it jumps: a
periodic sawtooth felt as a hitch every few seconds. Worse, the video loop and
the render loop snapped independently, so the picture and the c3d preview could
sit a whole threshold apart.

``slew_clock`` replaces the snap with a first-order tracking loop: advance by
real elapsed time (feed-forward), then close a fraction of the residual error
each update. The clock tracks the master with no steady-state lag and no jumps,
and every consumer of the same master converges to the same value — so the
picture, the playhead and the c3d preview stay locked to each other and to the
sound.
"""
from __future__ import annotations

import math

# Tracking time constant: the error decays with this e-folding time, regardless
# of how fast the caller ticks. ~80 ms smooths the master's 33 ms update steps
# into fluid motion while keeping the start-up lead (see below) small.
_TAU_S = 0.08
# Errors beyond this are treated as real discontinuities (a seek, a loop wrap,
# the first tick after a re-lock) and snapped, not slewed, so a genuine jump is
# not smeared across many frames.
_SNAP_THRESHOLD_S = 0.25


def slew_clock(
    local_s: float,
    target_s: float,
    dt_s: float,
    tau_s: float = _TAU_S,
    snap_threshold_s: float = _SNAP_THRESHOLD_S,
) -> float:
    """Advance a local clock by dt, then slew a fraction of the error to target.

    The correction fraction is derived from ``tau_s`` and ``dt_s`` so the
    tracking time constant is independent of the caller's tick rate: a loop
    spinning at 1 kHz and one at 60 Hz converge at the same real-time rate.

    Args:
        local_s: The local clock's previous value, in seconds.
        target_s: The master clock to track, in seconds.
        dt_s: Real time elapsed since the last update, in seconds.
        tau_s: Tracking time constant, in seconds. Larger = smoother but slower
            to correct; must be > 0.
        snap_threshold_s: Errors larger than this snap straight to ``target_s``
            instead of slewing.

    Returns:
        The new local clock value, in seconds.
    """
    error = target_s - local_s
    if abs(error) > snap_threshold_s:
        return target_s
    alpha = 1.0 - math.exp(-dt_s / tau_s) if tau_s > 0.0 else 1.0
    return local_s + dt_s + error * alpha
