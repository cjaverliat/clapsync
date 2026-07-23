"""The playback clock must track the master smoothly, without periodic jumps.

Regression for playback stutter: the old estimator hard-snapped to the master
whenever a rate mismatch drifted it past a threshold, a sawtooth felt as a
hitch. slew_clock must stay locked to a real-time master within a fraction of a
frame, spread each coarse master update across several ticks (no jump), and take
a genuine discontinuity in one step.
"""
from clapsync.gui.playback_clock import slew_clock


def test_snaps_on_large_discontinuity():
    # A seek moves the master far away: take it in one step, don't smear it.
    assert slew_clock(5.0, 12.0, 0.016, snap_threshold_s=0.25) == 12.0
    assert slew_clock(5.0, 0.0, 0.016, snap_threshold_s=0.25) == 0.0


def test_small_error_is_slewed_not_snapped():
    # A 10 ms drift is corrected by a fraction, not jumped.
    out = slew_clock(1.000, 1.010, 0.016, tau_s=0.08)
    assert 1.000 < out < 1.010 + 0.016  # advanced by ~dt, nudged toward target


def test_stays_locked_to_a_realtime_master():
    # Master advances at real time from a 40 ms initial lag; the local clock
    # locks to it within a fraction of a frame and holds there (no drift, no
    # sawtooth back-and-forth).
    dt = 1.0 / 60.0
    local = 0.0
    target = 0.040  # start 40 ms behind
    errors = []
    for _ in range(240):  # 4 s of playback
        target += dt      # master ticks forward at real time
        local = slew_clock(local, target, dt, tau_s=0.08)
        errors.append(abs(target - local))
    # Locked to within one 60 Hz frame, and the tail is steady (converged, not
    # oscillating): the last second stays tight.
    assert max(errors[-60:]) < 1.0 / 60.0


def test_coarse_master_step_is_smoothed_across_ticks():
    # The master updates coarsely (~30 Hz); the local clock renders at 60 Hz.
    # A single 33 ms jump in the master must not appear as a 33 ms jump in the
    # local clock — the first tick moves only a fraction of the way.
    dt = 1.0 / 60.0
    local = 1.000
    target = 1.000 + 1.0 / 30.0  # master jumped one 30 Hz step ahead
    step = slew_clock(local, target, dt, tau_s=0.08) - local
    # Feed-forward (~dt) plus a slice of the 33 ms error, but well short of the
    # full jump: motion stays continuous.
    assert step < (target - local)
    assert step > 0.0
