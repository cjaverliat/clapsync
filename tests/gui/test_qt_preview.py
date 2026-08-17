"""Pure offset/herding math for the hybrid preview controller."""
from clapsync.gui.qt_preview import herd_action, local_position_s


def test_local_position_subtracts_offset():
    assert local_position_s(5.0, 2.0) == 3.0


def test_local_position_floors_at_zero():
    # Before its offset a track has not started; local time never goes negative.
    assert local_position_s(1.0, 2.0) == 0.0


def test_herd_holds_within_threshold():
    # Seek-only on averaged drift; no rate trim (would wobble audible pitch).
    assert herd_action(0.0)[0] == "hold"
    assert herd_action(0.02)[0] == "hold"
    assert herd_action(-0.029)[0] == "hold"


def test_herd_seeks_beyond_threshold():
    assert herd_action(0.05) == ("seek", 0.0)
    assert herd_action(-0.2) == ("seek", 0.0)
