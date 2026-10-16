import numpy as np
import pandas as pd
import pytest

from cgmt.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_play(game_id=1, play_id=1, nfl_id=10, speed=5.0, n_pre=10, n_post=40, turn_at=None, turn_deg=90.0):
    """Synthetic 10 Hz play: still before the snap, then constant speed along +x.

    If `turn_at` (frames after snap) is given, the direction of travel rotates by
    `turn_deg` at that frame.
    """
    rows, x, y, heading = [], 0.0, 0.0, 90.0
    t0 = pd.Timestamp("2025-01-01T00:00:00")
    for i in range(n_pre + n_post):
        post = i >= n_pre
        k = i - n_pre
        if turn_at is not None and post and k == turn_at:
            heading += turn_deg
        s = speed if post and k > 0 else 0.0
        if post and k > 0:
            rad = np.deg2rad(90 - heading)  # NGS convention: 0 deg = +y, 90 deg = +x
            x += s * 0.1 * np.cos(rad)
            y += s * 0.1 * np.sin(rad)
        rows.append({"game_id": game_id, "play_id": play_id, "nfl_id": nfl_id,
                     "time": (t0 + pd.Timedelta(milliseconds=100 * i)).isoformat(),
                     "x": x, "y": y, "s": s, "a": 0.0, "dis": s * 0.1, "o": heading % 360,
                     "dir": heading % 360, "event": "ball_snap" if i == n_pre else pd.NA})
    return pd.DataFrame(rows)
