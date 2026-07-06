"""D4 (square dihedral) symmetry transforms for observations and policies.

Transform ids 0..7: `t % 4` counter-of-board 90-degree rotations applied
after an optional x-flip when `t >= 4`. All tables are precomputed at import.

Permutation convention: `PERM[t][i]` is where index `i` lands under transform
`t`, so a transformed distribution is `new_pi[PERM[t]] = pi`.

Placement plies and playing plies live in different index spaces (0..24 mean
board cells during setup but from-cell-(0,0) moves during play), so pi must
be permuted with SPACE_PERM during setup and ACTION_PERM during play — the
phase split is mandatory.
"""

from __future__ import annotations

import numpy as np

from santorini import utils
from santorini.az.fastgame import N, NUM_ACTIONS, NUM_SPACES

NUM_TRANSFORMS = 8


def apply_xy(t: int, x: int, y: int) -> tuple[int, int]:
    """Map a board coordinate through transform t (affine; works for any ints)."""
    if t >= 4:
        x = N - 1 - x
    for _ in range(t % 4):
        x, y = N - 1 - y, x
    return x, y


def _apply_dir(t: int, dx: int, dy: int) -> tuple[int, int]:
    """Linear part of transform t, for direction vectors."""
    if t >= 4:
        dx = -dx
    for _ in range(t % 4):
        dx, dy = -dy, dx
    return dx, dy


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    space = np.empty((NUM_TRANSFORMS, NUM_SPACES), dtype=np.int64)
    xmaj = np.empty((NUM_TRANSFORMS, NUM_SPACES), dtype=np.int64)
    dirs = np.empty((NUM_TRANSFORMS, 8), dtype=np.int64)
    action = np.empty((NUM_TRANSFORMS, NUM_ACTIONS), dtype=np.int64)

    for t in range(NUM_TRANSFORMS):
        for s in range(NUM_SPACES):
            x, y = utils.decode_space(s)
            tx, ty = apply_xy(t, x, y)
            space[t, s] = utils.encode_space((tx, ty))
        # obs arrays are indexed [x][y]: flat x-major index = x*N + y
        for x in range(N):
            for y in range(N):
                tx, ty = apply_xy(t, x, y)
                xmaj[t, x * N + y] = tx * N + ty
        for d, (dx, dy) in enumerate(utils.DIRS):
            dirs[t, d] = utils.DIRS.index(_apply_dir(t, dx, dy))
        for s in range(NUM_SPACES):
            for md in range(8):
                for bd in range(8):
                    a = s * 64 + md * 8 + bd
                    action[t, a] = space[t, s] * 64 + dirs[t, md] * 8 + dirs[t, bd]
    return space, xmaj, dirs, action


SPACE_PERM, _XMAJ_PERM, DIR_PERM, ACTION_PERM = _build_tables()


def transform_obs(obs: np.ndarray, t: int) -> np.ndarray:
    """Spatially permute a (5,5,C) [x][y]-indexed observation."""
    flat = obs.reshape(NUM_SPACES, -1)
    out = np.empty_like(flat)
    out[_XMAJ_PERM[t]] = flat
    return out.reshape(obs.shape)


def transform_action_ids(ids: np.ndarray, t: int, setup: bool) -> np.ndarray:
    """Permute sparse policy indices (placement ids during setup)."""
    return SPACE_PERM[t][ids] if setup else ACTION_PERM[t][ids]


def transform_pi(pi: np.ndarray, t: int, setup: bool) -> np.ndarray:
    """Permute a dense policy vector. During setup pi may be length 25 or a
    length-1600 vector whose mass sits in the first 25 entries."""
    out = np.zeros_like(pi)
    if setup:
        out[SPACE_PERM[t]] = pi[:NUM_SPACES]
    else:
        out[ACTION_PERM[t]] = pi
    return out
