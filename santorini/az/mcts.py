"""PUCT Monte-Carlo tree search over the fast game core.

Nodes are stored in a transposition table keyed by fastgame.key, so the tree
is reused automatically both within a search and across moves of the same
game (call reset() between games). Per node only compact arrays over the
legal actions are kept.

Values are always from the perspective of the player to move at a state and
are negated once per ply on backup; fastgame.step flips the player even on a
winning move, so a terminal state's value is -1 (the player to move lost)
and the uniform negation is correct.
"""

from __future__ import annotations

import numpy as np

from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.model import AZNet


class Node:
    __slots__ = ("ids", "P", "N", "W", "children")

    def __init__(self, ids: np.ndarray, priors: np.ndarray):
        self.ids = ids
        self.P = priors
        self.N = np.zeros(len(ids), dtype=np.int32)
        self.W = np.zeros(len(ids), dtype=np.float32)
        self.children: list[fastgame.State | None] = [None] * len(ids)


class MCTS:
    def __init__(
        self,
        net: AZNet,
        config: AZConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.net = net
        self.config = config if config is not None else AZConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.nodes: dict[bytes, Node] = {}

    def reset(self) -> None:
        self.nodes.clear()

    def run(
        self, state: fastgame.State, sims: int, add_noise: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run simulations from `state`; returns (legal ids, visit counts)."""
        if fastgame.is_terminal(state):
            raise ValueError("Cannot search a terminal state")
        root = self._ensure(state)
        root_priors = root.P
        if add_noise and len(root.ids) > 1:
            cfg = self.config
            noise = self.rng.dirichlet(np.full(len(root.ids), cfg.dirichlet_alpha))
            root_priors = (1 - cfg.dirichlet_frac) * root.P + cfg.dirichlet_frac * noise
        for _ in range(sims):
            self._simulate(state, root, root_priors)
        return root.ids, root.N.copy()

    def _ensure(self, state: fastgame.State) -> Node:
        k = fastgame.key(state)
        node = self.nodes.get(k)
        if node is None:
            node, _ = self._expand(state, k)
        return node

    def _expand(self, state: fastgame.State, k: bytes) -> tuple[Node, float]:
        ids = np.asarray(fastgame.legal_actions(state), dtype=np.int64)
        logits, value = self.net.predict(
            fastgame.canonical_obs(state), fastgame.is_setup(state)
        )
        lp = logits[ids].astype(np.float64)
        lp -= lp.max()
        priors = np.exp(lp)
        priors /= priors.sum()
        node = Node(ids, priors.astype(np.float32))
        self.nodes[k] = node
        return node, value

    def _simulate(
        self, state: fastgame.State, root: Node, root_priors: np.ndarray
    ) -> None:
        c_puct = self.config.c_puct
        node = root
        priors = root_priors
        path: list[tuple[Node, int]] = []
        while True:
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            # sqrt(N+1) instead of sqrt(N) so the very first simulation
            # already follows the priors rather than an arbitrary tie-break.
            u = c_puct * priors * np.sqrt(node.N.sum() + 1) / (1.0 + node.N)
            idx = int(np.argmax(q + u))
            path.append((node, idx))

            child = node.children[idx]
            if child is None:
                child = fastgame.step(state, int(node.ids[idx]))
                node.children[idx] = child
            state = child

            if fastgame.is_terminal(state):
                value = fastgame.terminal_value(state)
                break
            k = fastgame.key(state)
            nxt = self.nodes.get(k)
            if nxt is None:
                _, value = self._expand(state, k)
                break
            node = nxt
            priors = node.P

        for node, idx in reversed(path):
            value = -value
            node.N[idx] += 1
            node.W[idx] += value
