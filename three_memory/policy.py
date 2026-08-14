"""Tiny boxed policy: when to collect/write, whether to apply a matched record.

Features exclude door identity so the policy cannot memorize 'red → use_key'.
The file's action= tag still chooses the action (frozen grammar).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hashing import hash_arrays

COLLECT_IGNORE = 0
COLLECT_PEEK = 1
COLLECT_COMMIT = 2
COLLECT_NAMES = ("ignore", "peek", "commit")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - float(np.max(x))
    e = np.exp(z)
    return e / (float(np.sum(e)) + 1e-12)


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


class UsePolicy:
    """Linear collect (3-way) + apply gate + write gate. Cortex stays frozen elsewhere."""

    n_feat = 2  # s_hit, opportunity (w_hit or opened) — no door id, no novelty

    def __init__(self, seed: int = 7, lr: float = 0.15):
        rng = np.random.default_rng(seed)
        self.W_collect = rng.normal(0.0, 0.05, size=(self.n_feat, 3))
        self.b_collect = np.zeros(3, dtype=np.float64)
        # Untrained: prefer not applying the store (species prior wins).
        self.w_apply = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_apply = np.array(-1.2, dtype=np.float64)
        # Untrained: do not author a note from a life event.
        self.w_write = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_write = np.array(-1.2, dtype=np.float64)
        # Untrained: dump the pile (species prior + mixed notes). Learn to select.
        self.w_retrieve = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_retrieve = np.array(-1.2, dtype=np.float64)
        # Untrained: do not copy action= from the file into motor logits.
        self.w_use = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_use = np.array(-1.2, dtype=np.float64)
        # Untrained: apply every match (mix). Learn to pick one (newest).
        self.w_pick = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_pick = np.array(-1.2, dtype=np.float64)
        # Untrained: write {door} only. Learn to include action=.
        self.w_schema = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_schema = np.array(-1.2, dtype=np.float64)
        # Untrained recency prior: newest wins. Learn to prefer ok=1 over newest.
        self.w_rank = np.array([1.2, 0.0], dtype=np.float64)
        # Untrained: read action=, match door=. Learn alt names do= / here=.
        self.w_key = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_key = np.array(-1.2, dtype=np.float64)
        self.w_match = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_match = np.array(-1.2, dtype=np.float64)
        # Untrained: write action= / door=. Learn to emit do= / here=.
        self.w_wkey = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_wkey = np.array(-1.2, dtype=np.float64)
        self.w_wplace = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_wplace = np.array(-1.2, dtype=np.float64)
        # Untrained: keep filename-first / dump-all W hits. Learn newest one.
        self.w_wsel = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_wsel = np.array(-1.2, dtype=np.float64)
        # Untrained: keep filename-first stub. Learn the page that has action=/do=.
        self.w_wcomp = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_wcomp = np.array(-1.2, dtype=np.float64)
        # Untrained: among keys found in files, prefer any hit. Learn uncommon keys.
        # Not a {door, here} menu: candidates are whatever keys the files have.
        self.w_qname = np.array([1.2, 0.0], dtype=np.float64)
        # Untrained: copy the query key (place code). Learn another integer field in the file.
        self.w_vname = np.array([1.2, 0.0], dtype=np.float64)
        # Untrained: among unread files, prefer any that carry the current code. Learn rare keys.
        self.w_search = np.array([1.2, 0.0], dtype=np.float64)
        self.lr = lr
        self.n_updates = 0
        self._hash0 = self.weight_hash()

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.W_collect,
            self.b_collect,
            self.w_apply,
            np.asarray(self.b_apply).reshape(1),
            self.w_write,
            np.asarray(self.b_write).reshape(1),
            self.w_retrieve,
            np.asarray(self.b_retrieve).reshape(1),
            self.w_use,
            np.asarray(self.b_use).reshape(1),
            self.w_pick,
            np.asarray(self.b_pick).reshape(1),
            self.w_schema,
            np.asarray(self.b_schema).reshape(1),
            self.w_rank,
            self.w_key,
            np.asarray(self.b_key).reshape(1),
            self.w_match,
            np.asarray(self.b_match).reshape(1),
            self.w_wkey,
            np.asarray(self.b_wkey).reshape(1),
            self.w_wplace,
            np.asarray(self.b_wplace).reshape(1),
            self.w_wsel,
            np.asarray(self.b_wsel).reshape(1),
            self.w_wcomp,
            np.asarray(self.b_wcomp).reshape(1),
            self.w_qname,
            self.w_vname,
            self.w_search,
        )

    def weight_hash(self) -> str:
        return hash_arrays(self.arrays())

    def changed(self) -> bool:
        return self.weight_hash() != self._hash0

    @staticmethod
    def features(s_hit: bool, w_hit: bool, novelty: float = 0.0) -> np.ndarray:
        del novelty
        return np.array([1.0 if s_hit else 0.0, 1.0 if w_hit else 0.0], dtype=np.float64)

    def decide(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        logits = feat @ self.W_collect + self.b_collect
        probs = softmax(logits)
        p_apply = sigmoid(float(feat @ self.w_apply + self.b_apply))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            c = int(rng.integers(0, 3))
            apply = bool(rng.random() < 0.5)
        else:
            c = int(np.argmax(probs))
            apply = bool(p_apply >= 0.5)
        logp_c = float(np.log(probs[c] + 1e-12))
        logp_a = float(np.log((p_apply if apply else (1.0 - p_apply)) + 1e-12))
        return {
            "kind": "collect",
            "collect_mode": COLLECT_NAMES[c],
            "collect_idx": c,
            "apply": apply,
            "p_apply": p_apply,
            "probs_collect": probs.tolist(),
            "logp": logp_c + logp_a,
            "feat": feat.tolist(),
        }

    @staticmethod
    def retrieve_features(n_store: int, n_hits: int) -> np.ndarray:
        """Grown pile? Any match? No door id."""
        return np.array([1.0 if n_store >= 2 else 0.0, 1.0 if n_hits >= 1 else 0.0], dtype=np.float64)

    def decide_retrieve(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        p_select = sigmoid(float(feat @ self.w_retrieve + self.b_retrieve))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            select = bool(rng.random() < 0.5)
        else:
            select = bool(p_select >= 0.5)
        logp = float(np.log((p_select if select else (1.0 - p_select)) + 1e-12))
        return {
            "kind": "retrieve",
            "retrieve_mode": "select" if select else "dump",
            "select": select,
            "p_select": p_select,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_use(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Gate: copy file action= into logits[action], or ignore the tag."""
        p_use = sigmoid(float(feat @ self.w_use + self.b_use))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            use = bool(rng.random() < 0.5)
        else:
            use = bool(p_use >= 0.5)
        logp = float(np.log((p_use if use else (1.0 - p_use)) + 1e-12))
        return {
            "kind": "use",
            "use": use,
            "p_use": p_use,
            "logp": logp,
            "feat": feat.tolist(),
        }

    @staticmethod
    def pick_features(n_hits: int) -> np.ndarray:
        """Several matches? Any match? No door id, no action=."""
        return np.array([1.0 if n_hits >= 2 else 0.0, 1.0 if n_hits >= 1 else 0.0], dtype=np.float64)

    def decide_pick(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Gate: apply only the newest match, or copy every hit."""
        p_one = sigmoid(float(feat @ self.w_pick + self.b_pick))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            one = bool(rng.random() < 0.5)
        else:
            one = bool(p_one >= 0.5)
        logp = float(np.log((p_one if one else (1.0 - p_one)) + 1e-12))
        return {
            "kind": "pick",
            "one": one,
            "p_one": p_one,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_wsel(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: filename-first or dump-all. True: commit the newest W hit only."""
        p_alt = sigmoid(float(feat @ self.w_wsel + self.b_wsel))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "wsel",
            "wsel_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    @staticmethod
    def wcomp_features(has_payload: bool, n_hits: int) -> np.ndarray:
        """Any action=/do= in the W hits? Several matches? No door id, no integer."""
        return np.array([1.0 if has_payload else 0.0, 1.0 if n_hits >= 2 else 0.0], dtype=np.float64)

    def decide_wcomp(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: filename-first. True: commit the first hit that has action= or do=."""
        p_alt = sigmoid(float(feat @ self.w_wcomp + self.b_wcomp))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "wcomp",
            "wcomp_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_schema(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Gate: write {door, action} or {door} only. Integer still comes from the event."""
        p_complete = sigmoid(float(feat @ self.w_schema + self.b_schema))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            complete = bool(rng.random() < 0.5)
        else:
            complete = bool(p_complete >= 0.5)
        logp = float(np.log((p_complete if complete else (1.0 - p_complete)) + 1e-12))
        return {
            "kind": "schema",
            "complete": complete,
            "p_complete": p_complete,
            "logp": logp,
            "feat": feat.tolist(),
        }

    @staticmethod
    def rank_features(is_newest: bool, has_ok: bool) -> np.ndarray:
        """Recency vs success mark. No door id, no action=."""
        return np.array([1.0 if is_newest else 0.0, 1.0 if has_ok else 0.0], dtype=np.float64)

    def decide_rank(
        self,
        items: list[tuple[bool, bool]],
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Choose one hit. Untrained argmax is newest. Trained can prefer ok=1."""
        rng = rng or np.random.default_rng()
        if not items:
            return {"kind": "rank", "idx": 0, "feats": [], "logp": 0.0, "feat": [0.0, 0.0]}
        feats = np.stack([self.rank_features(n, o) for n, o in items], axis=0)
        scores = feats @ self.w_rank
        probs = softmax(scores)
        if float(rng.random()) < epsilon:
            idx = int(rng.integers(0, len(items)))
        else:
            idx = int(np.argmax(scores))
        logp = float(np.log(probs[idx] + 1e-12))
        return {
            "kind": "rank",
            "idx": idx,
            "feats": feats.tolist(),
            "feat": feats[idx].tolist(),
            "is_newest": bool(items[idx][0]),
            "has_ok": bool(items[idx][1]),
            "logp": logp,
        }

    def decide_key(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: copy action=. True: copy do=."""
        p_alt = sigmoid(float(feat @ self.w_key + self.b_key))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "key",
            "key_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    @staticmethod
    def qname_features(has_hit: bool, key_common: bool) -> np.ndarray:
        """Hit on this file-key at the current code? Key appears in many files? No name id, no door id."""
        return np.array([1.0 if has_hit else 0.0, 1.0 if key_common else 0.0], dtype=np.float64)

    def decide_qname(
        self,
        items: list[tuple[bool, bool]],
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Choose a query name among keys that exist in files. Untrained: any hit (first on ties)."""
        rng = rng or np.random.default_rng()
        if not items:
            return {"kind": "qname", "idx": 0, "feats": [], "logp": 0.0, "feat": [0.0, 0.0]}
        feats = np.stack([self.qname_features(h, c) for h, c in items], axis=0)
        scores = feats @ self.w_qname
        probs = softmax(scores)
        if float(rng.random()) < epsilon:
            idx = int(rng.integers(0, len(items)))
        else:
            idx = int(np.argmax(scores))
        logp = float(np.log(probs[idx] + 1e-12))
        return {
            "kind": "qname",
            "idx": idx,
            "feats": feats.tolist(),
            "feat": feats[idx].tolist(),
            "has_hit": bool(items[idx][0]),
            "key_common": bool(items[idx][1]),
            "logp": logp,
        }

    def decide_vname(
        self,
        items: list[tuple[bool, bool]],
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Choose a copy name among keys on the hit. Untrained: the query key."""
        rng = rng or np.random.default_rng()
        if not items:
            return {"kind": "vname", "idx": 0, "feats": [], "logp": 0.0, "feat": [0.0, 0.0]}
        feats = np.stack([self.qname_features(q, c) for q, c in items], axis=0)
        scores = feats @ self.w_vname
        probs = softmax(scores)
        if float(rng.random()) < epsilon:
            idx = int(rng.integers(0, len(items)))
        else:
            idx = int(np.argmax(scores))
        logp = float(np.log(probs[idx] + 1e-12))
        return {
            "kind": "vname",
            "idx": idx,
            "feats": feats.tolist(),
            "feat": feats[idx].tolist(),
            "is_query": bool(items[idx][0]),
            "key_common": bool(items[idx][1]),
            "logp": logp,
        }

    def decide_search(
        self,
        items: list[tuple[bool, bool]],
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Choose one file among a pool. Untrained: any file that carries the current code."""
        rng = rng or np.random.default_rng()
        if not items:
            return {"kind": "search", "idx": 0, "feats": [], "logp": 0.0, "feat": [0.0, 0.0]}
        feats = np.stack([self.qname_features(c, r) for c, r in items], axis=0)
        scores = feats @ self.w_search
        probs = softmax(scores)
        if float(rng.random()) < epsilon:
            idx = int(rng.integers(0, len(items)))
        else:
            idx = int(np.argmax(scores))
        logp = float(np.log(probs[idx] + 1e-12))
        return {
            "kind": "search",
            "idx": idx,
            "feats": feats.tolist(),
            "feat": feats[idx].tolist(),
            "has_code": bool(items[idx][0]),
            "has_rare": bool(items[idx][1]),
            "logp": logp,
        }

    def decide_match(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: match door=. True: match here=."""
        p_alt = sigmoid(float(feat @ self.w_match + self.b_match))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "match",
            "match_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_wkey(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: write action=. True: write do=."""
        p_alt = sigmoid(float(feat @ self.w_wkey + self.b_wkey))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "wkey",
            "wkey_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_wplace(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """False: write door=. True: write here=."""
        p_alt = sigmoid(float(feat @ self.w_wplace + self.b_wplace))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            alt = bool(rng.random() < 0.5)
        else:
            alt = bool(p_alt >= 0.5)
        logp = float(np.log((p_alt if alt else (1.0 - p_alt)) + 1e-12))
        return {
            "kind": "wplace",
            "wplace_alt": alt,
            "p_alt": p_alt,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def decide_write(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        p_write = sigmoid(float(feat @ self.w_write + self.b_write))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            write = bool(rng.random() < 0.5)
        else:
            write = bool(p_write >= 0.5)
        logp = float(np.log((p_write if write else (1.0 - p_write)) + 1e-12))
        return {
            "kind": "write",
            "write": write,
            "p_write": p_write,
            "logp": logp,
            "feat": feat.tolist(),
        }

    def update(self, traces: list[dict[str, Any]], advantage: float) -> None:
        """REINFORCE. Advantage is episode return minus baseline."""
        if not traces or abs(advantage) < 1e-12:
            return
        lr = self.lr * float(advantage)
        for tr in traces:
            if tr.get("kind") == "rank":
                feats = np.asarray(tr.get("feats") or [tr["feat"]], dtype=np.float64)
                if feats.size == 0:
                    continue
                scores = feats @ self.w_rank
                probs = softmax(scores)
                idx = int(tr["idx"])
                grad = -probs
                grad[idx] += 1.0
                self.w_rank += lr * (feats.T @ grad)
                self.n_updates += 1
                continue
            if tr.get("kind") == "qname":
                feats = np.asarray(tr.get("feats") or [tr["feat"]], dtype=np.float64)
                if feats.size == 0:
                    continue
                scores = feats @ self.w_qname
                probs = softmax(scores)
                idx = int(tr["idx"])
                grad = -probs
                grad[idx] += 1.0
                self.w_qname += lr * (feats.T @ grad)
                self.n_updates += 1
                continue
            if tr.get("kind") == "vname":
                feats = np.asarray(tr.get("feats") or [tr["feat"]], dtype=np.float64)
                if feats.size == 0:
                    continue
                scores = feats @ self.w_vname
                probs = softmax(scores)
                idx = int(tr["idx"])
                grad = -probs
                grad[idx] += 1.0
                self.w_vname += lr * (feats.T @ grad)
                self.n_updates += 1
                continue
            if tr.get("kind") == "search":
                feats = np.asarray(tr.get("feats") or [tr["feat"]], dtype=np.float64)
                if feats.size == 0:
                    continue
                scores = feats @ self.w_search
                probs = softmax(scores)
                idx = int(tr["idx"])
                grad = -probs
                grad[idx] += 1.0
                self.w_search += lr * (feats.T @ grad)
                self.n_updates += 1
                continue
            feat = np.asarray(tr["feat"], dtype=np.float64)
            if tr.get("kind") == "key":
                p = sigmoid(float(feat @ self.w_key + self.b_key))
                y = 1.0 if tr["key_alt"] else 0.0
                g = y - p
                self.w_key += lr * g * feat
                self.b_key = np.array(float(self.b_key) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "match":
                p = sigmoid(float(feat @ self.w_match + self.b_match))
                y = 1.0 if tr["match_alt"] else 0.0
                g = y - p
                self.w_match += lr * g * feat
                self.b_match = np.array(float(self.b_match) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "wkey":
                p = sigmoid(float(feat @ self.w_wkey + self.b_wkey))
                y = 1.0 if tr["wkey_alt"] else 0.0
                g = y - p
                self.w_wkey += lr * g * feat
                self.b_wkey = np.array(float(self.b_wkey) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "wplace":
                p = sigmoid(float(feat @ self.w_wplace + self.b_wplace))
                y = 1.0 if tr["wplace_alt"] else 0.0
                g = y - p
                self.w_wplace += lr * g * feat
                self.b_wplace = np.array(float(self.b_wplace) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "wsel":
                p = sigmoid(float(feat @ self.w_wsel + self.b_wsel))
                y = 1.0 if tr["wsel_alt"] else 0.0
                g = y - p
                self.w_wsel += lr * g * feat
                self.b_wsel = np.array(float(self.b_wsel) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "wcomp":
                p = sigmoid(float(feat @ self.w_wcomp + self.b_wcomp))
                y = 1.0 if tr["wcomp_alt"] else 0.0
                g = y - p
                self.w_wcomp += lr * g * feat
                self.b_wcomp = np.array(float(self.b_wcomp) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "pick":
                p = sigmoid(float(feat @ self.w_pick + self.b_pick))
                y = 1.0 if tr["one"] else 0.0
                g = y - p
                self.w_pick += lr * g * feat
                self.b_pick = np.array(float(self.b_pick) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "schema":
                p = sigmoid(float(feat @ self.w_schema + self.b_schema))
                y = 1.0 if tr["complete"] else 0.0
                g = y - p
                self.w_schema += lr * g * feat
                self.b_schema = np.array(float(self.b_schema) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "use":
                p = sigmoid(float(feat @ self.w_use + self.b_use))
                y = 1.0 if tr["use"] else 0.0
                g = y - p
                self.w_use += lr * g * feat
                self.b_use = np.array(float(self.b_use) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "write":
                p = sigmoid(float(feat @ self.w_write + self.b_write))
                y = 1.0 if tr["write"] else 0.0
                g = y - p
                self.w_write += lr * g * feat
                self.b_write = np.array(float(self.b_write) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") == "retrieve":
                p = sigmoid(float(feat @ self.w_retrieve + self.b_retrieve))
                y = 1.0 if tr["select"] else 0.0
                g = y - p
                self.w_retrieve += lr * g * feat
                self.b_retrieve = np.array(float(self.b_retrieve) + lr * g, dtype=np.float64)
                self.n_updates += 1
                continue
            if tr.get("kind") != "collect":
                raise ValueError(f"unknown policy trace kind: {tr.get('kind')}")
            probs = softmax(feat @ self.W_collect + self.b_collect)
            c = int(tr["collect_idx"])
            grad = -probs
            grad[c] += 1.0
            self.W_collect += lr * np.outer(feat, grad)
            self.b_collect += lr * grad
            p = sigmoid(float(feat @ self.w_apply + self.b_apply))
            y = 1.0 if tr["apply"] else 0.0
            g = y - p
            self.w_apply += lr * g * feat
            self.b_apply = np.array(float(self.b_apply) + lr * g, dtype=np.float64)
            self.n_updates += 1
