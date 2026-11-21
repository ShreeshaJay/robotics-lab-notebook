"""Task-string hygiene: empty strings, whitespace, duplicates after normalisation, near-duplicates, typo candidates."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

_WORD = re.compile(r"[a-z]+")
_spell = None


def _speller():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker
        _spell = SpellChecker(language="en", distance=1)
    return _spell


def normalize(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def task_hygiene(tasks: list[str], episodes_per_task: dict[str, int] | None = None, max_examples: int = 8) -> dict:
    episodes_per_task = episodes_per_task or {}
    empty = [t for t in tasks if not t.strip()]
    ws = [t for t in tasks if t.strip() and (t != t.strip() or "  " in t)]
    ctrl = [t for t in tasks if any(ord(c) < 32 for c in t)]
    groups = defaultdict(list)
    for t in tasks:
        if t.strip():
            groups[normalize(t)].append(t)
    exact_dups = [v for v in groups.values() if len(v) > 1]
    # typo candidates: words in no English word list, one edit from a word that is in the list
    vocab = Counter(w for t in tasks for w in set(_WORD.findall(t.lower())))
    sp = _speller()
    unknown = sp.unknown([w for w in vocab if len(w) >= 4])
    typos = {}
    for w in sorted(unknown):
        c = sp.correction(w)
        if c and c != w and (c in vocab or vocab[w] == 1):
            typos[w] = c
    # near duplicates: same key after stemming and typo correction, but different normalised strings
    near = defaultdict(set)
    for key in groups:
        k2 = " ".join(stem(typos.get(w, w)) for w in key.split())
        near[k2].add(key)
    near_dups = [sorted(v) for v in near.values() if len(v) > 1]
    n_eps = sum(episodes_per_task.values()) if episodes_per_task else None
    eps_empty = sum(episodes_per_task.get(t, 0) for t in empty) if episodes_per_task else None
    typo_tasks = [t for t in tasks if any(w in typos for w in _WORD.findall(t.lower()))]
    return {
        "n_tasks": len(tasks),
        "n_empty": len(empty), "episodes_with_empty_task": eps_empty,
        "episodes_with_empty_task_frac": (eps_empty / n_eps) if n_eps else None,
        "n_whitespace_issues": len(ws), "whitespace_examples": ws[:max_examples],
        "n_control_chars": len(ctrl),
        "n_duplicate_groups_after_normalisation": len(exact_dups), "duplicate_examples": exact_dups[:max_examples],
        "n_near_duplicate_groups": len(near_dups), "near_duplicate_examples": near_dups[:max_examples],
        "n_typo_words": len(typos), "typo_examples": dict(list(typos.items())[:max_examples * 2]),
        "n_tasks_with_typo_candidates": len(typo_tasks),
        "word_count_median": float(sorted(len(t.split()) for t in tasks)[len(tasks) // 2]) if tasks else 0.0,
    }
