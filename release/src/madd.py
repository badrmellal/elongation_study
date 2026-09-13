"""
Locate madd (held vowel) segments by CTC forced alignment against the DIACRITIZED
Quranic text, then classify each by tajwid class.

"""

from typing import List, Tuple, Dict
import numpy as np

# Arabic orthography 
FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SUKUN, SHADDA = "ْ", "ّ"
DAGGER_ALIF = "ٰ"      # ٰ
MADDA_SIGN = "ٓ"       # ٓ  (marks madd lazim / muttasil in mushaf script)
ALIF_MADDA = "آ"       # آ
ALIF, WAW, YA = "ا", "و", "ي"
HAMZAT = set("ءأإآؤئ")   # ء أ إ آ ؤ ئ
DIACRITICS = set("ًٌٍَُِّْٰٓـ")

# tajwid length in "counts" (harakat). 1 count = one short-vowel duration.
CLASS_COUNTS = {"tabii": 2, "silah": 2, "aarid": 4, "munfasil": 4, "muttasil": 5, "lazim": 6}


def find_madd(text: str) -> List[Dict]:
    """Return madd occurrences as {char_idx, letter, cls, counts}.

    char_idx indexes into text exactly as given (diacritics included), so it can
    be mapped through a forced alignment computed on the same string.
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        is_madd = False
        if ch == ALIF and prev == FATHA:
            is_madd = True
        elif ch == WAW and prev == DAMMA:
            is_madd = True
        elif ch == YA and prev == KASRA:
            is_madd = True
        elif ch in (DAGGER_ALIF, ALIF_MADDA):
            is_madd = True
        if not is_madd:
            continue

        # what follows decides the tajwid class, and therefore the expected length
        j = i + 1
        while j < n and text[j] in (MADDA_SIGN,):
            j += 1
        nxt = text[j] if j < n else ""
        nxt2 = text[j + 1] if j + 1 < n else ""

        if MADDA_SIGN in text[i + 1:i + 3]:
            cls = "lazim" if (nxt in (SHADDA, SUKUN) or nxt2 in (SHADDA, SUKUN)) else "muttasil"
            
        elif nxt in HAMZAT or nxt2 in HAMZAT:
            cls = "muttasil"
        elif nxt == " " and _first_letter_after_space(text, j) in HAMZAT:
            cls = "munfasil"
        elif nxt in (SHADDA, SUKUN):
            cls = "lazim"
        elif j >= n - 1:
            cls = "aarid"          # verse-final: lengthened at the pause
        else:
            cls = "tabii"
        out.append({"char_idx": i, "letter": ch, "cls": cls, "counts": CLASS_COUNTS[cls]})
    return out


def _first_letter_after_space(text: str, j: int) -> str:
    k = j + 1
    while k < len(text) and (text[k] == " " or text[k] in DIACRITICS):
        k += 1
    return text[k] if k < len(text) else ""


# CTC forced alignment (own implementation, no torchaudio)

def forced_align(logp: np.ndarray, target_ids: List[int], blank: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Viterbi alignment of `target_ids` to frames.

    logp: (T, V) log-probabilities. Returns (path, frame_to_tok) where
    frame_to_tok[t] is the index into target_ids, or -1 for blank.

    Standard CTC trellis over z = [b, y1, b, y2, ..., yU, b]; transitions are
    stay, advance one, and advance two (the last only when the target is not a
    blank and differs from the token two back, i.e. no blank needed between
    distinct symbols).
    """
    T = logp.shape[0]
    U = len(target_ids)
    S = 2 * U + 1
    z = np.full(S, blank, dtype=int)
    z[1::2] = target_ids

    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, z[0]]
    if S > 1:
        dp[0, 1] = logp[0, z[1]]

    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        adv1 = np.concatenate(([NEG], prev[:-1]))
        adv2 = np.concatenate(([NEG, NEG], prev[:-2]))
        allowed2 = np.zeros(S, dtype=bool)
        s_idx = np.arange(S)
        ok = (s_idx >= 2) & (z != blank)
        ok[2:] &= z[2:] != z[:-2]
        allowed2[ok] = True
        adv2 = np.where(allowed2, adv2, NEG)

        cand = np.stack([stay, adv1, adv2])          # (3, S)
        best = np.argmax(cand, axis=0)
        dp[t] = cand[best, s_idx] + logp[t, z]
        bp[t] = best

    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    frame_to_tok = np.where(z[path] == blank, -1, (path - 1) // 2)
    return path, frame_to_tok


def madd_intervals(text: str, target_ids: List[int], id_of_char: List[int],
                   frame_to_tok: np.ndarray, frame_s: float,
                   min_counts: int = 4) -> List[Dict]:
    """Map madd char positions to time intervals via the alignment.

    id_of_char[i] gives the position in target_ids of text character i (or -1 if
    the character was dropped when building the target, e.g. spaces).
    min_counts=4 keeps only the genuinely long classes (muttasil/munfasil/aarid/
    lazim), which are the ones held for a perceptible duration in mujawwad.
    """
    out = []
    for m in find_madd(text):
        if m["counts"] < min_counts:
            continue
        tok = id_of_char[m["char_idx"]] if m["char_idx"] < len(id_of_char) else -1
        if tok < 0:
            continue
        fr = np.flatnonzero(frame_to_tok == tok)
        if len(fr) == 0:
            continue
        # CTC alignments are PEAKY: the model emits one spike per token and blanks
        # elsewhere, so fr is typically a single frame and its width is meaningless
        # as a duration. The phone's territory is the span from its own spike to
        # the NEXT token's spike; the intervening blanks belong to the held vowel.
        nxt = np.flatnonzero(frame_to_tok == tok + 1)
        start_f = int(fr[0])
        end_f = int(nxt[0]) if len(nxt) else int(fr[-1] + 1)
        if end_f <= start_f:
            end_f = start_f + 1
        out.append({**m,
                    "start": float(start_f * frame_s),
                    "end": float(end_f * frame_s),
                    "dur_ms": float((end_f - start_f) * frame_s * 1000),
                    "spike_frames": int(len(fr))})
    return out
