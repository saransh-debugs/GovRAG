"""Put real values back into the LLM's answer, but only the ones this user may see.

`reveal` is the allowlist, built per request from:
  - vault values for tokens recorded on *this request's retrieved chunks*, filtered to
    the PII types the asker's roles may unmask (vault.load_for_chunks);
  - values the asker typed in their own question (they already know them).
Any other token - forbidden type, invented by the LLM, forged in a document - becomes a
display alias like "[PERSON 1]". Aliases are numbered per answer, so a user without
PERSON access can't use token IDs to tell that two answers mention the same person.

The LLM sometimes drops the brackets ("PERSON_3f9a1c2e4b7d9a01"), so bare tokens are
handled too: a bare token ID would otherwise leak the same cross-answer link.
"""

from __future__ import annotations

import re
from collections import Counter

from govrag.pii.tokens import TOKEN_RE

_BARE = re.compile(r"(?<![\w\[])([A-Z][A-Z0-9_]{0,40})_([0-9a-f]{16})(?![\w\]])")
# Longest possible bracketed token: "[" + 41-char type + "_" + 16 hex + "]".
MAX_TOKEN_LEN = 1 + 41 + 1 + 16 + 1
# A trailing piece of text that could still grow into a bare token.
_BARE_PREFIX = re.compile(r"[A-Z][A-Z0-9_]*(?:_[0-9a-f]{0,16})?$")


class Rehydrator:
    def __init__(self, reveal: dict[str, str]) -> None:
        self._reveal = reveal
        self._alias: dict[str, str] = {}
        self._n: Counter[str] = Counter()
        self.unmasked_types: set[str] = set()
        self.masked_types: set[str] = set()
        self._buf = ""

    def _replace(self, m: re.Match[str]) -> str:
        etype, tok = m.group(1), f"{m.group(1)}_{m.group(2)}"
        value = self._reveal.get(tok)
        if value is not None:
            self.unmasked_types.add(etype)
            return value
        self.masked_types.add(etype)
        if tok not in self._alias:
            self._n[etype] += 1
            self._alias[tok] = f"[{etype} {self._n[etype]}]"
        return self._alias[tok]

    def whole(self, text: str) -> str:
        return _BARE.sub(self._replace, TOKEN_RE.sub(self._replace, text))

    # --- streaming ---------------------------------------------------------------------

    def feed(self, piece: str) -> str:
        """Returns the text that is safe to emit now. Holds back anything that might be the
        start of a token split across pieces ("[PERS" + "ON_3f9a...]"), at most
        MAX_TOKEN_LEN chars, so a stray "[" can't stall the stream."""
        self._buf += piece
        cut = len(self._buf)
        i = self._buf.rfind("[")
        if i != -1 and "]" not in self._buf[i:] and len(self._buf) - i < MAX_TOKEN_LEN:
            cut = i
        else:
            m = _BARE_PREFIX.search(self._buf)
            if m and len(self._buf) - m.start() < MAX_TOKEN_LEN:
                cut = m.start()
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return self.whole(out)

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return self.whole(out)
