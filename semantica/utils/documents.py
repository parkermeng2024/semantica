"""Helpers for reading text out of parsed-document records.

``DocumentParser`` / ``DoclingParser`` return a dict carrying ``full_text``
(see docs/reference/parse.md), while some in-repo call sites historically
expected ``content`` and others ``text``. Reading the wrong key used to fall
back to ``str(item)``, which silently fed the *repr of the dict* into
normalization and extraction (doubling the text, because parsers emit both
``text`` and ``full_text``, and leaking metadata numbers into entities).
"""

from typing import Any

# Keys that may carry a parsed document's text, in resolution order.
TEXT_KEYS = ("content", "text", "full_text")


def document_text(item: Any) -> str:
    """Return the text carried by a parsed-document record.

    Resolution order is ``content`` -> ``text`` -> ``full_text`` for dicts and
    for objects exposing those attributes. Strings pass through unchanged and
    anything unrecognised falls back to ``str(item)`` so callers keep their
    previous behaviour.
    """
    if isinstance(item, str):
        return item

    values = (
        (item.get(key) for key in TEXT_KEYS)
        if isinstance(item, dict)
        else (getattr(item, key, None) for key in TEXT_KEYS)
    )
    for value in values:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value

    return str(item)
