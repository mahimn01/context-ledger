"""Short per-ledger aliases: "@" + Crockford base32 of a dense counter (@0, @1, ..., @Z, @10, ...).

An alias names a canonical ref: a span id, "fold:<fold_id>#<i>" (a fold sentence),
"fold:<fold_id>" (a whole fold segment) or "raw:<segment_id>". Aliases are introduced by
`alias_assigned` events and by `FoldRecord.aliases`, in ledger order; the k-th introduction
is "@" + b32(k). They are never reused or reassigned, so replaying the ledger rebuilds the
identical map.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ALIAS_RE = re.compile(r"^@[0-9A-HJKMNP-TV-Z]+$")


class UnknownAlias(KeyError):
    """A token looks like an alias but no alias with that name has been introduced."""

    def __init__(self, alias: str) -> None:
        super().__init__(alias)
        self.alias = alias

    def __str__(self) -> str:
        return f"unknown alias {self.alias}"


def b32(n: int) -> str:
    """Crockford base32 of n >= 0, uppercase, no padding: 0 -> "0", 31 -> "Z", 32 -> "10"."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"b32 needs an int >= 0, got {n!r}")
    digits = ""
    while True:
        n, r = divmod(n, 32)
        digits = CROCKFORD[r] + digits
        if n == 0:
            return digits


def alias_for(n: int) -> str:
    """The alias of the n-th introduction (0-based)."""
    return "@" + b32(n)


def is_alias(token: str) -> bool:
    return isinstance(token, str) and ALIAS_RE.match(token) is not None


class AliasMap:
    """Bidirectional alias <-> canonical ref map, built during replay.

    `introduce` never raises on a bad introduction (a reused alias or an already-aliased
    target); it counts it and keeps the first mapping, so replay of a tampered ledger goes on
    and the checker (P13) reports the problem."""

    def __init__(self) -> None:
        self._target: dict[str, str] = {}
        self._alias: dict[str, str] = {}
        self.count = 0

    def introduce(self, alias: str, target: str) -> bool:
        """Count one introduction; map it if neither side is taken. Returns whether it mapped."""
        self.count += 1
        if alias in self._target or target in self._alias:
            return False
        self._target[alias] = target
        self._alias[target] = alias
        return True

    def next_alias(self) -> str:
        return alias_for(self.count)

    def alias_of(self, target: str) -> str | None:
        return self._alias.get(target)

    def has_alias(self, alias: str) -> bool:
        return alias in self._target

    def target_of(self, alias: str) -> str:
        """The canonical ref an alias names; UnknownAlias if it was never introduced."""
        try:
            return self._target[alias]
        except KeyError:
            raise UnknownAlias(alias) from None

    def resolve(self, token: str) -> str:
        """Canonical ref for an alias or a canonical ref (returned unchanged). UnknownAlias if
        `token` is alias-shaped and unknown."""
        return self.target_of(token) if is_alias(token) else token

    def items(self) -> Iterator[tuple[str, str]]:
        """(alias, target) pairs that took effect, in introduction order."""
        return iter(self._target.items())

    def copy(self) -> AliasMap:
        out = AliasMap()
        out._target, out._alias, out.count = dict(self._target), dict(self._alias), self.count
        return out

    def __len__(self) -> int:
        return len(self._target)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AliasMap) and (self.count, self._target) == (other.count, other._target)

    def __repr__(self) -> str:
        return f"AliasMap(count={self.count}, {self._target!r})"
