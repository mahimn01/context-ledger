"""The live path to the real Anthropic API, and everything that keeps it safe.

A live call needs all four of: the `--live` flag; an environment that is not Claude Code and
does not make the SDK log headers (gate.py); a SpendGuard able to persist its hash-chained
spend ledger (spend.py); and a KeyProvider that returns the key (keys.py). The key lives only
in a `Secret` inside the process; `Secret.reveal()` is called in exactly one place, the live
factory (factory.py). Every arm receives a GuardedClient (guarded.py), never a raw client.
"""
