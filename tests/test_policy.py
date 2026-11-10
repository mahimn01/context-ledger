from __future__ import annotations

import json
from fractions import Fraction

import pytest

from context_ledger.fold import BackstopPolicy, BackstopPolicyV1, ReReadEstimator, policy_from_dict
from context_ledger.ids import canonical_json
from context_ledger.store import Ledger


def test_v1_round_trips_through_canonical_json():
    policy = BackstopPolicyV1(budget_tokens=3000, ttl="1h", estimator=ReReadEstimator("horizon", 12), tail_keep=3)
    d = policy.to_dict()
    assert d["id"] == "backstop-v1" and d["hard_bp"] == 8500 and d["estimator"] == {"kind": "horizon", "planned_calls": 12}

    def leaves(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from leaves(v)
        else:
            yield obj

    assert all(isinstance(v, (int, str)) and not isinstance(v, float) for v in leaves(d))
    decoded = policy_from_dict(json.loads(canonical_json(d)))
    assert decoded == policy and isinstance(decoded, BackstopPolicyV1)
    assert BackstopPolicy.from_dict(d) == BackstopPolicyV1.from_dict(d) == policy


@pytest.mark.parametrize(
    "changes",
    [
        {"target_bp": 0},
        {"target_bp": 7000},  # target must be below soft
        {"soft_bp": 9000},  # soft above hard
        {"hard_bp": 9500},  # hard must be below mandatory
        {"mandatory_bp": 10001},
        {"half_life_calls": 0},
        {"q0_bp": 10001},
        {"tail_keep": -1},
        {"ttl": "10m"},
        {"token_counter": "tiktoken"},
        {"model": ""},
        {"budget_tokens": 0},
    ],
)
def test_v1_validation(changes):
    with pytest.raises((ValueError, TypeError)):
        BackstopPolicyV1(budget_tokens=1000, **changes)


@pytest.mark.parametrize("value", [0.85, True, "8500", None])
def test_v1_ratios_must_be_plain_ints(value):
    with pytest.raises((ValueError, TypeError)):
        BackstopPolicyV1(budget_tokens=1000, hard_bp=value)


def test_v1_from_dict_is_strict():
    d = BackstopPolicyV1(budget_tokens=1000).to_dict()
    with pytest.raises(ValueError):
        policy_from_dict({**d, "surprise": 1})
    with pytest.raises(ValueError):
        policy_from_dict({k: v for k, v in d.items() if k != "tau_tokens"})
    with pytest.raises((ValueError, TypeError)):
        policy_from_dict({**d, "soft_bp": 7000.0})
    with pytest.raises(ValueError):
        policy_from_dict({**d, "id": "backstop-v9"})
    with pytest.raises(ValueError):
        policy_from_dict({**d, "estimator": {"kind": "lindy", "n": 3}})


def test_injected_float_in_policy_payload_is_rejected_at_append():
    policy = BackstopPolicyV1(budget_tokens=1000).to_dict()
    with Ledger(":memory:") as ledger:
        with pytest.raises(TypeError):
            ledger.append("fold", {"fold": {"policy": {**policy, "hard_bp": 8500.0}}})
        assert len(ledger) == 0


def test_v0_decoder_parses_m1_strings_exactly():
    d = {"id": "backstop-v0", "budget_tokens": 600, "hard": "0.85", "target": "0.5", "tail_keep": 2, "cold_m": 4}
    policy = policy_from_dict(d)
    assert isinstance(policy, BackstopPolicy) and policy.to_dict() == d
    assert Fraction(d["hard"]) == Fraction(17, 20) and policy.hard == 0.85
    for bad in ("nan", "inf", "0.8.5", ""):
        with pytest.raises((ValueError, ZeroDivisionError)):
            policy_from_dict({**d, "hard": bad})
    with pytest.raises(TypeError):
        policy_from_dict({**d, "hard": 0.85})
    with pytest.raises(ValueError):
        policy_from_dict({**d, "extra": 1})


def test_estimators():
    assert [ReReadEstimator("fixed", 3).rhat(n) for n in (0, 9)] == [3, 3]
    assert [ReReadEstimator("lindy", 5).rhat(n) for n in (0, 1, 4, 9)] == [1, 1, 4, 5]
    assert [ReReadEstimator("horizon", 6).rhat(n) for n in (0, 4, 6, 10)] == [6, 2, 1, 1]
    assert ReReadEstimator() == ReReadEstimator("lindy", 64)
    for kind, value in [("fixed", 0), ("lindy", 0), ("horizon", -1), ("magic", 1)]:
        with pytest.raises(ValueError):
            ReReadEstimator(kind, value)
