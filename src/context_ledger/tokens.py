"""Versioned token estimators, and the `tokenizer_set` event that puts one in force.

Every budget, backstop and invariant computation counts tokens with the estimator in force at
that point of the ledger, so the meaning of an old ledger never changes:

    bytes4-v0  ceil(UTF-8 bytes / 4), whatever the text. The default for any ledger without a
               `tokenizer_set` event (every M1 and M2 ledger). Policies name it "approx-bytes4".
    calib-v1   per-class affine fit against the API's count_tokens:
               tokens = ceil(a_c * bytes + b_c) for class c in python, prose, json, other.
               The fit (a_c) is the point estimate used for reporting; the SpendGuard
               fallback (and M3 ledgers) use the conservative upper envelope a_c_hi, so
               `count` is ceil(a_c_hi * bytes + b_c) (and at least 1 for non-empty text). The
               corpus sizes C-rho are counted with it, one file per text.
    calib-v2   (M4) the same calibration applied to the pieces of one prompt, the way the
               window W and fold savings need it: tokens = ceil(a_c * bytes), the point fit
               with NO per-text intercept, for `count` and `estimate` alike.
    calib-v3   (BUILD-V2) calib-v2 for content plus the envelope model (envelope.py): the
               structural tokens of the request, its messages and its content blocks, fitted
               against count_tokens. The window measure of every arm then is the whole prompt
               minus the arm's tool definitions (render.rendered_window_tokens for L,
               envelope.prompt_tokens for B3 and B4).

Why calib-v2. calib-v1's fit models one text sent as one user message; its intercept b_c is a
per-message term (and absorbs fitting noise: the 2026-10-31 live fit has b = 190 for python).
A window is many pieces in ONE prompt, so summing calib-v1 counts per segment charges b_c once
per piece and its envelope slope a_c_hi on every byte. In the 2026-10-31 W=6000 smoke run that
measured the ledger arm's window at about 2.4 times its count_tokens size (the compaction arm,
with fewer and larger pieces, about 1.6 times), and predicted fold savings of 2034, 2341 and
2402 tokens where count_tokens showed 448, 571 and 1675. The marginal tokens of a piece inside
a prompt are a_c * bytes: with the point fit the same three folds predict 436, 556 and 1399.
The point fit is used because W and fold savings are predictions of real tokens that must mean
the same thing in every arm; the envelope stays where a bound is needed (the SpendGuard).
Under calib-v2 the structural tokens of tool blocks (tool_result wrappers, ids) are not
modelled in any arm, so savings of folds over many tool blocks are under-predicted; calib-v3
(BUILD-V2) models them with the fitted envelope (envelope.py).

The class of a segment is chosen deterministically from its kind and its source's media type:
tool results and assistant content blocks are json; spans of .py sources are python; spans
of .rst, .txt and .md sources are prose; everything else is other. Coefficients are decimal
strings parsed exactly to Fractions, so nothing here is a float. Estimators are pure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .ids import canonical_json, sha256_hex

BYTES4 = "bytes4-v0"
CALIB = "calib-v1"
CALIB2 = "calib-v2"
CALIB3 = "calib-v3"
CLASSES = ("python", "prose", "json", "other")
CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "calibration"
_DECIMAL = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EstimatorError(ValueError):
    """A tokenizer_set payload or calibration file is malformed."""


class CalibrationMissing(EstimatorError):
    """No usable calib-v1 calibration exists for the model (none has been produced yet)."""


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def decimal_fraction(text: object, what: str) -> Fraction:
    """Parse a plain decimal string ("0.3125", "-4", "7.5") exactly; EstimatorError otherwise."""
    if not isinstance(text, str) or _DECIMAL.match(text) is None:
        raise EstimatorError(f"{what} must be a decimal string, got {text!r}")
    return Fraction(text)


def decimal_str(value: Fraction, places: int = 6) -> str:
    """`value` rounded half-up to `places` decimals as a plain decimal string (no exponent)."""
    scaled = Fraction(value) * 10**places
    n = (abs(scaled.numerator) * 2 + scaled.denominator) // (2 * scaled.denominator)
    sign = "-" if scaled < 0 and n else ""
    whole, frac = divmod(n, 10**places)
    text = f"{sign}{whole}.{frac:0{places}d}".rstrip("0").rstrip(".") if places else f"{sign}{whole}"
    return text or "0"


@dataclass(frozen=True)
class Bytes4Estimator:
    """bytes4-v0: ceil(UTF-8 bytes / 4) for every class."""

    estimator: str = BYTES4
    name: str = "approx-bytes4"  # the M2 policy token_counter name of the same rule

    def count_bytes(self, n: int, cls: str = "other") -> int:
        return (n + 3) // 4

    def estimate_bytes(self, n: int, cls: str = "other") -> int:
        return (n + 3) // 4

    def count(self, text: str, cls: str = "other") -> int:
        return self.count_bytes(len(text.encode("utf-8")), cls)

    def estimate(self, text: str, cls: str = "other") -> int:
        return self.count(text, cls)

    def payload(self) -> dict:
        return {"estimator": BYTES4, "version": 0, "model": None, "coefficients": {}, "calibration_sha256": None}


@dataclass(frozen=True)
class ClassFit:
    """tokens = ceil(a * bytes + b); `a_hi` is the 95th-percentile envelope of (tokens - b) / bytes."""

    a: Fraction
    b: Fraction
    a_hi: Fraction

    def to_dict(self, places: int = 6) -> dict:
        return {"a": decimal_str(self.a, places), "b": decimal_str(self.b, places), "a_hi": decimal_str(self.a_hi, places)}

    @classmethod
    def from_dict(cls, d: object, what: str) -> ClassFit:
        if not isinstance(d, dict) or set(d) != {"a", "b", "a_hi"}:
            raise EstimatorError(f"{what} must have exactly the keys a, b, a_hi")
        fit = cls(*(decimal_fraction(d[k], f"{what}.{k}") for k in ("a", "b", "a_hi")))
        if fit.a <= 0 or fit.a_hi < fit.a:
            raise EstimatorError(f"{what}: need 0 < a <= a_hi")
        return fit


@dataclass(frozen=True)
class CalibEstimator:
    """calib-v1 for one model. `coefficients` maps every class to its ClassFit."""

    model: str
    coefficients: dict[str, ClassFit] = field(hash=False)
    calibration_sha256: str
    estimator: str = CALIB
    name: str = CALIB

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise EstimatorError("calib-v1 needs a model")
        if set(self.coefficients) != set(CLASSES):
            raise EstimatorError(f"calib-v1 needs coefficients for exactly {list(CLASSES)}")
        if not isinstance(self.calibration_sha256, str) or _HEX64.match(self.calibration_sha256) is None:
            raise EstimatorError("calibration_sha256 must be 64 lowercase hex characters")

    def _apply(self, n: int, cls: str, a: Fraction) -> int:
        if n <= 0:
            return 0
        fit = self.coefficients[cls if cls in self.coefficients else "other"]
        return max(1, _ceil(a * n + fit.b))

    def count_bytes(self, n: int, cls: str = "other") -> int:
        """Conservative (upper-envelope) count: what budgets and the backstop use."""
        return self._apply(n, cls, self.coefficients[cls if cls in self.coefficients else "other"].a_hi)

    def estimate_bytes(self, n: int, cls: str = "other") -> int:
        """Point estimate from the least-squares fit: what reports use."""
        return self._apply(n, cls, self.coefficients[cls if cls in self.coefficients else "other"].a)

    def count(self, text: str, cls: str = "other") -> int:
        return self.count_bytes(len(text.encode("utf-8")), cls)

    def estimate(self, text: str, cls: str = "other") -> int:
        return self.estimate_bytes(len(text.encode("utf-8")), cls)

    def payload(self) -> dict:
        return {
            "estimator": CALIB,
            "version": 1,
            "model": self.model,
            "coefficients": {c: self.coefficients[c].to_dict() for c in CLASSES},
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True)
class MarginalCalibEstimator(CalibEstimator):
    """calib-v2: a calib-v1 calibration applied per piece of one prompt, ceil(a_c * bytes) (the
    point slope, no per-text intercept, at least 1 for non-empty text), for counting and
    estimating alike. See the module docstring."""

    estimator: str = CALIB2
    name: str = CALIB2

    def count_bytes(self, n: int, cls: str = "other") -> int:
        return 0 if n <= 0 else max(1, _ceil(self.coefficients[cls if cls in self.coefficients else "other"].a * n))

    estimate_bytes = count_bytes

    def payload(self) -> dict:
        return {**super().payload(), "estimator": CALIB2, "version": 2}


@dataclass(frozen=True)
class EnvelopeCalibEstimator(MarginalCalibEstimator):
    """calib-v3: calib-v2 content counts plus `envelope` (an envelope.Envelope), the structural
    tokens of a request; `envelope_sha256` is the hash of the envelope calibration file. The
    content count is calib-v2's; the window measures add the envelope (see envelope.py)."""

    envelope: object = None
    envelope_sha256: str = ""
    estimator: str = CALIB3
    name: str = CALIB3

    def __post_init__(self) -> None:
        super().__post_init__()
        from .envelope import Envelope

        if not isinstance(self.envelope, Envelope):
            raise EstimatorError("calib-v3 needs an envelope")
        if not isinstance(self.envelope_sha256, str) or _HEX64.match(self.envelope_sha256) is None:
            raise EstimatorError("envelope_sha256 must be 64 lowercase hex characters")

    def payload(self) -> dict:
        return {**super().payload(), "estimator": CALIB3, "version": 3, "envelope": self.envelope.to_dict(), "envelope_sha256": self.envelope_sha256}


Estimator = Bytes4Estimator | CalibEstimator | MarginalCalibEstimator | EnvelopeCalibEstimator
BYTES4_V0 = Bytes4Estimator()


def window_estimator(estimator: Estimator, envelope=None, envelope_sha256: str | None = None) -> Estimator:
    """The estimator the arms measure the window W with: for a calib-v1 calibration, calib-v3
    with `envelope` if one is given (BUILD-V2), else calib-v2 (same coefficients and
    calibration hash); anything else unchanged."""
    if type(estimator) is CalibEstimator:
        if envelope is not None:
            return EnvelopeCalibEstimator(
                estimator.model, estimator.coefficients, estimator.calibration_sha256, envelope=envelope, envelope_sha256=envelope_sha256 or envelope.sha256()
            )
        return MarginalCalibEstimator(estimator.model, estimator.coefficients, estimator.calibration_sha256)
    return estimator


def envelope_of(estimator) -> object | None:
    """The envelope.Envelope of a calib-v3 estimator, None for any other."""
    return getattr(estimator, "envelope", None)


def from_payload(p: dict | None) -> Estimator:
    """The estimator a `tokenizer_set` payload names (None means bytes4-v0). Strict: exactly the
    keys estimator, version, model, coefficients, calibration_sha256."""
    if p is None:
        return BYTES4_V0
    keys = {"estimator", "version", "model", "coefficients", "calibration_sha256"}
    if isinstance(p, dict) and p.get("estimator") == CALIB3:
        keys |= {"envelope", "envelope_sha256"}
    if not isinstance(p, dict) or set(p) != keys:
        raise EstimatorError(f"a tokenizer_set payload has exactly the keys {sorted(keys)}")
    if p["estimator"] == BYTES4:
        if p != BYTES4_V0.payload():
            raise EstimatorError("bytes4-v0 takes version 0, no model, no coefficients and no calibration")
        return BYTES4_V0
    kinds = {(CALIB, 1): CalibEstimator, (CALIB2, 2): MarginalCalibEstimator, (CALIB3, 3): EnvelopeCalibEstimator}
    kind = kinds.get((p["estimator"], p["version"]))
    if kind is None:
        raise EstimatorError(f"unknown estimator {p['estimator']!r} version {p['version']!r}")
    if not isinstance(p["coefficients"], dict):
        raise EstimatorError("coefficients must be a dict")
    fits = {c: ClassFit.from_dict(v, f"coefficients.{c}") for c, v in p["coefficients"].items()}
    if kind is EnvelopeCalibEstimator:
        from .envelope import Envelope, EnvelopeError

        try:
            envelope = Envelope.from_dict(p["envelope"])
        except EnvelopeError as exc:
            raise EstimatorError(str(exc)) from None
        estimator = kind(p["model"], fits, p["calibration_sha256"], envelope=envelope, envelope_sha256=p["envelope_sha256"])
    else:
        estimator = kind(p["model"], fits, p["calibration_sha256"])
    if estimator.payload() != p:
        raise EstimatorError("coefficients must be written in canonical decimal form")
    return estimator


def estimator_for(state) -> Estimator:
    """The estimator in force in a replayed WindowState."""
    return getattr(state, "estimator", None) or BYTES4_V0


def source_class(uri: str, media_type: str = "") -> str:
    """python for .py sources (or text/x-python), prose for .rst, .txt and .md, else other."""
    lower = uri.lower()
    if lower.endswith(".py") or media_type == "text/x-python":
        return "python"
    if lower.endswith((".rst", ".txt", ".md")):
        return "prose"
    return "other"


def segment_class(state, segment) -> str:
    """The estimator class of a window segment (see the module docstring)."""
    if segment.content_type == "blocks" or (segment.ref_kind == "raw" and segment.tool_use_id is not None):
        return "json"
    if segment.ref_kind == "span":
        span = state.spans.get(segment.ref)
        source = state.sources.get(span.source_id) if span is not None else None
        if source is not None:
            return source_class(source.uri, source.media_type)
    return "other"


def calibration_estimator(data: dict, *, allow_fake: bool = False) -> CalibEstimator:
    """The calib-v1 estimator of a calibration file's contents; its sha256 is the canonical JSON
    hash of the whole file. A calibration made with a fake counter is refused unless
    `allow_fake` (tests and offline dry runs only)."""
    if not isinstance(data, dict) or "coefficients" not in data or "model" not in data:
        raise EstimatorError("a calibration file needs model and coefficients")
    if data.get("counter") != "count_tokens" and not allow_fake:
        raise CalibrationMissing(f"calibration was made with the {data.get('counter')!r} counter, not count_tokens")
    fits = {c: ClassFit.from_dict(data["coefficients"].get(c), f"coefficients.{c}") for c in CLASSES}
    return CalibEstimator(data["model"], fits, sha256_hex(canonical_json(data)))


def load_calibration(model: str, directory: str | Path | None = None, *, allow_fake: bool = False) -> CalibEstimator:
    """The newest calibration/<model>-<YYYYMMDD>.json as a calib-v1 estimator. Raises
    CalibrationMissing, with the path it looked in, if there is none."""
    folder = Path(directory) if directory is not None else CALIBRATION_DIR
    found = sorted(folder.glob(f"{model}-[0-9]*.json")) if folder.is_dir() else []
    if not found:
        raise CalibrationMissing(f"no calib-v1 calibration for {model} in {folder}; run `calibrate --live` first")
    return calibration_estimator(json.loads(found[-1].read_text(encoding="utf-8")), allow_fake=allow_fake)


def estimator_or_fallback(model: str, directory: str | Path | None = None) -> tuple[Estimator, str | None]:
    """calib-v1 if a calibration exists, else bytes4-v0 with an explicit warning to print."""
    try:
        return load_calibration(model, directory), None
    except CalibrationMissing as exc:
        return BYTES4_V0, f"warning: {exc}; falling back to bytes4-v0 (ceil(bytes/4)), which underestimates Claude 4.7+ token counts"


__all__ = [
    "BYTES4",
    "BYTES4_V0",
    "CALIB",
    "CALIB2",
    "CALIB3",
    "CALIBRATION_DIR",
    "CLASSES",
    "Bytes4Estimator",
    "CalibEstimator",
    "CalibrationMissing",
    "ClassFit",
    "EnvelopeCalibEstimator",
    "Estimator",
    "EstimatorError",
    "MarginalCalibEstimator",
    "calibration_estimator",
    "decimal_fraction",
    "decimal_str",
    "envelope_of",
    "estimator_for",
    "estimator_or_fallback",
    "from_payload",
    "load_calibration",
    "segment_class",
    "source_class",
    "window_estimator",
]
