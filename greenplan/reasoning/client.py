"""Reasoning models.

* OpenRouterModel — DeepSeek via OpenRouter's OpenAI-compatible
  chat-completions endpoint. Inference ONLY: nothing here trains or
  fine-tunes the model; all "learning" happens in the prompt via the memory
  store. Key comes from the OPENROUTER_API_KEY env var, never hard-coded.
* MockModel — a fully offline stand-in (used by --mock) that fills the same
  role numerically: trend + seasonal extrapolation, bias-corrected using the
  retrieved memory records. It exercises the exact same pipeline, including
  genuine use of memory, at zero cost.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

import numpy as np
import requests

from ..config import ModelCfg
from ..features.soil import species_soil_ok
from ..features.trends import METRIC_BOUNDS as BOUNDS
from ..features.trends import METRICS, trend_seasonal_estimate
from . import prompts
from .species import SPECIES_KB, kb_markdown_table, validate_selection

log = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    pass


# Provider registry: env var holding the API key + default base URL. All of
# these speak the OpenAI chat-completions protocol, so one client serves them.
PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "nvidia": {  # NVIDIA NIM — https://integrate.api.nvidia.com/v1
        "key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
}


def extract_json(text: str) -> dict[str, Any]:
    """Strict-JSON parsing that tolerates the usual LLM wrapping: reasoning
    tags, code fences, and prose around the object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in model reply")


def _require_number(obj: dict[str, Any], key: str) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"key '{key}' must be a number, got {v!r}")
    return float(v)


class OpenRouterClient:
    """Thin chat-completions client: retries with backoff, timeout, and
    strict-JSON extraction with repair re-asks."""

    def __init__(self, cfg: ModelCfg) -> None:
        provider = PROVIDERS.get(cfg.provider)
        if provider is None:
            raise OpenRouterError(
                f"unknown model.provider {cfg.provider!r}; "
                f"expected one of {sorted(PROVIDERS)} or 'mock'"
            )
        key_env = provider["key_env"]
        key = os.environ.get(key_env)
        if not key:
            raise OpenRouterError(
                f"{key_env} is not set. Export your {cfg.provider} API key, "
                "or run with --mock (no key, no cost)."
            )
        self.cfg = cfg
        self._provider = cfg.provider
        self._key = key
        self._session = requests.Session()
        # base_url from config wins if given; otherwise the provider default.
        base = cfg.base_url or provider["base_url"]
        self._url = base.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if self._provider == "openrouter":  # OpenRouter-only ranking headers
            headers["HTTP-Referer"] = "https://github.com/local/greengrid"
            headers["X-Title"] = "GreenGrid"
        return headers

    @staticmethod
    def _read_stream(resp: requests.Response) -> str:
        """Assemble assistant text from an OpenAI-style SSE stream. The read
        timeout applies *per chunk*, so a slow model that keeps emitting tokens
        won't trip a single-shot timeout the way a non-streamed call does."""
        parts: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw[6:] if raw.startswith("data: ") else raw
            if line.strip() == "[DONE]":
                break
            try:
                delta = json.loads(line)["choices"][0]["delta"]
            except (KeyError, IndexError, ValueError, TypeError):
                continue  # keep-alive / non-delta line
            piece = delta.get("content")
            if piece:
                parts.append(piece)
        return "".join(parts)

    def _post(self, payload: dict[str, Any]) -> str:
        """POST with retry/backoff and return the assistant message text.
        Streams when cfg.stream is set (needed for slow reasoning models)."""
        stream = getattr(self.cfg, "stream", False)
        if stream:
            payload = {**payload, "stream": True}
        delay = 1.5
        last = "no attempt made"
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._session.post(
                    self._url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_s,
                    stream=stream,
                )
            except requests.RequestException as exc:
                last = f"network error: {exc}"
                log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 200:
                if stream:
                    try:
                        return self._read_stream(resp)
                    except requests.RequestException as exc:
                        last = f"stream interrupted: {exc}"
                        log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                        time.sleep(delay)
                        delay *= 2
                        continue
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as exc:
                    raise OpenRouterError(f"unexpected {self._provider} response shape: {exc}")
            if resp.status_code == 400 and "response_format" in resp.text and "response_format" in payload:
                log.warning("model rejected response_format; retrying without it")
                payload = {k: v for k, v in payload.items() if k != "response_format"}
                continue
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (408, 429, 500, 502, 503, 524):
                log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                time.sleep(delay)
                delay *= 2
                continue
            raise OpenRouterError(f"{self._provider} request failed: {last}")
        raise OpenRouterError(f"{self._provider} retries exhausted; last error: {last}")

    def chat_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = "no attempt made"
        for _ in range(self.cfg.json_repair_attempts):
            text = self._post(
                {
                    "model": self.cfg.name,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "response_format": {"type": "json_object"},
                }
            )
            try:
                obj = extract_json(text)
                if validate:
                    validate(obj)
                return obj
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                log.warning("invalid JSON from model (%s); re-asking", last_err)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {last_err}. "
                            "Reply again with STRICT JSON only, exactly matching the "
                            "schema in the system message. No prose, no code fences."
                        ),
                    }
                )
        raise OpenRouterError(f"model never produced valid JSON: {last_err}")


class OpenRouterModel:
    """DeepSeek-backed implementation of the reasoning interface."""

    def __init__(self, cfg: ModelCfg) -> None:
        self.client = OpenRouterClient(cfg)

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        def _validate(obj: dict[str, Any]) -> None:
            for m in METRICS:
                _require_number(obj, m)

        obj = self.client.chat_json(prompts.PREDICT_SYSTEM, prompts.predict_user(ctx), _validate)
        return {
            m: float(np.clip(_require_number(obj, m), *BOUNDS[m])) for m in METRICS
        }

    def lesson(self, ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        def _validate(obj: dict[str, Any]) -> None:
            if not isinstance(obj.get("lesson"), str) or not obj["lesson"].strip():
                raise ValueError("key 'lesson' must be a non-empty string")

        obj = self.client.chat_json(
            prompts.LESSON_SYSTEM, prompts.lesson_user(ctx, pred, actual), _validate
        )
        return obj["lesson"].strip()

    def recommend(
        self, ranked_rows: list[dict[str, Any]], lessons: list[str]
    ) -> list[dict[str, Any]]:
        given_order = [r["zone"] for r in ranked_rows]

        def _validate(obj: dict[str, Any]) -> None:
            zones = obj.get("zones")
            if not isinstance(zones, list) or not zones:
                raise ValueError("key 'zones' must be a non-empty list")
            for z in zones:
                if z.get("zone") not in given_order:
                    raise ValueError(f"unknown zone id {z.get('zone')!r}")
                picked = validate_selection(list(z.get("species") or []))
                if len(picked) < 3:
                    raise ValueError(
                        f"zone {z.get('zone')}: need 3-5 species chosen EXACTLY from the "
                        f"table's `common` column; valid picks were {picked}"
                    )

        obj = self.client.chat_json(
            prompts.RECOMMEND_SYSTEM,
            prompts.recommend_user(ranked_rows, lessons, kb_markdown_table()),
            _validate,
        )
        by_zone = {z["zone"]: z for z in obj["zones"]}
        out = []
        for row in ranked_rows:  # numeric order is authoritative, never the model's
            z = by_zone.get(row["zone"], {})
            out.append(
                {
                    "zone": row["zone"],
                    "justification": str(z.get("justification", "")).strip()
                    or "(model gave no justification)",
                    "species": validate_selection(list(z.get("species") or [])),
                }
            )
        return out

    def project_note(self, city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
        def _validate(obj: dict[str, Any]) -> None:
            if not isinstance(obj.get("note"), str) or not obj["note"].strip():
                raise ValueError("key 'note' must be a non-empty string")

        obj = self.client.chat_json(
            prompts.PROJECT_SYSTEM, prompts.project_user(city, n_months, sample_rows), _validate
        )
        return obj["note"].strip()


# ---------------------------------------------------------------------------
# Offline mock model
# ---------------------------------------------------------------------------


class MockModel:
    """Deterministic offline stand-in for DeepSeek.

    Prediction = Theil-Sen trend + detrended seasonal profile, then a genuine
    memory correction: the mean signed error of the retrieved past cases is
    subtracted, so as the memory store grows the mock really does get better —
    a numeric analog of the in-context learning the real model does in text.
    """

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        target = ctx["target_month"]
        target_cal = ctx["season"]
        for m in METRICS:
            pts = [(h["month"], h["cal_month"], h[m]) for h in ctx["history"] if h[m] is not None]
            est = trend_seasonal_estimate(pts, target, target_cal)
            if est is None:
                out[m] = float(np.clip(ctx["inputs"].get(f"{m}_latest", 0.0), *BOUNDS[m]))
                continue
            pred = est
            # memory correction: learn from own past signed errors, weighting
            # records by season proximity (errors are strongly seasonal) and
            # doubling down on same-zone history
            errs, wts = [], []
            for r in ctx.get("memories", []):
                e = r.get("signed_error", {}).get(m)
                if e is None:
                    continue
                r_season = r.get("inputs", {}).get("season", target_cal)
                w = 1.0 + np.cos(2 * np.pi * (r_season - target_cal) / 12.0)
                if r.get("zone") == ctx["zone"]:
                    w *= 1.5
                errs.append(e)
                wts.append(w)
            if errs and sum(wts) > 1e-9:
                pred -= 0.6 * float(np.average(errs, weights=wts))
            out[m] = float(np.clip(pred, *BOUNDS[m]))
        return out

    def lesson(self, ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        stats = ctx.get("stats", {})
        worst, worst_z = "aqi", -1.0
        for m in METRICS:
            std = stats.get(m, {}).get("std", 1.0) or 1.0
            z = abs(pred[m] - actual[m]) / std
            if z > worst_z:
                worst, worst_z = m, z
        direction = "Overestimated" if pred[worst] > actual[worst] else "Underestimated"
        return (
            f"{direction} {worst} in zone {ctx['zone']} for calendar month "
            f"{ctx['season']}; weight the seasonal cycle more against the raw trend."
        )

    def recommend(
        self, ranked_rows: list[dict[str, Any]], lessons: list[str]
    ) -> list[dict[str, Any]]:
        out = []
        for row in ranked_rows:
            need_tolerance = row.get("aqi_latest", 0) >= 120 or row.get("aqi_pred_delta", 0) > 0
            tight_space = row.get("plantable_space", 0.5) < 0.35
            dry = row.get("ndvi_slope", 0) < 0
            soil = row.get("soil")

            def score(sp: dict[str, str]) -> float:
                s = 0.0
                tol = {"high": 2.0, "medium": 1.0, "low": 0.0}[sp["pollution_tolerance"]]
                s += tol * (2.0 if need_tolerance else 1.0)
                if tight_space and sp["canopy"] in ("small", "medium"):
                    s += 1.5
                if not tight_space and sp["canopy"] == "large":
                    s += 1.0
                if dry and sp["water_need"] == "low":
                    s += 1.0
                if sp["native_status"] == "native":
                    s += 0.5
                return s

            # drop species whose soil pH/texture clashes with the zone; only
            # fall back to the full KB if the soil filter leaves too few
            compatible = [sp for sp in SPECIES_KB if species_soil_ok(sp, soil)]
            pool = compatible if len(compatible) >= 5 else SPECIES_KB
            ranked_sp = sorted(pool, key=score, reverse=True)
            species = [sp["common"] for sp in ranked_sp[:4]]
            soil_note = ""
            if isinstance(soil, dict) and soil.get("ph") is not None:
                soil_note = (
                    f" Soil pH {soil['ph']:.1f} ({soil.get('ph_class', '?')}), "
                    f"{soil.get('texture', 'unknown')} — species filtered for soil fit."
                )
            out.append(
                {
                    "zone": row["zone"],
                    "justification": (
                        f"Priority score {row['score']:.3f}: predicted AQI change "
                        f"{row['aqi_pred_delta']:+.1f}, traffic change {row['traffic_pred_delta']:+.1f}, "
                        f"NDVI trend {row['ndvi_slope'] * 12:+.4f}/yr with canopy at "
                        f"{row['ndvi_latest']:.2f} and plantable-space {row['plantable_space']:.2f}. "
                        "Species chosen for pollution tolerance and fit to available space."
                        + soil_note
                    ),
                    "species": validate_selection(species),
                }
            )
        return out

    def project_note(self, city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
        return (
            f"UNVALIDATED projection: {n_months} months of damped trend extrapolation for "
            f"{city}. No observations exist that far out, so error compounds and is "
            "uncheckable; policy shifts, climate variation and land-use change are not "
            "modeled. Treat as a scenario sketch, not a forecast."
        )


def build_model(cfg: ModelCfg, mock: bool) -> Any:
    if mock or cfg.provider == "mock":
        log.info("using MockModel (offline, no API key, no cost)")
        return MockModel()
    log.info("using %s model %s (inference only)", cfg.provider, cfg.name)
    return OpenRouterModel(cfg)
