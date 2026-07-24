"""Local HTTP bridge that puts the *real* trainable GreenGrid engine behind the
Front_End.html map — no framework, stdlib only, runs on the same venv.

On startup it loads the city panel, builds the model (offline MockModel unless a
provider key is set), runs the backtest/memory TRAINING loop (this is the
"trained on all the data" step), and caches the ranked zones + recommendations.

Endpoints (all CORS-open so a file:// page can call them):
  GET  /api/health      -> engine status, city, model, training skill
  GET  /api/zones       -> recommendations.geojson (trained-engine top zones)
  POST /api/recommend   -> body {lat,lon,aqi,ndvi|green,plantable}; returns the
                           engine's species pick + justification for that point,
                           using the curated species KB + soil/pollution logic.

The frontend keeps its API-key slots empty and separate; this server needs no
key to run (mock/offline). Set NVIDIA_API_KEY to train/predict with live NVIDIA
reasoning instead — the per-click species selection stays fast + deterministic.

Run:  python -m greenplan.server --config config/city.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so output is valid JSON
    (json.dumps emits bare NaN/Infinity, which browsers' JSON.parse reject)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj

from .config import load_config
from .engine import load_panel, recommend, train
from .features.h3grid import cell_boundary_lonlat, latlng_to_cell
from .reasoning.client import MockModel, build_model
from .reasoning.species import kb_by_name, validate_selection
from .training.memory import MemoryStore

log = logging.getLogger(__name__)


class Engine:
    """Loads + trains the pipeline once, then answers per-point queries."""

    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        # Two independent "mock" decisions:
        #  * adapters ALWAYS follow config (so real csv: streams like AQI are
        #    honored) — pass mock=False to load_panel.
        #  * the reasoning MODEL falls back to MockModel only when no LLM key is
        #    set, so the engine still runs fully offline.
        self.mock_model = not os.environ.get("NVIDIA_API_KEY") and not os.environ.get(
            "OPENROUTER_API_KEY"
        )
        self.kb = kb_by_name()
        # Per-click reasoning engine: the real LLM when a provider key is set,
        # the deterministic offline engine otherwise. This is the ONLY place an
        # LLM is used — it answers the handful of clicks a user actually reads.
        # _fallback catches rate-limits/timeouts so a click never hard-fails.
        self._picker = build_model(self.cfg.model, self.mock_model)
        self._fallback = MockModel()
        self._build_and_train()

    def _build_and_train(self) -> None:
        cfg = self.cfg
        log.info("loading panel for %s (adapters from config; mock_model=%s)…",
                 cfg.city.name, self.mock_model)
        self.panel = load_panel(cfg, mock=False)  # honor configured adapters
        # Batch forecasting/training is ALWAYS the offline deterministic model:
        # predict_future runs once per zone (146 cells) plus ~120 training calls.
        # Routing that through the LLM would mean ~270 requests at startup, which
        # this account answers with HTTP 429. The LLM is reserved for per-click
        # reasoning (self._picker) — the part a user actually reads.
        self.model = build_model(cfg.model, True)
        self.memory = MemoryStore(cfg.resolve(cfg.training.memory_path))
        log.info("training (backtest + memory) …")
        self.train_report = train(cfg, self.panel, self.model, self.memory)
        log.info("running recommendation pass …")
        result = recommend(cfg, self.panel, self.model, self.memory, self.train_report)
        self.ranked = result["ranked"]
        self.recommendations = result["recommendations"]
        # cache the trained-engine geojson for the map overlay
        geo_path = cfg.resolve(cfg.run.outputs_dir) / "recommendations.geojson"
        try:
            self.zones_geojson = json.loads(geo_path.read_text(encoding="utf-8"))
        except Exception:
            self.zones_geojson = {"type": "FeatureCollection", "features": []}
        self.lessons = [
            r.get("lesson", "").strip()
            for r in reversed(self.memory.records)
            if r.get("lesson", "").strip()
        ][:12]
        self._build_greenloss()
        log.info(
            "engine ready: %d zones, %d memory records, model=%s | greenloss %s",
            len(self.ranked), len(self.memory), type(self.model).__name__,
            self.greenloss["count"],
        )

    # Zone status thresholds on real NDVI (documented, not magic):
    #   GREEN_NDVI  — at/above this the cell is currently vegetated
    #   LOSS_DELTA  — forecast yr-on-yr NDVI change at/below this = meaningful loss
    GREEN_NDVI = 0.28
    LOSS_DELTA = -0.02

    def _build_greenloss(self) -> None:
        """All ranked cells as H3 polygons tagged green / yellow / red, from the
        trained engine's forecast. YELLOW (currently green but predicted to
        decline) is the whole point — it comes from ndvi_pred_delta, not a
        snapshot. GREEN/RED here are the engine's NDVI-based view; the browser
        refines them per-pixel (VARI) and with OSM (buildings/roads/water)."""
        feats = []
        for r in self.ranked.itertuples():
            nl, dd = float(r.ndvi_latest), float(r.ndvi_pred_delta)
            if not math.isfinite(nl) or not math.isfinite(float(r.score)):
                continue  # cell has no real NDVI coverage — omit, don't guess a colour
            if nl >= self.GREEN_NDVI and dd <= self.LOSS_DELTA:
                status = "yellow"          # has green, forecast to lose it
            elif nl >= self.GREEN_NDVI:
                status = "green"           # vegetated, stable/improving
            else:
                status = "red"             # low vegetation (plantable candidate)
            try:
                geom = {"type": "Polygon", "coordinates": [cell_boundary_lonlat(r.zone)]}
            except Exception:
                geom = None
            feats.append({
                "type": "Feature", "geometry": geom,
                "properties": {
                    "zone": r.zone, "rank": int(r.rank),
                    "score": round(float(r.score), 4),
                    "ndvi_latest": round(nl, 3),
                    "ndvi_slope_per_year": round(float(r.ndvi_slope) * 12, 4),
                    "ndvi_pred_delta": round(dd, 4),
                    "status": status,
                },
            })
        count = {s: sum(1 for f in feats if f["properties"]["status"] == s)
                 for s in ("green", "yellow", "red")}
        self.greenloss = {
            "type": "FeatureCollection",
            "features": feats,
            "thresholds": {"green_ndvi": self.GREEN_NDVI, "loss_delta": self.LOSS_DELTA},
            "count": count,
            "total_zones": len(feats),
        }

    def health(self) -> dict[str, Any]:
        tr = self.train_report or {}
        return {
            "ok": True,
            "city": self.cfg.city.name,
            "model": type(self.model).__name__,
            "mock_model": self.mock_model,
            "aqi_source": self.cfg.adapters.aqi,
            "zones": int(len(self.ranked)),
            "memory_records": int(len(self.memory)),
            "trained": int(len(self.memory)) > 0,
            "retrained_this_session": self.train_report is not None,
            "memory_helped": tr.get("memory_helped"),
            "ndvi_source": self.cfg.adapters.green_cover,
            "reasoning": "nvidia-llm" if not self.mock_model else "offline-engine",
            "reasoning_model": self.cfg.model.name if not self.mock_model else None,
            "greenloss": self.greenloss["count"],
        }

    def recommend_point(self, body: dict[str, Any]) -> dict[str, Any]:
        """Species pick + justification for one clicked point, from the engine."""
        aqi = _num(body.get("aqi"), 90.0)
        # accept ndvi (0..1) directly, or a 0..100 "green"/canopy percentage
        ndvi = body.get("ndvi")
        if ndvi is None and body.get("green") is not None:
            ndvi = _num(body.get("green"), 40.0) / 100.0
        if ndvi is None and body.get("vegPct") is not None:
            ndvi = _num(body.get("vegPct"), 40.0) / 100.0
        ndvi = float(np.clip(_num(ndvi, 0.4), 0.0, 1.0))
        plantable = float(np.clip(_num(body.get("plantable"), 1.0 - ndvi), 0.0, 1.0))
        lat, lon = body.get("lat"), body.get("lon")
        try:
            zone = latlng_to_cell(float(lat), float(lon), self.cfg.grid.h3_resolution)
        except Exception:
            zone = "point"

        # crude single-point MCDA-ish score just for the justification text
        score = float(np.clip(0.3 * (aqi / 300.0) + 0.35 * (1.0 - ndvi) + 0.1 * plantable, 0, 1))
        row = {
            "zone": zone, "score": round(score, 3),
            "aqi_latest": round(aqi, 1), "aqi_pred_delta": 0.0,
            "traffic_latest": 0.0, "traffic_pred_delta": 0.0,
            "ndvi_latest": round(ndvi, 3), "ndvi_slope": 0.0,
            "plantable_space": round(plantable, 2), "soil": None,
        }
        used = "nvidia-llm" if not self.mock_model else "offline-engine"
        try:
            recs = self._picker.recommend([row], self.lessons)
        except Exception as exc:  # 429 / timeout / invalid JSON from the provider
            log.warning("LLM failed (%s); using offline engine", exc)
            recs = self._fallback.recommend([row], self.lessons)
            used = "offline-engine (LLM unavailable)"
        names = validate_selection(recs[0]["species"]) if recs else []
        species = [self._species_card(n) for n in names]
        return {
            "source": used,
            "zone": zone,
            "justification": recs[0]["justification"] if recs else "",
            "species": species,
            "score": row["score"],
        }

    def _species_card(self, name: str) -> dict[str, Any]:
        k = self.kb.get(name, {})
        return {
            "common": name,
            "botanical": k.get("botanical", ""),
            "native_status": k.get("native_status", ""),
            "canopy": k.get("canopy", ""),
            "pollution_tolerance": k.get("pollution_tolerance", ""),
            "water_need": k.get("water_need", ""),
            "context": k.get("context", ""),
            "soil_ph": k.get("soil_ph", ""),
        }


def _num(v: Any, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GreenGridEngine/1.0"

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, obj: Any, code: int = 200) -> None:
            payload = json.dumps(_clean(obj)).encode("utf-8")
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/health"):
                self._json(engine.health())
            elif self.path.startswith("/api/greenloss"):
                self._json(engine.greenloss)
            elif self.path.startswith("/api/zones"):
                self._json(engine.zones_geojson)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/api/recommend"):
                self._json({"error": "not found"}, 404)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                self._json(engine.recommend_point(body))
            except Exception as exc:  # keep the server alive on any bad request
                log.warning("recommend failed: %s", exc)
                self._json({"error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="GreenGrid local engine server")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = Engine(args.config)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    log.info("GreenGrid engine serving on http://%s:%d  (Ctrl+C to stop)", args.host, args.port)
    log.info("  GET  /api/health   GET /api/zones   POST /api/recommend")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        httpd.server_close()


if __name__ == "__main__":
    main()
