"""Bare-land candidate sites: the bridge from hexagon-level ranking to actual
map points where trees can go.

Input is a CSV of bare-ground patches (e.g. centroids of the "Bare ground"
class from Esri's 10 m Land Cover, exported over the city bbox). Each patch is
snapped to its H3 cell so we can (a) report specific coordinates inside each
top-priority zone and (b) replace the crude ``1 - NDVI`` plantable-space proxy
with a real bare-area fraction per cell.

Accuracy note: 10 m land-cover locates a patch to ~±5 m. That is enough to send
a crew to the right plot, NOT to mark an individual planting pit — a human still
confirms the exact spot on a sub-metre basemap (Esri World Imagery). The brief
prints the coordinates to go inspect.

CSV columns (extras ignored, names case-insensitive):
  * ``lat`` + ``lon``            — patch centroid (required)
  * ``patch_area_m2`` / ``area`` / ``value`` / ``mean`` — bare area, optional
                                   (defaults to one 10 m pixel = 100 m²)
  * ``confidence``               — 0..1 classifier confidence, optional
  * ``landcover_class``          — land-cover group name, optional
  * ``plantable``                — 0/1 from the land-cover filter, optional

The land-cover filter, and the two gates it has to clear
--------------------------------------------------------
Low NDVI is not the same thing as bare ground: water, rock and flat roofs sit
below the threshold too. ``scripts/train_landcover.py`` trains a classifier on
ESA WorldCover labels to tell them apart and
``sentinel2_ndvi_export.py --classify-sites`` writes its verdict into the
columns above.

Those columns are read here but **only acted on if the model earned it**, and
earning it takes two separate pieces of evidence (see ``landcover_gate``):
held-out skill against the NDVI rule, and continued agreement with its own
training labels on the actual candidate sites. If either fails, the columns
are kept for inspection, every site is treated as unflagged, and confidences
fall back to 1.0 — so a losing classifier cannot sneak into the ranking
through the confidence sort either. That is the same discipline
``provider: hybrid`` applies to the forecast challenger: a model in the repo
with a published score, not a model in the pipeline on faith.

On the shipped Ahmedabad data the model clears the first gate comfortably and
fails the second, so it is benched. That is the gate working, not the gate
being decorative.

Flagged sites are never dropped. They stay in the file, in the GeoJSON and in
the brief, marked, because a screening model that silently deletes a real
planting site is worse than one that is visibly wrong.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .h3grid import cell_area_m2, latlng_to_cell

log = logging.getLogger(__name__)

_DEFAULT_PIXEL_M2 = 100.0  # one 10 m Sentinel-2 / Esri land-cover pixel


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, f"no {path.name} at {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as exc:
        return None, f"unreadable {path}: {exc}"


def landcover_gate(landcover_dir: str | Path | None) -> dict[str, Any]:
    """May the land-cover classifier act on the candidate sites?

    TWO criteria, and it has to clear both. Anything missing or unreadable is
    a closed gate — a model that cannot show its score does not get to filter
    planting sites.

    1. **Held-out skill** (`report.json`): beat the tuned NDVI-threshold rule
       it would replace, on spatially disjoint blocks.
    2. **Deployment agreement** (`site_audit.json`): still agree with its own
       training labels on the candidate-site population.

    Criterion 2 exists because criterion 1 cannot see the failure that
    matters here. The held-out score is measured on a random spatial sample of
    the whole scene; the candidate sites are the extreme low-NDVI tail picked
    by a completely different rule. A model can be strong on the first and
    worthless on the second, and for this repo's Ahmedabad data that is
    exactly what happened — see the README. One gate would have shipped it."""
    if not landcover_dir:
        return {"passed": False, "reason": "no sites.landcover_dir configured"}
    d = Path(landcover_dir)

    report, err = _read_json(d / "report.json")
    if report is None:
        return {"passed": False, "reason": err}
    gate = dict(report.get("gate") or {})
    model = (report.get("model") or {}).get("openvino") or {}
    base = report.get("baseline_ndvi_threshold") or {}
    out: dict[str, Any] = {
        "held_out_passed": bool(gate.get("passed")),
        "accuracy": model.get("accuracy"),
        "baseline_accuracy": base.get("accuracy"),
    }
    if not out["held_out_passed"]:
        return out | {
            "passed": False,
            "reason": f"held-out accuracy {out['accuracy']} did not beat the "
                      f"NDVI baseline {out['baseline_accuracy']}",
        }

    audit, err = _read_json(d / "site_audit.json")
    if audit is None:
        return out | {
            "passed": False,
            "reason": f"{err} — the classifier has not been audited on the "
                      "candidate sites it would filter (run "
                      "sentinel2_ndvi_export.py --classify-sites)",
        }
    out |= {
        "audit_passed": bool(audit.get("passed")),
        "site_agreement": audit.get("agreement"),
        "required_agreement": audit.get("required_agreement"),
        "worldcover_plantable_rate": audit.get("worldcover_plantable_rate"),
        "classifier_plantable_rate": audit.get("classifier_plantable_rate"),
    }
    if not out["audit_passed"]:
        return out | {
            "passed": False,
            "reason": (
                f"held-out accuracy {out['accuracy']} but only "
                f"{out['site_agreement']} agreement with ESA WorldCover on the "
                f"candidate sites themselves (needs {out['required_agreement']}): "
                f"it calls {out['classifier_plantable_rate']} of them plantable "
                f"where the labels say {out['worldcover_plantable_rate']}"
            ),
        }
    return out | {"passed": True, "reason": "both gates passed"}


class CandidateSites:
    """Bare-ground patches indexed by H3 cell."""

    def __init__(self, sites: pd.DataFrame, resolution: int,
                 gate: dict[str, Any] | None = None) -> None:
        self.resolution = resolution
        # columns: lat, lon, cell, patch_area_m2, confidence, landcover_class,
        #          flagged
        self.sites = sites
        self.gate = gate or {"passed": False, "reason": "no classifier"}

    @property
    def empty(self) -> bool:
        return self.sites.empty

    @property
    def filtered(self) -> bool:
        """True when a land-cover classifier passed its gate and its flags are
        actually being acted on."""
        return bool(self.gate.get("passed")) and bool(self.sites["flagged"].any())

    def zones(self) -> set[str]:
        return set(self.sites["cell"].unique())

    def plantable_fraction_by_cell(self) -> dict[str, float]:
        """bare patch area / hexagon area, per cell, clipped to [0, 1].

        A cell fully covered by bare patches -> 1.0; a sealed cell with no
        patches never appears here (callers fall back to the proxy).

        Patches the land-cover classifier flagged as not-plantable — water,
        roofs, existing canopy — contribute no area, because they are not
        plantable space. Without a passed gate nothing is flagged and this is
        the plain sum it always was."""
        out: dict[str, float] = {}
        usable = self.sites[~self.sites["flagged"]]
        for cell, g in usable.groupby("cell"):
            area = cell_area_m2(cell)
            if area <= 0:
                continue
            out[cell] = float(np.clip(g["patch_area_m2"].sum() / area, 0.0, 1.0))
        return out

    def sites_for_zone(self, zone: str, k: int) -> list[dict[str, Any]]:
        """The k biggest / most-confident patches in a zone, largest first.

        Flagged patches sort last but are still returned when the zone has
        fewer than k unflagged ones: a crew told "these three are what the
        model likes, these two it thinks are roofs" can check for itself."""
        g = self.sites[self.sites["cell"] == zone]
        if g.empty:
            return []
        g = g.sort_values(
            ["flagged", "confidence", "patch_area_m2"],
            ascending=[True, False, False],
        ).head(k)
        return [
            {
                "lat": round(float(r.lat), 6),
                "lon": round(float(r.lon), 6),
                "patch_area_m2": round(float(r.patch_area_m2), 1),
                "confidence": round(float(r.confidence), 3),
                "landcover_class": str(r.landcover_class),
                "flagged_not_plantable": bool(r.flagged),
            }
            for r in g.itertuples()
        ]


def load_candidate_sites(
    path: str | Path,
    resolution: int,
    min_patch_m2: float = 0.0,
    landcover_dir: str | Path | None = None,
) -> CandidateSites:
    """Read a bare-patch CSV and snap every patch to its H3 cell."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"sites CSV not found: {path}. Export bare-ground patches (see the "
            "README 'Finding planting sites' section) or unset sites.candidates_csv."
        )
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(f"{path}: need 'lat' and 'lon' columns; found {list(df.columns)}")

    area_col = next(
        (c for c in ("patch_area_m2", "area", "value", "mean") if c in df.columns), None
    )
    area = (
        pd.to_numeric(df[area_col], errors="coerce")
        if area_col
        else pd.Series(_DEFAULT_PIXEL_M2, index=df.index)
    )
    conf = (
        pd.to_numeric(df["confidence"], errors="coerce")
        if "confidence" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    klass = (
        df["landcover_class"].astype(str)
        if "landcover_class" in df.columns
        else pd.Series("unclassified", index=df.index)
    )
    plantable = (
        pd.to_numeric(df["plantable"], errors="coerce")
        if "plantable" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # THE GATE. The classifier's columns are honoured only if it cleared both
    # criteria in landcover_gate(): held-out skill over the NDVI-threshold
    # rule, AND continued agreement with its own labels on these very sites.
    classified = "plantable" in df.columns or "landcover_class" in df.columns
    gate = landcover_gate(landcover_dir) if classified else {
        "passed": False, "reason": "sites CSV carries no land-cover columns"
    }
    if classified and gate["passed"]:
        flagged = plantable.fillna(1) < 0.5
        log.info(
            "land-cover filter ACTIVE (held-out accuracy %s vs NDVI baseline %s): "
            "%d of %d candidates flagged not-plantable — flagged, not dropped",
            gate.get("accuracy"), gate.get("baseline_accuracy"),
            int(flagged.sum()), len(df),
        )
    else:
        # Benched, exactly like the forecast challenger: the columns survive
        # for inspection, but nothing downstream acts on them — including the
        # confidence sort, which falls back to neutral.
        flagged = pd.Series(False, index=df.index)
        if classified:
            log.info(
                "land-cover filter BENCHED (%s): its columns are kept for "
                "inspection but no site is filtered or re-ranked by them",
                gate.get("reason"),
            )
            conf = pd.Series(1.0, index=df.index)

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "patch_area_m2": area.fillna(_DEFAULT_PIXEL_M2),
            "confidence": conf.fillna(1.0).clip(0.0, 1.0),
            "landcover_class": klass,
            "flagged": flagged.astype(bool),
        }
    ).dropna(subset=["lat", "lon"])
    out = out[out["patch_area_m2"] >= float(min_patch_m2)]
    out["cell"] = [
        latlng_to_cell(la, lo, resolution) for la, lo in zip(out["lat"], out["lon"])
    ]
    log.info(
        "candidate sites: %d patches in %d cells from %s",
        len(out), out["cell"].nunique(), path.name,
    )
    return CandidateSites(out.reset_index(drop=True), resolution, gate)
