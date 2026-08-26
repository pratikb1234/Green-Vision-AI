# Green Vision

**Where should this city plant trees next, and what should go in the ground?**

Green Vision answers that in two parts that share one subject and one grid:

| Part | What it is | Runs |
|---|---|---|
| `greenplan/` | **The priority engine.** Ranks every hexagonal cell in the city by where green cover is declining fastest against worsening air quality. This is the AI. | Python, offline, one command |
| `index.html` | **The design studio.** Click a place, read live conditions across 100 km², draw a plot, place trees, cost it, project 25 years. | Browser, no server |

The studio reads the engine's `recommendations.geojson` in its **Priority**
view, so the ranking, the per-cell reasoning and the species picks all surface
on the map — see Part 2.

---

## Part 1 — the priority engine

### Run it right now (no API key, no cost)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m greenplan run --config config/city.yaml --mock --recommend
```

`--mock` swaps in synthetic adapters **and** an offline stand-in model, so the
whole pipeline runs with zero network calls. To run on the **real** Ahmedabad
data with the stand-in model, set `model.provider: mock` in the config and drop
`--mock` — the CSV adapters then load and you get a genuine 146-cell result.

### Run it with a real model — locally, on Intel OpenVINO

This is the default. No API key, no network, no per-token cost.

```bash
python scripts/fetch_openvino_model.py          # ~1 GB, one time
.venv/Scripts/python -m greenplan run --config config/city.yaml --recommend
```

Intel publishes instruct models already converted to OpenVINO IR and
weight-compressed to **INT4**, which is what makes this practical: ~1 GB on
disk instead of ~3 GB at FP16, **2.5 s to load and ~4 s per reply on a plain
CPU**, no GPU anywhere. `model.device` passes straight through to OpenVINO, so
the same build targets `CPU`, an integrated `GPU`, or an `NPU` unchanged.

`scripts/fetch_openvino_model.py --list` shows the catalogue (TinyLlama 1.1B →
Phi-3.5-mini 3.8B).

### Or with a hosted model

```bash
$env:NVIDIA_API_KEY="..."        # Linux/macOS: export NVIDIA_API_KEY=...
.venv/Scripts/python -m greenplan run --config config/city.yaml --recommend
```

Set `model.provider` to `nvidia` (NIM, `meta/llama-3.1-70b-instruct`) or
`openrouter`. Keys are read from the environment and never hard-coded.

### Measured: local model vs. the naive baseline

30 backtest iterations, 12-month horizon, on the real 146-cell panel:

| | Qwen2.5-1.5B INT4 (local) | Offline stand-in (Theil–Sen + seasonality) |
|---|---|---|
| MAE — AQI | 33.3 | **13.5** |
| MAE — NDVI | 0.103 | **0.042** |
| Skill vs baseline | **−0.385 → −0.333** | +0.075 → +0.084 |
| `memory_helped` | false | true |
| Malformed JSON replies | 0 / 30 | n/a |

**Read this before quoting any accuracy number.** A 1.5B model is *worse than
a trend line* at numeric extrapolation, and no amount of prompting fixes that —
it is the wrong tool for the regression half of the task. What it does do
reliably is produce well-formed output (0 repairs needed in 30 iterations) and
write the per-cell justification and species picks, which is the half a
language model is actually good at. Its memory loop does work — skill improves
+0.052 from the first half of training to the second — it simply starts from
too far behind to overtake the baseline.

If you need the ranking to be as accurate as possible, run a hosted model. If
you need it to run anywhere, for free, with nothing leaving the machine, run
OpenVINO. That trade is real and this table is the honest version of it.

### What a run does

1. **Adapters** load three per-cell monthly streams:
   - `data/ahmedabad_ndvi.csv` — **real** NASA MOD13Q1 vegetation index (250 m,
     16-day) via ORNL DAAC, keyless
   - `data/ahmedabad_aqi.csv` — **real** US AQI from the Open-Meteo Air-Quality
     archive, keyless
   - `data/ahmedabad_traffic_placeholder.csv` — an **inert placeholder**. No
     free historical traffic source exists for arbitrary coordinates. Its MCDA
     weight is `0.0`, so it never influences a ranking. Disclosed, not real.
2. **Grid** merges all three onto H3 resolution 7 (~5 km² cells).
   Measured panel: **146 cells × 42 months, 2023-01 → 2026-06.**
3. **Training** backtests: predict a past month from history strictly before a
   cutoff, score against what really happened, ask the model for a one-line
   lesson, append to `models/{city}/memory.jsonl`. Later prompts retrieve the
   most relevant records. **In-context learning — no weight updates, no
   fine-tuning, no retraining.**
4. **`--recommend`** ranks all cells by a numeric MCDA score, has the model
   justify the ranking and pick species, and writes `recommendations.geojson`,
   `recommendations.csv`, and `planting_brief.txt`.

### MCDA weights (`config/city.yaml`)

| Criterion | Weight |
|---|---|
| AQI worsening | 0.40 |
| NDVI decline | 0.35 |
| Low green cover | 0.15 |
| Plantable space | 0.10 |
| Traffic worsening | **0.0** (inert placeholder) |

### Validation honesty

A horizon can only be *checked* if it fits inside the data:

```bash
.venv/Scripts/python -m greenplan horizon --config config/city.yaml
# 42 months of history, min_history=18 -> largest honestly validatable horizon: 23 months.
```

Requesting a longer `--horizon` fails loudly. Anything beyond the ceiling must
go through `--project N`, which is labeled **UNVALIDATED** in the output file
itself.

### Soil

Real, and wired in. `data/ahmedabad_soilgrids.csv` holds pH, sand/silt/clay,
organic carbon and nitrogen per H3 cell from **ISRIC SoilGrids v2.0** (250 m,
free, no key). Species selection respects soil pH and texture alongside
pollution load. Refresh with:

```bash
python scripts/soilgrids_export.py --config config/city.yaml   --out data/ahmedabad_soilgrids.csv
```

SoilGrids masks built-up land, so cells in the dense core come back empty at
their centre; the exporter re-samples four offsets around each miss and
recovers most of them. Whatever is still empty falls through to pollution-only
matching rather than inventing a value.

### Still not wired up

- **Soil moisture.** `soil.moisture_csv` stays commented out: NASA SMAP needs
  an Earthdata login, which breaks the no-key property.
- **Bare-ground site finder.** `sites.enabled: true` but `candidates_csv` is
  commented out, so it falls back to the `1 - NDVI` plantable proxy and emits no
  `planting_sites.geojson`.
- **10 years of history.** Not possible from free keyless sources. Open-Meteo's
  air-quality archive returns nothing before **2023-01** (verified back to
  2013), which caps the panel at 42 months regardless of how far MODIS goes
  back. Any claim of a decade of data is wrong.

---

## Part 2 — the design studio

Open `index.html` in any modern browser. No install, no server, no build step.

- **100 km² area of interest.** Clicking the map draws a circle of
  r = √(100/π) km ≈ 5.64 km (`GV.CFG.AOI_KM2`) and scopes every reading to it,
  sampling air quality at 9 points spread across the area.
- **Live, keyless data.** Open-Meteo weather + air quality, Open-Meteo archive
  (2020–2024) for rainfall, OpenStreetMap via Overpass for the feature census,
  Nominatim for place names, Esri World Imagery for canopy.
- **Canopy from imagery.** `vegScore()` computes a greenness index from the RGB
  of Esri tiles and renders a red→green heatmap. This is **current** imagery —
  the studio has no historical satellite stack. History lives in the engine.
- **Species catalogue.** 16 curated India-climate species matched on planting
  goal, pollution tolerance (`pol` 1–5), drought tolerance, and local rainfall.
  **Not** on soil.
- **Cost and 25-year projection.** Indicative 2026 Indian rates. The projection
  is explicitly labeled `PROJECTED, not forecast` and `UNVALIDATED` in the source
  — logistic canopy growth, survival curves, species lifespan, saturating
  cooling. Treat it as a defensible shape, not a prediction.
- **Priority view.** The fourth view button loads the engine's
  `recommendations.geojson` and draws each ranked H3 cell as a colour-coded
  hexagon — warm where planting is most urgent, which is deliberately *not*
  the green ramp used for current canopy. Click a cell for its priority score,
  the predicted AQI and canopy change, the species picked for that cell, and
  the model's own plain-English justification, then hand off to the Studio to
  design on it.

  Served over `http://`, the file is fetched automatically from
  `outputs/<city>/recommendations.geojson`. Opened as `file://`, the browser
  forbids that read, so the page asks you to pick the file — it is parsed
  in-page and never uploaded.
- **Storage.** Designs are saved to browser LocalStorage. Nothing is uploaded
  unless you set `COMMUNITY_URL` and publish deliberately.

### Configuration

There are **two** config objects in `index.html`:

| Object | Line | Holds |
|---|---|---|
| `CFG` | ~177 | map start, tile URLs, Overpass endpoints |
| `GV.CFG` | ~965 | `AOI_KM2`, `RATES`, `TOMTOM_KEY`, `AUTH_URL`, `COMMUNITY_URL`, `GOOGLE_CLIENT_ID` |

```javascript
GV.CFG = {
  AOI_KM2: 100,
  AUTH_URL: "",           // blank => local-only sign-in
  COMMUNITY_URL: "",      // blank => gallery shows only your own designs
  GOOGLE_CLIENT_ID: "",   // for "Continue with Google"
  TOMTOM_KEY: "",         // blank => traffic is MODELLED from OSM topology, not measured
  RATES: { labour_day: 650, mali_month: 14000, water_kl: 45,
           contingency: 0.12, design_fee: 0.08, gst: 0.18 }
};
```

### Privacy, stated accurately

Designs never leave the browser, and with `provider: openvino` the engine's
reasoning never leaves the machine either — no key, no endpoint, no per-cell
figures in anyone's logs.

What still goes out is the map itself: every click sends coordinates to
Open-Meteo, Overpass, Nominatim and Esri, plus Google/unpkg/cdnjs for tiles and
libraries. Switching `model.provider` to a hosted option also sends per-cell
aggregates to that provider.

So: **local-first, and fully local for inference — but the map layer is not
offline.** Say that, rather than "runs entirely on the host computer".

---

## Repo notes

- `Front_End.html` is the earlier committed UI. `index.html` supersedes it.
- `temp_script.js` is a stale standalone duplicate of the studio module. It is
  not referenced by `index.html` and re-declares `const CFG`, so loading both
  would throw. Safe to delete.
