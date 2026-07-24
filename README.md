# GreenGrid

An AI decision engine that tells city planners **where to plant trees and what
to plant there**, from three per-zone data streams: satellite green cover
(NDVI), traffic congestion, and AQI. No frontend, no dashboards — it produces
GeoJSON + CSV + a plain-text planting brief.

DeepSeek (via the OpenRouter API) is the reasoning brain, used for
**inference only** — nothing ever trains or fine-tunes the model. The system
"learns" through an in-context memory loop: it backtests itself against
history, records every prediction/actual/error/lesson to a JSONL memory file,
and re-injects the most relevant records into each new prompt.

## Run it right now (no API key, no cost)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python -m greenplan run --config config/city.yaml --mock --recommend
```

`--mock` swaps in synthetic-but-realistic adapters **and** an offline mock
model, so the entire pipeline — H3 aggregation, backtest training, memory,
ranking, species selection, all output files — runs end-to-end with zero
network calls. Or with Docker:

```bash
docker build -t greengrid .
docker run --rm -v "$PWD/outputs:/app/outputs" greengrid
```

## What a run does

1. **Adapters** fetch three streams of rows `zone, month, traffic, aqi, ndvi`
   (month = integer index, oldest first; each adapter fills only its metric).
2. **Features** merges the streams onto a common H3 hexagon grid
   (`grid.h3_resolution`, default 9) and computes robust Theil–Sen trends.
3. **Training** runs the backtest loop (below) unless the retrain policy says
   the memory is still fresh.
4. **`--recommend`** predicts each zone `horizon` months past the end of the
   data, ranks all zones with a numeric MCDA score, has DeepSeek justify the
   ranking and pick 3–5 species per top zone (matched to pollution load **and
   soil**, when soil is supplied), and writes `recommendations.geojson`,
   `recommendations.csv`, `planting_brief.txt`, and — when a bare-ground sites
   CSV is configured — `planting_sites.geojson` with point-level coordinates.
5. **`--project N`** additionally emits a long-range damped-trend forecast
   (e.g. `--project 480` ≈ 40 years) clearly labeled **UNVALIDATED**, because
   no real data exists that far out to check it against.

## The training loop (in-context learning, no weight updates)

Each backtest iteration:

1. picks a zone and a **cutoff month in the past**;
2. shows the model only history **before** the cutoff (plus the most relevant
   `memory_k` records retrieved from memory: its own past predictions,
   actuals, errors, and lessons);
3. asks for strict-JSON predictions of traffic / AQI / NDVI at
   `cutoff + horizon`;
4. looks up what **really happened** (still inside recorded history) and
   scores the prediction (MAE per metric);
5. asks the model for a one-line **lesson** about the pattern it missed;
6. appends everything to `models/memory.jsonl`.

At the end it reports, first half of iterations vs second half: (a) raw
normalized error, and (b) **skill vs a memory-free naive baseline**
(Theil–Sen trend + seasonality) computed on the *same* tasks — the fair
comparison, since the two halves may randomly draw tasks of different
difficulty. `memory_helped` is true when, by the second half of training, the
model **with** memory beats the memory-free baseline. With the mock model
(which applies a genuine season-weighted bias correction learned from
retrieved memory records) that skill is consistently positive; with DeepSeek
the improvement comes from the model pattern-matching on its own past cases
re-injected into the prompt.

**Validation honesty:** a horizon can only be *checked* if it fits inside the
data. Find your ceiling with:

```bash
python -m greenplan horizon --config config/city.yaml --mock
# e.g. "120 months of history ... largest honestly validatable horizon: 101 months"
```

Requesting `--horizon` beyond that ceiling fails loudly instead of silently
producing uncheckable "training". Anything longer must go through
`--project N`, which is labeled UNVALIDATED in the output file itself.

## Running with DeepSeek for real

```bash
export OPENROUTER_API_KEY=sk-or-...   # never hard-coded, env var only
python -m greenplan run --config config/city.yaml --recommend
```

* Model is `model.name` in the config: `deepseek/deepseek-chat` (default) or
  `deepseek/deepseek-r1`.
* The client retries with backoff on 408/429/5xx and network errors, enforces
  a timeout, strips code fences / `<think>` blocks, validates the JSON
  schema, and re-asks on invalid JSON.
* Cost scales with `--iterations` (2 calls per iteration) plus one predict
  call per zone and one recommendation call under `--recommend`. Check
  current OpenRouter pricing before large runs.
* The model **explains and selects from provided options only**: the numeric
  MCDA ranking is authoritative, and species must be exact names from the
  table in `greenplan/reasoning/species.py` (anything else is rejected and
  re-asked).

## Plugging in your real data

Implement `fetch()` in the three empty subclasses in
[greenplan/adapters/base.py](greenplan/adapters/base.py) —
`GreenCoverAdapter`, `TrafficAdapter`, `AQIAdapter` — then set
`adapters.<stream>: custom` in the config. Rules:

* return columns `zone, month, traffic, aqi, ndvi`; fill only your metric,
  leave the others NaN;
* `month` is a consecutive integer index, 0 = oldest, aligned across all
  three adapters (same month 0);
* optionally implement `zone_geometry()` returning `{zone: (lat, lon)}` so
  your zones snap onto the H3 grid — without it, your zone ids are used
  as-is and GeoJSON geometry is omitted;
* set `data.start_month` to the calendar month of index 0 so seasonality is
  interpreted correctly.

There's also a ready-made CSV path — set an adapter to `"csv:<path>"` and
point it at a file with columns `month` + a value column (`ndvi`/`value`/
`mean`) + either `zone` or `lat`+`lon`:

```yaml
adapters:
  green_cover: "csv:data/ahmedabad_ndvi_monthly.csv"
```

**Getting real NDVI via Esri (ArcGIS Pro):**

1. Add a **Living Atlas Sentinel-2 / Landsat multispectral** image service
   (these carry the near-infrared band; the sub-metre World Imagery *basemap*
   is a mixed-date mosaic and is **not** a monthly time series — use it only
   for the visual site check below).
2. Compute NDVI = `(NIR − Red) / (NIR + Red)` with **Band Arithmetic**, build a
   monthly composite, `Sample` it onto a point grid over the bbox, and export
   the attribute table to CSV.
3. Reshape the export to month-indexed rows:
   ```bash
   python scripts/esri_ndvi_to_greengrid.py export.csv data/ahmedabad_ndvi.csv --start 2016-01
   ```
   Then set `adapters.green_cover: "csv:data/ahmedabad_ndvi.csv"`.

The imagery is served by Esri, but the NDVI pixels are ESA Sentinel-2 / USGS
Landsat — Esri is the portal, Sentinel/Landsat the provenance. (A free Google
Earth Engine alternative, [scripts/gee_ndvi_export.js](scripts/gee_ndvi_export.js),
is also included.)

**Match `grid.h3_resolution` to your data density.** Resolution 9 cells are
~200 m wide — right for dense data, but a 1.1 km NDVI grid or a handful of
AQI stations will leave most cells covered by only one stream, and training
needs cells where all three metrics overlap. With sparse real data, drop to
resolution 7 (~2.4 km cells) so the streams land in shared cells.

Mock and custom adapters can be mixed per stream while you wire sources up.

## Finding exact planting sites (Esri 10 m Land Cover)

The ranking scores hexagons; this step turns the top ones into **point
coordinates of actual bare ground** and writes `planting_sites.geojson`.

1. Open Esri's **Sentinel-2 10 m Land Cover** (Land Cover Explorer, or the
   Living Atlas layer in ArcGIS Pro). Isolate the **"Bare ground"** class over
   your bbox and export the bare pixels' centroids to a CSV with columns
   `lat, lon, patch_area_m2` (area optional; defaults to one 100 m² pixel).
2. Point the config at it:
   ```yaml
   sites:
     enabled: true
     candidates_csv: data/ahmedabad_sites.csv
   ```

Each patch snaps to its H3 cell, so `plantable_space` in the ranking becomes a
**real bare-area fraction** (cells with no detected bare ground → 0, so export
across the whole bbox), and each top zone gets specific site coordinates in the
brief and GeoJSON.

**Accuracy ceiling — read this.** 10 m land cover locates a patch to about
**±5 m**: enough to send a crew to the right plot, not to mark an individual
pit. Confirm each spot visually on the **sub-metre Esri World Imagery basemap**
(and on the ground — ownership, utilities and access are not modelled) before
digging. The brief prints the coordinates to inspect and says so.

## Soil data (SoilGrids + NASA SMAP)

Supplying soil lets species selection respect **pH and texture** (e.g.
Ahmedabad's alkaline soils rule some species out) and consider **moisture**.

* **Chemistry & texture — SoilGrids 250 m** (free, CC BY 4.0): export
  `phh2o, soc, nitrogen, sand, silt, clay` (0–30 cm) over the bbox via ArcGIS
  Pro or the SoilGrids WCS to a CSV with `lat, lon` + those columns. Integer
  scaling (pH×10, texture in g/kg) is auto-detected. Set
  `soil.soilgrids_csv: data/ahmedabad_soilgrids.csv`.
* **Moisture — NASA SMAP** (free, Earthdata login): point-sample the 1 km
  downscaled product via AppEEARS, then
  ```bash
  python scripts/smap_to_greengrid.py smap_export.csv data/ahmedabad_smap.csv --since 2020-01
  ```
  Set `soil.moisture_csv: data/ahmedabad_smap.csv`.

Honest scope: **moisture is measured**; **pH / texture / carbon are SoilGrids'
*modelled* estimates** — good for screening species, not a substitute for a
soil lab test. True mineralogy is not recoverable from these sources. Any
missing soil stream simply leaves that field blank and never blocks a run.

## Adding another city

Everything is config-driven, so a new city is one file:

```bash
cp config/city.yaml config/surat.yaml   # edit city.name and city.bbox
python -m greenplan run --config config/surat.yaml --mock --recommend
```

The `{city}` placeholder in `memory_path` and `outputs_dir` expands to the
city name, so each city keeps its own learned memory
(`models/<city>/memory.jsonl`) and its own outputs (`outputs/<city>/`) —
lessons learned in one city never leak into another's prompts.

## Configuration (config/city.yaml)

| Section | What it controls |
|---|---|
| `city` | name + bbox (mock zones are scattered inside the bbox) |
| `grid.h3_resolution` | H3 cell size of the common grid (default 9) |
| `data` | `start_month`, and mock history length / zone count |
| `adapters` | `mock`, `csv:<path>`, or `custom` per stream |
| `sites` | bare-ground `candidates_csv`, `min_patch_m2`, `max_sites_per_zone`, plantability thresholds (site finder) |
| `soil` | `soilgrids_csv` (pH/texture/carbon), `moisture_csv` (SMAP) |
| `training` | iterations, horizon, min history, context window, `memory_k`, memory path, retrain policy (`always` / `if_new_months` / `never`) |
| `mcda.weights` | AQI worsening, traffic worsening, NDVI decline, low canopy, plantable space |
| `model` | provider, model name, base URL, temperature, timeout, retries |
| `run` | seed (mock data and backtest sampling are fully reproducible), outputs dir |

CLI flags `--iterations`, `--horizon`, `--memory-k`, `--seed` override the
config; `--recommend` and `--project N` enable those stages; `--mock` forces
everything offline.

## Outputs

* `recommendations.geojson` — top-N H3 cells as polygons with priority score,
  predicted year-over-year deltas, `n_sites`, soil summary, species list, and
  justification.
* `planting_sites.geojson` — **only when `sites.candidates_csv` is set**: one
  point per bare patch in each top zone, with patch area, soil, matched
  species, and a ±5 m / "confirm before digging" note. This is the
  point-level "where exactly to plant" output.
* `recommendations.csv` — every zone with score components (predicted deltas
  are year-over-year, so seasonality doesn't masquerade as "worsening") plus
  `n_sites` and soil columns.
* `planting_brief.txt` — planner-readable brief: training summary, top zones
  with site coordinates and soil, species, lessons learned, caveats.
* `projection_UNVALIDATED_<N>mo.csv` — only with `--project N`.
* `models/memory.jsonl` — the full question/answer/error/lesson memory.

## Project layout

```
greenplan/
  adapters/    base.py (DataAdapter + 3 empty subclasses), mock.py, csvfile.py
  features/    h3grid.py (H3 aggregation), trends.py (Theil–Sen, features),
               sites.py (bare-land site finder), soil.py (SoilGrids + SMAP)
  training/    backtest.py (the loop), memory.py (JSONL store + retrieval)
  reasoning/   client.py (OpenRouter + mock model), prompts.py, species.py
  engine.py    orchestration — engine.run(cfg)
  cli.py       python -m greenplan ...
scripts/       esri_ndvi_to_greengrid.py, smap_to_greengrid.py, GEE exporters
config/city.yaml
data/          your exported CSVs (NDVI, sites, soil) — you create this
models/        memory.jsonl + retrain metadata (created at runtime)
outputs/       GeoJSON / CSV / brief (created at runtime)
```

## Known limitations (read before operational use)

* **Plantable space** is a placeholder proxy (1 − NDVI) **until you supply
  `sites.candidates_csv`**, after which it is real bare-ground area. Even then,
  site coordinates are ±5 m — confirm on a sub-metre basemap and on the ground
  before planting.
* **Soil pH/texture are SoilGrids *modelled* estimates** (250 m), not lab
  measurements; moisture is from SMAP. Good for screening, not a soil test.
* **The species table is a draft** — including the soil tolerance columns.
  Verify every row against a horticulture source (state Forest Department
  nursery lists, ICFRE guidance) — see the TODO in
  `greenplan/reasoning/species.py`.
* Long-range `--project` output is trend extrapolation, not a forecast; it is
  labeled UNVALIDATED for a reason.
