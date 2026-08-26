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

### Measured: the numeric model bake-off

Same held-out task for every contender — predict a cell's metrics 12 months
past a cutoff, seeing only history strictly before it; skill is measured
against a Theil–Sen + seasonality baseline on the same tasks. Real 146-cell
panel:

| | Statistical forecaster (trend + season + memory) | Qwen2.5-1.5B INT4 (local LLM) | Trained MLP (`greenplan.forecast`) | Trained forest (`--model rf`, 200 trees) |
|---|---|---|---|---|
| MAE — AQI | **13.5** | 33.3 | 41.3 | 29.0 |
| MAE — NDVI | **0.042** | 0.103 | 0.173 | 0.032 |
| Skill vs baseline | **+0.075 → +0.084** | −0.385 → −0.333 | −1.41 | −0.50 |
| `memory_helped` | true | false | n/a | n/a |
| Train time (stock sklearn, M3 Pro) | — | — | 0.5–0.7 s | 0.4–1.1 s |

**Read this before quoting any accuracy number.** Each contender's MAE is
scored on its own held-out tasks, so MAEs are not comparable across columns —
the forest's 0.032 NDVI does **not** beat the champion's 0.042. The
comparable number is **skill**: model error vs the same Theil–Sen +
seasonality baseline on the same tasks, and every trained challenger is
negative. A 1.5B model is *worse than a trend line* at numeric extrapolation,
and no amount of prompting fixes that. A neural network trained on the panel
is worse still — with 42 months of history, the residuals left by the robust
baseline are dominated by city-wide shocks (weather, season anomalies) that
repeat too few times to learn; a ridge probe on zone-relative residuals
confirms there is no spatial signal hiding either (−0.03). A 200-tree random
forest on the identical residual task (`--model rf`) recovers most of the
MLP's deficit — NDVI lands within noise of the baseline (−0.05) — but the
combined score still loses to the trend line, so it is published here and
benched by the same gate. The statistical forecaster's memory loop, by
contrast, genuinely works: skill improves +0.05 from the first half of
training to the second.

So the **default provider is `hybrid`**: numbers from the measured champion,
words — per-cell justifications, species picks with soil fit, projection
caveats — from the local LLM, which produced 0 malformed replies in 30
iterations at exactly that job. Champion selection is re-checked from
evidence on every run: `python -m greenplan.forecast.train` scores a trained
challenger (the MLP, or a random forest with `--model rf`) on held-out months
and writes its report; the day a longer panel lets one report positive skill,
`hybrid` deploys it automatically — the challenger harness (ONNX export,
OpenVINO inference with measured sklearn parity, `CPU`/`GPU`/`NPU` via the
same `model.device` knob) is production-ready and waiting for data.

Two measured notes on that harness. skl2onnx converts a forest to a single
`ai.onnx.ml.TreeEnsembleRegressor` op, which OpenVINO's ONNX frontend cannot
convert ("No conversion rule found", 2026.3) — so `onnx_trees.py` lowers the
fitted trees to standard ONNX ops (Gather/Where/ReduceMean) that every
OpenVINO device executes. And on ARM Macs OpenVINO's CPU plugin defaults to
f16 inference, which flips tree splits near a threshold (measured: 0.05 max
error); both the trainer and `OVForecaster` pin f32, after which the exported
graphs match sklearn to 2.1e-07 (forest) and 9.5e-07 (MLP) on the held-out
set — the parity number is re-measured and written into `report.json` on
every training run.

### Intel Extension for Scikit-learn: measured, not assumed

The trainer takes `--intel`, which calls `sklearnex.patch_sklearn()` before
any sklearn import so accelerated estimators dispatch to Intel oneDAL, and
`report.json` records whether the patch was actually active next to the
measured train time — re-running the same command with and without the flag
*is* the acceleration measurement:

```bash
python -m greenplan.forecast.train --config config/city.yaml --model rf
python -m greenplan.forecast.train --config config/city.yaml --model rf --intel
```

What we can state honestly today:

- **The MLP challenger gains nothing from sklearnex by design.**
  `MLPRegressor` is not on the extension's accelerated-estimator list (no
  neural network is — checked against the sklearnex 2026.1 docs), so for
  `--model mlp` the patch is a documented no-op. That is exactly why the
  forest challenger exists: `RandomForestRegressor` **is** on the list, so
  `--model rf --intel` exercises oneDAL for real.
- **The timings in the table are stock scikit-learn.** This repo's dev
  machine is an Apple-Silicon Mac, and scikit-learn-intelex ships wheels for
  Windows/Linux x86_64 (Python ≤ 3.13) only — `pip install
  scikit-learn-intelex` here returns "No matching distribution found"
  (measured, recorded in `requirements.txt`). The `--intel` flag logs that
  and falls back to stock sklearn rather than pretending. An
  oneDAL-accelerated timing requires the same two commands on an Intel
  x86_64 machine; until someone runs them, no speedup number is claimed —
  and at about a second of stock train time on this panel there is little to
  accelerate anyway. The flag will matter, if ever, on a much longer panel.

### Compress a model yourself (NNCF)

`fetch_openvino_model.py` downloads models Intel already compressed. To do
the compression locally — any Hugging Face instruct model → INT4 OpenVINO IR
via Intel's NNCF — use:

```bash
pip install "optimum[openvino]"
python scripts/compress_model_nncf.py --base Qwen/Qwen2.5-0.5B-Instruct --verify
```

Measured on this repo: Qwen2.5-0.5B FP16 → INT4 in 292 s on a laptop CPU,
343 MB on disk, loads and answers in ~6 s. No Intel hardware is needed to
*run* the compression; the result runs on any OpenVINO device.

### Benchmark every OpenVINO device on this machine

```bash
python scripts/bench_devices.py
```

Enumerates the devices the runtime can see, loads the same model on each, and
prints load time, time-to-first-token, throughput, and whether the reply
parsed as strict JSON. On a plain laptop that is one CPU row; on an Intel
Core Ultra "AI PC" the same command benchmarks the integrated GPU and NPU
with zero code changes — `model.device` passes straight through.

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

### Sentinel-2 at 10 m: where inside a cell

MODIS (250 m) is the temporal backbone — long, keyless, pre-composited. What
it cannot say is where INSIDE a ~5 km² priority cell the bare ground actually
is. `scripts/sentinel2_ndvi_export.py` adds that: true NDVI (near-infrared vs
red) from the most recent cloud-free **Sentinel-2 L2A** scene at **10 m**,
via AWS's Earth Search STAC API — still no key, still free.

```bash
python scripts/sentinel2_ndvi_export.py --config config/city.yaml \
    --out-cells data/ahmedabad_s2_ndvi.csv --out-sites data/ahmedabad_sites.csv
```

`--out-sites` writes the lowest-NDVI 10 m ground per cell as candidate
planting sites; `sites.candidates_csv` in the config points at it, replacing
the `1 - NDVI` proxy with measured bare ground (measured for Ahmedabad:
2,190 candidates from a 3.9 %-cloud scene). Honest limits, in the file
header itself: one scene is a snapshot, not a composite, and low NDVI can be
water/rock/roofs at scene time — candidates for a human to verify, not a
survey. The two-scale rule: **MODIS answers "how is this cell changing",
Sentinel-2 answers "where in it is the ground".**

### A second city is one config file

Everything downstream of `config/<city>.yaml` is city-agnostic. Standing up
Delhi:

```bash
# copy config/city.yaml -> config/delhi.yaml, change name + bbox + csv paths
python scripts/openmeteo_aqi_export.py  --config config/delhi.yaml --start 2023-01 --end 2026-06 --out data/delhi_aqi.csv
python scripts/soilgrids_export.py      --config config/delhi.yaml --out data/delhi_soilgrids.csv
python scripts/modis_ndvi_export.py     --config config/delhi.yaml --start 2023-01 --end 2026-06 --out data/delhi_ndvi.csv
python scripts/sentinel2_ndvi_export.py --config config/delhi.yaml --out-sites data/delhi_sites.csv
python -m greenplan run --config config/delhi.yaml --recommend
```

Every data source is global (NASA, Open-Meteo, SoilGrids, OSM, Sentinel-2),
so this recipe works for any city on Earth with a bounding box — and each
city's `run` backtests itself, so each city ships with its own measured
skill, never Ahmedabad's borrowed one.

### Still not wired up

- **Soil moisture.** `soil.moisture_csv` stays commented out: NASA SMAP needs
  an Earthdata login, which breaks the no-key property.
- **10 years of history.** Not possible from free keyless sources. Open-Meteo's
  air-quality archive returns nothing before **2023-01** (verified back to
  2013), which caps the panel at 42 months regardless of how far MODIS goes
  back. Any claim of a decade of data is wrong. (Sentinel-2's archive reaches
  2015 but needs cloud compositing per month — the honest route to a longer
  panel, not yet built.)

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
