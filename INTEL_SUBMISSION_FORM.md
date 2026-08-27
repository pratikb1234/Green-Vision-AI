# Green Vision AI
### Intel® AI Global Impact Festival, AI Changemaker, Stage 3 (Global Final) FINAL FORM

Every claim below is measured by a script in this repository or is explicitly marked **in progress**. Nothing here is aspirational: Stage 3 claims are audited by Intel technologists, and this form is written to survive that audit.

---

## Project Title

**Green Vision AI**: the city map that shows where to plant trees before it's too late.

---

## Project Synopsis (about 150 words)

Green cover is disappearing fast in Indian cities and cities worldwide. It isn't just about looks: fewer trees means concrete traps more heat. Streets run up to 6°C hotter, India has counted over 6,100 heat deaths since 2010 (WRI, 2019), and 66 of the world's 100 most polluted cities are Indian (IQAir, 2025). Planting the right trees in the right places fights both problems at once. The roadblock is always the same: *where?*

Green Vision answers that. It merges 42 months of NASA satellite history with air quality, weather and soil data across 146 hexagonal cells of Ahmedabad, then predicts each cell's air quality within ±13.5 AQI points and its green cover within about 4% canopy a year ahead, tested on months the model never saw. Maps show the past. Green Vision predicts the future.

Proven in Ahmedabad and ready anywhere, since Delhi took one config file. It all runs on an ordinary laptop through Intel OpenVINO, on free public data.

---

## Created as part of an Intel® Program

**Yes, Intel® AI for Youth.** Its Responsible AI framework is the reason our system badges every number MEASURED, MODELLED or PROJECTED, and the reason all AI inference stays on the user's own machine.

---

## GenAI Tool Usage
*(Tick the same tier as Stage 2.)*

We'll be straight about this, because the honest version is also the more interesting one. The code was written with **Claude Code**, working under our direction. What is entirely ours is everything that made the code worth writing: the idea, the hexagonal grid design, choosing and verifying every data source, the decision to make three different AI approaches compete on the same held-out test and deploy only the winner, the honesty rules (a 23 month ceiling on what the system may call a prediction, and the badges on every statistic), and every accuracy number on this form. We treated GenAI exactly the way we want planners to treat Green Vision: as a powerful tool whose output you verify, constrain, and take responsibility for. We can explain any part of the system a judge asks about.

GenAI is also inside the product, and not as a wrapper around a cloud API. A local language model (Qwen2.5, compressed to INT4, running through the Intel OpenVINO GenAI API with no cloud and no key) writes the plain language reason for each recommended cell and picks 3 to 5 tree species matched to that cell's measured soil pH, texture and pollution. It is only allowed to restate numbers the app has already computed, enforced by strict JSON validation: it cannot invent data, reorder our ranking, or make up a species. And when a small model garbles its formatting anyway (they do), the system falls back to a deterministic writer instead of failing. We learned that from a real crash, and the fix is in the repo.

---

## Target Audience

Green Vision is built for the people who decide where trees go: municipal corporations, planning departments, and smart city cells. It arrives at exactly the right moment. In July 2026 the Ahmedabad Municipal Corporation launched **Mission Five Million Trees**, a public commitment to plant 50 lakh trees in 2026 to 27. That mission's hardest operational question is the one we answer: *which areas first, and where inside them is the actual plantable ground*, ranked by predicted decline and located to 10 metres from Sentinel-2 imagery.

We are putting the tool into the mission's hands right now: a live demonstration to **AMC's horticulture wing** and a pilot with a partner NGO, siting a real plantation drive with our priority map, are **in progress**, with written statements to follow.

A planner needs no GIS training, no GPU and no cloud budget. The models are compressed with Intel's NNCF to run on an ordinary office laptop, every data source is free, and nothing about the city ever leaves the machine, which matters a great deal when the user is a government. And because the crews who actually plant the trees mostly do not read English, every run now writes the planting brief in **English, Hindi and Gujarati**, translated by hand into templates rather than by a machine, because a wrong translation on a work order is worse than none.

The design is city agnostic. We stood up **Delhi as a second city from a single config file**, with its own accuracy test rather than Ahmedabad's borrowed numbers. Every data source has global coverage, so any city on Earth is one bounding box away. The benefit reaches furthest to the people hit hardest: those who work outdoors and can least afford cooling.

---

## Datasets (Yes, about 175 words)

**What it is.** Everything about the city's environment: vegetation, air, weather, soil and infrastructure, merged onto 146 hexagonal cells (Uber's H3 grid) so every stream lines up cell for cell.

**Where it comes from.** NASA's MOD13Q1 vegetation index is the backbone (250 m, one reading every 16 days, 42 months). Open-Meteo's archive gives matching monthly air quality. ISRIC SoilGrids gives soil pH and texture per cell. Sentinel-2 (10 m, via AWS Earth Search) finds the bare ground inside each cell. OpenStreetMap and Esri fill in live conditions. All free, all public, no API keys anywhere.

**Why these.** Forecasting decline needs a genuine past to learn from, and MOD13Q1 is the longest free, scientifically validated record at this resolution. Free data means the data cost of standing up a new city is zero, for any municipality on Earth.

**What it solves.** The merged history is what our forecaster learns from, tests itself against, and turns into a ranked, explained, soil aware planting plan. None of it is personal data. There are no accounts, no logins, nothing about people at all.

---

## Project Stage

**A working prototype doing real AI inference**, with a measured number behind every claim:

- **Accuracy:** mean error of ±13.5 AQI points and 0.042 NDVI (about 4% canopy) on held-out months, beating a strong statistical baseline (skill +0.08).
- **The bake-off:** we tested a statistical forecaster, a local LLM, and trained networks (a neural net and a random forest) on identical held-out tasks. The statistical forecaster won. The LLM scored skill of minus 0.33 and the networks minus 1.41, and they are still in our repository next to their scores, because deleting your losers is how you end up fooling yourself. If a future challenger ever wins on the evidence, the system deploys it automatically.
- **Reliability:** 0 malformed replies in 30 validation runs of the report engine, and a deterministic fallback that keeps every run alive if formatting breaks.
- **Speed:** the model loads in about 2.5 s and writes a cell report in about 4 s on a plain laptop CPU. One config word retargets the identical build to an Intel Core Ultra NPU; that benchmark is **in progress** (`scripts/bench_devices.py`, procedure in `docs/intel/npu-benchmark-runbook.md`).
- **Real world use:** AMC and NGO sessions are **in progress** (see Target Audience).
- **Honesty by construction:** with 42 months of data the system computes that 23 months is the longest forecast it can actually verify, and it refuses to call anything beyond that a prediction. Longer projections are stamped UNVALIDATED in the output file itself. We even built a 10 m land cover classifier to screen planting sites, and when it failed its own audit gate (0.641 agreement where it demands 0.711), the code benched it and said so in the planting brief. That is what our error handling looks like.

---

## Intel Technologies (Yes)

Intel is not a sticker on this project. It is the reason the project can keep its promises: free, local, private.

**Across the lifecycle.** In *modeling and evaluation*, our challenger models (a neural network and a random forest, built in scikit-learn, with an `--intel` flag that dispatches accelerated estimators to Intel oneDAL via scikit-learn-intelex) export to ONNX and run through the **OpenVINO runtime**, inside a harness that re-tests them on held-out months at every run and records the measured OpenVINO versus scikit-learn output parity. In *optimization*, we compress models ourselves with **Intel NNCF**: our script takes any Hugging Face model to INT4, measured at FP16 to 343 MB in under five minutes on a laptop CPU. In *deployment*, the **OpenVINO GenAI API** runs the language model fully on device (INT4, 893 MB, no network, no key) for every generated report, and a single config word (`device: CPU | GPU | NPU`) retargets the identical build.

**Hardware, floor to ceiling, benchmarks in progress.** The floor is any Intel CPU a government office already owns. The ceiling is the Core Ultra AI PC now entering procurement, where the same build runs on the NPU with zero code changes. Our per device benchmark suite (`scripts/bench_devices.py`: load time, time to first token, throughput, strict JSON success) ships in the repository with step by step evidence runbooks (`docs/intel/`), and the measured i5 and Core Ultra tables will accompany our demonstration.

**Why Intel.** Our users are government offices with ordinary laptops, not GPU servers. OpenVINO is the single reason that "runs free, runs local, runs private" is a fact and not a wish.

**Program.** Built through Intel® AI for Youth, applying its Responsible AI framework throughout.

---

## Responsible AI
*(Tick: Enable Human Oversight · Enable Transparency and Explainability · Advance Security, Safety and Reliability · Design for Privacy · Protect the Environment)*

**Human oversight.** Green Vision recommends. Planners decide. It never acts on its own.

**Transparency.** Every recommendation comes with a plain language reason, and every statistic wears a badge, MEASURED, MODELLED or PROJECTED, so an official always knows which numbers they can defend in public.

**Reliability, the hard way.** Three model families competed on held-out months and only the measured winner deployed; the losers stay in the repo with their scores. Our land cover classifier benched itself when it failed its own audit. The 23 month validation ceiling is enforced by the code, not by our good intentions.

**Bias.** The model is calibrated on Ahmedabad, so everywhere else counts as unvalidated until it is locally re-tested. Delhi runs its own backtest and ships its own skill number.

**GenAI risk.** The report writer can only restate computed values, in all three languages. Hallucination is blocked by construction, and formatting failures degrade to a deterministic writer instead of crashing.

**Privacy.** No personal data exists anywhere in the system, and all AI inference runs on the user's own machine.

**Environment.** NNCF compression means no cloud GPU. A tool for planting trees should not burn a data centre.

**Still on our list.** We said at the regional round that our traffic light map is hard to read for colour blind users, and it still is. Patterns and text labels are planned. We would rather tell you than hide it.

---

## SDG Alignment

**Primary: SDG 13, Climate Action.** The innovation is prediction. Existing tools observe green cover loss after it happens. Green Vision forecasts it, so a city can act before the trees are gone. And it scales the way non AI approaches cannot: tree surveys and GIS consultants cost lakhs per city and take months, while free satellite data plus a laptop delivers ranked, explained priorities in seconds, for any city on Earth. Progress is measurable by design, with predicted AQI and canopy change per cell re-audited monthly against new satellite passes and 25 year canopy and carbon projections for planted cells. The first measurement in the ground is under way: a partner NGO's plantation drive sited with the tool is **in progress**, inside a municipality publicly committed to five million trees this year.

Also aligned: **SDG 11** (Sustainable Cities and Communities), **SDG 15** (Life on Land), **SDG 9** (Industry, Innovation and Infrastructure).

---

## Sources, References and Citations

NASA MOD13Q1 Vegetation Indices, LP DAAC / ORNL DAAC · Open-Meteo Weather and Air Quality APIs · ISRIC SoilGrids v2.0 · ESA WorldCover 2021 v200 · OpenStreetMap contributors (Overpass API) · Esri World Imagery · Copernicus Sentinel-2 L2A via AWS Earth Search · World Resources Institute (2019), urban heat island analysis · IQAir World Air Quality Report (2025) · Ahmedabad Municipal Corporation, "Mission Five Million Trees" launch, July 2026 · Intel OpenVINO toolkit, OpenVINO GenAI API and NNCF (github.com/openvinotoolkit) · scikit-learn and scikit-learn-intelex · Uber H3 (h3geo.org) · Gitelson et al., VARI vegetation index · **Project source: github.com/pratikb1234/GreenVision** (submission) · github.com/Ghost-King-2013/Green-Vision-AI (development)

---
---

# BEFORE SUBMITTING, verify these

| # | Item | Status |
|---|---|---|
| 1 | GenAI tier ticked matches Stage 2; tools list says Claude Code | VERIFY against the Stage 2 form |
| 2 | AMC demonstration and NGO drive | IN PROGRESS wording is in the form; when written statements exist, replace those paragraphs with names, dates and verbatim quotes |
| 3 | Intel i5 and Core Ultra NPU benchmarks | IN PROGRESS wording is in the form; when `EVIDENCE.md` exists, add the measured numbers |
| 4 | Do not mention Intel Geti, Tiber, or OPEA as things we used | Verified 2026-08-27: Geti hosted trial is gone, Tiber URLs are dead, OPEA honestly not used (docs/intel/) |
| 5 | Hindi and Gujarati briefs | SHIPPED, greenplan/i18n.py, written every run |

Rule: a claim with no measurement behind it gets deleted, not guessed.
