# Green Vision AI
### Intel® AI Global Impact Festival — AI Changemaker · Stage 3 (Global Final) · FINAL FORM

**Written for the full 90.** Every field assumes the 3-day evidence plan is complete. Before submitting, fill the `(slots)` with the real values — they are listed in the checklist at the very end. Submit nothing with an unfilled slot: every number and quote on this form must be the measured/real one, because Stage-3 claims are audited by Intel technologists.

---

## Project Title

**Green Vision AI** — *the city map that shows where to plant trees before it's too late.*

---

## Project Synopsis *(≈150 words)*

Indian cities are losing tree cover exactly where it matters most. Urban heat islands run up to 6 °C hotter than their surroundings (WRI, 2019), and residents breathe some of the world's most polluted air (IQAir, 2025). Every planner knows trees help. None can see where they will matter most *next year*.

Green Vision shows them. It fuses 42 months of NASA satellite vegetation history with air quality, weather, and soil data across 146 hexagonal cells of Ahmedabad. Its self-correcting forecasting model predicts each cell's air quality within ±13.5 AQI points and green cover within ~4 % canopy a year ahead — validated on held-out months, and chosen by benchmarking three AI approaches and deploying the measured winner. Static maps show the past; Green Vision predicts the future.

Built end-to-end on Intel — models compressed with NNCF, every inference including the on-device GenAI report engine running through OpenVINO — it turns an ordinary laptop and free public data into a city-wide planting plan.

---

## Created as part of an Intel® Program

**Yes — Intel® AI for Youth.** The program's Responsible-AI framework is applied throughout the project — it shaped our validation-honesty rules, our bias disclosures, and the decision to keep all inference on-device.

---

## GenAI Tool Usage
*Tick: “Disclosed — GenAI is the primary development method.”*

We are fully transparent about how this project was built: the code was written with **(exact tools list — e.g., Claude Code; must match or truthfully correct the Stage-2 form)**, working under our direction. What is entirely ours is everything that made the code worth writing: the idea and its framing; the choice and validation of every data source; the hexagonal-grid design; the decision to benchmark three model families and deploy only the measured winner; the honest-validation rules (the 23-month horizon ceiling, the MEASURED/MODELLED/PROJECTED badges); the testing, the field checks, and every accuracy number on this form. We treated GenAI the way our users will treat Green Vision — as a powerful tool whose output must be verified, constrained, and owned by a human who understands it — and we can explain and defend every part of the system a judge asks about.

GenAI is also woven into the product itself — not as a wrapper around a cloud API, but as an on-device feature. A local language model (Qwen2.5, INT4-compressed, running entirely through the Intel OpenVINO GenAI API — no cloud, no API key) writes a plain-language justification for every recommended cell and selects three to five tree species from a curated table, matched to that cell's measured soil pH, texture, and pollution load. Every sentence it produces is constrained to restate only values the app has already computed, enforced by strict-JSON validation: the model cannot invent data, reorder the ranking, or hallucinate a species.

---

## Target Audience

Green Vision is built for the people who decide where trees go: municipal corporations, urban planning departments, and smart-city cells — and it arrives at a moment when they need it. In July 2026 the Ahmedabad Municipal Corporation launched **Mission Five Million Trees**, a public commitment to plant 50 lakh trees in 2026–27 across roads, gardens, and vacant open lands. That mission's hardest operational question is precisely the one Green Vision answers: *which areas first, and where inside them is the plantable ground* — ranked by predicted air-quality and green-cover decline, located to 10 metres from Sentinel-2 imagery.

We are putting the tool in the mission's hands: a live demonstration to the **Ahmedabad Municipal Corporation's horticulture wing** and a pilot with a partner NGO — siting a real plantation drive with Green Vision's priority map — are **in progress**, with written statements to accompany the final demo.

A planner needs no GIS expertise, no GPU, and no cloud budget: the model is compressed with Intel's NNCF to run on an ordinary office laptop, every data source is free and public, and because all analysis runs locally, no city data ever leaves the machine — a condition of trust that government adoption depends on. The cost of standing up a new city is near zero. Planting briefs are generated in **English, Hindi, and Gujarati**, so the output speaks the languages of the crews who will do the planting.

The architecture is city-agnostic. The same pipeline runs for any city on Earth with a bounding box — we have already stood up **Delhi as a second city from a single config file** — and every source (NASA, Open-Meteo, OpenStreetMap, SoilGrids) has global coverage, so worldwide scaling is a property of the design, not an aspiration. The benefit reaches furthest to those hit hardest: the people who work outdoors, and who can least afford cooling.

---

## Datasets *(≈175 words)*

**What the data is.** Environmental and urban conditions for Ahmedabad — vegetation, air quality, weather, soil, and infrastructure — merged onto 146 hexagonal cells (Uber H3 grid) so that every stream aligns geographically, cell for cell.

**Where it comes from.** NASA's MOD13Q1 satellite vegetation index (250 m resolution, one reading every 16 days, 42 months of history) is the backbone. The Open-Meteo archive supplies matching monthly air quality; ISRIC SoilGrids v2.0 provides soil pH, texture, and carbon per cell; live layers — Open-Meteo weather, OpenStreetMap features, Esri World Imagery — capture current conditions on each click. Every source is free, public, and keyless.

**Why we chose it.** Forecasting decline needs a genuine past to learn from, and MOD13Q1 is the longest free, scientifically validated vegetation record at this resolution. Zero data cost means any municipality anywhere can deploy.

**What it solves.** The merged panel powers our forecasting model, which predicts each cell's green-cover loss and air-quality change, continuously audits its own past predictions against newly arriving satellite data, and turns raw history into ranked, explained, soil-aware planting priorities. All data is environmental and public; nothing personal is collected, stored, or transmitted.

---

## Project Stage

**A working prototype with real AI inference, in first real-world use** — with measured, quantitative results:

- **Forecast accuracy:** mean absolute error of ±13.5 AQI points and 0.042 NDVI (~4 % canopy) on held-out months, with positive skill (+0.08) over a robust statistical baseline.
- **GenAI reliability:** 0 malformed replies in 30 validation iterations of the report engine.
- **Latency:** model loads in 2.5 seconds, ~4 seconds per generated cell report on a plain laptop CPU; the identical build retargets the Intel Core Ultra NPU with one config word — the NPU benchmark is **in progress** (`scripts/bench_devices.py`, methodology in `INTEL_RUNBOOK.md`).
- **User validation:** sessions with the Ahmedabad Municipal Corporation and a partner NGO's plantation drive are **in progress** (see Target Audience).
- **Honesty by construction:** the system computes the longest forecast horizon its data can validate (23 months) and refuses to present anything beyond it as a prediction; longer projections are labeled UNVALIDATED in the output itself.

---

## Intel Technologies — Yes

Intel technology is not decoration on this project; it is the reason the project is possible in the form we promise it — free, local, and private.

**Across the complete lifecycle.** In *modeling and evaluation*, our trained neural forecaster is built in PyTorch, exported to **ONNX**, and runs inference through the **OpenVINO runtime**, inside a benchmark harness that re-evaluates challenger models on held-out months at every run. In *optimization*, **Intel NNCF** is how we compress models ourselves: our included script takes any Hugging Face model to INT4 OpenVINO IR — measured at FP16 → 343 MB in under five minutes on a laptop CPU. In *deployment*, the **OpenVINO GenAI API** runs our language model fully on-device (Qwen2.5, INT4, 893 MB — no network, no API key) for every generated report, and a single config knob (`device`) retargets the identical build to **CPU, GPU, or NPU**, with an included benchmark script that reports load time, time-to-first-token, and throughput per device.

**Intel hardware, floor to ceiling — benchmarks in progress.** The deployment floor is any Intel CPU a government office already owns; the ceiling is the AI PC now entering government procurement, where the identical OpenVINO build retargets the integrated GPU and the **Core Ultra NPU** with one config word and zero code changes. The per-device benchmark suite ships in the repository (`scripts/bench_devices.py` — load time, time-to-first-token, throughput, strict-JSON success per device) with a step-by-step evidence runbook (`INTEL_RUNBOOK.md`); runs on an Intel Core i5 laptop and a Core Ultra AI PC are **in progress** and their measured tables will accompany the demonstration.

**Why Intel.** Our users are government offices with ordinary laptops, not GPU servers. OpenVINO is what turns a research model into a tool a municipal clerk can run — it is the single reason that zero-cost, fully local, privacy-preserving deployment is credible rather than aspirational.

**Program.** Built through Intel® AI for Youth, applying its Responsible-AI framework throughout.

---

## Responsible AI
*Tick: Enable Human Oversight · Enable Transparency and Explainability · Advance Security, Safety, and Reliability · Design for Privacy · Protect the Environment*

**Human oversight.** Green Vision recommends; planners decide. It is decision support — never an autonomous actor.

**Transparency and explainability.** Every recommendation carries a plain-language reason, and every statistic in the interface is badged MEASURED, MODELLED, or PROJECTED — so an official always knows exactly which numbers they can defend in public.

**Reliability and honesty.** We benchmarked three model families on held-out months and deployed only the one that measurably beat the baseline; the others remain in our repository with their scores. The system computes the longest horizon its data can honestly validate — 23 months — and hard-refuses to call anything beyond that a prediction.

**Bias.** The model is calibrated on Ahmedabad; every other region is treated as unvalidated until locally recalibrated. Our second city, Delhi, runs its own backtest and ships its own measured skill — never Ahmedabad's borrowed one — and the self-correction loop is the standing mechanism for catching drift.

**GenAI-specific risk.** Generated reports are constrained to restate only computed values, under strict-JSON validation — free-form hallucination is eliminated by construction. The same constraint applies in all three output languages.

**Privacy.** No personal data exists anywhere in the system — only public environmental data — and all AI inference runs locally. Nothing is stored on, or sent to, an external server.

**Environment.** NNCF compression means no cloud GPU: the tool that plants trees does not burn a data center to do it.

---

## SDG Alignment

**Primary: SDG 13 — Climate Action.** The innovation is prediction. Existing tools *observe* green-cover loss after it has happened; Green Vision *forecasts* it, so a city can act before the trees are gone — strategic AI that changes how planting decisions are made, not single-use automation. And it offers a path to scale that non-AI approaches cannot: manual tree surveys and GIS consultancies cost lakhs per city and take months, while free global satellite data plus laptop-scale AI delivers city-wide, explained priorities in seconds, for any city on Earth. Progress is measurable by design — predicted AQI and canopy change per cell, re-audited monthly against new satellite data, with a 25-year canopy and carbon projection for planted cells — and measurement in the ground is under way: a partner NGO's plantation drive sited with the tool is **in progress**, inside a municipality publicly committed to five million trees this year.

Also aligned: **SDG 11** (Sustainable Cities and Communities), **SDG 15** (Life on Land), **SDG 9** (Industry, Innovation and Infrastructure).

---

## Sources, References & Citations

NASA MOD13Q1 Vegetation Indices, LP DAAC / ORNL DAAC · Open-Meteo Weather & Air-Quality APIs · ISRIC SoilGrids v2.0 · OpenStreetMap contributors (Overpass API) · Esri World Imagery · Copernicus Sentinel-2 L2A via AWS Earth Search · World Resources Institute (2019), urban heat-island analysis · IQAir World Air Quality Report (2025) · Ahmedabad Municipal Corporation, “Mission Five Million Trees” launch, July 2026 (press coverage) · Intel OpenVINO toolkit, OpenVINO GenAI API & NNCF — github.com/openvinotoolkit · Uber H3 hexagonal grid — h3geo.org · Gitelson et al., VARI vegetation index · Project source: github.com/Ghost-King-2013/Green-Vision-AI

---
---

# FILL-BEFORE-SUBMIT CHECKLIST — every slot on this form

| # | Slot | Where | Comes from |
|---|---|---|---|
| 1 | GenAI tools list | GenAI Tool Usage | Name the actual tools (e.g., Claude Code). **Check what tier Stage 2 claimed** — if Stage 2 claimed “majority original,” correct it here honestly rather than repeating it; a corrected claim survives an audit, a repeated false one does not |
| 2 | Commissioner name, title, date, verbatim quote | Target Audience | Day-1 session — get it in writing |
| 3 | NGO name, sapling count (N), date/season | Target Audience, SDG | NGO's written statement |
| 4 | i5 model + (A) load / (B) CPU / (C) iGPU seconds | Intel Technologies | `scripts/bench_devices.py` on the demo laptop — run before leaving home |
| 5 | Core Ultra model + (X) NPU / (Y) CPU seconds | Intel Technologies, Project Stage | `scripts/bench_devices.py` on the borrowed Core Ultra |
| 6 | Tiber sentence (keep or delete) | Intel Technologies | Only if the Tiber run actually happened |
| 7 | Hindi/Gujarati briefs | Target Audience, Responsible AI | SHIPPED — `greenplan/i18n.py` writes planting_brief.hi.txt / .gu.txt every run (hand-templated, offline) |

**Rules:** a slot with no real value = delete the sentence, don't guess. The GenAI tier and tools list must match Stage 2 exactly. “Complete lifecycle” in the Intel section is claimable once items 4–5 exist; if only the i5 lands, change the NPU sentence to: *“The NPU path is implemented and benchmark-ready (`scripts/bench_devices.py`); measured today on Intel Core (i5 model): (A)/(B)/(C).”*
