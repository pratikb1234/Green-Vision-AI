"""Sentinel-2 NDVI at 10 m for a city — free, NO API key.

MODIS (250 m, 42-month archive) stays the engine's TEMPORAL backbone: it is
pre-composited, keyless, and long enough to forecast from. What it cannot say
is where INSIDE a ~5 km2 priority cell the bare ground actually is. This
script adds that spatial layer: true NDVI (B08 near-infrared vs B04 red) from
the most recent cloud-free Sentinel-2 scene, 625x sharper than MODIS.

Keyless access is what makes this possible now: AWS's Earth Search STAC API
(https://earth-search.aws.element84.com/v1, Element 84) serves Sentinel-2 L2A
as cloud-optimized GeoTIFFs — no Copernicus account, no token. rasterio
reads just the bbox window over HTTPS, so a whole city costs a few tens of MB.

Two outputs:
  1. --out-cells:  zone,lat,lon,month,ndvi — current per-H3-cell mean NDVI on
     the SAME grid as the engine (one snapshot month, for comparison layers).
  2. --out-sites:  lat,lon,patch_area_m2 — the lowest-NDVI 10 m ground inside
     each cell: candidate planting sites for `sites.candidates_csv` in
     config/city.yaml, replacing the 1-NDVI proxy with measured bare ground.

Usage:
    python scripts/sentinel2_ndvi_export.py --config config/city.yaml \
        --out-cells data/ahmedabad_s2_ndvi.csv \
        --out-sites data/ahmedabad_sites.csv

Honest limits, stated: one scene is a snapshot, not a composite — pass
--max-cloud (default 10%%) and the script picks the most recent scene under
it; scene date is printed and written into the header. NDVI < the bare
threshold can also be water/rock/roofs at scene time; sites are candidates
for a human to verify, not survey results.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenplan.config import load_config  # noqa: E402
from greenplan.features.h3grid import cell_center, cells_covering_bbox  # noqa: E402

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"


def find_scene(bbox: list[float], max_cloud: float) -> dict:
    """Most recent Sentinel-2 L2A scene covering the bbox under the cloud cap."""
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 12,
    }
    req = urllib.request.Request(
        STAC_SEARCH,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.load(r)["features"]
    if not feats:
        raise SystemExit(
            f"no Sentinel-2 scene with cloud cover < {max_cloud}% over this bbox; "
            "raise --max-cloud"
        )
    # prefer the scene whose footprint contains the bbox centre (UTM tiles
    # overlap at edges); fall back to the newest
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for f in feats:
        fb = f["bbox"]
        if fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]:
            return f
    return feats[0]


def read_band(href: str, bbox: list[float]) -> tuple[np.ndarray, object]:
    """Windowed read of one COG band over the lon/lat bbox."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(l, b, r, t, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        return data, (transform, src.crs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentinel-2 10 m NDVI export (keyless)")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--out-cells", default=None, help="per-H3-cell mean NDVI csv")
    ap.add_argument("--out-sites", default=None, help="low-NDVI candidate sites csv")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--bare-ndvi", type=float, default=0.15,
                    help="NDVI below this counts as bare, plantable ground")
    ap.add_argument("--sites-per-cell", type=int, default=15)
    args = ap.parse_args()
    if not (args.out_cells or args.out_sites):
        ap.error("nothing to do: pass --out-cells and/or --out-sites")

    cfg = load_config(args.config)
    bbox = list(cfg.city.bbox)
    res = cfg.grid.h3_resolution

    scene = find_scene(bbox, args.max_cloud)
    date = scene["properties"]["datetime"][:10]
    cloud = scene["properties"].get("eo:cloud_cover")
    print(f"scene {scene['id']}  date {date}  cloud {cloud:.1f}%")

    red, _ = read_band(scene["assets"]["red"]["href"], bbox)
    nir, georef = read_band(scene["assets"]["nir"]["href"], bbox)
    transform, crs = georef
    h = min(red.shape[0], nir.shape[0])
    w = min(red.shape[1], nir.shape[1])
    red, nir = red[:h, :w], nir[:h, :w]

    valid = (red > 0) & (nir > 0)  # 0 = L2A nodata
    ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
    print(f"grid {w}x{h} px @10m — mean NDVI {np.nanmean(ndvi):.3f}")

    # geolocate every pixel centre once (rows/cols -> lon/lat)
    from rasterio.warp import transform as warp_transform

    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lons, lats = warp_transform(crs, "EPSG:4326", xs.ravel(), ys.ravel())
    lons = np.asarray(lons).reshape(h, w)
    lats = np.asarray(lats).reshape(h, w)

    # assign pixels to H3 cells via each cell's bounding box mask (fast, no
    # per-pixel python h3 calls); cells overlap boxes slightly at hex edges,
    # which shifts a mean by <1% — fine for a comparison layer.
    cells = cells_covering_bbox(tuple(bbox), res)
    print(f"{len(cells)} H3 res-{res} cells")

    import h3 as h3lib

    cell_rows = []
    site_rows = []
    for cell in cells:
        boundary = np.array(h3lib.cell_to_boundary(cell))  # (lat, lon) pairs
        la0, la1 = boundary[:, 0].min(), boundary[:, 0].max()
        lo0, lo1 = boundary[:, 1].min(), boundary[:, 1].max()
        mask = (lats >= la0) & (lats <= la1) & (lons >= lo0) & (lons <= lo1)
        vals = ndvi[mask]
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        lat_c, lon_c = cell_center(cell)
        cell_rows.append((cell, lat_c, lon_c, float(np.mean(vals))))

        if args.out_sites:
            bare = mask & (ndvi < args.bare_ndvi)
            by, bx = np.nonzero(bare)
            if len(by):
                order = np.argsort(ndvi[by, bx])[: args.sites_per_cell]
                for i in order:
                    site_rows.append(
                        (float(lats[by[i], bx[i]]), float(lons[by[i], bx[i]]), 100.0)
                    )

    header = (
        f"# Sentinel-2 L2A 10 m NDVI via AWS Earth Search STAC (no key).\n"
        f"# scene {scene['id']} date {date} cloud {cloud:.1f}%. "
        f"SNAPSHOT of one scene, not a composite.\n"
    )
    if args.out_cells:
        with open(args.out_cells, "w", encoding="utf-8") as f:
            f.write(header + "zone,lat,lon,month,ndvi\n")
            for cell, lat_c, lon_c, v in cell_rows:
                f.write(f"{cell},{lat_c:.5f},{lon_c:.5f},0,{v:.4f}\n")
        print(f"wrote {args.out_cells}: {len(cell_rows)} cell means")
    if args.out_sites:
        with open(args.out_sites, "w", encoding="utf-8") as f:
            f.write(header + "lat,lon,patch_area_m2\n")
            for lat, lon, area in site_rows:
                f.write(f"{lat:.6f},{lon:.6f},{area:.0f}\n")
        print(
            f"wrote {args.out_sites}: {len(site_rows)} candidate sites — set\n"
            f"  sites.candidates_csv: {args.out_sites}\nin config to use them"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
