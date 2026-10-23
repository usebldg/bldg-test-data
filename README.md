# bldg test data

Floor plan images with ground truth, and real BIM models, used to test and benchmark
[bldg](https://github.com/usebldg/bldg) (buildings AI agents can edit). Every file here may be used commercially,
under the licence of its folder ([LICENSE.md](LICENSE.md)).

| folder | samples | ground truth | licence |
|---|---:|---|---|
| [`synthetic/`](synthetic/) | 24 | walls, openings with full door swing, rooms, labels | CC0 1.0 (our own generator, [GENERATOR.md](synthetic/GENERATOR.md)) |
| [`swiss-dwellings/`](swiss-dwellings/) | 43 | walls, openings (no swing), rooms, stairs | CC BY 4.0, Archilyse AG ([LICENSE.md](swiss-dwellings/LICENSE.md)) |
| [`wikimedia-commons/`](wikimedia-commons/) | 3 | none (images only) | public domain, CC BY-SA 3.0 / 4.0 ([LICENSE.md](wikimedia-commons/LICENSE.md)) |
| [`ifc/`](ifc/) | 12 | IFC models (IFC2X3, IFC4) as other BIM tools export them, for IFC importers | CC BY 4.0, MIT, NIBS reuse terms ([LICENSE.md](ifc/LICENSE.md)) |

Swiss Dwellings: 28 residential floors and 12 office or commercial ones (offices, shops, a medical practice, a
workshop), most of them turned 4 to 42 degrees in the source.

bldg's published benchmarks used 31 non-synthetic plans of an earlier local set: 12 from CubiCasa5K, 12 from
FloorPlanCAD and 4 from CVC-FP, which are not here (CC BY-NC 4.0: no commercial use), and 3 Swiss Dwellings plans,
which are (`swiss-dwellings/realworld_08`, `_09`, `_10`).

## Layout

```
<folder>/<name>.jpg          the input image
<folder>/<name>.gt.json      the ground truth (floorplan/v0, below)
<folder>/<name>.orig.csv     the source's own annotation rows, where small enough to keep (Swiss Dwellings)
<folder>/<name>.commons.json the Wikimedia Commons file record (title, author, licence, source page)
<folder>/manifest.json       per-sample source, author, licence, style and ground-truth fidelity
all_manifest.json            the per-folder manifests merged, with counts (the image samples)
splits.json                  a fixed dev / test split ({"dev": [...], "test": [...]}, keys <folder>/<name>)
ifc/<name>.ifczip            an IFC model (a zip holding one .ifc); ifc/manifest.json: source, licence, changes
tools/                       the scripts that made the set (below)
```

A sample is named by its key, `<folder>/<name>` (`synthetic/synthetic_19`). The first samples keep their names from
the set's first layout (`swiss-dwellings/realworld_08`); the other Swiss Dwellings floors are named by their source
floor id (`swiss-dwellings/floor_1627`). In `splits.json` a name ending in an odd number is dev, an even one test.

## Tools

Everything here can be made again from its source with the scripts in [`tools/`](tools/) (Python 3.12; numpy,
opencv-python-headless, shapely, scipy):

| script | what |
|---|---|
| `synthetic_gen.py`, `synthetic_verify.py` | generate `synthetic/` (byte for byte) and check every drawn door against its ground truth ([GENERATOR.md](synthetic/GENERATOR.md)) |
| `realworld_convert_msd.py`, `swiss_dwellings_batch.py` | render a Swiss Dwellings floor with its ground truth; the batch script lists the chosen floors and how they were chosen (byte for byte with opencv-python-headless 4.10.0.84) |
| `realworld_fetch_commons.py` | fetch a Wikimedia Commons image with its licence record |
| `ifc_scrub.py`, `ifc_pack.py`, `ifc_manifest.py`, `ifc_scan_pii.py` | remove personal data from an IFC model, pack it, write `ifc/manifest.json`, scan for what is left |
| `build_manifest.py` | merge the folder manifests into `all_manifest.json` and write `splits.json` |
| `fpgt.py` | validate (`validate GT.json ...`) or draw (`overlay IMAGE GT OUT.png`) floorplan/v0 ground truth; `realworld_verify_doors.py`: a contact sheet of every door |
| `fp_eval.py`, `fp_eval_selftest.py`, `fp_eval_adversarial.py` | score floorplan/v0 predictions against the ground truth (walls, doors and their swing, windows, rooms), and its self-tests |

```bash
python3 -I tools/fp_eval.py --pred-dir PRED_DIR --split test --md report.md   # one <name>.json per sample
```

## Use with bldg

Clone it next to the bldg repository; bldg's tests, benchmarks and figure scripts find it there (`tests/testdata.py`;
`BLDG_TESTDATA` points them elsewhere):

```bash
git clone https://github.com/usebldg/bldg-test-data ../bldg-test-data
```

## Ground truth: floorplan/v0

One JSON file per image. Coordinates are image pixels, x right, y down from the top-left corner.

- `image`: `file`, `width`, `height`. `scale.mm_per_px`: millimetres per pixel where known (`source` says how).
- `walls`: `id`, `centerline` (two points, junction to junction), `thickness_px` / `thickness_mm`, `kind`
  (exterior, interior, ...), `polygon` (the wall's outline, trimmed at the face of the wall it meets).
- `openings`: `id`, `category` (door, window, opening), `type`, `wall_id`, `jambs` (the two centerline points at
  its ends), `width_px` / `width_mm`, and for doors `leaves`: `hinge`, `closed_tip`, `open_tip` per leaf, plus
  `swing_into`, `handedness`, `connects` where known.
- `rooms`: `id`, `polygon`, `label`, `type`, `area_m2`. `stairs`, `labels` (`text`, `bbox`, `kind`, `room`).
- `gt_fidelity`: what the sample's ground truth can be scored on. Score door direction only where
  `door_swing == "full"`; an absent or null field means the source does not provide it, not "none".
- `conventions`: the exact meaning of jambs, leaves and polygons for that sample.

## Samples

| key | ground truth | walls | doors (swing) | windows | rooms | licence |
|---|---|---:|---:|---:|---:|---|
| `synthetic/synthetic_00` | yes | 6 | 3 (full) | 8 | 3 | CC0 1.0 |
| `synthetic/synthetic_01` | yes | 7 | 4 (full) | 7 | 4 | CC0 1.0 |
| `synthetic/synthetic_02` | yes | 10 | 6 (full) | 9 | 5 | CC0 1.0 |
| `synthetic/synthetic_03` | yes | 9 | 6 (full) | 10 | 5 | CC0 1.0 |
| `synthetic/synthetic_04` | yes | 11 | 6 (full) | 10 | 6 | CC0 1.0 |
| `synthetic/synthetic_05` | yes | 9 | 8 (full) | 10 | 6 | CC0 1.0 |
| `synthetic/synthetic_06` | yes | 12 | 7 (full) | 11 | 7 | CC0 1.0 |
| `synthetic/synthetic_07` | yes | 12 | 8 (full) | 11 | 7 | CC0 1.0 |
| `synthetic/synthetic_08` | yes | 14 | 10 (full) | 10 | 8 | CC0 1.0 |
| `synthetic/synthetic_09` | yes | 11 | 11 (full) | 11 | 8 | CC0 1.0 |
| `synthetic/synthetic_10` | yes | 14 | 10 (full) | 11 | 9 | CC0 1.0 |
| `synthetic/synthetic_11` | yes | 13 | 10 (full) | 11 | 9 | CC0 1.0 |
| `synthetic/synthetic_12` | yes | 6 | 3 (full) | 6 | 3 | CC0 1.0 |
| `synthetic/synthetic_13` | yes | 9 | 3 (full) | 8 | 4 | CC0 1.0 |
| `synthetic/synthetic_14` | yes | 9 | 6 (full) | 8 | 5 | CC0 1.0 |
| `synthetic/synthetic_15` | yes | 11 | 4 (full) | 10 | 6 | CC0 1.0 |
| `synthetic/synthetic_16` | yes | 9 | 7 (full) | 9 | 6 | CC0 1.0 |
| `synthetic/synthetic_17` | yes | 13 | 8 (full) | 11 | 7 | CC0 1.0 |
| `synthetic/synthetic_18` | yes | 12 | 9 (full) | 9 | 7 | CC0 1.0 |
| `synthetic/synthetic_19` | yes | 13 | 8 (full) | 13 | 8 | CC0 1.0 |
| `synthetic/synthetic_20` | yes | 11 | 8 (full) | 12 | 8 | CC0 1.0 |
| `synthetic/synthetic_21` | yes | 15 | 10 (full) | 12 | 9 | CC0 1.0 |
| `synthetic/synthetic_22` | yes | 8 | 5 (full) | 10 | 5 | CC0 1.0 |
| `synthetic/synthetic_23` | yes | 16 | 10 (full) | 12 | 9 | CC0 1.0 |
| `swiss-dwellings/realworld_08` | yes | 56 | 14 (none) | 14 | 21 | CC BY 4.0 |
| `swiss-dwellings/realworld_09` | yes | 61 | 19 (none) | 15 | 22 | CC BY 4.0 |
| `swiss-dwellings/realworld_10` | yes | 41 | 15 (none) | 15 | 19 | CC BY 4.0 |
| `swiss-dwellings/floor_509` | yes | 72 | 14 (none) | 12 | 24 | CC BY 4.0 |
| `swiss-dwellings/floor_1627` | yes | 82 | 19 (none) | 16 | 25 | CC BY 4.0 |
| `swiss-dwellings/floor_1771` | yes | 90 | 17 (none) | 11 | 21 | CC BY 4.0 |
| `swiss-dwellings/floor_1795` | yes | 97 | 35 (none) | 18 | 44 | CC BY 4.0 |
| `swiss-dwellings/floor_2304` | yes | 36 | 10 (none) | 4 | 15 | CC BY 4.0 |
| `swiss-dwellings/floor_3320` | yes | 180 | 42 (none) | 14 | 41 | CC BY 4.0 |
| `swiss-dwellings/floor_4889` | yes | 103 | 46 (none) | 31 | 33 | CC BY 4.0 |
| `swiss-dwellings/floor_5954` | yes | 55 | 17 (none) | 14 | 22 | CC BY 4.0 |
| `swiss-dwellings/floor_6002` | yes | 77 | 33 (none) | 20 | 37 | CC BY 4.0 |
| `swiss-dwellings/floor_6407` | yes | 23 | 8 (none) | 7 | 13 | CC BY 4.0 |
| `swiss-dwellings/floor_6863` | yes | 82 | 34 (none) | 20 | 30 | CC BY 4.0 |
| `swiss-dwellings/floor_7396` | yes | 62 | 21 (none) | 23 | 25 | CC BY 4.0 |
| `swiss-dwellings/floor_8166` | yes | 53 | 22 (none) | 10 | 27 | CC BY 4.0 |
| `swiss-dwellings/floor_8432` | yes | 53 | 22 (none) | 12 | 26 | CC BY 4.0 |
| `swiss-dwellings/floor_9106` | yes | 40 | 16 (none) | 9 | 19 | CC BY 4.0 |
| `swiss-dwellings/floor_9776` | yes | 57 | 11 (none) | 6 | 24 | CC BY 4.0 |
| `swiss-dwellings/floor_10355` | yes | 66 | 25 (none) | 14 | 33 | CC BY 4.0 |
| `swiss-dwellings/floor_11019` | yes | 77 | 24 (none) | 15 | 29 | CC BY 4.0 |
| `swiss-dwellings/floor_11977` | yes | 44 | 20 (none) | 12 | 22 | CC BY 4.0 |
| `swiss-dwellings/floor_12960` | yes | 76 | 15 (none) | 16 | 31 | CC BY 4.0 |
| `swiss-dwellings/floor_13773` | yes | 108 | 35 (none) | 18 | 42 | CC BY 4.0 |
| `swiss-dwellings/floor_13837` | yes | 91 | 28 (none) | 13 | 30 | CC BY 4.0 |
| `swiss-dwellings/floor_14814` | yes | 156 | 46 (none) | 12 | 56 | CC BY 4.0 |
| `swiss-dwellings/floor_15050` | yes | 53 | 10 (none) | 11 | 14 | CC BY 4.0 |
| `swiss-dwellings/floor_15108` | yes | 17 | 6 (none) | 8 | 10 | CC BY 4.0 |
| `swiss-dwellings/floor_15408` | yes | 75 | 25 (none) | 16 | 32 | CC BY 4.0 |
| `swiss-dwellings/floor_24468` | yes | 90 | 34 (none) | 16 | 43 | CC BY 4.0 |
| `swiss-dwellings/floor_42450` | yes | 32 | 6 (none) | 6 | 11 | CC BY 4.0 |
| `swiss-dwellings/floor_43081` | yes | 65 | 22 (none) | 13 | 28 | CC BY 4.0 |
| `swiss-dwellings/floor_47886` | yes | 43 | 14 (none) | 15 | 12 | CC BY 4.0 |
| `swiss-dwellings/floor_48780` | yes | 36 | 14 (none) | 14 | 15 | CC BY 4.0 |
| `swiss-dwellings/floor_48978` | yes | 41 | 17 (none) | 12 | 19 | CC BY 4.0 |
| `swiss-dwellings/floor_48983` | yes | 45 | 10 (none) | 12 | 15 | CC BY 4.0 |
| `swiss-dwellings/floor_49063` | yes | 54 | 16 (none) | 13 | 17 | CC BY 4.0 |
| `swiss-dwellings/floor_49870` | yes | 68 | 9 (none) | 9 | 9 | CC BY 4.0 |
| `swiss-dwellings/floor_49889` | yes | 43 | 14 (none) | 15 | 12 | CC BY 4.0 |
| `swiss-dwellings/floor_49980` | yes | 143 | 19 (none) | 14 | 23 | CC BY 4.0 |
| `swiss-dwellings/floor_50411` | yes | 123 | 28 (none) | 20 | 38 | CC BY 4.0 |
| `swiss-dwellings/floor_50496` | yes | 68 | 18 (none) | 13 | 23 | CC BY 4.0 |
| `swiss-dwellings/floor_51300` | yes | 25 | 12 (none) | 6 | 9 | CC BY 4.0 |
| `wikimedia-commons/realworld_11` | image only | 0 | 0 (none) | 0 | 0 | Public domain |
| `wikimedia-commons/realworld_13` | image only | 0 | 0 (none) | 0 | 0 | CC BY-SA 4.0 |
| `wikimedia-commons/realworld_14` | image only | 0 | 0 (none) | 0 | 0 | CC BY-SA 3.0 |

## Licence

Mixed, by folder: see [LICENSE.md](LICENSE.md). Attribution is required for `swiss-dwellings/`, for the two
CC BY-SA images in `wikimedia-commons/` (adaptations of those two must keep CC BY-SA) and for every model in
`ifc/`. The scripts in `tools/` are Apache-2.0.
