# bldg test data

Floor plan images with ground truth, used to test and benchmark [bldg](https://github.com/usebldg/bldg)
(buildings AI agents can edit). Every file here may be used commercially, under the licence of its folder
([LICENSE.md](LICENSE.md)).

| folder | samples | ground truth | licence |
|---|---:|---|---|
| [`synthetic/`](synthetic/) | 24 | walls, openings with full door swing, rooms, labels | CC0 1.0 (our own generator, [GENERATOR.md](synthetic/GENERATOR.md)) |
| [`swiss-dwellings/`](swiss-dwellings/) | 3 | walls, openings (no swing), rooms, stairs | CC BY 4.0, Archilyse AG ([LICENSE.md](swiss-dwellings/LICENSE.md)) |
| [`wikimedia-commons/`](wikimedia-commons/) | 3 | none (images only) | public domain, CC BY-SA 3.0 / 4.0 ([LICENSE.md](wikimedia-commons/LICENSE.md)) |

bldg's published benchmarks also use 31 samples from CubiCasa5K, FloorPlanCAD, CVC-FP and SESYD. Their licences do
not allow commercial use or redistribution (CC BY-NC 4.0, or no licence), so they are not here.

## Layout

```
<folder>/<name>.jpg          the input image
<folder>/<name>.gt.json      the ground truth (floorplan/v0, below)
<folder>/<name>.orig.csv     the source's own annotation rows, where small enough to keep (Swiss Dwellings)
<folder>/<name>.commons.json the Wikimedia Commons file record (title, author, licence, source page)
<folder>/manifest.json       per-sample source, author, licence, style and ground-truth fidelity
all_manifest.json            the per-folder manifests merged, with counts
splits.json                  a fixed dev / test split ({"dev": [...], "test": [...]}, keys <folder>/<name>)
```

A sample is named by its key, `<folder>/<name>` (`synthetic/synthetic_19`). The names are kept from the set's first
layout, so `swiss-dwellings/realworld_08` keeps its `realworld_` name.

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
| `wikimedia-commons/realworld_11` | image only | 0 | 0 (none) | 0 | 0 | Public domain |
| `wikimedia-commons/realworld_13` | image only | 0 | 0 (none) | 0 | 0 | CC BY-SA 4.0 |
| `wikimedia-commons/realworld_14` | image only | 0 | 0 (none) | 0 | 0 | CC BY-SA 3.0 |

## Licence

Mixed, by folder: see [LICENSE.md](LICENSE.md). Attribution is required for `swiss-dwellings/` and for the two
CC BY-SA images in `wikimedia-commons/`; adaptations of those two images must keep CC BY-SA.
