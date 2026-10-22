# Synthetic floor plans (`synthetic_00` … `synthetic_23`)

This set is our own procedurally generated data, released with no restrictions. It contains no third-party content and nothing was downloaded. Its job is to be the door-swing benchmark. Every door leaf (hinge, closed tip, open tip) comes straight from the geometry that was drawn, so `gt_fidelity.door_swing = "full"` holds exactly.

## Regenerate

```
python3 -m venv .venv   # only if it is missing
.venv/bin/pip install shapely numpy opencv-python-headless
.venv/bin/python -I tools/synthetic_gen.py --all --out testdata/synthetic      # all 24 + manifest.json
.venv/bin/python -I tools/synthetic_gen.py --index 7 --out /tmp/one            # a single sample
python3 -I tools/fpgt.py validate testdata/synthetic/*.gt.json
```

The output is deterministic: two runs give byte-identical JPGs and JSON with the same library versions (shapely 2.2, opencv 5.0, numpy 2.5). Each sample uses `seed = 20261013 + 7919 * index` with Python's `random.Random`. Degradation noise uses `numpy.default_rng(seed)`.

## Pipeline

1. **Layout (mm, x right, y down).**
   - The footprint holds about 11.5 to 17 m² per room, with an aspect ratio of 1.2 to 1.75.
   - It is a rectangle, or an L-shape made by removing one corner notch that is 30-45 % by 30-50 % of the footprint.
   - When the corridor option is on, a 1.15-1.5 m strip is cut along the long axis of the largest zone.
   - The remaining zones are split recursively (BSP) into 3-9 rooms. Rooms are at least 1.9 m on every side, with an aspect ratio of 3 or less, snapped to a 50 mm grid.
   - The layout is transposed with 50 % probability.
   - **Diagonal walls** come from chamfering 1-2 convex exterior corners with 45° walls. The cut-off triangle always lies inside a single room.
2. **Room types.** The largest room is living. The smallest is bath. One mid-sized room is the kitchen when there are 4 or more rooms. Optional extras are a second bath (WC), storage and a study/office. Everything else is a bedroom.
3. **Walls.**
   - Exterior walls are 250-400 mm thick, and the outline edges are offset with mitred corners.
   - Interior walls are 100-150 mm thick. They are the room edges that do not lie on the outline, with collinear pieces merged into maximal straight runs.
   - GT `centerline` runs junction to junction. The wall polygons are trimmed at the face of the wall they abut.
4. **Openings.** Each one is checked against the room's clear polygon, other door sweeps, and keep-clear zones (600 mm on the non-swing side, 500 mm for sliding, pocket and plain openings). Gaps stay at least 120 mm apart.
   - Interior doors: Kruskal over room adjacency. An edge is only accepted if a door actually fits, so every room is reachable.
     - **single**: 700-950 mm, or 700-800 mm for baths and storage. The hinge is a or b at random. The leaf swings into the more private room 75 % of the time.
     - **double**: 1300-1600 mm, as two leaves.
     - **sliding**: 1300-1800 mm, as two offset bypass panels inside the wall thickness.
     - **pocket**: 700-900 mm. The panel is drawn 45 % retracted into a visible slot cut in the wall.
     - **plain opening**: `category: opening`, `type: none`.
   - Entrance: a single (sometimes double) door, 900-1000 mm, preferably in the corridor. 35 % of entrance doors swing outward.
   - Optional patio door from the living room or a bedroom: sliding, double or single, possibly opening outward.
   - Windows are 3-line symbols in exterior walls: both faces plus the centre line, plus jamb ticks. They are 900-2400 mm (600-900 mm for baths), 1-2 per exterior wall piece, and also placed on diagonal walls.
5. **Rendering** (OpenCV, anti-aliased, sub-pixel). The scale is chosen so the long side matches `long_px`, which makes `mm_per_px` about 7.4 to 29.
   - Wall styles: `solid` (black fill), `gray` (grey fill and outline), `hollow` (double outline), `hatched` (45° hatch and outline).
   - Doors are drawn with the leaf (a line or a thin rectangle) and a quarter arc, which is dashed in some samples.
   - Clutter options:
     - room names (OpenCV `FONT_HERSHEY_*` ids; note that OpenCV 5.0 renders these with fonts embedded in its binary, Rubik under SIL OFL 1.1, which places no restriction on rendered images), with area text (`area`)
     - dimension chains on 2 sides, with tick slashes and mm numbers (vertical text is rotated)
     - furniture: bed + nightstands, wardrobe, desk + chair, sofa, armchair, TV unit, coffee table, dining set, kitchen counter with sink and hob, fridge, toilet, basin, tub, shower, shelves
     - `tiles` (floor grid in bath and kitchen)
     - `graphpaper` (page grid)
     - `axisgrid` (dash-dot structural grid with A/B/1/2 bubbles, crossing the plan)
     - `stairs` (treads, arrow, "UP")
     - `color` (pastel room fills)
     - `title`
     - `scalebar` (a 5 m bar; its GT endpoints are in `scale_bar`)
     - `north` arrow
6. **Degradation**, applied to samples 12-23 (level 1-12). The steps run in this order:
   - rotation about the image centre (1-4°, counter-clockwise on screen when positive). GT is pushed through the same 2x3 matrix, which is stored in `generator.rotation_matrix`.
   - fade (ink contrast)
   - paper tint
   - uneven lighting (a smooth 4x4 random gain field)
   - Gaussian blur (sigma 0.6-1.4)
   - dust specks
   - Gaussian noise (sigma 4-12)
   - JPEG quality 70 down to 40

   Clean samples are saved at JPEG quality 95.

## Knob table

| idx | rooms | L | diag | corr | walls | text | dims | furn | extras | long px | degrade |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 3 | - | 0 | - | solid | - | - | - | | 1000 | - |
| 01 | 4 | - | 0 | - | hollow | - | - | - | | 1200 | - |
| 02 | 5 | L | 0 | - | hatched | y | - | - | | 1400 | - |
| 03 | 5 | - | 1 | y | solid | y | y | - | | 1600 | - |
| 04 | 6 | L | 0 | y | hollow | y | y | - | scalebar | 1800 | - |
| 05 | 6 | - | 0 | - | gray | - | - | y | | 1500 | - |
| 06 | 7 | L | 0 | y | hatched | y | - | y | tiles | 2000 | - |
| 07 | 7 | - | 2 | y | solid | y | y | y | scalebar, north | 2200 | - |
| 08 | 8 | L | 1 | y | hollow | y | y | y | stairs, area | 2400 | - |
| 09 | 8 | - | 0 | y | hatched | y | y | y | tiles, axisgrid | 2600 | - |
| 10 | 9 | L | 0 | y | solid | y | y | y | color, area, title | 2800 | - |
| 11 | 9 | - | 1 | y | hollow | y | y | y | graphpaper, stairs, scalebar | 3200 | - |
| 12 | 3 | - | 0 | - | solid | y | - | - | | 1000 | L1: q70, rot 1.0 |
| 13 | 4 | L | 0 | - | hollow | y | y | - | | 1300 | L2: q66, blur 0.6 |
| 14 | 5 | - | 1 | - | hatched | y | y | y | | 1600 | L3: q63, rot -1.5, noise 4 |
| 15 | 6 | L | 0 | y | gray | y | y | y | tiles | 1800 | L4: q60, tint, light 0.15 |
| 16 | 6 | - | 0 | y | solid | y | y | y | color | 2000 | L5: q58, rot 2, blur 0.8, noise 5 |
| 17 | 7 | L | 1 | y | hollow | y | y | y | stairs, title | 2200 | L6: q55, tint, light 0.2, noise 6 |
| 18 | 7 | - | 2 | y | hatched | y | y | y | axisgrid, area | 2400 | L7: q52, rot -2.5, blur 1.0, tint |
| 19 | 8 | L | 0 | y | solid | y | y | y | graphpaper, tiles | 2000 | L8: q50, rot 3, noise 8, light 0.25 |
| 20 | 8 | - | 0 | y | hollow | y | y | y | scalebar, north, title | 1600 | L9: q48, blur 1.1, tint, light 0.3, specks |
| 21 | 9 | L | 1 | y | gray | y | y | y | stairs, tiles, area | 1400 | L10: q45, rot -3.5, blur 1.2, noise 9, tint, fade 0.8 |
| 22 | 5 | - | 0 | - | hatched | y | y | y | color, graphpaper | 1100 | L11: q42, rot 4, noise 10, light 0.35, specks, tint |
| 23 | 9 | L | 2 | y | solid | y | y | y | axisgrid, stairs, title, scalebar | 900 | L12: q40, rot -4, blur 1.4, noise 12, tint, light 0.4, fade 0.75, specks |

The "diag" column is the number of chamfers requested. A chamfer is skipped when no eligible corner room exists, and the actual count is in `generator.diagonal_walls_in_rooms`.

## GT conventions

These conventions are repeated in every file under `conventions`.

- `jambs` are the two ends of the opening on the wall **centreline**, with a = `jambs[0]` and b = `jambs[1]`.
- `leaves[].hinge` and `closed_tip` are the jamb corners on the **wall face on the swing side**, which is where the leaf and arc are drawn. `open_tip = hinge + width × swing normal`, and the leaf is drawn fully open at 90°. As a result, `|hinge - closed_tip| = |hinge - open_tip|` = the leaf width.
- `handedness` uses the schema convention. The viewer stands on the side the door swings **away from** and faces the door. A hinge on the viewer's left is `LH`, and on the right is `RH`. With this convention `LHR` and `RHR` never occur. Only single-leaf doors get a handedness; double doors get `null`.
- `swing_into` is a room id or `"outside"`. For swing doors, `connects` is `[non-swing-side room, swing_into]`.
- Pocket doors have `slide.track_side` set to the jamb (`a` or `b`) whose side holds the pocket, and `slide.direction` set to the unit vector the panel moves to open. Bypass sliding doors have `track_side` and `direction` set to `null`, with `panels: 2`.
- Room `polygon` is the clear polygon at the wall faces, and `polygon_centerline` is the polygon at the wall centrelines. Polygon winding gives a positive shoelace area in image coordinates (x right, y down). That is mathematically counter-clockwise in the image frame, which looks clockwise on screen.
- `scale.mm_per_px` is exact. `labels[].bbox` is the axis-aligned box in the final, possibly rotated, image, clipped to the image. Labels that rotation pushes less than 75 % into the image are dropped (verifier fix: sample 23 had an axis bubble "A" with bbox y=-3.6..5.9 whose glyph is cut off).
- Extra keys beyond the schema: `width_mm`, `thickness_mm`, `role` (entrance/patio), `label_drawn`, `area_m2`, `scale_bar`, `generator`, `conventions`.

## Totals (v1)

- 24 samples containing 260 walls, 154 rooms, 239 windows, 7 plain openings and 5 stairs.
- 170 doors in total:
  - 124 single, all with swing: 61 LH and 56 RH into rooms or corridors, 7 swinging outside. All four combinations of hinge at a/b and swing to either side of the jamb line occur at least 23 times each.
  - 11 double
  - 19 sliding
  - 16 pocket
- 135 doors carry full leaves.

## Independent verification

`python3 -I tools/synthetic_verify.py testdata/synthetic` re-derives every door's 4 (single) or 2 (double) possible hinge/swing hypotheses from the jambs, wall normal and wall thickness only, scores the ink on each hypothesis' arc and leaf in the JPG, and requires the GT leaf to be the best one. It also checks swing_into/connects against room polygons, handedness, pocket direction, window symbols, wall faces/gaps, room edges, labels, scale and manifest counts. Result: 135/135 swing doors confirmed. The remaining flags are known detector false positives (pastel fill or heavy blur near door gaps in samples 22-23, and window glass-band tests at 900 px).
