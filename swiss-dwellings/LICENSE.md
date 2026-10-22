# Swiss Dwellings: CC BY 4.0

The 43 samples in this folder are adapted from:

> Archilyse AG: M. Standfest, M. Franzen, Y. Schroeder, L. Gonzalez Medina, Y. Villanueva Hernandez, J. H. Buck, Y.-L. Tan, M. Niedzwiecka, R. Colmegna. *Swiss Dwellings* v3.0.0. Zenodo. doi:[10.5281/zenodo.7788422](https://doi.org/10.5281/zenodo.7788422).
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| sample | source floor (Swiss Dwellings v3.0.0 geometries.csv) | use |
|---|---|---|
| `realworld_08` | floor_id=1588 site_id=210 | flats (2 units) |
| `realworld_09` | floor_id=4024 site_id=19 | flats (3 units) |
| `realworld_10` | floor_id=42195 site_id=11111 | flats (2 units) |
| `floor_509` | floor_id=509 site_id=101 | offices and flats |
| `floor_1627` | floor_id=1627 site_id=220 | flats (2 units) |
| `floor_1771` | floor_id=1771 site_id=245 | flats (1 unit) |
| `floor_1795` | floor_id=1795 site_id=248 | flats (3 units) |
| `floor_2304` | floor_id=2304 site_id=299 | flats (2 units) |
| `floor_3320` | floor_id=3320 site_id=355 | flats (4 units) |
| `floor_4889` | floor_id=4889 site_id=461 | flats (2 units) |
| `floor_5954` | floor_id=5954 site_id=633 | flats (3 units) |
| `floor_6002` | floor_id=6002 site_id=635 | flats (4 units) |
| `floor_6407` | floor_id=6407 site_id=696 | flats (1 unit) |
| `floor_6863` | floor_id=6863 site_id=738 | flats (1 unit) |
| `floor_7396` | floor_id=7396 site_id=758 | offices and a flat |
| `floor_8166` | floor_id=8166 site_id=868 | flats (2 units) |
| `floor_8432` | floor_id=8432 site_id=894 | flats (3 units) |
| `floor_9106` | floor_id=9106 site_id=971 | flats (2 units) |
| `floor_9776` | floor_id=9776 site_id=1054 | flats (1 unit) |
| `floor_10355` | floor_id=10355 site_id=1100 | medical practice and flats |
| `floor_11019` | floor_id=11019 site_id=725 | flats (3 units) |
| `floor_11977` | floor_id=11977 site_id=1301 | flats (4 units) |
| `floor_12960` | floor_id=12960 site_id=1607 | flats (6 units) |
| `floor_13773` | floor_id=13773 site_id=764 | flats (4 units) |
| `floor_13837` | floor_id=13837 site_id=722 | flats (3 units) |
| `floor_14814` | floor_id=14814 site_id=1222 | flats (6 units) |
| `floor_15050` | floor_id=15050 site_id=2292 | flats (1 unit) |
| `floor_15108` | floor_id=15108 site_id=2295 | flats (2 units) |
| `floor_15408` | floor_id=15408 site_id=1995 | flats (5 units) |
| `floor_24468` | floor_id=24468 site_id=3699 | flats (4 units) |
| `floor_42450` | floor_id=42450 site_id=11126 | flats (1 unit) |
| `floor_43081` | floor_id=43081 site_id=11163 | flats (3 units) |
| `floor_47886` | floor_id=47886 site_id=11487 | shop |
| `floor_48780` | floor_id=48780 site_id=11667 | offices and a workshop |
| `floor_48978` | floor_id=48978 site_id=11706 | offices |
| `floor_48983` | floor_id=48983 site_id=11707 | offices |
| `floor_49063` | floor_id=49063 site_id=11720 | shop |
| `floor_49870` | floor_id=49870 site_id=11787 | offices |
| `floor_49889` | floor_id=49889 site_id=11786 | factory floor and offices |
| `floor_49980` | floor_id=49980 site_id=11364 | flats (2 units) |
| `floor_50411` | floor_id=50411 site_id=11860 | open-plan office and flats |
| `floor_50496` | floor_id=50496 site_id=11865 | flats (2 units) |
| `floor_51300` | floor_id=51300 site_id=11905 | offices and a shop |

**Changes.** One floor of each site was selected, for variety: flats of one to six units, offices, shops, a medical
practice, a workshop. The images are our own renders of its geometry (doors drawn as outlined rectangles in the wall:
the source has no door swing). The ground truth is that geometry converted to floorplan/v0 in image pixels; room
polygons keep only their outer ring (the few holes, columns or shafts inside a staircase or foyer, are named in
`manifest.json`). Each `*.orig.csv` keeps the source rows that were used. All 43 samples are made by
[`tools/realworld_convert_msd.py`](../tools/realworld_convert_msd.py); the `floor_*` ones through
[`tools/swiss_dwellings_batch.py`](../tools/swiss_dwellings_batch.py), which lists them and how they were chosen.

Our renders and converted ground truth are offered under the same licence, CC BY 4.0. Archilyse AG does not endorse
this repository or bldg.
