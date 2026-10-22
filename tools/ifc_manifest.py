"""Write ifc/manifest.json for the packed files: per file its source, licence, credit, schema, sizes, hashes, entity
counts and the changes made.  python -I tools/ifc_manifest.py RAW_DIR SCRUBBED_DIR ZIPPED_DIR scrub_records.jsonl OUT.json"""
import hashlib
import json
import re
import sys
from pathlib import Path

import ifcopenshell

raw_dir, scr_dir, zip_dir, rec_path, out = map(Path, sys.argv[1:6])
recs = {Path(json.loads(l)["file"]).stem: json.loads(l)["changes"] for l in rec_path.read_text().splitlines() if l}

BSC = "https://github.com/buildingsmart-community/Community-Sample-Test-Files"
BSC_MEDIA = "https://media.githubusercontent.com/media/buildingsmart-community/Community-Sample-Test-Files/main/"
HF = "https://huggingface.co/datasets/sylvainHellin/ifc-bench"
HF_REV = "9ba471a3e38b353a81a1dc9ecb4aa56d2ce0a7fa"
CCBY = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
MIT = ("MIT", "https://opensource.org/license/mit")

SRC = {
    "duplex_apartment": dict(
        raw="bsc_duplex", title="Duplex Apartment, architectural model", use="residential",
        building="two-storey duplex apartment building (two dwellings)",
        source_page=f"{BSC}/tree/main/IFC%202.3.0.1%20(IFC%202x3)/Duplex%20Apartment",
        file_url=BSC_MEDIA + "IFC%202.3.0.1%20(IFC%202x3)/Duplex%20Apartment/Duplex_A_20110907.ifc",
        license=CCBY, copyright="(C) original authors (repository LICENSE)",
        credit='BSI (2020) "Duplex Apartment Test Files," buildingSMART International, ' + BSC,
        notes="First published in Germany, provided to the US Army ERDC-CERL; hosted by the US National Institute of "
              "Building Sciences 2009-2020. The folder README asks for the credit above and a note of changes."),
    "medical_dental_clinic": dict(
        raw="bsc_clinic", title="Medical-Dental Clinic, architectural model", use="healthcare",
        building="clinic with medical, dental, pediatric and imaging suites (a real building, redacted)",
        source_page=f"{BSC}/tree/main/IFC%202.3.0.1%20(IFC%202x3)/Medical-Dental%20Clinic",
        file_url=BSC_MEDIA + "IFC%202.3.0.1%20(IFC%202x3)/Medical-Dental%20Clinic/Clinic_Architectural.ifc",
        license=CCBY, copyright="(C) original authors (repository LICENSE)",
        credit='BSI (2020) "Medical-Dental Test Files," buildingSMART International, ' + BSC,
        notes="Provided to the US Army ERDC-CERL; hosted by the US National Institute of Building Sciences "
              "2010-2020. The folder README asks for the credit above and a note of changes."),
    "esplanades_tallinn": dict(
        raw="bsc_esplanades", title="Esplanades: example model from Estonia (Maleva 18, Tallinn)",
        use="residential", building="apartment building, preliminary design (eelprojekt)",
        source_page=f"{BSC}/tree/main/IFC%202.3.0.1%20(IFC%202x3)/Esplanades",
        file_url=BSC_MEDIA + "IFC%202.3.0.1%20(IFC%202x3)/Esplanades/1807_EP_AR_v18.ifc",
        license=CCBY, copyright="(c) Esplan, http://www.esplan.ee/en",
        credit="Esplan (http://www.esplan.ee/en), example model from Estonia, buildingSMART Community Sample & "
               "Test Files, " + BSC, notes="Folder readme: \"(c) Esplan ... Published under the Creative Commons "
                                           "Attribution 4.0 International License.\""),
    "gym_amersfoort": dict(
        raw="bsc_gym", title="Gymzaal Amersfoort (example project location)", use="sports",
        building="sports hall in the Netherlands, IFC4 Reference View export",
        source_page=f"{BSC}/tree/main/IFC%204.0.2.1%20(IFC%204)/Example%20project%20location",
        file_url=BSC_MEDIA + "IFC%204.0.2.1%20(IFC%204)/Example%20project%20location/example%20project%20location.ifc",
        license=CCBY, copyright="(C) original authors (repository LICENSE)",
        credit="Gymzaal Amersfoort, buildingSMART Community Sample & Test Files, " + BSC + "; creators named in "
               "the original file: Hans Hendriks (deBIMspecialist), with manual corrections by Helga Tauscher "
               "(HTW Dresden, Project LevelOut)",
        notes="No licence in the folder readme: the repository LICENSE (CC BY 4.0, \"(C) original authors\") and its "
              "README (contributions are published under CC BY 4.0) cover it. The file links no door or window to "
              "its void (no IfcRelFillsElement)."),
    "wbdg_office": dict(
        raw="wbdg_office", title="Office Building, Common Building Information Model Files, architectural model",
        use="office", building="two-storey office building",
        source_page=f"{HF}/tree/{HF_REV}/projects/wbdg_office",
        file_url=f"{HF}/resolve/{HF_REV}/projects/wbdg_office/arc.ifc",
        license=("NIBS / buildingSMART alliance Common BIM Files reuse terms (attribution)",
                 "https://web.archive.org/web/20130310082112/http://www.buildingsmartalliance.org/index.php/projects/commonbimfiles"),
        copyright="U.S. Army Corps of Engineers ERDC-CERL and the buildingSMART alliance (NIBS)",
        credit="U.S. Army Corps of Engineers ERDC-CERL and buildingSMART alliance (2011). Common Building "
               "Information Model Files: Office Building (2011-09-14 IFC 2x3 Coordination MVD). National Institute "
               "of Building Sciences, Washington, DC. Obtained via IFC-Bench (S. Hellin, Technical University of "
               "Munich, 2025), " + HF,
        notes="The original WBDG page is offline; the terms are those of the archived project page (quoted in "
              "LICENSE.md): copy, distribute, display, use and make derivative works with proper citation; no cost "
              "or use restriction, for-profit or not."),
    "rwth_digitalhub": dict(
        raw="rwth_digitalhub", title="DigitalHub, architectural model (Fachmodell ARC), version 2",
        use="office / university", building="RWTH Aachen E3D DigitalHub building, IFC4 Reference View export",
        source_page="https://github.com/RWTH-E3D/DigitalHub/tree/master/Version_2",
        file_url="https://raw.githubusercontent.com/RWTH-E3D/DigitalHub/master/Version_2/DigitalHub_FM-ARC_v2.ifc",
        license=MIT, copyright="Copyright (c) 2020 RWTH Aachen University - E3D Institute of Energy Efficiency and "
                               "Sustainable Building",
        credit="RWTH Aachen University, E3D Institute of Energy Efficiency and Sustainable Building: DigitalHub, "
               "https://github.com/RWTH-E3D/DigitalHub (MIT)", notes=""),
    "bimwhale_large_building": dict(
        raw="whale_large", title="BIM Whale sample: Large Building", use="generic sample",
        building="two simple buildings with dummy property data",
        source_page="https://github.com/andrewisen/bim-whale-ifc-samples/tree/main/LargeBuilding",
        file_url="https://raw.githubusercontent.com/andrewisen/bim-whale-ifc-samples/main/LargeBuilding/IFC/LargeBuilding.ifc",
        license=MIT, copyright='Copyright (c) 2020 André "MG" Wisén',
        credit='André "MG" Wisén, BIM Whale IFC samples, https://github.com/andrewisen/bim-whale-ifc-samples (MIT)',
        notes="The MIT original, not the CC0-labelled Kaggle re-upload."),
    **{f"zenodo6397164_model{i}": dict(
        raw=f"zenodo_model{i}", title=f"IFC architectural building models: model{i}", use="residential",
        building="small multi-storey building",
        source_page="https://zenodo.org/records/6397164",
        file_url=f"https://zenodo.org/api/records/6397164/files/model{i}.ifc/content",
        license=CCBY, copyright="Tafraout Salim",
        credit="Tafraout Salim (2022). IFC architectural building models. Zenodo. doi:10.5281/zenodo.6397164",
        notes="Zenodo record licence: cc-by-4.0. No IfcSpace in the file.") for i in (1, 2, 3)},
    "tum_fantasy_hotel": dict(
        raw="tum_hotel", title="Fantasy hotel 1, architectural model (student coursework)", use="hotel",
        building="hotel, IFC4 Reference View export",
        source_page=f"{HF}/tree/{HF_REV}/projects/fantasy_hotel_1",
        file_url=f"{HF}/resolve/{HF_REV}/projects/fantasy_hotel_1/arc.ifc",
        license=MIT, copyright="Copyright (c) 2025 TUM Students, BIM Fundamentals SS2025",
        credit="TUM Students, BIM Fundamentals SS2025 (MIT), via IFC-Bench (S. Hellin, Technical University of "
               "Munich, 2025), " + HF, notes="projects/fantasy_hotel_1/license.txt: MIT."),
    "tum_fantasy_residential": dict(
        raw="tum_residential", title="Fantasy residential building 1, architectural model (student coursework)",
        use="residential", building="residential building, IFC4 Reference View export",
        source_page=f"{HF}/tree/{HF_REV}/projects/fantasy_residential_building_1",
        file_url=f"{HF}/resolve/{HF_REV}/projects/fantasy_residential_building_1/arc.ifc",
        license=MIT, copyright="Copyright (c) 2025 TUM Students, BIM Fundamentals SS2025",
        credit="TUM Students, BIM Fundamentals SS2025 (MIT), via IFC-Bench (S. Hellin, Technical University of "
               "Munich, 2025), " + HF, notes="projects/fantasy_residential_building_1/license.txt: MIT."),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def n(f, *classes) -> int:
    total = 0
    for c in classes:
        try:
            total += len(f.by_type(c))
        except RuntimeError:
            pass
    return total


rows = []
for stem, m in SRC.items():
    raw, ifc, z = raw_dir / f"{m['raw']}.ifc", scr_dir / f"{stem}.ifc", zip_dir / f"{stem}.ifczip"
    f = ifcopenshell.open(str(ifc))
    head = ifc.read_text(encoding="latin-1")[:4000]
    view = re.search(r"ViewDefinition \[([^\]]*)\]", head)
    apps = f.by_type("IfcApplication")
    rows.append({
        "file": z.name, "ifc_file": ifc.name, "title": m["title"], "use": m["use"], "building": m["building"],
        "schema": f.schema, "view_definition": view.group(1).strip() if view else None,
        "authoring_tool": apps[0].ApplicationFullName if apps else None,
        "source_page": m["source_page"], "file_url": m["file_url"], "source_file_sha256": sha(raw),
        "license": m["license"][0], "license_url": m["license"][1], "copyright": m["copyright"], "credit": m["credit"],
        "size_bytes": z.stat().st_size, "sha256": sha(z), "ifc_size_bytes": ifc.stat().st_size, "ifc_sha256": sha(ifc),
        "counts": {"storeys": n(f, "IfcBuildingStorey"), "walls": n(f, "IfcWall"), "curtain_walls": n(f, "IfcCurtainWall"),
                   "railings": n(f, "IfcRailing"), "doors": n(f, "IfcDoor"), "windows": n(f, "IfcWindow"),
                   "spaces": n(f, "IfcSpace"), "openings": n(f, "IfcOpeningElement"),
                   "filled_openings": n(f, "IfcRelFillsElement"), "stairs": n(f, "IfcStair"),
                   "columns": n(f, "IfcColumn"), "slabs": n(f, "IfcSlab"), "zones": n(f, "IfcZone")},
        "changes": recs[stem] + [f"renamed from {m['file_url'].removesuffix('/content').rsplit('/', 1)[-1].replace('%20', ' ')}"
                                 f" and packed as {z.name} (one member, {ifc.name}; ifcopenshell opens it directly)",
                                 "the building's geometry, properties and relationships unchanged"],
        "notes": m["notes"]})
out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
print(out, len(rows), sum(r["size_bytes"] for r in rows))
