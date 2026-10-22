# IFC models

Twelve architectural BIM models exported by common BIM tools (IFC2X3 and IFC4), kept as test inputs for IFC
importers. Each keeps its own licence; every one of them allows commercial use, redistribution and changes, with
credit. `manifest.json` lists per file its source, licence, credit, schema, sizes, SHA-256 (of the original download,
the stored `.ifc` and the `.ifczip`), entity counts and the changes made.

| file | model | credit | licence |
|---|---|---|---|
| `duplex_apartment.ifczip` | Duplex Apartment, architectural | BSI (2020) "Duplex Apartment Test Files," buildingSMART International, <https://github.com/buildingsmart-community/Community-Sample-Test-Files> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `medical_dental_clinic.ifczip` | Medical-Dental Clinic, architectural | BSI (2020) "Medical-Dental Test Files," buildingSMART International, <https://github.com/buildingsmart-community/Community-Sample-Test-Files> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `esplanades_tallinn.ifczip` | Esplanades, example model from Estonia | (c) Esplan, <http://www.esplan.ee/en>, via the buildingSMART Community Sample & Test Files | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `gym_amersfoort.ifczip` | Gymzaal Amersfoort (example project location) | buildingSMART Community Sample & Test Files contributors; creators named in the original file: Hans Hendriks (deBIMspecialist), with corrections by Helga Tauscher (HTW Dresden, Project LevelOut) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (repository licence) |
| `wbdg_office.ifczip` | Office Building, Common BIM Files, architectural | U.S. Army Corps of Engineers ERDC-CERL and buildingSMART alliance (2011). Common Building Information Model Files: Office Building (2011-09-14 IFC 2x3 Coordination MVD). National Institute of Building Sciences, Washington, DC. Obtained via IFC-Bench, <https://huggingface.co/datasets/sylvainHellin/ifc-bench> | NIBS Common BIM Files reuse terms (below) |
| `rwth_digitalhub.ifczip` | DigitalHub, architectural (Fachmodell ARC, version 2) | RWTH Aachen University, E3D Institute of Energy Efficiency and Sustainable Building, <https://github.com/RWTH-E3D/DigitalHub> | MIT (below) |
| `bimwhale_large_building.ifczip` | BIM Whale sample: Large Building | André "MG" Wisén, <https://github.com/andrewisen/bim-whale-ifc-samples> | MIT (below) |
| `zenodo6397164_model1.ifczip`, `_model2`, `_model3` | IFC architectural building models 1 to 3 | Tafraout Salim (2022). IFC architectural building models. Zenodo. [doi:10.5281/zenodo.6397164](https://doi.org/10.5281/zenodo.6397164) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `tum_fantasy_hotel.ifczip` | Fantasy hotel 1, architectural (student coursework) | TUM Students, BIM Fundamentals SS2025, via IFC-Bench (S. Hellin, Technical University of Munich, 2025), <https://huggingface.co/datasets/sylvainHellin/ifc-bench> | MIT (below) |
| `tum_fantasy_residential.ifczip` | Fantasy residential building 1, architectural (student coursework) | as above | MIT (below) |

**Changes.** Every file was changed in two ways, and in no other:

1. Personal and infrastructure data were removed: the header's comment block (the exporting computer's host name, a
   database path in the user's profile, the toolkit's licence id); the `FILE_NAME` name (a server path, or a file
   name holding a person's name), author, organization and authorization; the identification and names of every
   `IfcPerson` (set to `anonymised`); and an `IfcTelecomAddress` with an e-mail address (`gym_amersfoort` only).
   `manifest.json` lists what was removed from each file.
2. The file was renamed and packed as an `.ifczip`: a zip archive with the one `.ifc` inside (the buildingSMART
   container format; IfcOpenShell and most BIM tools open it directly).

The building's geometry, properties and relationships are unchanged: each file imports to the same plan as the
original download. Where a licence above names a creator who was named in a removed field, the credit in this table
keeps that name.

## NIBS Common BIM Files reuse terms

`wbdg_office.ifczip`, quoted from the original project page (now offline; archived copy:
<https://web.archive.org/web/20130310082112/http://www.buildingsmartalliance.org/index.php/projects/commonbimfiles>):

> People downloading these models may copy, distribute, display, and use these models and make derivative works
> based on these models if they provide proper citation of the source of these models through proper academic
> citation whenever these models are presented. Any future examples or extensions to the models subsequently
> published on this website may also be used according to the terms above. There is no cost or use restriction on
> these models in any for-profit or not-for-profit.

## MIT licence

`rwth_digitalhub.ifczip`: Copyright (c) 2020 RWTH Aachen University - E3D Institute of Energy Efficiency and
Sustainable Building

`bimwhale_large_building.ifczip`: Copyright (c) 2020 André "MG" Wisén

`tum_fantasy_hotel.ifczip`, `tum_fantasy_residential.ifczip`: Copyright (c) 2025 TUM Students, BIM Fundamentals
SS2025

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of
> the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
> THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

None of the creators or publishers above endorses this repository or bldg.
