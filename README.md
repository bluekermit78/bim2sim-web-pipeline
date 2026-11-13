# BIM2SIM Web-Pipeline (IFC -> EnergyPlus / OpenStudio / EVEBI)

Automatisierte Konvertierungs- und Simulations-Pipeline auf Basis von **IfcOpenShell / OpenCASCADE**, **bim2sim** und **EnergyPlus 9.4** mit direktem Export für **OpenStudio (gbXML 7.03)** und **ENVISYS EVEBI 14.3.8**.

## Kernmodule
- `app.py` – FastAPI Web-UI & Job-Steuerung
- `sb_generator.py` – OpenCASCADE Pre-Processor (erzeugt 2nd Level Space Boundaries inkl. Satteldach-/Giebeldach-Verschmelzung und `IfcSpace`-Ebenenprojektion)
- `idf_sanitizer.py` – Heilt EnergyPlus-IDF-Geometrien (Fenster-Subsurfaces, Normalenvektoren, Zonen-Grundflächen & Volumina)
- `gbxml_exporter.py` – Erzeugt valide `gbXML v7.03` (`_gbxml.xml`) für OpenStudio 3.11 & EVEBI 14.3.8
- `update_preview.py` – Parser für EnergyPlus-Ergebnisse (Heizlast, Kühllast, Jahreswärmebedarf, Raumtemperaturen)
- `geometry_report.py` – HTML-Geometriebericht (NGF, Volumen, mittlere Raumhöhe $\bar{h} = V/A$ für EVEBI, Bauteil-U-Werte)
- `bim2sim_patches.diff` – Notwendige Patches am `bim2sim`-Framework (wird via `./apply_bim2sim_patches.sh` angewendet)
