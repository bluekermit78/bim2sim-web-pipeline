import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import numpy as np

try:
    import ifcopenshell
except ImportError:
    ifcopenshell = None


def _clean_id(raw: str, prefix: str = "id") -> str:
    cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(raw).strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _poly_normal_and_area(pts):
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0
    p0 = np.array(pts[0], dtype=float)
    cross_sum = np.zeros(3, dtype=float)
    for i in range(1, len(pts) - 1):
        v1 = np.array(pts[i], dtype=float) - p0
        v2 = np.array(pts[i + 1], dtype=float) - p0
        cross_sum += np.cross(v1, v2)
    norm_len = float(np.linalg.norm(cross_sum))
    if norm_len < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return cross_sum / norm_len, 0.5 * norm_len


def _poly_dimensions(pts, area):
    """Berechnet reale Breite und Höhe eines Polygons für RectangularGeometry."""
    if len(pts) < 3:
        return 1.0, 1.0
    arr = np.array(pts, dtype=float)
    dz = float(np.max(arr[:, 2]) - np.min(arr[:, 2]))
    if dz > 0.1:  # Vertikale oder geneigte Fläche (Wand / Fenster / Schrägdach)
        height = dz
        width = max(0.1, area / height)
    else:         # Horizontale Fläche (Boden / Flachdach)
        dx = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
        dy = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
        width = max(0.1, dx)
        height = max(0.1, dy)
    return round(width, 3), round(height, 3)


def _azimuth_and_tilt(normal):
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    nz_clamped = max(-1.0, min(1.0, nz))
    tilt = math.degrees(math.acos(nz_clamped))
    if math.hypot(nx, ny) < 1e-6:
        azimuth = 0.0
    else:
        azimuth = (math.degrees(math.atan2(nx, ny)) + 360.0) % 360.0
    return round(azimuth, 2), round(tilt, 2)


def _add_polyloop(parent_el, pts):
    loop = ET.SubElement(parent_el, "PolyLoop")
    for p in pts:
        cp = ET.SubElement(loop, "CartesianPoint")
        for coord in p[:3]:
            c_el = ET.SubElement(cp, "Coordinate")
            c_el.text = f"{float(coord):.4f}"
    return loop


def _parse_idf_objects(idf_path: Path):
    objects = {}
    txt = idf_path.read_text(encoding="utf-8", errors="ignore")
    for raw_block in txt.split(";"):
        lines = []
        for line in raw_block.splitlines():
            code_part = line.split("!")[0].strip()
            if code_part:
                lines.append(code_part)
        if not lines:
            continue
        fields = [f.strip() for f in " ".join(lines).split(",")]
        obj_type = fields[0].upper()
        objects.setdefault(obj_type, []).append(fields[1:])
    return objects


def _extract_ifc_space_names(ifc_path: Path) -> dict:
    mapping = {}
    if not ifcopenshell or not ifc_path or not ifc_path.exists():
        return mapping
    try:
        f = ifcopenshell.open(str(ifc_path))
        for sp in f.by_type("IfcSpace"):
            guid = sp.GlobalId
            name = getattr(sp, "LongName", None) or getattr(sp, "Name", None) or guid
            if getattr(sp, "Name", None) and getattr(sp, "LongName", None) and sp.Name != sp.LongName:
                label = f"{sp.Name} - {sp.LongName}"
            else:
                label = str(name)
            mapping[guid.upper()] = label
    except Exception:
        pass
    return mapping


def generate_gbxml_from_job(project_dir: Path, output_gbxml_path: Path, ifc_path: Path = None) -> Path:
    idf_files = [f for f in project_dir.rglob("*.idf") if "Minimal.idf" not in str(f)]
    if not idf_files:
        raise FileNotFoundError("Keine exportierte IDF-Datei für den gbXML-Export gefunden.")
    idf_path = idf_files[0]
    idf_data = _parse_idf_objects(idf_path)
    ifc_space_names = _extract_ifc_space_names(ifc_path)

    # 1. Materialien aus IDF lesen
    materials = {}
    for m in idf_data.get("MATERIAL", []):
        if len(m) >= 6:
            m_name = m[0]
            m_id = _clean_id(m_name, "Mat")
            try:
                thick = float(m[2])
                cond = float(m[3])
                dens = float(m[4])
                spec_heat = float(m[5])
            except ValueError:
                thick, cond, dens, spec_heat = 0.2, 0.5, 1200.0, 1000.0
            materials[m_name.upper()] = {
                "id": m_id,
                "name": m_name,
                "thickness": max(0.002, thick),
                "conductivity": max(0.01, cond),
                "density": max(10.0, dens),
                "specific_heat": max(100.0, spec_heat),
                "r_value": round(thick / max(0.01, cond), 4)
            }

    glazings = {}
    for g in idf_data.get("WINDOWMATERIAL:GLAZING", []):
        if len(g) >= 4:
            g_name = g[0]
            try:
                thick = float(g[3]) if g[3] else 0.006
            except ValueError:
                thick = 0.006
            glazings[g_name.upper()] = {"name": g_name, "thickness": thick}

    # 2. Konstruktionen (Opak & Fenster) aufbauen
    constructions = {}
    window_types = {}
    for c in idf_data.get("CONSTRUCTION", []):
        if not c:
            continue
        c_name = c[0]
        layers = [lyr for lyr in c[1:] if lyr]
        if c_name.upper().startswith("WINDOW") or any(l.upper() in glazings for l in layers):
            wt_id = _clean_id(c_name, "WinType")
            window_types[c_name.upper()] = {
                "id": wt_id,
                "name": c_name,
                "u_value": 1.3,
                "shgc": 0.60,
                "vt": 0.70
            }
        else:
            c_id = _clean_id(c_name, "Cons")
            r_total = 0.17
            mat_refs = []
            for lyr in layers:
                m_info = materials.get(lyr.upper())
                if m_info:
                    r_total += m_info["r_value"]
                    mat_refs.append(m_info["id"])
            u_val = round(1.0 / max(0.15, r_total), 4)
            constructions[c_name.upper()] = {
                "id": c_id,
                "name": c_name,
                "u_value": u_val,
                "materials": mat_refs
            }

    if not constructions:
        materials["DEFAULT_MAT"] = {
            "id": "Mat_Default", "name": "Standard_Mauerwerk",
            "thickness": 0.30, "conductivity": 0.25, "density": 1400.0, "specific_heat": 1000.0, "r_value": 1.2
        }
        constructions["DEFAULT_CONS"] = {
            "id": "Cons_Default", "name": "Standard_Aussenwand",
            "u_value": 0.73, "materials": ["Mat_Default"]
        }
    default_cons_id = next(iter(constructions.values()))["id"]

    if not window_types:
        window_types["DEFAULT_WIN"] = {
            "id": "WinType_Default", "name": "WSV_2fach",
            "u_value": 1.3, "shgc": 0.60, "vt": 0.70
        }
    default_win_id = next(iter(window_types.values()))["id"]

    # 3. Zonen / Spaces aus IDF lesen
    zones = {}
    for z in idf_data.get("ZONE", []):
        if not z:
            continue
        z_name = z[0]
        z_key = z_name.upper()
        sp_id = _clean_id(z_name, "Space")
        vol = 0.0
        area = 0.0
        if len(z) > 7 and z[7] and z[7].lower() != "autocalculate":
            try:
                vol = float(z[7])
            except ValueError:
                pass
        if len(z) > 8 and z[8] and z[8].lower() != "autocalculate":
            try:
                area = float(z[8])
            except ValueError:
                pass
        zones[z_key] = {
            "id": sp_id,
            "guid": z_name,
            "name": ifc_space_names.get(z_key, f"Raum_{z_name[:8]}"),
            "volume": vol,
            "area": area,
            "surfaces": []
        }

    # 4. Fenster / Türen (FenestrationSurface:Detailed) einlesen
    #    WICHTIG: fen[0..7] = Metadaten, fen[8] = Number_of_Vertices, fen[9:] = X,Y,Z-Koordinaten!
    openings_by_base = {}
    for fen in idf_data.get("FENESTRATIONSURFACE:DETAILED", []):
        if len(fen) < 18:
            continue
        f_name = fen[0]
        f_type = fen[1].upper()
        f_cons = fen[2].upper()
        base_name = fen[3].upper()
        coords_raw = []
        for val in fen[9:]:
            if val != "":
                try:
                    coords_raw.append(float(val))
                except ValueError:
                    pass
        pts = [coords_raw[i:i+3] for i in range(0, (len(coords_raw) // 3) * 3, 3)]
        if len(pts) >= 3:
            openings_by_base.setdefault(base_name, []).append({
                "id": _clean_id(f_name, "Open"),
                "name": f_name,
                "is_door": "DOOR" in f_type,
                "cons_key": f_cons,
                "pts": pts
            })

    # 5. Opake Flächen (BuildingSurface:Detailed) einlesen
    #    WICHTIG: s[0..8] = 9 Metadaten-Felder, s[9] = Number_of_Vertices (z.B. 4),
    #    erst ab s[10:] beginnen Vertex_1_X, Vertex_1_Y, Vertex_1_Z!
    surfaces = []
    for s in idf_data.get("BUILDINGSURFACE:DETAILED", []):
        if len(s) < 19:
            continue
        s_name = s[0]
        s_type = s[1].upper()     # WALL, FLOOR, ROOF, CEILING
        s_cons = s[2].upper()
        s_zone = s[3].upper()
        s_bc = s[4].upper()       # OUTDOORS, GROUND, SURFACE, ADIABATIC
        s_bc_obj = s[5].upper() if len(s) > 5 else ""

        coords_raw = []
        for val in s[10:]:
            if val != "":
                try:
                    coords_raw.append(float(val))
                except ValueError:
                    pass
        pts = [coords_raw[i:i+3] for i in range(0, (len(coords_raw) // 3) * 3, 3)]
        if len(pts) < 3:
            continue

        normal, area = _poly_normal_and_area(pts)
        az, tilt = _azimuth_and_tilt(normal)
        avg_z = sum(p[2] for p in pts) / len(pts)

        surfaces.append({
            "id": _clean_id(s_name, "Surf"),
            "name": s_name,
            "key": s_name.upper(),
            "idf_type": s_type,
            "cons_key": s_cons,
            "zone_key": s_zone,
            "bc": s_bc,
            "bc_obj": s_bc_obj,
            "pts": pts,
            "area": area,
            "azimuth": az,
            "tilt": tilt,
            "avg_z": avg_z
        })

    # 5b. Horizontale Flächen je Zone anhand der echten Z-Koordinaten eindeutig in FLOOR (unten) und ROOF (oben) einordnen
    for zk, zd in zones.items():
        z_surfs = [s for s in surfaces if s["zone_key"] == zk]
        all_z = [p[2] for s in z_surfs for p in s["pts"]]
        if not all_z:
            continue
        z_min_z, z_max_z = min(all_z), max(all_z)
        z_mid = 0.5 * (z_min_z + z_max_z)

        for s in z_surfs:
            n, _ = _poly_normal_and_area(s["pts"])
            if abs(float(n[2])) > 0.35:  # Horizontale / geneigte Fläche
                if s["avg_z"] < z_mid:
                    s["idf_type"] = "FLOOR"
                    if s["bc"] != "SURFACE":
                        s["bc"] = "GROUND"
                    if float(n[2]) > 0:
                        s["pts"] = list(reversed(s["pts"]))
                        n, _ = _poly_normal_and_area(s["pts"])
                    s["azimuth"], s["tilt"] = _azimuth_and_tilt(n)
                else:
                    if s["bc"] != "SURFACE":
                        s["idf_type"] = "ROOF"
                        s["bc"] = "OUTDOORS"
                    if float(n[2]) < 0:
                        s["pts"] = list(reversed(s["pts"]))
                        n, _ = _poly_normal_and_area(s["pts"])
                    s["azimuth"], s["tilt"] = _azimuth_and_tilt(n)

        # Prüfen, ob Bodenplatte (FLOOR bei z_min_z) oder Dach (ROOF bei z_max_z) in der IDF fehlen:
        has_floor = any(s["idf_type"] == "FLOOR" for s in z_surfs)
        has_roof = any(s["idf_type"] in ("ROOF", "CEILING") for s in z_surfs)

        if not has_floor and has_roof and (z_max_z - z_min_z) > 1.0:
            roof_s = next(s for s in z_surfs if s["idf_type"] in ("ROOF", "CEILING"))
            fl_pts = [[p[0], p[1], z_min_z] for p in roof_s["pts"]]
            fn, f_area = _poly_normal_and_area(fl_pts)
            if float(fn[2]) > 0:
                fl_pts = list(reversed(fl_pts))
                fn, f_area = _poly_normal_and_area(fl_pts)
            f_az, f_tilt = _azimuth_and_tilt(fn)
            new_floor = {
                "id": f"Surf_Floor_{zd['id']}",
                "name": f"Bodenplatte_{zd['name']}",
                "key": f"AUTOFLOOR_{zk}",
                "idf_type": "FLOOR",
                "cons_key": "PROJECT FLOOR",
                "zone_key": zk,
                "bc": "GROUND",
                "bc_obj": "",
                "pts": fl_pts,
                "area": f_area,
                "azimuth": f_az,
                "tilt": f_tilt,
                "avg_z": z_min_z
            }
            surfaces.append(new_floor)
            z_surfs.append(new_floor)

        elif not has_roof and has_floor and (z_max_z - z_min_z) > 1.0:
            floor_s = next(s for s in z_surfs if s["idf_type"] == "FLOOR")
            rf_pts = [[p[0], p[1], z_max_z] for p in floor_s["pts"]]
            rn, r_area = _poly_normal_and_area(rf_pts)
            if float(rn[2]) < 0:
                rf_pts = list(reversed(rf_pts))
                rn, r_area = _poly_normal_and_area(rf_pts)
            r_az, r_tilt = _azimuth_and_tilt(rn)
            new_roof = {
                "id": f"Surf_Roof_{zd['id']}",
                "name": f"Dach_{zd['name']}",
                "key": f"AUTOROOF_{zk}",
                "idf_type": "ROOF",
                "cons_key": "PROJECT FLAT ROOF",
                "zone_key": zk,
                "bc": "OUTDOORS",
                "bc_obj": "",
                "pts": rf_pts,
                "area": r_area,
                "azimuth": r_az,
                "tilt": r_tilt,
                "avg_z": z_max_z
            }
            surfaces.append(new_roof)
            z_surfs.append(new_roof)

        zd["surfaces"] = [(s["idf_type"], s["pts"], s["area"], s["avg_z"]) for s in z_surfs]
        floor_areas = [s["area"] for s in z_surfs if s["idf_type"] == "FLOOR"]
        if floor_areas:
            zd["area"] = round(sum(floor_areas), 2)
        elif zd["area"] <= 0.01:
            roof_areas = [s["area"] for s in z_surfs if s["idf_type"] in ("ROOF", "CEILING")]
            zd["area"] = round(sum(roof_areas), 2) if roof_areas else 25.0

        if zd["volume"] <= 0.01:
            # 1. Versuche das exakt berechnete EnergyPlus-Volumen (wie in der Web-UI) zu übernehmen:
            try:
                from update_preview import parse_energyplus_summary
                ep_sum = parse_energyplus_summary(project_dir)
                ep_z = ep_sum.get("zones", {}).get(zk, {})
                if ep_z.get("volume_m3", 0.0) > 0.1:
                    zd["volume"] = float(ep_z["volume_m3"])
            except Exception:
                pass
        if zd["volume"] <= 0.01:
            # 2. Fallback: Geometrische Berechnung aus Trauf- und Firsthöhe (Quader + Giebelprisma)
            eave_zs = [min(p[2] for p in s["pts"]) for s in z_surfs if s["idf_type"] == "ROOF" and (max(p[2] for p in s["pts"]) - min(p[2] for p in s["pts"])) > 0.3]
            if eave_zs:
                z_eave = sum(eave_zs) / len(eave_zs)
                zd["volume"] = round(zd["area"] * ((z_eave - z_min_z) + 0.5 * (z_max_z - z_eave)), 2)
            else:
                zd["volume"] = round(zd["area"] * max(1.0, (z_max_z - z_min_z)), 2)

    surf_to_zone = {s["key"]: s["zone_key"] for s in surfaces}

    # 6. gbXML-Baum (Version 7.03 für OpenStudio 3.11 & EVEBI 14.3.8) aufbauen
    root = ET.Element("gbXML", {
        "xmlns": "http://www.gbxml.org/schema",
        "temperatureUnit": "C",
        "lengthUnit": "Meters",
        "areaUnit": "SquareMeters",
        "volumeUnit": "CubicMeters",
        "useSIUnitsForResults": "true",
        "version": "7.03"
    })

    campus = ET.SubElement(root, "Campus", {"id": "Campus_1"})
    loc = ET.SubElement(campus, "Location")
    ET.SubElement(loc, "Name").text = "Lippstadt_NRW_DE"
    ET.SubElement(loc, "Latitude").text = "51.67"
    ET.SubElement(loc, "Longitude").text = "8.34"
    ET.SubElement(loc, "Elevation").text = "79.0"

    total_bldg_area = round(sum(z["area"] for z in zones.values()), 2)
    bldg = ET.SubElement(campus, "Building", {"id": "Building_1", "buildingType": "SingleFamily"})
    ET.SubElement(bldg, "Name").text = idf_path.stem
    ET.SubElement(bldg, "Area").text = f"{total_bldg_area:.2f}"

    storey = ET.SubElement(bldg, "BuildingStorey", {"id": "Storey_1"})
    ET.SubElement(storey, "Name").text = "Erdgeschoss"
    ET.SubElement(storey, "Level").text = "0.00"

    for zk, zd in zones.items():
        sp_el = ET.SubElement(bldg, "Space", {
            "id": zd["id"],
            "buildingStoreyIdRef": "Storey_1",
            "zoneIdRef": f"Zone_{zd['id']}"
        })
        ET.SubElement(sp_el, "Name").text = zd["name"]
        ET.SubElement(sp_el, "Area").text = f"{zd['area']:.2f}"
        ET.SubElement(sp_el, "Volume").text = f"{zd['volume']:.2f}"

        shell_geom = ET.SubElement(sp_el, "ShellGeometry", {"id": f"Shell_{zd['id']}", "unit": "Meters"})
        closed_shell = ET.SubElement(shell_geom, "ClosedShell")
        for st, pts, a, az in zd["surfaces"]:
            _add_polyloop(closed_shell, pts)

    processed_pairs = set()
    for s in surfaces:
        pair_key = tuple(sorted([s["key"], s["bc_obj"]])) if s["bc"] == "SURFACE" and s["bc_obj"] else (s["key"],)
        if len(pair_key) == 2 and pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        idf_t = s["idf_type"]
        bc = s["bc"]
        adj_zone_1 = zones.get(s["zone_key"], {}).get("id")
        adj_zone_2 = zones.get(surf_to_zone.get(s["bc_obj"], ""), {}).get("id") if bc == "SURFACE" else None

        if idf_t == "ROOF":
            gb_type = "Roof"
            exposed = "true"
        elif idf_t == "CEILING":
            gb_type = "InteriorCeiling" if adj_zone_2 else "Roof"
            exposed = "false" if adj_zone_2 else "true"
        elif idf_t == "FLOOR":
            if bc == "GROUND" or not adj_zone_2:
                gb_type = "SlabOnGrade"
                exposed = "false"
            else:
                gb_type = "InteriorFloor"
                exposed = "false"
        else:
            if bc == "SURFACE" and adj_zone_2:
                gb_type = "InteriorWall"
                exposed = "false"
            elif bc == "GROUND":
                gb_type = "UndergroundWall"
                exposed = "false"
            else:
                gb_type = "ExteriorWall"
                exposed = "true"

        cons_id = constructions.get(s["cons_key"], {}).get("id", default_cons_id)
        surf_el = ET.SubElement(campus, "Surface", {
            "id": s["id"],
            "surfaceType": gb_type,
            "constructionIdRef": cons_id,
            "exposedToSun": exposed
        })
        ET.SubElement(surf_el, "Name").text = s["name"]

        if adj_zone_1:
            ET.SubElement(surf_el, "AdjacentSpaceId", {"spaceIdRef": adj_zone_1})
        if adj_zone_2 and adj_zone_2 != adj_zone_1:
            ET.SubElement(surf_el, "AdjacentSpaceId", {"spaceIdRef": adj_zone_2})

        w_surf, h_surf = _poly_dimensions(s["pts"], s["area"])
        rect_geom = ET.SubElement(surf_el, "RectangularGeometry")
        ET.SubElement(rect_geom, "Azimuth").text = f"{s['azimuth']:.2f}"
        ET.SubElement(rect_geom, "Tilt").text = f"{s['tilt']:.2f}"
        ET.SubElement(rect_geom, "Width").text = f"{w_surf:.3f}"
        ET.SubElement(rect_geom, "Height").text = f"{h_surf:.3f}"

        planar = ET.SubElement(surf_el, "PlanarGeometry")
        _add_polyloop(planar, s["pts"])

        for op in openings_by_base.get(s["key"], []):
            op_type = "NonSlidingDoor" if op["is_door"] else "FixedWindow"
            op_attrs = {"id": op["id"], "openingType": op_type}
            if not op["is_door"]:
                op_attrs["windowTypeIdRef"] = window_types.get(op["cons_key"], {}).get("id", default_win_id)
            else:
                op_attrs["constructionIdRef"] = constructions.get(op["cons_key"], {}).get("id", default_cons_id)

            op_el = ET.SubElement(surf_el, "Opening", op_attrs)
            ET.SubElement(op_el, "Name").text = op["name"]
            _, op_area = _poly_normal_and_area(op["pts"])
            w_op, h_op = _poly_dimensions(op["pts"], op_area)
            op_rg = ET.SubElement(op_el, "RectangularGeometry")
            ET.SubElement(op_rg, "Azimuth").text = f"{s['azimuth']:.2f}"
            ET.SubElement(op_rg, "Tilt").text = f"{s['tilt']:.2f}"
            ET.SubElement(op_rg, "Width").text = f"{w_op:.3f}"
            ET.SubElement(op_rg, "Height").text = f"{h_op:.3f}"
            op_planar = ET.SubElement(op_el, "PlanarGeometry")
            _add_polyloop(op_planar, op["pts"])

        ET.SubElement(surf_el, "CADObjectId").text = s["name"]

    for c_key, c_info in constructions.items():
        c_el = ET.SubElement(root, "Construction", {"id": c_info["id"]})
        ET.SubElement(c_el, "Name").text = c_info["name"]
        u_el = ET.SubElement(c_el, "U-value", {"unit": "WPerSquareMeterK"})
        u_el.text = f"{c_info['u_value']:.4f}"
        if c_info["materials"]:
            ET.SubElement(c_el, "LayerId", {"layerIdRef": f"Layer_{c_info['id']}"})

    for c_key, c_info in constructions.items():
        if c_info["materials"]:
            l_el = ET.SubElement(root, "Layer", {"id": f"Layer_{c_info['id']}"})
            for m_id in c_info["materials"]:
                ET.SubElement(l_el, "MaterialId", {"materialIdRef": m_id})

    for m_key, m_info in materials.items():
        m_el = ET.SubElement(root, "Material", {"id": m_info["id"]})
        ET.SubElement(m_el, "Name").text = m_info["name"]
        ET.SubElement(m_el, "Thickness", {"unit": "Meters"}).text = f"{m_info['thickness']:.4f}"
        ET.SubElement(m_el, "Conductivity", {"unit": "WPerMeterK"}).text = f"{m_info['conductivity']:.4f}"
        ET.SubElement(m_el, "Density", {"unit": "KgPerCubicMeter"}).text = f"{m_info['density']:.1f}"
        ET.SubElement(m_el, "SpecificHeat", {"unit": "JPerKgK"}).text = f"{m_info['specific_heat']:.1f}"
        ET.SubElement(m_el, "R-value", {"unit": "SquareMeterKPerW"}).text = f"{m_info['r_value']:.4f}"

    for wt_key, wt_info in window_types.items():
        wt_el = ET.SubElement(root, "WindowType", {"id": wt_info["id"]})
        ET.SubElement(wt_el, "Name").text = wt_info["name"]
        ET.SubElement(wt_el, "U-value", {"unit": "WPerSquareMeterK"}).text = f"{wt_info['u_value']:.2f}"
        ET.SubElement(wt_el, "SolarHeatGainCoeff", {"unit": "Fraction"}).text = f"{wt_info['shgc']:.2f}"
        ET.SubElement(wt_el, "Transmittance", {"type": "Visible", "unit": "Fraction"}).text = f"{wt_info['vt']:.2f}"

    for zk, zd in zones.items():
        tz_el = ET.SubElement(root, "Zone", {"id": f"Zone_{zd['id']}"})
        ET.SubElement(tz_el, "Name").text = zd["name"]
        ET.SubElement(tz_el, "DesignHeatT", {"unit": "C"}).text = "20.0"
        ET.SubElement(tz_el, "DesignCoolT", {"unit": "C"}).text = "26.0"

    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8")
    output_gbxml_path.write_bytes(pretty_xml)
    return output_gbxml_path
