import math
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import numpy as np

try:
    import ifcopenshell
except ImportError:
    ifcopenshell = None


def _clean_id(raw: str, prefix: str = "ID") -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(raw).strip())
    if not s:
        return f"{prefix}_0"
    if not s.startswith(f"{prefix}_"):
        s = f"{prefix}_{s}"
    return s


def _make_coord(val: float) -> ET.Element:
    el = ET.Element("Coordinate")
    el.text = f"{float(val):.4f}"
    return el


def _map_surface_type(idf_type: str, bc: str, tilt: float = 90.0) -> str:
    t = idf_type.upper()
    b = bc.upper()
    if "ROOF" in t or (t == "CEILING" and b == "OUTDOORS"):
        return "Roof"
    if "FLOOR" in t:
        return "SlabOnGrade" if b == "GROUND" else "InteriorFloor"
    if "CEILING" in t:
        return "Ceiling"
    if b == "OUTDOORS":
        return "ExteriorWall"
    if b == "GROUND":
        return "UndergroundWall"
    return "InteriorWall"


def _parse_idf_objects(idf_path: Path) -> dict:
    txt = idf_path.read_text(encoding="utf-8", errors="ignore")
    lines = [re.sub(r"!.*$", "", line).strip() for line in txt.splitlines()]
    clean = " ".join(l for l in lines if l)
    objects = {}
    for stmt in clean.split(";"):
        stmt = stmt.strip()
        if not stmt or "," not in stmt:
            continue
        fields = [f.strip() for f in stmt.split(",")]
        obj_type = fields[0].upper()
        objects.setdefault(obj_type, []).append(fields[1:])
    return objects


def _poly_normal_and_area(pts: list) -> tuple:
    if len(pts) < 3:
        return (0.0, 0.0, 1.0), 0.0
    arr = np.array(pts, dtype=float)
    p0 = arr[0]
    total_cross = np.zeros(3)
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - p0
        v2 = arr[i + 1] - p0
        total_cross += np.cross(v1, v2)
    norm_len = float(np.linalg.norm(total_cross))
    if norm_len < 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    normal = tuple((total_cross / norm_len).tolist())
    area = 0.5 * norm_len
    return normal, area


def _poly_dimensions(pts: list, area: float) -> tuple:
    if not pts or area <= 1e-6:
        return 1.0, 1.0
    arr = np.array(pts, dtype=float)
    z_span = float(np.max(arr[:, 2]) - np.min(arr[:, 2]))
    xy_span = float(np.hypot(np.max(arr[:, 0]) - np.min(arr[:, 0]), np.max(arr[:, 1]) - np.min(arr[:, 1])))
    if z_span > 0.1 and xy_span > 0.1:
        w = xy_span
        h = max(0.1, area / w)
    else:
        w = max(0.1, xy_span if xy_span > 0.1 else math.sqrt(area))
        h = max(0.1, area / w)
    return round(w, 3), round(h, 3)


def _azimuth_and_tilt(normal: tuple) -> tuple:
    nx, ny, nz = normal
    nz_clamped = max(-1.0, min(1.0, nz))
    tilt = math.degrees(math.acos(nz_clamped))
    if abs(nx) < 1e-6 and abs(ny) < 1e-6:
        azimuth = 0.0
    else:
        azimuth = (math.degrees(math.atan2(nx, ny)) + 360.0) % 360.0
    return round(azimuth, 2), round(tilt, 2)


def _extract_ifc_element_types(ifc_path: Path) -> dict:
    """
    Liest aus allen IFC-Dateien des Jobs zu jeder SpaceBoundary-GUID und Bauteil-GUID den in
    Blender/Bonsai vergebenen Bauteiltyp-Namen (IfcWindowType, IfcDoorType, IfcWallType, Name, ObjectType).
    """
    elem_map = {}
    if not ifcopenshell or not ifc_path:
        return elem_map

    ifc_files = [Path(ifc_path)]
    try:
        for parent in Path(ifc_path).resolve().parents:
            if parent.name == "jobs" or parent.parent.name == "jobs":
                job_root = parent if parent.parent.name == "jobs" else Path(ifc_path).resolve().parents[1]
                for cand in job_root.rglob("*.ifc"):
                    if cand not in ifc_files:
                        ifc_files.append(cand)
                break
    except Exception:
        pass

    def _get_type_label(el):
        if not el:
            return None
        for attr_name in ("IsTypedBy", "IsDefinedBy"):
            for rel in getattr(el, attr_name, []) or []:
                if rel.is_a("IfcRelDefinesByType") and getattr(rel, "RelatingType", None):
                    t_obj = rel.RelatingType
                    t_name = getattr(t_obj, "Name", None) or getattr(t_obj, "ElementType", None)
                    if t_name:
                        return str(t_name).strip()
        for rel in getattr(el, "HasAssociations", []) or []:
            if rel.is_a("IfcRelAssociatesMaterial") and getattr(rel, "RelatingMaterial", None):
                mat = rel.RelatingMaterial
                if mat.is_a("IfcMaterialLayerSetUsage") and getattr(mat, "ForLayerSet", None):
                    mat = mat.ForLayerSet
                ls_name = getattr(mat, "LayerSetName", None) or getattr(mat, "Name", None)
                if ls_name and not str(ls_name).startswith(("LayerSet_Ifc", "Standard", "Isolierverglasung")):
                    return str(ls_name).strip()
        obj_type = getattr(el, "ObjectType", None)
        if obj_type and str(obj_type).strip().lower() not in ("wall", "window", "slab", "roof", "door", "baseslab"):
            return str(obj_type).strip()
        el_name = getattr(el, "Name", None)
        if el_name and str(el_name).strip().lower() not in ("wall", "window", "slab", "roof", "door"):
            return str(el_name).strip()
        return el.is_a().replace("Ifc", "")

    for fpath in ifc_files:
        if not fpath or not fpath.exists():
            continue
        try:
            f = ifcopenshell.open(str(fpath))
            for el in f.by_type("IfcBuildingElement"):
                lbl = _get_type_label(el)
                if lbl:
                    elem_map[el.GlobalId.upper()] = {
                        "type_label": lbl,
                        "el_name": getattr(el, "Name", None) or el.GlobalId[:8],
                        "ifc_class": el.is_a(),
                    }
            for sb in f.by_type("IfcRelSpaceBoundary"):
                rel_el = getattr(sb, "RelatedBuildingElement", None)
                if rel_el:
                    lbl = _get_type_label(rel_el)
                    elem_map[sb.GlobalId.upper()] = {
                        "type_label": lbl or rel_el.is_a().replace("Ifc", ""),
                        "el_name": getattr(rel_el, "Name", None) or rel_el.GlobalId[:8],
                        "ifc_class": rel_el.is_a(),
                    }
        except Exception:
            pass
    return elem_map


def _extract_ifc_space_names(ifc_path: Path) -> dict:
    names = {}
    if not ifcopenshell or not ifc_path or not Path(ifc_path).exists():
        return names
    try:
        f = ifcopenshell.open(str(ifc_path))
        for sp in f.by_type("IfcSpace"):
            guid = sp.GlobalId.upper()
            n = getattr(sp, "LongName", None) or getattr(sp, "Name", None) or guid
            names[guid] = str(n)
    except Exception:
        pass
    return names


def generate_gbxml_from_job(project_dir: Path, output_gbxml_path: Path, ifc_path: Path = None) -> Path:
    idf_files = [f for f in project_dir.rglob("*.idf") if "Minimal.idf" not in str(f)]
    if not idf_files:
        raise FileNotFoundError("Keine exportierte IDF-Datei im Projektordner gefunden.")
    idf_path = idf_files[0]
    idf_data = _parse_idf_objects(idf_path)

    if ifc_path is None or not Path(ifc_path).exists():
        cand_ifcs = list(project_dir.rglob("*.ifc")) + list(project_dir.parent.glob("input/*.ifc"))
        ifc_path = cand_ifcs[0] if cand_ifcs else None

    ifc_space_names = _extract_ifc_space_names(ifc_path)
    ifc_elem_types = _extract_ifc_element_types(ifc_path)

    # 1. Materialien aus IDF lesen
    materials = {}
    for m in idf_data.get("MATERIAL", []):
        if len(m) < 6:
            continue
        m_name = m[0]
        try:
            thick = max(0.005, float(m[2]))
            cond = max(0.01, float(m[3]))
            dens = max(10.0, float(m[4]))
            sh = max(100.0, float(m[5]))
        except ValueError:
            thick, cond, dens, sh = 0.20, 0.35, 1200.0, 1000.0
        r_val = round(thick / cond, 4)
        materials[m_name.upper()] = {
            "id": _clean_id(m_name, "Mat"),
            "name": m_name,
            "thickness": thick,
            "conductivity": cond,
            "density": dens,
            "specific_heat": sh,
            "r_value": r_val,
        }

    if not materials:
        materials["DEFAULT_MAT"] = {
            "id": "Mat_Default",
            "name": "Standardbaustoff",
            "thickness": 0.25,
            "conductivity": 0.30,
            "density": 1400.0,
            "specific_heat": 1000.0,
            "r_value": 0.8333,
        }

    # 2. Konstruktionen & Fenstertypen aus IDF lesen
    constructions = {}
    window_types = {}
    for c in idf_data.get("CONSTRUCTION", []):
        if not c:
            continue
        cname = c[0]
        layers = [l.upper() for l in c[1:] if l]
        if cname.upper().startswith("WINDOW"):
            window_types[cname.upper()] = {
                "id": _clean_id(cname, "WinType"),
                "name": cname,
                "u_value": 1.30,
                "shgc": 0.60,
                "vt": 0.70,
            }
        else:
            valid_mats = [materials[l] for l in layers if l in materials]
            if not valid_mats:
                valid_mats = [next(iter(materials.values()))]
            r_total = 0.17 + sum(m["r_value"] for m in valid_mats)
            u_val = round(1.0 / max(0.15, r_total), 4)
            constructions[cname.upper()] = {
                "id": _clean_id(cname, "Cons"),
                "name": cname,
                "u_value": u_val,
                "materials": valid_mats,
            }

    if not constructions:
        first_mat = next(iter(materials.values()))
        constructions["DEFAULT_CONS"] = {
            "id": "Cons_Default",
            "name": "Standardkonstruktion",
            "u_value": 0.35,
            "materials": [first_mat],
        }
    if not window_types:
        window_types["DEFAULT_WIN"] = {
            "id": "WinType_Default",
            "name": "Standardfenster_2fach",
            "u_value": 1.30,
            "shgc": 0.60,
            "vt": 0.70,
        }

    # 3. Zonen / Spaces aus IDF lesen
    zones = {}
    for idx, z in enumerate(idf_data.get("ZONE", []), 1):
        if not z:
            continue
        zname = z[0]
        vol = 0.0
        area = 0.0
        # EnergyPlus ZONE Felder: z[7]=Ceiling Height, z[8]=Volume, z[9]=Floor Area
        if len(z) > 8 and z[8] and z[8].lower() != "autocalculate":
            try:
                vol = float(z[8])
            except ValueError:
                pass
        if len(z) > 9 and z[9] and z[9].lower() != "autocalculate":
            try:
                area = float(z[9])
            except ValueError:
                pass
        label = ifc_space_names.get(zname.upper(), f"Raum_{idx}_{zname[:8]}")
        zones[zname.upper()] = {
            "id": _clean_id(zname, "Space"),
            "name": label,
            "cad_id": zname,
            "volume": vol,
            "area": area,
        }

    # 4. Fenster & Türen (FENESTRATIONSURFACE:DETAILED) nach Basis-Wand gruppieren
    openings_by_base = {}
    for fen in idf_data.get("FENESTRATIONSURFACE:DETAILED", []):
        if len(fen) < 18:
            continue
        f_name = fen[0]
        f_type = fen[1].upper()
        f_cons = fen[2].upper()
        base_name = fen[3].upper()
        coords = []
        for val in fen[9:]:
            if val != "":
                try:
                    coords.append(float(val))
                except ValueError:
                    pass
        pts = [coords[i:i+3] for i in range(0, (len(coords) // 3) * 3, 3)]
        if len(pts) < 3:
            continue
        norm, area = _poly_normal_and_area(pts)
        w, h = _poly_dimensions(pts, area)
        az, tilt = _azimuth_and_tilt(norm)
        openings_by_base.setdefault(base_name, []).append({
            "id": _clean_id(f_name, "Open"),
            "name": f_name,
            "is_door": ("DOOR" in f_type or "DOOR" in f_cons),
            "cons_key": f_cons,
            "pts": pts,
            "norm": norm,
            "area": round(area, 3),
            "width": w,
            "height": h,
            "azimuth": az,
            "tilt": tilt,
        })

    # 5. Opake Hüllflächen (BUILDINGSURFACE:DETAILED) lesen
    surfaces = []
    surfaces_by_zone = {}
    for s in idf_data.get("BUILDINGSURFACE:DETAILED", []):
        if len(s) < 19:
            continue
        s_name = s[0]
        s_type = s[1].upper()
        s_cons = s[2].upper()
        s_zone = s[3].upper()
        s_bc = s[4].upper()
        coords = []
        for val in s[10:]:
            if val != "":
                try:
                    coords.append(float(val))
                except ValueError:
                    pass
        pts = [coords[i:i+3] for i in range(0, (len(coords) // 3) * 3, 3)]
        if len(pts) < 3:
            continue
        norm, area = _poly_normal_and_area(pts)
        w, h = _poly_dimensions(pts, area)
        az, tilt = _azimuth_and_tilt(norm)
        avg_z = sum(p[2] for p in pts) / len(pts)

        s_dict = {
            "id": _clean_id(s_name, "Surf"),
            "key": s_name.upper(),
            "name": s_name,
            "idf_type": s_type,
            "cons_key": s_cons,
            "zone_key": s_zone,
            "bc": s_bc,
            "pts": pts,
            "norm": norm,
            "area": round(area, 3),
            "width": w,
            "height": h,
            "azimuth": az,
            "tilt": tilt,
            "avg_z": avg_z,
            "openings": openings_by_base.get(s_name.upper(), []),
        }
        surfaces.append(s_dict)
        surfaces_by_zone.setdefault(s_zone, []).append(s_dict)

    # 6. Zonen-Geometrie, Floor/Roof-Klassifikation und exaktes Volumen (aus EnergyPlus / Giebelformel)
    try:
        from update_preview import parse_energyplus_summary
        ep_summary = parse_energyplus_summary(project_dir)
    except Exception:
        ep_summary = {}

    for zk, zd in zones.items():
        z_surfs = surfaces_by_zone.get(zk, [])
        if not z_surfs:
            continue
        all_z = [p[2] for s in z_surfs for p in s["pts"]]
        z_min_z, z_max_z = min(all_z), max(all_z)
        z_mid = 0.5 * (z_min_z + z_max_z)

        has_floor = False
        has_roof = False
        roof_s = None
        for s in z_surfs:
            if s["tilt"] < 65.0 or s["tilt"] > 115.0:
                if s["avg_z"] < z_mid:
                    s["idf_type"] = "FLOOR"
                    s["bc"] = "GROUND"
                    s["tilt"] = 180.0
                    has_floor = True
                else:
                    s["idf_type"] = "ROOF"
                    s["bc"] = "OUTDOORS"
                    has_roof = True
                    roof_s = s

        if not has_floor and has_roof and roof_s is not None:
            fl_pts = [[p[0], p[1], z_min_z] for p in reversed(roof_s["pts"])]
            fl_norm, fl_area = _poly_normal_and_area(fl_pts)
            fl_w, fl_h = _poly_dimensions(fl_pts, fl_area)
            fl_dict = {
                "id": _clean_id(f"AutoFloor_{zk[:8]}", "Surf"),
                "key": f"AUTOFLOOR_{zk}",
                "name": f"Floor_{zk[:8]}",
                "idf_type": "FLOOR",
                "cons_key": roof_s["cons_key"],
                "zone_key": zk,
                "bc": "GROUND",
                "pts": fl_pts,
                "norm": fl_norm,
                "area": round(fl_area, 3),
                "width": fl_w,
                "height": fl_h,
                "azimuth": 0.0,
                "tilt": 180.0,
                "avg_z": z_min_z,
                "openings": [],
            }
            surfaces.append(fl_dict)
            z_surfs.append(fl_dict)

        floors = [s for s in z_surfs if s["idf_type"] == "FLOOR"]
        if floors:
            zd["area"] = round(sum(s["area"] for s in floors), 2)
        elif ep_z := ep_summary.get("zones", {}).get(zk, {}):
            if ep_z.get("area_m2", 0.0) > 0.1:
                zd["area"] = round(float(ep_z["area_m2"]), 2)

        ep_z = ep_summary.get("zones", {}).get(zk, {})
        if ep_z.get("volume_m3", 0.0) > 0.1:
            zd["volume"] = round(float(ep_z["volume_m3"]), 2)
        if zd["volume"] <= 0.01:
            eave_zs = [min(p[2] for p in s["pts"]) for s in z_surfs if s["idf_type"] == "ROOF" and (max(p[2] for p in s["pts"]) - min(p[2] for p in s["pts"])) > 0.25]
            if eave_zs:
                z_eave = sum(eave_zs) / len(eave_zs)
                zd["volume"] = round(zd["area"] * ((z_eave - z_min_z) + 0.5 * (z_max_z - z_eave)), 2)
            else:
                zd["volume"] = round(zd["area"] * max(1.0, (z_max_z - z_min_z)), 2)

    # 7. XML-Baum nach gbXML v7.03 Schema aufbauen
    root = ET.Element("gbXML", {
        "xmlns": "http://www.gbxml.org/schema",
        "version": "7.03",
        "temperatureUnit": "C",
        "lengthUnit": "Meters",
        "areaUnit": "SquareMeters",
        "volumeUnit": "CubicMeters",
        "useSIUnitsForResults": "true",
    })

    campus = ET.SubElement(root, "Campus", {"id": "Campus_1"})
    ET.SubElement(campus, "Name").text = idf_path.stem
    loc = ET.SubElement(campus, "Location")
    ET.SubElement(loc, "Name").text = "Lippstadt_DE"
    ET.SubElement(loc, "Latitude").text = "51.67"
    ET.SubElement(loc, "Longitude").text = "8.34"

    total_bldg_area = round(sum(z["area"] for z in zones.values()), 2)
    bldg = ET.SubElement(campus, "Building", {"id": "Bldg_1", "buildingType": "Office"})
    ET.SubElement(bldg, "Name").text = idf_path.stem
    ET.SubElement(bldg, "Area").text = f"{max(10.0, total_bldg_area):.2f}"
    storey = ET.SubElement(bldg, "BuildingStorey", {"id": "Storey_1"})
    ET.SubElement(storey, "Name").text = "Erdgeschoss"
    ET.SubElement(storey, "Level").text = "0.00"

    for zk, zd in zones.items():
        sp_el = ET.SubElement(bldg, "Space", {
            "id": zd["id"],
            "buildingStoreyIdRef": "Storey_1",
            "zoneIdRef": f"Zone_{zd['id']}",
        })
        ET.SubElement(sp_el, "Name").text = zd["name"]
        ET.SubElement(sp_el, "Area").text = f"{max(1.0, zd['area']):.2f}"
        ET.SubElement(sp_el, "Volume").text = f"{max(1.0, zd['volume']):.2f}"

        z_surfs = surfaces_by_zone.get(zk, [])
        if z_surfs:
            sg = ET.SubElement(sp_el, "ShellGeometry", {"id": f"SG_{zd['id']}", "unit": "Meters"})
            cs = ET.SubElement(sg, "ClosedShell")
            for s in z_surfs:
                pl = ET.SubElement(cs, "PolyLoop")
                for pt in s["pts"]:
                    cp = ET.SubElement(pl, "CartesianPoint")
                    for val in pt:
                        ET.SubElement(cp, "Coordinate").text = f"{val:.4f}"

        ET.SubElement(sp_el, "CADObjectId").text = zd["cad_id"]

    # 8. Surfaces + untergeordnete Openings (XSD-konform NACH PlanarGeometry!)
    for s in surfaces:
        gb_type = _map_surface_type(s["idf_type"], s["bc"], s["tilt"])
        exposed = "true" if s["bc"] == "OUTDOORS" else "false"
        ifc_info = ifc_elem_types.get(s["key"])
        base_c = constructions.get(s["cons_key"], next(iter(constructions.values())))

        if ifc_info and ifc_info.get("type_label"):
            t_lbl = ifc_info["type_label"]
            custom_c_key = f"{s['cons_key']}__{t_lbl.upper()}"
            if custom_c_key not in constructions:
                constructions[custom_c_key] = {
                    "id": _clean_id(f"{t_lbl}_{s['cons_key'][:10]}", "Cons"),
                    "name": f"{t_lbl} ({base_c['name']})",
                    "u_value": base_c["u_value"],
                    "materials": list(base_c["materials"]),
                }
            cons_id = constructions[custom_c_key]["id"]
            s_display_name = f"{t_lbl}_{s['name'][:8]}"
        else:
            cons_id = base_c["id"]
            s_display_name = s["name"]

        surf_el = ET.SubElement(campus, "Surface", {
            "id": s["id"],
            "surfaceType": gb_type,
            "constructionIdRef": cons_id,
            "exposedToSun": exposed,
        })
        ET.SubElement(surf_el, "Name").text = s_display_name

        z_id = zones.get(s["zone_key"], {}).get("id")
        if z_id:
            ET.SubElement(surf_el, "AdjacentSpaceId", {"spaceIdRef": z_id})
            if s["bc"] not in ("OUTDOORS", "GROUND"):
                other_z = zones.get(s["bc"], {}).get("id", z_id)
                ET.SubElement(surf_el, "AdjacentSpaceId", {"spaceIdRef": other_z})

        rg = ET.SubElement(surf_el, "RectangularGeometry")
        ET.SubElement(rg, "Azimuth").text = f"{s['azimuth']:.2f}"
        ET.SubElement(rg, "CartesianPoint").extend([_make_coord(v) for v in s["pts"][0]])
        ET.SubElement(rg, "Tilt").text = f"{s['tilt']:.2f}"
        ET.SubElement(rg, "Width").text = f"{s['width']:.3f}"
        ET.SubElement(rg, "Height").text = f"{s['height']:.3f}"

        pg = ET.SubElement(surf_el, "PlanarGeometry")
        pl = ET.SubElement(pg, "PolyLoop")
        for pt in s["pts"]:
            cp = ET.SubElement(pl, "CartesianPoint")
            for val in pt:
                ET.SubElement(cp, "Coordinate").text = f"{val:.4f}"

        s_p0 = s["pts"][0]
        s_norm = s["norm"]
        for op in s["openings"]:
            op_type = "NonSlidingDoor" if op["is_door"] else "FixedWindow"
            op_attrs = {
                "id": op["id"],
                "openingType": op_type,
            }
            op_ifc_info = ifc_elem_types.get(op["name"].upper())
            t_lbl = op_ifc_info.get("type_label") if op_ifc_info else None

            if not op["is_door"]:
                base_wt = window_types.get(op["cons_key"], next(iter(window_types.values())))
                if t_lbl:
                    custom_wt_key = f"{op['cons_key']}__{t_lbl.upper()}"
                    if custom_wt_key not in window_types:
                        window_types[custom_wt_key] = {
                            "id": _clean_id(t_lbl, "WinType"),
                            "name": f"{t_lbl}",
                            "u_value": base_wt["u_value"],
                            "shgc": base_wt["shgc"],
                            "vt": base_wt["vt"],
                        }
                    op_attrs["windowTypeIdRef"] = window_types[custom_wt_key]["id"]
                else:
                    op_attrs["windowTypeIdRef"] = base_wt["id"]
            else:
                door_c = constructions.get(op["cons_key"], base_c)
                if t_lbl:
                    custom_dc_key = f"{op['cons_key']}__{t_lbl.upper()}"
                    if custom_dc_key not in constructions:
                        constructions[custom_dc_key] = {
                            "id": _clean_id(f"{t_lbl}_Door", "Cons"),
                            "name": f"{t_lbl} ({door_c['name']})",
                            "u_value": door_c["u_value"],
                            "materials": list(door_c["materials"]),
                        }
                    op_attrs["constructionIdRef"] = constructions[custom_dc_key]["id"]
                else:
                    op_attrs["constructionIdRef"] = door_c["id"]

            op_el = ET.SubElement(surf_el, "Opening", op_attrs)
            op_disp_name = f"{t_lbl}_{op['name'][:8]}" if t_lbl else op["name"]
            ET.SubElement(op_el, "Name").text = op_disp_name

            op_pts_proj = []
            for pt in op["pts"]:
                d_err = (pt[0] - s_p0[0])*s_norm[0] + (pt[1] - s_p0[1])*s_norm[1] + (pt[2] - s_p0[2])*s_norm[2]
                op_pts_proj.append([
                    pt[0] - d_err * s_norm[0],
                    pt[1] - d_err * s_norm[1],
                    pt[2] - d_err * s_norm[2],
                ])
            op_norm, _ = _poly_normal_and_area(op_pts_proj)
            if (op_norm[0]*s_norm[0] + op_norm[1]*s_norm[1] + op_norm[2]*s_norm[2]) < 0:
                op_pts_proj = list(reversed(op_pts_proj))

            org = ET.SubElement(op_el, "RectangularGeometry")
            ET.SubElement(org, "Azimuth").text = f"{s['azimuth']:.2f}"
            ET.SubElement(org, "CartesianPoint").extend([_make_coord(v) for v in op_pts_proj[0]])
            ET.SubElement(org, "Tilt").text = f"{s['tilt']:.2f}"
            ET.SubElement(org, "Width").text = f"{op['width']:.3f}"
            ET.SubElement(org, "Height").text = f"{op['height']:.3f}"

            opg = ET.SubElement(op_el, "PlanarGeometry")
            opl = ET.SubElement(opg, "PolyLoop")
            for pt in op_pts_proj:
                ocp = ET.SubElement(opl, "CartesianPoint")
                for val in pt:
                    ET.SubElement(ocp, "Coordinate").text = f"{val:.4f}"

        ET.SubElement(surf_el, "CADObjectId").text = s["name"]

    # 9. WindowTypes, Constructions, Layers, Materials & Zones anhängen
    for wt in window_types.values():
        wt_el = ET.SubElement(root, "WindowType", {"id": wt["id"]})
        ET.SubElement(wt_el, "Name").text = wt["name"]
        ET.SubElement(wt_el, "U-value", {"unit": "WPerSquareMeterK"}).text = f"{wt['u_value']:.3f}"
        ET.SubElement(wt_el, "SolarHeatGainCoeff", {"unit": "Fraction"}).text = f"{wt['shgc']:.3f}"
        ET.SubElement(wt_el, "Transmittance", {"type": "Visible", "unit": "Fraction"}).text = f"{wt['vt']:.3f}"

    written_layers = set()
    for c in constructions.values():
        c_el = ET.SubElement(root, "Construction", {"id": c["id"]})
        ET.SubElement(c_el, "Name").text = c["name"]
        ET.SubElement(c_el, "U-value", {"unit": "WPerSquareMeterK"}).text = f"{c['u_value']:.4f}"
        lid = f"Layer_{c['id']}"
        ET.SubElement(c_el, "LayerId", {"layerIdRef": lid})
        if lid not in written_layers:
            written_layers.add(lid)
            l_el = ET.SubElement(root, "Layer", {"id": lid})
            for m in c["materials"]:
                ET.SubElement(l_el, "MaterialId", {"materialIdRef": m["id"]})

    for m in materials.values():
        m_el = ET.SubElement(root, "Material", {"id": m["id"]})
        ET.SubElement(m_el, "Name").text = m["name"]
        ET.SubElement(m_el, "Thickness", {"unit": "Meters"}).text = f"{m['thickness']:.4f}"
        ET.SubElement(m_el, "Conductivity", {"unit": "WPerMeterK"}).text = f"{m['conductivity']:.4f}"
        ET.SubElement(m_el, "Density", {"unit": "KgPerCubicMeter"}).text = f"{m['density']:.1f}"
        ET.SubElement(m_el, "SpecificHeat", {"unit": "JPerKgK"}).text = f"{m['specific_heat']:.1f}"
        ET.SubElement(m_el, "R-value", {"unit": "SquareMeterKPerW"}).text = f"{m['r_value']:.4f}"

    for zk, zd in zones.items():
        z_el = ET.SubElement(root, "Zone", {"id": f"Zone_{zd['id']}"})
        ET.SubElement(z_el, "Name").text = zd["name"]
        ET.SubElement(z_el, "DesignHeatT", {"unit": "C"}).text = "20.0"
        ET.SubElement(z_el, "DesignCoolT", {"unit": "C"}).text = "26.0"

    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8")
    output_gbxml_path.write_bytes(pretty_xml)
    return output_gbxml_path
