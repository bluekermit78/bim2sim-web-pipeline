import numpy as np
from pathlib import Path

def _calc_normal(pts):
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    p0 = np.array(pts[0], dtype=float)
    p1 = np.array(pts[1], dtype=float)
    p2 = np.array(pts[2], dtype=float)
    n = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(n)
    return n / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])

def _calc_poly_area_3d(pts) -> float:
    if len(pts) < 3:
        return 0.0
    p0 = np.array(pts[0], dtype=float)
    total = np.zeros(3, dtype=float)
    for i in range(1, len(pts) - 1):
        v1 = np.array(pts[i], dtype=float) - p0
        v2 = np.array(pts[i + 1], dtype=float) - p0
        total += np.cross(v1, v2)
    return 0.5 * float(np.linalg.norm(total))

def _set_vertices(obj, pts):
    obj.Number_of_Vertices = len(pts)
    for idx, pt in enumerate(pts, start=1):
        setattr(obj, f"Vertex_{idx}_Xcoordinate", round(float(pt[0]), 4))
        setattr(obj, f"Vertex_{idx}_Ycoordinate", round(float(pt[1]), 4))
        setattr(obj, f"Vertex_{idx}_Zcoordinate", round(float(pt[2]), 4))

def sanitize_idf(idf, logger=None, export_dir=None):
    # 0a. Output:Diagnostics bereinigen
    for d in list(idf.idfobjects.get("OUTPUT:DIAGNOSTICS", [])):
        idf.removeidfobject(d)

    # 0b. Doppelte Objekte gleichen Typs und gleichen Namens entfernen
    dedup_removed = 0
    for obj_type, obj_list in list(idf.idfobjects.items()):
        if len(obj_list) <= 1:
            continue
        seen_names = set()
        for obj in reversed(list(obj_list)):
            name_val = getattr(obj, "Name", None)
            if name_val:
                key = str(name_val).strip().upper()
                if key in seen_names:
                    idf.removeidfobject(obj)
                    dedup_removed += 1
                else:
                    seen_names.add(key)

    bldg_surfaces = list(idf.idfobjects.get("BUILDINGSURFACE:DETAILED", []))
    opaque_consts = [
        c.Name for c in idf.idfobjects.get("CONSTRUCTION", [])
        if not c.Name.upper().startswith("WINDOW")
    ]
    default_wall_const = opaque_consts[0] if opaque_consts else "Project Wall"

    # 1. Alle BuildingSurface:Detailed reparieren, die fälschlich eine WINDOW_-Construction haben
    converted_walls = 0
    for s in bldg_surfaces:
        cname = str(getattr(s, "Construction_Name", ""))
        if cname.upper().startswith("WINDOW"):
            coords = list(getattr(s, "coords", []))
            if len(coords) >= 3:
                orig_name = s.Name
                host_name = f"HostWall_{orig_name[:18]}"
                s.Name = host_name
                s.Surface_Type = "Wall"
                s.Construction_Name = default_wall_const
                if str(getattr(s, "Outside_Boundary_Condition", "")).upper() == "SURFACE":
                    s.Outside_Boundary_Condition = "Outdoors"
                    s.Outside_Boundary_Condition_Object = ""
                    s.Sun_Exposure = "SunExposed"
                    s.Wind_Exposure = "WindExposed"

                win_pts = coords[:4]
                cx = sum(p[0] for p in win_pts) / len(win_pts)
                cy = sum(p[1] for p in win_pts) / len(win_pts)
                cz = sum(p[2] for p in win_pts) / len(win_pts)
                shrunk = [
                    (cx + (p[0] - cx) * 0.96, cy + (p[1] - cy) * 0.96, cz + (p[2] - cz) * 0.96)
                    for p in win_pts
                ]
                fen = idf.newidfobject(
                    "FENESTRATIONSURFACE:DETAILED",
                    Name=orig_name,
                    Surface_Type="Window",
                    Construction_Name=cname,
                    Building_Surface_Name=host_name,
                    View_Factor_to_Ground="autocalculate",
                )
                _set_vertices(fen, shrunk)
                converted_walls += 1

    # 1b. Horizontale Flächen je Zone nach echter Z-Höhe in Floor (unten) und Roof (oben) trennen
    surf_by_zone = {}
    for s in idf.idfobjects.get("BUILDINGSURFACE:DETAILED", []):
        zname = str(getattr(s, "Zone_Name", "")).strip()
        if zname:
            surf_by_zone.setdefault(zname.upper(), []).append(s)

    for z_obj in idf.idfobjects.get("ZONE", []):
        zk = str(getattr(z_obj, "Name", "")).strip().upper()
        z_surfs = surf_by_zone.get(zk, [])
        all_z = [p[2] for s in z_surfs for p in getattr(s, "coords", [])]
        if not all_z:
            continue
        z_min_zone, z_max_zone = min(all_z), max(all_z)
        z_mid = 0.5 * (z_min_zone + z_max_zone)

        floor_area = 0.0
        for s in z_surfs:
            coords = list(getattr(s, "coords", []))
            if len(coords) >= 3:
                n = _calc_normal(coords)
                if abs(float(n[2])) > 0.35:
                    avg_z = sum(p[2] for p in coords) / len(coords)
                    if avg_z < z_mid:
                        s.Surface_Type = "Floor"
                        if float(n[2]) > 0:
                            _set_vertices(s, list(reversed(coords)))
                        floor_area += _calc_poly_area_3d(coords)
                    else:
                        if str(getattr(s, "Outside_Boundary_Condition", "")).upper() != "SURFACE":
                            s.Surface_Type = "Roof"
                            s.Outside_Boundary_Condition = "Outdoors"
                            s.Outside_Boundary_Condition_Object = ""
                            s.Sun_Exposure = "SunExposed"
                            s.Wind_Exposure = "WindExposed"
                        if float(n[2]) < 0:
                            _set_vertices(s, list(reversed(coords)))
        # Falls <Slab> in bim2sim wegen fehlendem LayerSet nicht exportiert wurde -> Bodenplatte aus Dach-Grundriss auf z_min_zone erzeugen!
        if floor_area == 0.0 and (z_max_zone - z_min_zone) > 1.0:
            roof_candidates = [
                (s, list(getattr(s, "coords", [])))
                for s in z_surfs
                if len(getattr(s, "coords", [])) >= 3 and abs(float(_calc_normal(getattr(s, "coords", []))[2])) > 0.35
            ]
            if roof_candidates:
                _, r_coords = roof_candidates[0]
                fl_coords = [(p[0], p[1], z_min_zone) for p in r_coords]
                if float(_calc_normal(fl_coords)[2]) > 0:
                    fl_coords = list(reversed(fl_coords))
                default_floor_c = next((c for c in opaque_consts if "FLOOR" in c.upper() or "SLAB" in c.upper()), default_wall_const)
                fl_surf = idf.newidfobject(
                    "BUILDINGSURFACE:DETAILED",
                    Name=f"AutoFloor_{z_obj.Name[:13]}",
                    Surface_Type="Floor",
                    Construction_Name=default_floor_c,
                    Zone_Name=z_obj.Name,
                    Outside_Boundary_Condition="Ground",
                    Outside_Boundary_Condition_Object="",
                    Sun_Exposure="NoSun",
                    Wind_Exposure="NoWind",
                    View_Factor_to_Ground="autocalculate",
                )
                _set_vertices(fl_surf, fl_coords)
                floor_area = _calc_poly_area_3d(fl_coords)

        if floor_area > 0:
            z_obj.Floor_Area = round(floor_area, 2)

    # 2. Alle FenestrationSurface:Detailed prüfen: Zeigt der Normalenvektor in dieselbe Richtung wie die Base Surface?
    base_map = {s.Name: s for s in idf.idfobjects.get("BUILDINGSURFACE:DETAILED", [])}
    flipped_subsurfaces = 0
    for fen in list(idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", [])):
        base_name = getattr(fen, "Building_Surface_Name", "")
        base_surf = base_map.get(base_name)
        if not base_surf:
            continue
        f_coords = list(getattr(fen, "coords", []))
        b_coords = list(getattr(base_surf, "coords", []))
        if len(f_coords) >= 3 and len(b_coords) >= 3:
            n_fen = _calc_normal(f_coords)
            n_base = _calc_normal(b_coords)
            if float(np.dot(n_fen, n_base)) < 0.0:
                _set_vertices(fen, list(reversed(f_coords)))
                flipped_subsurfaces += 1

    # 3. Exportierte Job-IDF im Projektordner aktualisieren
    if export_dir:
        for job_idf in Path(export_dir).rglob("*.idf"):
            if "Minimal.idf" not in str(job_idf):
                idf.savecopy(str(job_idf))

    msg = (
        f"[IDF-Sanitizer] {dedup_removed} Duplikate entfernt, "
        f"{converted_walls} Fenster-Walls in FenestrationSurface umgewandelt, "
        f"{flipped_subsurfaces} Subsurface-Normalen gedreht."
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
