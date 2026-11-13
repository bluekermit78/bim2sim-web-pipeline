import math
from pathlib import Path
import numpy as np
import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.geom

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeBox
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Vec, gp_Pnt
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.BRepTools import breptools, BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def _face_area_and_normal(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()
    surf = BRepAdaptor_Surface(face, True)
    if surf.GetType() != GeomAbs_Plane:
        return area, None
    pln = surf.Plane()
    norm = pln.Axis().Direction()
    if face.Orientation() == TopAbs_REVERSED:
        return area, (-norm.X(), -norm.Y(), -norm.Z())
    return area, (norm.X(), norm.Y(), norm.Z())


def _extract_face_outer_points_3d(face, target_normal=None):
    """
    Extrahiert die bereinigten, streng gegen den Uhrzeigersinn geordneten Eckpunkte der
    2D-Konvexhülle einer ebenen Fläche. Entfernt automatisch Tür-Einbuchtungen an der
    Wand-Unterkante, sodass die Trägerwand für EnergyPlus/OpenStudio/EVEBI ein sauberes
    4-Eck (bzw. 5-Eck bei Giebelwänden) bleibt!
    """
    outer_wire = breptools.OuterWire(face)
    exp = TopExp_Explorer(outer_wire, TopAbs_VERTEX)
    raw_pts = []
    while exp.More():
        v = topods.Vertex(exp.Current())
        p = BRep_Tool.Pnt(v)
        pt = (round(p.X(), 4), round(p.Y(), 4), round(p.Z(), 4))
        if not any(math.dist(pt, existing) < 0.01 for existing in raw_pts):
            raw_pts.append(pt)
        exp.Next()

    if len(raw_pts) < 3:
        return raw_pts

    if target_normal is None:
        _, target_normal = _face_area_and_normal(face)
    if target_normal is None:
        return raw_pts

    nx, ny, nz = target_normal
    if abs(nz) < 0.9:
        rx, ry, rz = -ny, nx, 0.0
    else:
        rx, ry, rz = 1.0, 0.0, 0.0
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx, ry, rz = rx/rl, ry/rl, rz/rl
    ux = ny*rz - nz*ry
    uy = nz*rx - nx*rz
    uz = nx*ry - ny*rx

    cx = sum(p[0] for p in raw_pts) / len(raw_pts)
    cy = sum(p[1] for p in raw_pts) / len(raw_pts)
    cz = sum(p[2] for p in raw_pts) / len(raw_pts)

    pts_2d = []
    for p in raw_pts:
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        u = round(dx*rx + dy*ry + dz*rz, 5)
        v = round(dx*ux + dy*uy + dz*uz, 5)
        pts_2d.append((u, v, p))

    # 2D Convex Hull (Andrew's Monotone Chain) -> eliminiert nach innen ragende Tür-Ausschnitt-Ecken!
    pts_2d.sort(key=lambda item: (item[0], item[1]))

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts_2d:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 1e-4:
            lower.pop()
        lower.append(pt)

    upper = []
    for pt in reversed(pts_2d):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 1e-4:
            upper.pop()
        upper.append(pt)

    hull = lower[:-1] + upper[:-1]
    sorted_pts = [item[2] for item in hull] if len(hull) >= 3 else [item[2] for item in pts_2d]

    # Kollineare Zwischenpunkte auf geraden Kanten herausfiltern
    cleaned = []
    n_pts = len(sorted_pts)
    for i in range(n_pts):
        p_prev = np.array(sorted_pts[(i - 1) % n_pts], dtype=float)
        p_curr = np.array(sorted_pts[i], dtype=float)
        p_next = np.array(sorted_pts[(i + 1) % n_pts], dtype=float)
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 > 0.01 and l2 > 0.01:
            cross_len = float(np.linalg.norm(np.cross(v1 / l1, v2 / l2)))
            if cross_len > 0.02:
                cleaned.append(sorted_pts[i])

    return cleaned if len(cleaned) >= 3 else sorted_pts


def _create_ifc_curve_bounded_plane(ifc_file, pts_3d, normal_3d):
    if len(pts_3d) < 3:
        return None
    p0 = pts_3d[0]
    nx, ny, nz = normal_3d
    if abs(nz) < 0.9:
        rx, ry, rz = -ny, nx, 0.0
    else:
        rx, ry, rz = 1.0, 0.0, 0.0
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx, ry, rz = rx/rl, ry/rl, rz/rl
    ux = ny*rz - nz*ry
    uy = nz*rx - nx*rz
    uz = nx*ry - ny*rx

    pts_2d = []
    for p in pts_3d:
        dx, dy, dz = p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]
        u = round(dx*rx + dy*ry + dz*rz, 4)
        v = round(dx*ux + dy*uy + dz*uz, 4)
        pts_2d.append((u, v))
    pts_2d.append(pts_2d[0])

    ifc_p0 = ifc_file.createIfcCartesianPoint([float(p0[0]), float(p0[1]), float(p0[2])])
    ifc_axis = ifc_file.createIfcDirection([float(nx), float(ny), float(nz)])
    ifc_ref = ifc_file.createIfcDirection([float(rx), float(ry), float(rz)])
    placement = ifc_file.createIfcAxis2Placement3D(ifc_p0, ifc_axis, ifc_ref)
    plane = ifc_file.createIfcPlane(placement)

    ifc_pts_2d = [ifc_file.createIfcCartesianPoint([float(u), float(v)]) for u, v in pts_2d]
    polyline = ifc_file.createIfcPolyline(ifc_pts_2d)
    return ifc_file.createIfcCurveBoundedPlane(plane, polyline, [])


def ensure_thermal_materials(ifc_file, log_func=print):
    defaults = {
        "IfcWall": ("Mauerwerk_Dämmung", 0.30, 0.25, 1200.0, 1000.0),
        "IfcWallStandardCase": ("Mauerwerk_Dämmung", 0.30, 0.25, 1200.0, 1000.0),
        "IfcSlab": ("Stahlbeton_Dämmung", 0.25, 0.35, 2000.0, 1000.0),
        "IfcRoof": ("Dachaufbau_Gedämmt", 0.28, 0.18, 600.0, 1200.0),
        "IfcWindow": ("Isolierverglasung", 0.024, 0.80, 2500.0, 840.0),
        "IfcDoor": ("Außentür", 0.06, 0.40, 800.0, 1500.0),
    }

    def _enrich_material(mat, def_tuple):
        _, _, lam, rho, cp = def_tuple
        existing_pnames = set()
        for pset in getattr(mat, "HasProperties", []) or []:
            for prop in getattr(pset, "Properties", []) or []:
                existing_pnames.add(prop.Name)
        new_props = []
        if "ThermalConductivity" not in existing_pnames:
            new_props.append(ifc_file.createIfcPropertySingleValue(
                "ThermalConductivity", None, ifc_file.create_entity("IfcThermalConductivityMeasure", lam), None
            ))
        if "MassDensity" not in existing_pnames:
            new_props.append(ifc_file.createIfcPropertySingleValue(
                "MassDensity", None, ifc_file.create_entity("IfcMassDensityMeasure", rho), None
            ))
        if "SpecificHeatCapacity" not in existing_pnames:
            new_props.append(ifc_file.createIfcPropertySingleValue(
                "SpecificHeatCapacity", None, ifc_file.create_entity("IfcSpecificHeatCapacityMeasure", cp), None
            ))
        if new_props:
            ifc_file.createIfcMaterialProperties("Pset_MaterialThermal", None, new_props, mat)

    for mat in ifc_file.by_type("IfcMaterial"):
        _enrich_material(mat, ("Standardmaterial", 0.20, 0.35, 1500.0, 1000.0))

    # Bei 1-Zonen-Modellen alle Wände auf IsExternal=True setzen (damit Giebelwände nicht als InnerWall übersprungen werden!)
    if len(ifc_file.by_type("IfcSpace")) <= 1:
        for w in ifc_file.by_type("IfcWall"):
            for rel in getattr(w, "IsDefinedBy", []) or []:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if pset and pset.is_a("IfcPropertySet"):
                        for prop in getattr(pset, "HasProperties", []) or []:
                            if prop.Name == "IsExternal" and hasattr(prop, "NominalValue") and prop.NominalValue:
                                prop.NominalValue.wrappedValue = True

    # Bodenplatten als BASESLAB markieren, damit bim2sim sie als GroundFloor (mit eigenem LayerSet) erkennt!
    for slab in ifc_file.by_type("IfcSlab"):
        if getattr(slab, "PredefinedType", None) in (None, "FLOOR", "NOTDEFINED"):
            slab.PredefinedType = "BASESLAB"

    created_layersets = {}
    fixed_elements = 0

    for ifc_type, def_tuple in defaults.items():
        mat_name, thick, lam, rho, cp = def_tuple
        for el in ifc_file.by_type(ifc_type):
            has_valid_ls = False
            for rel in getattr(el, "HasAssociations", []) or []:
                if rel.is_a("IfcRelAssociatesMaterial"):
                    mat_rel = rel.RelatingMaterial
                    if mat_rel and mat_rel.is_a("IfcMaterialLayerSet"):
                        has_valid_ls = True
                    elif mat_rel and mat_rel.is_a("IfcMaterialLayerSetUsage") and getattr(mat_rel, "ForLayerSet", None):
                        rel.RelatingMaterial = mat_rel.ForLayerSet
                        has_valid_ls = True

            if not has_valid_ls:
                for rel in list(getattr(el, "HasAssociations", []) or []):
                    if rel.is_a("IfcRelAssociatesMaterial"):
                        ifc_file.remove(rel)
                if ifc_type not in created_layersets:
                    mat = ifc_file.createIfcMaterial(mat_name)
                    _enrich_material(mat, def_tuple)
                    layer = ifc_file.createIfcMaterialLayer(mat, thick, None)
                    ls = ifc_file.createIfcMaterialLayerSet([layer], f"LayerSet_{ifc_type}")
                    created_layersets[ifc_type] = ls
                ifc_file.createIfcRelAssociatesMaterial(
                    ifcopenshell.guid.new(), None, None, None, [el], created_layersets[ifc_type]
                )
                fixed_elements += 1

    log_func(f"[SB-PreProcessor] {fixed_elements} Bauteile direkt mit IfcMaterialLayerSet und Thermik-Kennwerten ausgestattet.")


def ensure_space_boundaries(ifc_path: Path, log_func=print) -> int:
    ifc_file = ifcopenshell.open(str(ifc_path))
    ensure_thermal_materials(ifc_file, log_func=log_func)

    existing_sbs = ifc_file.by_type("IfcRelSpaceBoundary")
    if len(existing_sbs) > 0:
        ifc_file.write(str(ifc_path))
        return len(existing_sbs)

    spaces = ifc_file.by_type("IfcSpace")
    if not spaces:
        ifc_file.write(str(ifc_path))
        return 0

    log_func(f"[SB-PreProcessor] 0 Space Boundaries gefunden. Berechne 2nd Level Space Boundaries fuer {len(spaces)} IfcSpaces...")

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    settings.set(settings.USE_WORLD_COORDS, True)

    target_types = ("IfcWall", "IfcSlab", "IfcRoof", "IfcWindow", "IfcDoor")
    elements = []
    max_roof_z = -1e9
    for t in target_types:
        for el in ifc_file.by_type(t):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el).geometry
                bbox = Bnd_Box()
                brepbndlib.Add(shape, bbox)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                if el.is_a("IfcRoof") and zmax > max_roof_z:
                    max_roof_z = zmax
                elements.append((el, shape, bbox))
            except Exception:
                continue

    log_func(f"[SB-PreProcessor] {len(elements)} Bauteil-Geometrien erfolgreich in OpenCASCADE geladen.")

    owner_history = ifc_file.by_type("IfcOwnerHistory")[0] if ifc_file.by_type("IfcOwnerHistory") else None
    created_count = 0

    for space in spaces:
        try:
            sp_shape = ifcopenshell.geom.create_shape(settings, space).geometry
            sp_bbox = Bnd_Box()
            brepbndlib.Add(sp_shape, sp_bbox)
            s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = sp_bbox.Get()
        except Exception as e:
            log_func(f"[SB-PreProcessor] Warnung: Konnte IfcSpace {space.GlobalId} nicht meshen: {e}")
            continue

        # Prüfen, ob über dem IfcSpace ein Giebeldach (IfcRoof) liegt, das höher als der IfcSpace-Kasten (s_zmax) ragt:
        has_gable_attic = (max_roof_z > s_zmax + 0.35) and (len(spaces) == 1)

        exp = TopExp_Explorer(sp_shape, TopAbs_FACE)
        sp_faces = []
        while exp.More():
            f = topods.Face(exp.Current())
            area, norm = _face_area_and_normal(f)
            if area > 0.05 and norm is not None:
                sp_faces.append((f, area, norm))
            exp.Next()

        for sp_face, sp_area, norm in sp_faces:
            # Standard-Suchtiefe 0.55 m; bei der oberen Deckenfläche unter einem Giebeldach bis über den Dachfirst extrudieren!
            if has_gable_attic and norm[2] > 0.5:
                probe_depth = (max_roof_z - s_zmax) + 0.50
            else:
                probe_depth = 0.55

            vec = gp_Vec(norm[0] * probe_depth, norm[1] * probe_depth, norm[2] * probe_depth)
            prism = BRepPrimAPI_MakePrism(sp_face, vec).Shape()

            # Falls Giebelwände oberhalb von s_zmax existieren, erweitern wir für vertikale Wand-Suchprismen die Höhe bis max_roof_z:
            if has_gable_attic and abs(norm[2]) < 0.2:
                f_bbox = Bnd_Box()
                brepbndlib.Add(prism, f_bbox)
                pxmin, pymin, pzmin, pxmax, pymax, pzmax = f_bbox.Get()
                prism = BRepPrimAPI_MakeBox(gp_Pnt(pxmin, pymin, pzmin), gp_Pnt(pxmax, pymax, max_roof_z)).Shape()

            p_bbox = Bnd_Box()
            brepbndlib.Add(prism, p_bbox)

            for el, el_shape, el_bbox in elements:
                if el.is_a("IfcRoof") and norm[2] < 0.25:
                    continue
                if el.is_a("IfcSlab") and abs(norm[2]) < 0.5:
                    continue
                if p_bbox.IsOut(el_bbox):
                    continue

                common = BRepAlgoAPI_Common(prism, el_shape)
                if not common.IsDone():
                    continue
                c_shape = common.Shape()

                f_exp = TopExp_Explorer(c_shape, TopAbs_FACE)
                candidate_faces = []
                while f_exp.More():
                    cf = topods.Face(f_exp.Current())
                    c_area, c_norm = _face_area_and_normal(cf)
                    if c_norm is not None and c_area > 0.05:
                        dot = abs(c_norm[0]*norm[0] + c_norm[1]*norm[1] + c_norm[2]*norm[2])
                        min_dot = 0.35 if el.is_a("IfcRoof") else 0.80
                        if dot >= min_dot:
                            candidate_faces.append((cf, c_area, c_norm))
                    f_exp.Next()

                if not candidate_faces:
                    continue

                # Für IfcRoof (oft trianguliertes Mesh mit mehreren Dreiecken pro Dachschräge!)
                # fassen wir alle komplanaren Dreiecke der Dach-Innenseite je Dachneigung (n_key) zu EINEM geschlossenen Polygon zusammen!
                rcx, rcy, rcz = 0.5*(s_xmin+s_xmax), 0.5*(s_ymin+s_ymax), 0.5*(s_zmin+s_zmax)
                if el.is_a("IfcRoof"):
                    planes_by_norm = {}
                    for cf, c_area, c_norm in candidate_faces:
                        if (c_norm[0]*norm[0] + c_norm[1]*norm[1] + c_norm[2]*norm[2]) < 0:
                            c_norm = (-c_norm[0], -c_norm[1], -c_norm[2])
                        # Nur echte Dachschrägen / Flachdächer nehmen (keine senkrechten Stirnkanten der Dachplatte!)
                        if abs(c_norm[2]) < 0.35:
                            continue
                        pts_cand = _extract_face_outer_points_3d(cf, c_norm)
                        if len(pts_cand) < 3:
                            continue
                        n_key = (round(c_norm[0], 2), round(c_norm[1], 2), round(c_norm[2], 2))
                        # Ebenen-Abstand d vom Raumzentrum entlang der Normalen (trennt Dach-Unterseite von Dach-Oberseite!)
                        fcx = sum(p[0] for p in pts_cand) / len(pts_cand)
                        fcy = sum(p[1] for p in pts_cand) / len(pts_cand)
                        fcz = sum(p[2] for p in pts_cand) / len(pts_cand)
                        plane_d = round((fcx - rcx)*c_norm[0] + (fcy - rcy)*c_norm[1] + (fcz - rcz)*c_norm[2], 2)
                        planes_by_norm.setdefault(n_key, {}).setdefault(plane_d, []).append((cf, c_area, c_norm, pts_cand))

                    selected = []
                    for n_key, d_dict in planes_by_norm.items():
                        # Wähle die Ebene mit der größten Gesamtfläche; bei gleicher Fläche die innere Ebene (kleineres plane_d)
                        valid_ds = [d for d in d_dict.keys() if sum(x[1] for x in d_dict[d]) > 0.5]
                        best_d = min(valid_ds) if valid_ds else min(d_dict.keys())
                        face_group = d_dict[best_d]
                        total_area = sum(x[1] for x in face_group)
                        rep_face, _, rep_norm, _ = face_group[0]

                        # Alle Eckpunkte der koplanaren Dreiecke dieser Dachseite sammeln, deduplizieren und als 1 Außenpolygon sortieren
                        merged_raw = []
                        for _, _, _, f_pts in face_group:
                            for pt in f_pts:
                                if not any(math.dist(pt, ex) < 0.015 for ex in merged_raw):
                                    merged_raw.append(pt)

                        if len(merged_raw) >= 3:
                            nx, ny, nz = rep_norm
                            rx, ry, rz = (-ny, nx, 0.0) if abs(nz) < 0.9 else (1.0, 0.0, 0.0)
                            rl = math.sqrt(rx*rx + ry*ry + rz*rz)
                            rx, ry, rz = rx/rl, ry/rl, rz/rl
                            ux, uy, uz = ny*rz - nz*ry, nz*rx - nx*rz, nx*ry - ny*rx
                            mcx = sum(p[0] for p in merged_raw) / len(merged_raw)
                            mcy = sum(p[1] for p in merged_raw) / len(merged_raw)
                            mcz = sum(p[2] for p in merged_raw) / len(merged_raw)
                            ang_list = []
                            for p in merged_raw:
                                dx, dy, dz = p[0] - mcx, p[1] - mcy, p[2] - mcz
                                u, v = dx*rx + dy*ry + dz*rz, dx*ux + dy*uy + dz*uz
                                ang_list.append((math.atan2(v, u), p))
                            ang_list.sort(key=lambda item: item[0])
                            s_pts = [item[1] for item in ang_list]

                            # Kollineare Zwischenpunkte auf den langen Dachkanten entfernen -> ergibt exakt 4 Eckpunkte pro Satteldach-Seite!
                            clean_pts = []
                            np_len = len(s_pts)
                            for i in range(np_len):
                                p_prev = np.array(s_pts[(i - 1) % np_len], dtype=float)
                                p_curr = np.array(s_pts[i], dtype=float)
                                p_next = np.array(s_pts[(i + 1) % np_len], dtype=float)
                                v1, v2 = p_curr - p_prev, p_next - p_curr
                                l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                                if l1 > 0.01 and l2 > 0.01:
                                    if float(np.linalg.norm(np.cross(v1 / l1, v2 / l2))) > 0.02:
                                        clean_pts.append(s_pts[i])
                            final_pts = clean_pts if len(clean_pts) >= 3 else s_pts
                            selected.append((rep_face, total_area, rep_norm, final_pts, best_d))
                else:
                    by_normal = {}
                    for cf, c_area, c_norm in candidate_faces:
                        if (c_norm[0]*norm[0] + c_norm[1]*norm[1] + c_norm[2]*norm[2]) < 0:
                            c_norm = (-c_norm[0], -c_norm[1], -c_norm[2])
                        n_key = (round(c_norm[0], 1), round(c_norm[1], 1), round(c_norm[2], 1))
                        pts_cand = _extract_face_outer_points_3d(cf, c_norm)
                        if len(pts_cand) < 3:
                            continue
                        fcx = sum(p[0] for p in pts_cand) / len(pts_cand)
                        fcy = sum(p[1] for p in pts_cand) / len(pts_cand)
                        fcz = sum(p[2] for p in pts_cand) / len(pts_cand)
                        dist_along_norm = (fcx - rcx)*c_norm[0] + (fcy - rcy)*c_norm[1] + (fcz - rcz)*c_norm[2]
                        if n_key not in by_normal:
                            by_normal[n_key] = (cf, c_area, c_norm, pts_cand, dist_along_norm)
                        else:
                            prev_cf, prev_area, prev_norm, prev_pts, prev_dist = by_normal[n_key]
                            if c_area > prev_area * 1.15 or (c_area >= prev_area * 0.85 and dist_along_norm < prev_dist):
                                by_normal[n_key] = (cf, c_area, c_norm, pts_cand, dist_along_norm)
                    selected = [max(by_normal.values(), key=lambda x: x[1])] if by_normal else []

                sp_pts_ref = _extract_face_outer_points_3d(sp_face, norm)
                sp_p0 = sp_pts_ref[0] if sp_pts_ref else None

                for best_face, best_area, best_norm, pts_3d, _ in selected:
                    # Wenn die Raumfläche (sp_face) parallel zur Bauteilfläche ist (dot > 0.95),
                    # projiziere alle Punkte exakt auf die Ebene von sp_face (verhindert jeden Dachdicken-Versatz!)
                    dot_sp = abs(best_norm[0]*norm[0] + best_norm[1]*norm[1] + best_norm[2]*norm[2])
                    if sp_p0 is not None and dot_sp > 0.95:
                        proj_pts = []
                        for pt in pts_3d:
                            d_err = (pt[0] - sp_p0[0])*norm[0] + (pt[1] - sp_p0[1])*norm[1] + (pt[2] - sp_p0[2])*norm[2]
                            proj_pts.append((
                                round(pt[0] - d_err * norm[0], 4),
                                round(pt[1] - d_err * norm[1], 4),
                                round(pt[2] - d_err * norm[2], 4),
                            ))
                        pts_3d = proj_pts
                    cbp = _create_ifc_curve_bounded_plane(ifc_file, pts_3d, best_norm)
                    if cbp is None:
                        continue

                    conn_geom = ifc_file.createIfcConnectionSurfaceGeometry(cbp, None)
                    sb_kwargs = {
                        "GlobalId": ifcopenshell.guid.new(),
                        "OwnerHistory": owner_history,
                        "Name": "2ndLevel",
                        "Description": f"AutoSB_{el.is_a()}",
                        "RelatingSpace": space,
                        "RelatedBuildingElement": el,
                        "ConnectionGeometry": conn_geom,
                        "PhysicalOrVirtualBoundary": "PHYSICAL",
                        "InternalOrExternalBoundary": "EXTERNAL",
                    }
                    if ifc_file.schema == "IFC4":
                        sb_kwargs["ParentBoundary"] = None
                        sb_kwargs["CorrespondingBoundary"] = None
                        ifc_file.create_entity("IfcRelSpaceBoundary2ndLevel", **sb_kwargs)
                    else:
                        ifc_file.create_entity("IfcRelSpaceBoundary", **sb_kwargs)
                    created_count += 1

    if created_count > 0:
        ifc_file.write(str(ifc_path))
        log_func(f"[SB-PreProcessor] ERFOLG: {created_count} IfcRelSpaceBoundary2ndLevel-Elemente in {ifc_path.name} geschrieben!")
    return created_count
