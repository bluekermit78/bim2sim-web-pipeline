import math
from pathlib import Path
import numpy as np

from gbxml_exporter import (
    _parse_idf_objects,
    _extract_ifc_space_names,
    _extract_ifc_element_types, _extract_ifc_space_bboxes,
    _poly_normal_and_area,
    _poly_dimensions,
    _azimuth_and_tilt,
)
from update_preview import parse_energyplus_summary


def _compass_dir(azimuth: float, tilt: float) -> str:
    if tilt < 15.0:
        return "Horizontal (Dach)"
    if tilt > 165.0:
        return "Horizontal (Boden)"
    dirs = ["N", "NO", "O", "SO", "S", "SW", "W", "NW"]
    idx = int(((azimuth + 22.5) % 360.0) // 45.0)
    return f"{dirs[idx]} ({azimuth:.0f}°)"


def generate_geometry_html(job_id: str, job_dir: Path, filename: str) -> str:
    project_dir = job_dir / "bim2sim_project"
    in_ifcs = list((job_dir / "bim2sim_project").rglob("*.ifc")) + list((job_dir / "input").glob("*.ifc"))
    ifc_path = in_ifcs[0] if in_ifcs else None

    idf_files = [f for f in project_dir.rglob("*.idf") if "Minimal.idf" not in str(f)] if project_dir.exists() else []
    if not idf_files:
        return "<h3>Keine exportierte IDF-Datei für diesen Job gefunden.</h3>"

    idf_path = idf_files[0]
    idf_data = _parse_idf_objects(idf_path)
    arch_ifcs = list(project_dir.rglob('*.ifc'))
    sb_ifc_path = arch_ifcs[0] if arch_ifcs else ifc_path
    ifc_names = _extract_ifc_space_names(sb_ifc_path)
    ifc_elem_types = _extract_ifc_element_types(sb_ifc_path)
    ifc_bboxes = _extract_ifc_space_bboxes(sb_ifc_path)
    ep_summary = parse_energyplus_summary(project_dir)

    # 1. Materialien & Konstruktionen (U-Werte) berechnen
    materials = {}
    for m in idf_data.get("MATERIAL", []):
        if len(m) >= 6:
            try:
                d = float(m[2])
                lam = float(m[3])
                materials[m[0].upper()] = d / max(0.005, lam)
            except ValueError:
                pass

    u_values = {}
    for c in idf_data.get("CONSTRUCTION", []):
        if not c:
            continue
        cname = c[0].upper()
        layers = [l.upper() for l in c[1:] if l]
        if cname.startswith("WINDOW"):
            u_values[cname] = 1.30
        else:
            r_tot = 0.17 + sum(materials.get(l, 0.5) for l in layers)
            u_values[cname] = round(1.0 / max(0.15, r_tot), 3)

    # 2. Fenster / Türen nach Basisfläche sammeln
    openings_by_base = {}
    for fen in idf_data.get("FENESTRATIONSURFACE:DETAILED", []):
        if len(fen) < 18:
            continue
        f_name, f_type, f_cons, base_name = fen[0], fen[1].upper(), fen[2].upper(), fen[3].upper()
        coords = []
        for val in fen[9:]:
            if val != "":
                try:
                    coords.append(float(val))
                except ValueError:
                    pass
        pts = [coords[i:i+3] for i in range(0, (len(coords) // 3) * 3, 3)]
        if len(pts) >= 3:
            norm, area = _poly_normal_and_area(pts)
            w, h = _poly_dimensions(pts, area)
            openings_by_base.setdefault(base_name, []).append({
                "name": f_name,
                "type": "Tür" if "DOOR" in f_type else "Fenster",
                "cons": (ifc_elem_types.get(f_name.upper(), {}).get("type_label") or fen[2]),
                "u_val": u_values.get(f_cons, 1.30),
                "area": area,
                "w": w,
                "h": h,
                "z_min": min(p[2] for p in pts),
                "z_max": max(p[2] for p in pts),
            })

    # 3. Opake Flächen sammeln und nach Zonen gruppieren
    surfaces_by_zone = {}
    for s in idf_data.get("BUILDINGSURFACE:DETAILED", []):
        if len(s) < 19:
            continue
        s_name, s_type, s_cons, s_zone, s_bc = s[0], s[1].upper(), s[2], s[3].upper(), s[4].upper()
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
        norm, gross_area = _poly_normal_and_area(pts)
        az, tilt = _azimuth_and_tilt(norm)
        w, h = _poly_dimensions(pts, gross_area)
        ops = openings_by_base.get(s_name.upper(), [])
        op_area = sum(o["area"] for o in ops)
        net_area = max(0.0, gross_area - op_area)
        zs = [p[2] for p in pts]

        surfaces_by_zone.setdefault(s_zone, []).append({
            "name": s_name,
            "type": s_type, "el_guid": ifc_elem_types.get(s_name.upper(), {}).get("el_guid", ""), "ifc_class": ifc_elem_types.get(s_name.upper(), {}).get("ifc_class", ""), "norm": norm,
            "bc": s_bc,
            "cons": (ifc_elem_types.get(s_name.upper(), {}).get("type_label") or s_cons),
            "u_val": u_values.get(s_cons.upper(), 0.28),
            "pts": pts,
            "gross_area": gross_area,
            "op_area": op_area,
            "net_area": net_area,
            "w": w,
            "h": h,
            "az": az,
            "tilt": tilt,
            "compass": _compass_dir(az, tilt),
            "z_min": min(zs),
            "z_max": max(zs),
            "avg_z": sum(zs) / len(zs),
            "n_pts": len(pts),
            "openings": ops,
        })

    # 4. Zonen-Kennwerte (inkl. mittlerer Höhe h_mean = V / A, Traufhöhe, Firsthöhe) berechnen
    zone_rows = []
    all_surfaces_flat = []

    total_ngf = 0.0
    total_vol = 0.0
    total_env_area = 0.0
    total_win_area = 0.0

    for zk, z_surfs in surfaces_by_zone.items():
        all_z = [p[2] for s in z_surfs for p in s["pts"]]
        z_min_zone = min(all_z) if all_z else 0.0
        z_max_zone = max(all_z) if all_z else 0.0
        z_mid = 0.5 * (z_min_zone + z_max_zone)

        # Typ-Feinabstimmung (Floor vs. Roof)
        for s in z_surfs:
            if s["tilt"] < 65.0 or s["tilt"] > 115.0:
                if s["avg_z"] < z_mid:
                    s["type"] = "FLOOR"
                else:
                    s["type"] = "ROOF"
                s["compass"] = _compass_dir(s["az"], s["tilt"])

        # Falls Bodenplatte in IDF fehlte, für Statistik aus Grundriss ergänzen
        floors = [s for s in z_surfs if s["type"] == "FLOOR"]
        roofs = [s for s in z_surfs if s["type"] in ("ROOF", "CEILING")]

        ep_z = ep_summary.get("zones", {}).get(zk, {})
        ngf = sum(s["gross_area"] for s in floors) if floors else float(ep_z.get("area_m2", 0.0))
        if ngf <= 0.01 and roofs:
            ngf = sum(s["gross_area"] * abs(math.cos(math.radians(s["tilt"]))) for s in roofs)

        # Traufhöhe (relativ zu z_min_zone) ermitteln
        eave_candidates = []
        for s in z_surfs:
            if s["type"] == "ROOF" and (s["z_max"] - s["z_min"]) > 0.25:
                eave_candidates.append(s["z_min"] - z_min_zone)
            elif s["type"] == "WALL" and s["n_pts"] == 4 and abs(s["z_min"] - z_min_zone) < 0.15 and s["z_max"] < z_max_zone - 0.25:
                eave_candidates.append(s["z_max"] - z_min_zone)

        h_max = max(0.0, z_max_zone - z_min_zone)
        h_eave = (sum(eave_candidates) / len(eave_candidates)) if eave_candidates else h_max

        # Volumen: aus EnergyPlus (falls vorhanden und plausibel) oder exakt aus Trauf-/Firsthöhe
        vol_ep = float(ep_z.get("volume_m3", 0.0))
        vol_geom = ngf * (h_eave + 0.5 * max(0.0, h_max - h_eave)) if ngf > 0 else 0.0
        vol = vol_ep if vol_ep > 0.1 else vol_geom

        # Mittlere Raum-/Zonenhöhe (entscheidend für EVEBI!)
        h_mean = (vol / ngf) if ngf > 0.01 else 0.0

        # Thermische Hüllfläche dieser Zone (Außenbauteile + Erdreich)
        env_area = sum(s["gross_area"] for s in z_surfs if s["bc"] in ("OUTDOORS", "GROUND"))
        if not floors and ngf > 0:
            env_area += ngf  # Bodenplatte mitzählen
        win_area = sum(s["op_area"] for s in z_surfs)
        wall_net_area = sum(s["net_area"] for s in z_surfs if s["type"] == "WALL")
        roof_area = sum(s["gross_area"] for s in roofs)

        ifc_label = ifc_names.get(zk, f"Raum / Zone ({zk[:10]})")

        zone_rows.append({
            "guid": zk,
            "label": ifc_label,
            "ngf": round(ngf, 2),
            "vol": round(vol, 2),
            "vol_geom": round(vol_geom, 2),
            "h_mean": round(h_mean, 3),
            "h_eave": round(h_eave, 3),
            "h_max": round(h_max, 3),
            "z_min": round(z_min_zone, 2),
            "z_max": round(z_max_zone, 2),
            "wall_net": round(wall_net_area, 2),
            "win_area": round(win_area, 2),
            "roof_area": round(roof_area, 2),
            "env_area": round(env_area, 2),
        })

        total_ngf += ngf
        total_vol += vol
        total_env_area += env_area
        total_win_area += win_area

        for s in z_surfs:
            s["zone_label"] = ifc_label
            s["rel_z_min"] = round(s["z_min"] - z_min_zone, 2)
            s["rel_z_max"] = round(s["z_max"] - z_min_zone, 2)
            all_surfaces_flat.append(s)

    bldg_h_mean = (total_vol / total_ngf) if total_ngf > 0.01 else 0.0
    av_ratio = (total_env_area / total_vol) if total_vol > 0.01 else 0.0
    win_ratio = (total_win_area / total_env_area * 100.0) if total_env_area > 0.01 else 0.0

    # HTML rendern
    z_trs = ""
    for zr in zone_rows:
        z_trs += f"""
        <tr>
            <td><b>{zr['label']}</b><br><span style="color:#64748b;font-size:11px;">GUID: {zr['guid']}</span></td>
            <td class="num"><b>{zr['ngf']:.2f} m²</b></td>
            <td class="num"><b>{zr['vol']:.2f} m³</b></td>
            <td class="num highlight-cell"><b>{zr['h_mean']:.3f} m</b></td>
            <td class="num">{zr['h_eave']:.2f} m</td>
            <td class="num">{zr['h_max']:.2f} m</td>
            <td class="num">{zr['wall_net']:.2f} m²</td>
            <td class="num">{zr['roof_area']:.2f} m²</td>
            <td class="num">{zr['win_area']:.2f} m²</td>
            <td class="num">{zr['env_area']:.2f} m²</td>
        </tr>"""

    s_trs = ""
    type_labels = {"WALL": "Außenwand", "ROOF": "Dachfläche", "FLOOR": "Bodenplatte", "CEILING": "Decke"}
    for s in sorted(all_surfaces_flat, key=lambda x: (x["type"], x["az"])):
        t_disp = type_labels.get(s["type"], s["type"])
        if s["type"] == "WALL" and s["n_pts"] == 5:
            t_disp = "Giebelwand (5-Eck)"
        elif s["type"] == "WALL" and s["n_pts"] == 3:
            t_disp = "Giebeldreieck"
        elif s["type"] == "ROOF" and s["tilt"] > 5.0:
            t_disp = f"Dachschräge ({s['tilt']:.1f}°)"

        s_trs += f"""
        <tr>
            <td><b>{t_disp}</b></td>
            <td><code>{s['name']}</code></td>
            <td><span style='background:#f1f5f9;padding:2px 6px;border-radius:4px;font-weight:600;color:#0f172a;'>{s['cons']}</span></td>
            <td>{s['zone_label']}</td>
            <td>{s['compass']}</td>
            <td class="num">{s['tilt']:.1f}°</td>
            <td class="num">{s['w']:.2f} × {s['h']:.2f} m</td>
            <td class="num">{s['rel_z_min']:.2f} .. {s['rel_z_max']:.2f} m</td>
            <td class="num">{s['gross_area']:.2f} m²</td>
            <td class="num">{s['op_area']:.2f} m²</td>
            <td class="num"><b>{s['net_area']:.2f} m²</b></td>
            <td class="num">{s['u_val']:.2f}</td>
        </tr>"""
        for op in s["openings"]:
            s_trs += f"""
        <tr style="background:#f8fafc; color:#475569; font-size:12.5px;">
            <td style="padding-left:24px;">↳ 🪟 {op['type']}</td>
            <td><code>{op['name']}</code></td>
            <td><span style='background:#e0f2fe;padding:2px 6px;border-radius:4px;color:#0369a1;'>{op['cons']}</span></td>
            <td>{s['zone_label']}</td>
            <td>{s['compass']}</td>
            <td class="num">{s['tilt']:.1f}°</td>
            <td class="num">{op['w']:.2f} × {op['h']:.2f} m</td>
            <td class="num">-</td>
            <td class="num">{op['area']:.2f} m²</td>
            <td class="num">-</td>
            <td class="num"><b>{op['area']:.2f} m²</b></td>
            <td class="num">{op['u_val']:.2f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Geometrie & Maße – {filename}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 28px auto; padding: 0 20px; background: #f4f6f9; color: #1e293b; }}
        .card {{ background: #fff; padding: 24px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 22px; }}
        h1 {{ margin-top: 0; font-size: 22px; color: #0f172a; }}
        h2 {{ margin-top: 0; font-size: 17px; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-top: 14px; }}
        .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
        .kpi.highlight {{ background: #eff6ff; border: 2px solid #3b82f6; }}
        .kpi-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.4px; }}
        .kpi-val {{ font-size: 21px; font-weight: 700; color: #0f172a; margin-top: 5px; }}
        .kpi.highlight .kpi-val {{ color: #1d4ed8; }}
        .kpi-sub {{ font-size: 12px; color: #475569; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 10px; }}
        th {{ text-align: left; padding: 10px 8px; background: #f1f5f9; border-bottom: 2px solid #cbd5e1; color: #334155; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 9px 8px; border-bottom: 1px solid #e2e8f0; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .highlight-cell {{ background: #eff6ff; color: #1d4ed8; font-size: 14.5px; }}
        code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; font-size: 12px; }}
        .hint {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; font-size: 13.5px; margin-top: 14px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📐 Gebäude-, Zonen- und Raumgeometrie: {filename}</h1>
        <p style="color:#475569; margin:4px 0 0 0;">Job-ID: <code>{job_id}</code> — Auswertung aller 3D-Hüllflächen, Volumina und mittleren Höhen für <b>EVEBI 14.3.8</b> &amp; <b>OpenStudio</b></p>

        <div class="kpi-grid">
            <div class="kpi">
                <div class="kpi-label">Netto-Grundfläche (NGF)</div>
                <div class="kpi-val">{total_ngf:.2f} m²</div>
                <div class="kpi-sub">Beheizte Bodenfläche</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Netto-Raumvolumen (V)</div>
                <div class="kpi-val">{total_vol:.2f} m³</div>
                <div class="kpi-sub">Inkl. Giebel-/Dachraum</div>
            </div>
            <div class="kpi highlight">
                <div class="kpi-label">⭐ Mittlere Höhe (h_mittel = V/A)</div>
                <div class="kpi-val">{bldg_h_mean:.3f} m</div>
                <div class="kpi-sub">Eingabewert für EVEBI!</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Thermische Hüllfläche (A)</div>
                <div class="kpi-val">{total_env_area:.2f} m²</div>
                <div class="kpi-sub">A/V-Verhältnis: {av_ratio:.2f} m⁻¹</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Fensterfläche (Gesamt)</div>
                <div class="kpi-val">{total_win_area:.2f} m²</div>
                <div class="kpi-sub">Anteil an Hülle: {win_ratio:.1f} %</div>
            </div>
        </div>

        <div class="hint">
            💡 <b>Hinweis für ENVISYS EVEBI 14.3.8:</b> Falls EVEBI nach dem gbXML-Import für das Giebeldach-Geschoss die volle Firsthöhe ansetzt, tragen Sie in EVEBI unter <i>Geschoss / Zone / Raum &rarr; Mittlere lichte Raumhöhe</i> einfach den blau markierten Wert <b>{bldg_h_mean:.3f} m</b> ein – damit rechnet EVEBI exakt mit <b>{total_ngf:.2f} m² × {bldg_h_mean:.3f} m = {total_vol:.2f} m³</b>!
        </div>
    </div>

    <div class="card">
        <h2>1. Zonen- &amp; Raumübersicht (IfcSpace) inkl. mittlerer Höhe</h2>
        <table>
            <thead>
                <tr>
                    <th>Zone / Raum (IfcSpace)</th>
                    <th class="num">Grundfläche (A)</th>
                    <th class="num">Volumen (V)</th>
                    <th class="num" style="background:#dbeafe;color:#1e40af;">Ø Höhe (V/A)</th>
                    <th class="num">Traufhöhe</th>
                    <th class="num">Firsthöhe</th>
                    <th class="num">Wand (netto)</th>
                    <th class="num">Dachfläche</th>
                    <th class="num">Fenster</th>
                    <th class="num">Hüllfläche (Σ)</th>
                </tr>
            </thead>
            <tbody>
                {z_trs}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>2. Alle Hüll- und Bauteilflächen im Detail (Wände, Giebel, Dachschrägen, Boden, Fenster)</h2>
        <table>
            <thead>
                <tr>
                    <th>Bauteiltyp</th>
                    <th>IDF / CAD-ID</th>
                    <th>Typ / Konstruktion</th>
                    <th>Zone / Raum</th>
                    <th>Orientierung</th>
                    <th class="num">Neigung</th>
                    <th class="num">Breite × Höhe</th>
                    <th class="num">Höhenlage (Z)</th>
                    <th class="num">Bruttofläche</th>
                    <th class="num">Abzug (Fenster)</th>
                    <th class="num">Nettofläche</th>
                    <th class="num">U-Wert [W/m²K]</th>
                </tr>
            </thead>
            <tbody>
                {s_trs}
            </tbody>
        </table>
    </div>
</body>
</html>"""
