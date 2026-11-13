import re
import csv
import numpy as np
from pathlib import Path


def _calc_idf_zone_floor_areas(project_dir: Path) -> dict:
    """Liest als Fallback die Grundfläche jeder Zone direkt aus den Boden-/Decken-Polygonen der .idf-Datei."""
    areas = {}
    idf_files = [f for f in project_dir.rglob("*.idf") if "Minimal.idf" not in str(f)]
    if not idf_files:
        return areas
    try:
        txt = idf_files[0].read_text(encoding="utf-8", errors="ignore")
        blocks = [b.strip() for b in txt.split(";") if "BuildingSurface:Detailed" in b]
        horiz_by_zone = {}
        for b in blocks:
            lines = [l.split("!")[0].strip() for l in b.splitlines() if l.split("!")[0].strip()]
            fields = [f.strip() for f in " ".join(lines).split(",")]
            if len(fields) < 15 or fields[0].upper() != "BUILDINGSURFACE:DETAILED":
                continue
            zname = fields[4].upper()
            coords_raw = []
            for val in fields[10:]:
                try:
                    coords_raw.append(float(val))
                except ValueError:
                    pass
            pts = [coords_raw[i:i+3] for i in range(0, (len(coords_raw) // 3) * 3, 3)]
            if len(pts) >= 3:
                p0 = np.array(pts[0], dtype=float)
                cross_sum = np.zeros(3, dtype=float)
                for i in range(1, len(pts) - 1):
                    cross_sum += np.cross(np.array(pts[i]) - p0, np.array(pts[i + 1]) - p0)
                norm_len = float(np.linalg.norm(cross_sum))
                if norm_len > 1e-6:
                    nz = abs(cross_sum[2] / norm_len)
                    if nz > 0.7:  # Horizontale oder leicht geneigte Fläche -> Grundriss-Projektionsfläche (XY)
                        xy_area = 0.5 * abs(float(cross_sum[2]))
                        avg_z = sum(p[2] for p in pts) / len(pts)
                        horiz_by_zone.setdefault(zname, []).append((avg_z, xy_area))

        for zname, items in horiz_by_zone.items():
            items.sort(key=lambda x: x[0])
            min_z = items[0][0]
            # Summe aller Flächen auf dem untersten Niveau (falls Boden vorhanden, sonst Projektion der Decke)
            bottom_areas = [a for z, a in items if abs(z - min_z) < 0.3]
            areas[zname] = round(sum(bottom_areas), 2)
    except Exception:
        pass
    return areas


def parse_energyplus_summary(project_dir: Path) -> dict:
    summary = {
        "zones": {},
        "total_heating_w": 0.0,
        "total_cooling_w": 0.0,
        "total_area_m2": 0.0,
        "annual_heating_kwh": 0.0,
        "annual_cooling_kwh": 0.0,
        "has_table_html": False
    }

    htm_files = list(project_dir.rglob("*tbl.htm*")) + list(project_dir.rglob("*Table.htm*"))
    if htm_files:
        summary["has_table_html"] = True
        htm_text = htm_files[0].read_text(encoding="utf-8", errors="ignore")

        tables = list(re.finditer(r"<table[^>]*>(.*?)</table>", htm_text, flags=re.DOTALL | re.IGNORECASE))
        for idx, m in enumerate(tables):
            start_pos = tables[idx - 1].end() if idx > 0 else max(0, m.start() - 1500)
            context = re.sub(r"<[^>]+>", " ", htm_text[start_pos:m.start()])
            context_clean = " ".join(context.split()).lower()

            rows = []
            for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), flags=re.DOTALL | re.IGNORECASE):
                cells = [
                    " ".join(re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").split())
                    for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.DOTALL | re.IGNORECASE)
                ]
                if any(cells):
                    rows.append(cells)

            if not rows:
                continue

            header_str = " ".join(rows[0]).lower()

            if "zone summary" in context_clean or ("area [m2]" in header_str and "volume [m3]" in header_str):
                for row in rows[1:]:
                    if len(row) >= 5:
                        zname = row[0].strip()
                        if not zname or zname.lower() in ("total", "conditioned total", "unconditioned total", "not part of total", "zone"):
                            continue
                        try:
                            area = float(row[1])
                            vol = float(row[4]) if len(row) > 4 and row[4].replace(".", "", 1).isdigit() else float(row[3])
                            z_dict = summary["zones"].setdefault(zname.upper(), {"label": zname})
                            z_dict["area_m2"] = round(area, 2)
                            z_dict["volume_m3"] = round(vol, 2)
                        except ValueError:
                            pass

            elif "zone sensible heating" in context_clean or ("calculated design load" in header_str and "heating" in context_clean):
                for row in rows[1:]:
                    if len(row) >= 3:
                        zname = row[0].strip()
                        if not zname or "total" in zname.lower():
                            continue
                        for col_val in (row[1], row[2]):
                            try:
                                val = float(col_val)
                                if val > 0:
                                    z_dict = summary["zones"].setdefault(zname.upper(), {"label": zname})
                                    z_dict["heating_load_w"] = round(val, 1)
                                    break
                            except ValueError:
                                pass

            elif "zone sensible cooling" in context_clean or ("calculated design load" in header_str and "cooling" in context_clean):
                for row in rows[1:]:
                    if len(row) >= 3:
                        zname = row[0].strip()
                        if not zname or "total" in zname.lower():
                            continue
                        for col_val in (row[1], row[2]):
                            try:
                                val = float(col_val)
                                if val > 0:
                                    z_dict = summary["zones"].setdefault(zname.upper(), {"label": zname})
                                    z_dict["cooling_load_w"] = round(val, 1)
                                    break
                            except ValueError:
                                pass

    csv_files = list(project_dir.rglob("eplusout.csv"))
    if csv_files:
        try:
            with open(csv_files[0], "r", encoding="utf-8", errors="ignore") as cf:
                reader = csv.reader(cf)
                headers = [h.strip() for h in next(reader, [])]

                temp_cols = {}
                heat_rate_cols = {}
                cool_rate_cols = {}
                heat_energy_cols = {}
                cool_energy_cols = {}

                for idx, h in enumerate(headers):
                    h_up = h.upper()
                    if ":" not in h_up:
                        continue
                    prefix = h.split(":")[0].strip()
                    z_key = prefix.split()[0].upper()
                    if z_key in ("ENVIRONMENT", "SITE", "EMS", " WHOLEBUILDING", "FACILITY"):
                        continue

                    if "TEMPERATURE" in h_up and ("ZONE" in h_up or "OPERATIVE" in h_up or "AIR" in h_up):
                        temp_cols.setdefault(z_key, []).append(idx)
                    elif "HEATING RATE" in h_up and "[W]" in h_up:
                        heat_rate_cols.setdefault(z_key, []).append(idx)
                    elif "COOLING RATE" in h_up and "[W]" in h_up:
                        cool_rate_cols.setdefault(z_key, []).append(idx)
                    elif "HEATING ENERGY" in h_up and "[J]" in h_up:
                        heat_energy_cols.setdefault(z_key, []).append(idx)
                    elif "COOLING ENERGY" in h_up and "[J]" in h_up:
                        cool_energy_cols.setdefault(z_key, []).append(idx)

                all_z_keys = set(temp_cols) | set(heat_rate_cols) | set(cool_rate_cols) | set(heat_energy_cols) | set(cool_energy_cols)
                temps = {k: [] for k in all_z_keys}
                max_heat_w = {k: 0.0 for k in all_z_keys}
                max_cool_w = {k: 0.0 for k in all_z_keys}
                sum_heat_j = {k: 0.0 for k in all_z_keys}
                sum_cool_j = {k: 0.0 for k in all_z_keys}
                sum_heat_wh = {k: 0.0 for k in all_z_keys}
                sum_cool_wh = {k: 0.0 for k in all_z_keys}
                row_count = 0

                for row in reader:
                    row_count += 1
                    for zk in all_z_keys:
                        for ci in temp_cols.get(zk, [])[:1]:
                            if ci < len(row) and row[ci].strip():
                                try:
                                    temps[zk].append(float(row[ci]))
                                except ValueError:
                                    pass
                        for ci in heat_rate_cols.get(zk, [])[:1]:
                            if ci < len(row) and row[ci].strip():
                                try:
                                    v = float(row[ci])
                                    if v > max_heat_w[zk]:
                                        max_heat_w[zk] = v
                                    sum_heat_wh[zk] += v
                                except ValueError:
                                    pass
                        for ci in cool_rate_cols.get(zk, [])[:1]:
                            if ci < len(row) and row[ci].strip():
                                try:
                                    v = float(row[ci])
                                    if v > max_cool_w[zk]:
                                        max_cool_w[zk] = v
                                    sum_cool_wh[zk] += v
                                except ValueError:
                                    pass
                        for ci in heat_energy_cols.get(zk, [])[:1]:
                            if ci < len(row) and row[ci].strip():
                                try:
                                    sum_heat_j[zk] += float(row[ci])
                                except ValueError:
                                    pass
                        for ci in cool_energy_cols.get(zk, [])[:1]:
                            if ci < len(row) and row[ci].strip():
                                try:
                                    sum_cool_j[zk] += float(row[ci])
                                except ValueError:
                                    pass

                dt_hours = 8760.0 / row_count if row_count > 1000 else 1.0

                for zk in all_z_keys:
                    z_dict = summary["zones"].setdefault(zk, {"label": zk})
                    if temps[zk]:
                        z_dict["t_min"] = round(min(temps[zk]), 1)
                        z_dict["t_mean"] = round(sum(temps[zk]) / len(temps[zk]), 1)
                        z_dict["t_max"] = round(max(temps[zk]), 1)
                    if max_heat_w[zk] > 0 and not z_dict.get("heating_load_w"):
                        z_dict["heating_load_w"] = round(max_heat_w[zk], 1)
                    if max_cool_w[zk] > 0 and not z_dict.get("cooling_load_w"):
                        z_dict["cooling_load_w"] = round(max_cool_w[zk], 1)

                    if sum_heat_j[zk] > 0:
                        z_dict["heating_kwh"] = round(sum_heat_j[zk] / 3.6e6, 1)
                    elif sum_heat_wh[zk] > 0 and row_count > 1000:
                        z_dict["heating_kwh"] = round((sum_heat_wh[zk] * dt_hours) / 1000.0, 1)

                    if sum_cool_j[zk] > 0:
                        z_dict["cooling_kwh"] = round(sum_cool_j[zk] / 3.6e6, 1)
                    elif sum_cool_wh[zk] > 0 and row_count > 1000:
                        z_dict["cooling_kwh"] = round((sum_cool_wh[zk] * dt_hours) / 1000.0, 1)
        except Exception as e:
            print("CSV Parse Error:", e)

    # 3. Fallback: Falls EnergyPlus area_m2 = 0.0 geliefert hat, Grundfläche direkt aus IDF-Polygonen berechnen
    idf_areas = _calc_idf_zone_floor_areas(project_dir)
    for zk, zd in summary["zones"].items():
        if not zd.get("area_m2") or zd.get("area_m2") == 0.0:
            if zk in idf_areas and idf_areas[zk] > 0:
                zd["area_m2"] = idf_areas[zk]
            elif zd.get("volume_m3", 0.0) > 0:
                # Fallback über Standard-Raumhöhe, falls keine horizontale Fläche gefunden wurde
                zd["area_m2"] = round(zd["volume_m3"] / 2.8, 2)

    # 4. Summen bilden
    for zk, zd in summary["zones"].items():
        summary["total_area_m2"] += zd.get("area_m2", 0.0)
        summary["total_heating_w"] += zd.get("heating_load_w", 0.0)
        summary["total_cooling_w"] += zd.get("cooling_load_w", 0.0)
        summary["annual_heating_kwh"] += zd.get("heating_kwh", 0.0)
        summary["annual_cooling_kwh"] += zd.get("cooling_kwh", 0.0)

    summary["total_area_m2"] = round(summary["total_area_m2"], 2)
    summary["total_heating_w"] = round(summary["total_heating_w"], 1)
    summary["total_cooling_w"] = round(summary["total_cooling_w"], 1)
    summary["annual_heating_kwh"] = round(summary["annual_heating_kwh"], 1) if summary["annual_heating_kwh"] > 0 else None
    summary["annual_cooling_kwh"] = round(summary["annual_cooling_kwh"], 1) if summary["annual_cooling_kwh"] > 0 else None
    return summary

if __name__ == "__main__":
    test_dir = Path("/opt/bim2sim_workspace/jobs/07d772ec-bace-456b-9f44-1d5bda7636b1/bim2sim_project")
    if test_dir.exists():
        import pprint
        print("\n=== Korrigierte Vorschau-Daten für Job 07d772ec ===")
        pprint.pprint(parse_energyplus_summary(test_dir))
