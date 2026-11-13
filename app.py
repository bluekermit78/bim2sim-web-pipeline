import os
import re
import csv
import uuid
import shutil
import logging
import traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from bim2sim import Project
from bim2sim.utilities.types import IFCDomain, LOD
from sb_generator import ensure_space_boundaries

app = FastAPI(title="bim2sim IFC4-to-EnergyPlus Converter")

WORKSPACE_DIR = Path("/opt/bim2sim_workspace/jobs")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EP_PATH = Path("/usr/local/EnergyPlus-9-4-0")
REPO_DIR = Path("/opt/bim2sim_workspace/bim2sim_repo")

epw_candidates = list(REPO_DIR.rglob("*Aachen*.epw")) + list(REPO_DIR.rglob("*.epw")) + list(DEFAULT_EP_PATH.rglob("*.epw"))
DEFAULT_EPW = epw_candidates[0] if epw_candidates else (DEFAULT_EP_PATH / "WeatherData" / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw")

JOBS = {}


def pick_default_answer(decision):
    cls_name = decision.__class__.__name__
    key_lower = str(getattr(decision, "key", "")).lower()

    if hasattr(decision, "items") and decision.items:
        items = list(decision.items)
        for pref in ["Office", "Single office", "Büro", "Residential", "Single family house"]:
            if pref in items:
                return pref
        return items[0]

    if cls_name == "BoolDecision":
        return True

    if cls_name in ("RealDecision", "FloatDecision", "IntDecision"):
        if getattr(decision, "default", None) is not None:
            return decision.default
        if "year" in key_lower or "construction" in key_lower:
            return 2015
        return 1

    if getattr(decision, "default", None) is not None:
        return decision.default
    return "default"


def _extract_html_table(html_text: str, table_anchor: str):
    """Extrahiert die erste HTML-Tabelle nach einer bestimmten Überschrift in eplustbl.htm."""
    pos = html_text.find(table_anchor)
    if pos == -1:
        return []
    sub = html_text[pos:]
    m = re.search(r"<table[^>]*>(.*?)</table>", sub, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), flags=re.DOTALL | re.IGNORECASE):
        cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.DOTALL | re.IGNORECASE)
        ]
        if cells:
            rows.append(cells)
    return rows


def parse_energyplus_summary(project_dir: Path) -> dict:
    """Liest Heiz-/Kühllasten, Zonenflächen und Temperaturen direkt aus eplustbl.htm und eplusout.csv."""
    summary = {
        "zones": {},
        "total_heating_w": 0.0,
        "total_cooling_w": 0.0,
        "total_area_m2": 0.0,
        "annual_heating_kwh": None,
        "annual_cooling_kwh": None,
        "has_table_html": False
    }

    # 1. Aus eplustbl.htm Zonen-Geometrie und Sizing-Heizlast/Kühllast lesen
    htm_files = list(project_dir.rglob("eplustbl.htm"))
    if htm_files:
        summary["has_table_html"] = True
        htm_text = htm_files[0].read_text(encoding="utf-8", errors="ignore")

        # Zone Summary (Fläche & Volumen)
        z_rows = _extract_html_table(htm_text, "Zone Summary")
        for row in z_rows[1:]:
            if len(row) >= 6 and row[0] not in ("Total", "Conditioned Total", "Unconditioned Total", "Not Part of Total"):
                zname = row[0]
                try:
                    area = float(row[1])
                    vol = float(row[4])
                    summary["zones"].setdefault(zname, {})["area_m2"] = round(area, 2)
                    summary["zones"][zname]["volume_m3"] = round(vol, 2)
                    summary["total_area_m2"] += area
                except ValueError:
                    pass

        # Zone Sensible Heating (Spitzenheizlast)
        h_rows = _extract_html_table(htm_text, "Zone Sensible Heating")
        for row in h_rows[1:]:
            if len(row) >= 3:
                zname = row[0].upper()
                try:
                    # Spalte 1 = Calculated Design Load [W], Spalte 2 = User Design Load [W]
                    load_w = float(row[1])
                    target_z = next((k for k in summary["zones"] if k.upper() == zname), row[0])
                    summary["zones"].setdefault(target_z, {})["heating_load_w"] = round(load_w, 1)
                    summary["total_heating_w"] += load_w
                except ValueError:
                    pass

        # Zone Sensible Cooling (Spitzenkühllast)
        c_rows = _extract_html_table(htm_text, "Zone Sensible Cooling")
        for row in c_rows[1:]:
            if len(row) >= 3:
                zname = row[0].upper()
                try:
                    load_w = float(row[1])
                    target_z = next((k for k in summary["zones"] if k.upper() == zname), row[0])
                    summary["zones"].setdefault(target_z, {})["cooling_load_w"] = round(load_w, 1)
                    summary["total_cooling_w"] += load_w
                except ValueError:
                    pass

    # 2. Aus eplusout.csv Min/Max/Mittelwert der Raumtemperatur & ggf. Jahresarbeit lesen
    csv_files = list(project_dir.rglob("eplusout.csv"))
    if csv_files:
        try:
            with open(csv_files[0], "r", encoding="utf-8", errors="ignore") as cf:
                reader = csv.reader(cf)
                headers = [h.strip() for h in next(reader, [])]
                temp_cols = {}
                heat_energy_cols = []
                cool_energy_cols = []

                for idx, h in enumerate(headers):
                    h_up = h.upper()
                    if "ZONE MEAN AIR TEMPERATURE" in h_up or "ZONE OPERATIVE TEMPERATURE" in h_up:
                        z_prefix = h.split(":")[0].strip()
                        temp_cols.setdefault(z_prefix, []).append(idx)
                    elif "IDEAL LOADS AIR SYSTEM TOTAL HEATING ENERGY" in h_up and "[J]" in h_up:
                        heat_energy_cols.append(idx)
                    elif "IDEAL LOADS AIR SYSTEM TOTAL COOLING ENERGY" in h_up and "[J]" in h_up:
                        cool_energy_cols.append(idx)

                temps_by_zone = {z: [] for z in temp_cols}
                heat_j = 0.0
                cool_j = 0.0
                row_count = 0

                for row in reader:
                    row_count += 1
                    for z_prefix, col_idxs in temp_cols.items():
                        col_i = col_idxs[0]
                        if col_i < len(row) and row[col_i].strip():
                            try:
                                temps_by_zone[z_prefix].append(float(row[col_i]))
                            except ValueError:
                                pass
                    for hc in heat_energy_cols:
                        if hc < len(row) and row[hc].strip():
                            try:
                                heat_j += float(row[hc])
                            except ValueError:
                                pass
                    for cc in cool_energy_cols:
                        if cc < len(row) and row[cc].strip():
                            try:
                                cool_j += float(row[cc])
                            except ValueError:
                                pass

                for z_prefix, vals in temps_by_zone.items():
                    if vals:
                        target_z = next((k for k in summary["zones"] if k.upper() == z_prefix.upper()), z_prefix)
                        summary["zones"].setdefault(target_z, {})["t_min"] = round(min(vals), 1)
                        summary["zones"][target_z]["t_mean"] = round(sum(vals) / len(vals), 1)
                        summary["zones"][target_z]["t_max"] = round(max(vals), 1)

                # Falls Jahressimulation lief (> 1000 Zeitschritte), Jahresheizwärmebedarf in kWh ausweisen
                if row_count > 1000 and (heat_j > 0 or cool_j > 0):
                    summary["annual_heating_kwh"] = round(heat_j / 3.6e6, 1)
                    summary["annual_cooling_kwh"] = round(cool_j / 3.6e6, 1)
        except Exception:
            pass

    summary["total_heating_w"] = round(summary["total_heating_w"], 1)
    summary["total_cooling_w"] = round(summary["total_cooling_w"], 1)
    summary["total_area_m2"] = round(summary["total_area_m2"], 2)
    return summary


def run_bim2sim_pipeline(
    job_id: str,
    ifc_path: Path,
    epw_path: Path,
    add_sb: bool,
    correct_sb: bool,
    close_gaps: bool,
    run_sim: bool
):
    job_dir = WORKSPACE_DIR / job_id
    project_dir = job_dir / "bim2sim_project"
    log_file = job_dir / "conversion.log"
    decision_log = job_dir / "decisions_log.txt"

    JOBS[job_id]["status"] = "running"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"=== Starte bim2sim Konvertierung für Job {job_id} ===\n")
            lf.write(f"IFC-Datei: {ifc_path.name}\n")
            lf.write(f"Wetterdatei: {epw_path}\n\n")

            def _sb_log(msg):
                lf.write(msg + "\n")
                lf.flush()

            ensure_space_boundaries(ifc_path, log_func=_sb_log)

        ifc_paths = {IFCDomain.arch: ifc_path}

        project = Project.create(
            project_folder=project_dir,
            ifc_paths=ifc_paths,
            plugin="energyplus"
        )

        # Schlankes Preset: Keine automatischen Plots/DataFrames im ZIP,
        # Jahressimulation nur wenn Checkbox 'run_sim' aktiv ist!
        sim_settings = project.sim_settings
        for attr, val in [
            ("ep_install_path", DEFAULT_EP_PATH),
            ("weather_file_path", epw_path),
            ("add_space_boundaries", add_sb),
            ("correct_space_boundaries", correct_sb),
            ("close_space_boundary_gaps", close_gaps),
            ("run_full_simulation", run_sim),
            ("create_plots", False),
            ("layers_and_materials", LOD.low),
        ]:
            if hasattr(sim_settings, attr):
                setattr(sim_settings, attr, val)

        with open(decision_log, "a", encoding="utf-8") as df:
            for decision_bunch in project.run(cleanup=False):
                for decision in decision_bunch:
                    val = pick_default_answer(decision)
                    decision.value = val
                    df.write(f"[AUTO-DECISION] Key: {decision.key} ({decision.__class__.__name__})\n")
                    df.write(f"  Frage:   {getattr(decision, 'question', 'N/A')}\n")
                    df.write(f"  Optionen:{getattr(decision, 'items', 'N/A')}\n")
                    df.write(f"  Gewählt: {val}\n\n")
                    df.flush()

        project.finalize(True)
        JOBS[job_id]["status"] = "completed"

    except Exception:
        err_trace = traceback.format_exc()
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[EXCEPTION TRACEBACK]\n{err_trace}\n")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = err_trace
    finally:
        idf_files = list(project_dir.rglob("*.idf"))
        if idf_files:
            JOBS[job_id]["idf_file"] = str(idf_files[0])
        if project_dir.exists():
            from update_preview import parse_energyplus_summary as _new_parse
            JOBS[job_id]["results"] = _new_parse(project_dir)
            zip_base = job_dir / f"result_{job_id}"
            shutil.make_archive(str(zip_base), "zip", project_dir)
            JOBS[job_id]["zip_file"] = f"{zip_base}.zip"
        root_logger.removeHandler(file_handler)
        file_handler.close()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="utf-8">
        <title>bim2sim IFC4 -> EnergyPlus Pipeline</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #f8fafc; color: #1e293b; }
            .card { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
            h1 { margin-top: 0; font-size: 1.5rem; color: #0f172a; }
            label { display: block; margin: 0.75rem 0 0.25rem; font-weight: 600; }
            .checkbox-group label { display: flex; align-items: center; gap: 0.5rem; font-weight: normal; margin: 0.4rem 0; }
            button { background: #2563eb; color: white; border: none; padding: 0.65rem 1.25rem; border-radius: 6px; font-weight: 600; cursor: pointer; margin-top: 1rem; }
            button:hover { background: #1d4ed8; }
            .job-item { border-left: 4px solid #cbd5e1; padding: 1rem; margin: 1rem 0; background: #f1f5f9; border-radius: 0 6px 6px 0; }
            .status-running { border-left-color: #eab308; }
            .status-completed { border-left-color: #22c55e; }
            .status-failed { border-left-color: #ef4444; }
            .links a { display: inline-block; margin-right: 0.85rem; margin-top: 0.6rem; color: #2563eb; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
            .links a:hover { text-decoration: underline; }
            pre { background: #0f172a; color: #f87171; padding: 0.75rem; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; max-height: 250px; }
            .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.75rem; margin-top: 0.85rem; }
            .kpi-box { background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.8rem; }
            .kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }
            .kpi-val { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem; }
            table.res-table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; background: white; font-size: 0.85rem; border-radius: 6px; overflow: hidden; }
            table.res-table th, table.res-table td { padding: 0.5rem 0.65rem; border-bottom: 1px solid #e2e8f0; text-align: right; }
            table.res-table th:first-child, table.res-table td:first-child { text-align: left; font-weight: 600; }
            table.res-table th { background: #e2e8f0; color: #334155; font-size: 0.78rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>bim2sim: Bonsai IFC4 &rarr; EnergyPlus (.idf)</h1>
            <form id="uploadForm">
                <label>1. IFC4-Datei aus Bonsai (.ifc):</label>
                <input type="file" name="ifc_file" accept=".ifc" required>

                <label>2. Optional: Eigene Wetterdatei (.epw):</label>
                <input type="file" name="epw_file" accept=".epw">

                <label>3. Geometrie-Aufbereitung & Simulation:</label>
                <div class="checkbox-group">
                    <label><input type="checkbox" name="add_sb" checked> Fehlende 2nd Level Space Boundaries generieren (add_space_boundaries)</label>
                    <label><input type="checkbox" name="correct_sb" checked> Fehlerhafte Geometrien / Space Boundaries heilen (correct_space_boundaries)</label>
                    <label><input type="checkbox" name="close_gaps" checked> Lücken zwischen Space Boundaries schließen (close_space_boundary_gaps)</label>
                    <label><input type="checkbox" name="run_sim"> Komplette 8760h-Jahressimulation rechnen (ohne Haken: schneller Sizing-Lauf für Spitzenheizlast)</label>
                </div>
                <button type="submit">Konvertierung starten</button>
            </form>
        </div>

        <div class="card">
            <h2>Aktuelle Durchläufe & Ergebnis-Vorschau</h2>
            <div id="jobsList">Lade Jobs...</div>
        </div>

        <script>
            const form = document.getElementById('uploadForm');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const btn = form.querySelector('button');
                btn.disabled = true;
                btn.textContent = 'Wird hochgeladen...';
                await fetch('/api/convert', { method: 'POST', body: formData });
                btn.disabled = false;
                btn.textContent = 'Konvertierung starten';
                form.reset();
                loadJobs();
            });

            function renderPreview(res) {
                if (!res || !res.zones || Object.keys(res.zones).length === 0) return '';
                const specificHeat = res.total_area_m2 > 0 ? (res.total_heating_w / res.total_area_m2).toFixed(1) : '-';
                let kpis = `
                    <div class="kpi-grid">
                        <div class="kpi-box"><div class="kpi-label">Netto-Grundfläche</div><div class="kpi-val">${res.total_area_m2} m²</div></div>
                        <div class="kpi-box"><div class="kpi-label">Auslegungs-Heizlast</div><div class="kpi-val">${res.total_heating_w} W (${specificHeat} W/m²)</div></div>
                        <div class="kpi-box"><div class="kpi-label">Auslegungs-Kühllast</div><div class="kpi-val">${res.total_cooling_w} W</div></div>
                        ${res.annual_heating_kwh !== null ? `<div class="kpi-box"><div class="kpi-label">Jahres-Heizwärme</div><div class="kpi-val">${res.annual_heating_kwh} kWh/a</div></div>` : ''}
                    </div>
                `;
                const rows = Object.entries(res.zones).map(([zname, z]) => `
                    <tr>
                        <td>${zname}</td>
                        <td>${z.area_m2 ?? '-'} m²</td>
                        <td>${z.volume_m3 ?? '-'} m³</td>
                        <td><strong>${z.heating_load_w ?? '-'} W</strong></td>
                        <td>${z.cooling_load_w ?? '-'} W</td>
                        <td>${z.t_min !== undefined ? `${z.t_min} / <strong>${z.t_mean}</strong> / ${z.t_max} °C` : '-'}</td>
                    </tr>
                `).join('');
                return kpis + `
                    <table class="res-table">
                        <thead>
                            <tr>
                                <th>Thermische Zone (IfcSpace)</th>
                                <th>Fläche</th>
                                <th>Volumen</th>
                                <th>Heizlast (Auslegung)</th>
                                <th>Kühllast (Auslegung)</th>
                                <th>Raumtemp. (Min / Ø / Max)</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                `;
            }

            async function loadJobs() {
                const res = await fetch('/api/jobs');
                const jobs = await res.json();
                const container = document.getElementById('jobsList');
                const keys = Object.keys(jobs).reverse();
                if (keys.length === 0) {
                    container.innerHTML = '<p>Noch keine Konvertierungen gestartet.</p>';
                    return;
                }
                container.innerHTML = keys.map(id => {
                    const j = jobs[id];
                    let links = `<a href="/api/download/${id}/log" target="_blank">📄 Logbuch</a> <a href="/api/download/${id}/geometry" target="_blank" style="background:#eff6ff;border-color:#93c5fd;color:#1d4ed8;font-weight:600;">📐 Geometrie & Maße (EVEBI)</a>`;
                    links += `<a href="/api/download/${id}/decisions" target="_blank">⚙️ Auto-Decisions</a>`;
                    if (j.status === 'completed' || j.idf_file) {
                        if (j.idf_file) {
                            links += `<a href="/api/download/${id}/idf">⬇️ Nur .idf herunterladen</a>`;
                            links += `<a href="/api/download/${id}/gbxml">🌐 Als gbXML (.xml) herunterladen (OpenStudio / EVEBI)</a>`;
                        }
                        if (j.results && j.results.has_table_html) links += `<a href="/api/download/${id}/report" target="_blank">📊 Vollständiger EnergyPlus-Report (HTML)</a>`;
                        if (j.zip_file) links += `<a href="/api/download/${id}/zip">📦 Projekt (.zip)</a>`;
                    }
                    return `
                        <div class="job-item status-${j.status}">
                            <div><strong>Job ${id.slice(0,8)}</strong> – Datei: <code>${j.filename}</code> | Status: <strong>${j.status.toUpperCase()}</strong></div>
                            ${j.error ? `<pre>${j.error}</pre>` : ''}
                            ${renderPreview(j.results)}
                            <div class="links">${links}</div>
                        </div>
                    `;
                }).join('');
            }
            setInterval(loadJobs, 3000);
            loadJobs();
        </script>
    </body>
    </html>
    """


@app.post("/api/convert")
async def start_conversion(
    background_tasks: BackgroundTasks,
    ifc_file: UploadFile = File(...),
    epw_file: UploadFile = File(None),
    add_sb: bool = Form(False),
    correct_sb: bool = Form(False),
    close_gaps: bool = Form(False),
    run_sim: bool = Form(False)
):
    job_id = str(uuid.uuid4())
    job_dir = WORKSPACE_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ifc_path = input_dir / ifc_file.filename
    with open(ifc_path, "wb") as f:
        shutil.copyfileobj(ifc_file.file, f)

    if epw_file and epw_file.filename:
        epw_path = input_dir / epw_file.filename
        with open(epw_path, "wb") as f:
            shutil.copyfileobj(epw_file.file, f)
    else:
        epw_path = DEFAULT_EPW

    JOBS[job_id] = {
        "filename": ifc_file.filename,
        "status": "queued",
        "idf_file": None,
        "zip_file": None,
        "results": None,
        "error": None
    }

    background_tasks.add_task(
        run_bim2sim_pipeline,
        job_id,
        ifc_path,
        epw_path,
        add_sb,
        correct_sb,
        close_gaps,
        run_sim
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs():
    from update_preview import parse_energyplus_summary as _new_parse
    for jdir in sorted(WORKSPACE_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if jdir.is_dir() and jdir.name not in JOBS:
            pdir = jdir / "bim2sim_project"
            in_files = list((jdir / "input").glob("*.ifc")) if (jdir / "input").exists() else []
            idf_files = list(pdir.rglob("*.idf")) if pdir.exists() else []
            zip_files = list(jdir.glob("result_*.zip"))
            if in_files or idf_files:
                JOBS[jdir.name] = {
                    "filename": in_files[0].name if in_files else "Projekt.ifc",
                    "status": "completed" if idf_files else "failed",
                    "idf_file": str(idf_files[0]) if idf_files else None,
                    "zip_file": str(zip_files[0]) if zip_files else None,
                    "results": _new_parse(pdir) if pdir.exists() else None,
                    "error": None
                }
        elif jdir.name in JOBS and JOBS[jdir.name].get("status") == "completed":
            pdir = jdir / "bim2sim_project"
            if pdir.exists():
                JOBS[jdir.name]["results"] = _new_parse(pdir)
    return JOBS


@app.get("/api/download/{job_id}/{file_type}")
def download_result(job_id: str, file_type: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

    job_dir = WORKSPACE_DIR / job_id
    project_dir = job_dir / "bim2sim_project"

    if file_type == "geometry":
        from fastapi.responses import HTMLResponse
        from geometry_report import generate_geometry_html
        html_content = generate_geometry_html(job_id, job_dir, JOBS[job_id].get("filename", "Modell.ifc"))
        return HTMLResponse(content=html_content)

    if file_type == "gbxml":
        from gbxml_exporter import generate_gbxml_from_job
        gbxml_path = job_dir / f"{Path(JOBS[job_id]['filename']).stem}_gbxml.xml"
        in_ifcs = list((job_dir / "input").glob("*.ifc"))
        ifc_src = in_ifcs[0] if in_ifcs else None
        try:
            generate_gbxml_from_job(project_dir, gbxml_path, ifc_path=ifc_src)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"gbXML-Generierung fehlgeschlagen: {e}")
        return FileResponse(gbxml_path, filename=gbxml_path.name, media_type="application/xml")

    elif file_type == "idf":
        idf_path = JOBS[job_id].get("idf_file")
        if not idf_path or not Path(idf_path).exists():
            raise HTTPException(status_code=404, detail="IDF-Datei nicht vorhanden")
        return FileResponse(idf_path, filename=f"{Path(JOBS[job_id]['filename']).stem}.idf")

    elif file_type == "report":
        htm_files = list(project_dir.rglob("eplustbl.htm"))
        if not htm_files:
            raise HTTPException(status_code=404, detail="EnergyPlus HTML-Report nicht vorhanden")
        return FileResponse(htm_files[0], media_type="text/html")

    elif file_type == "zip":
        zip_path = JOBS[job_id].get("zip_file")
        if not zip_path or not Path(zip_path).exists():
            raise HTTPException(status_code=404, detail="ZIP-Archiv nicht vorhanden")
        return FileResponse(zip_path, filename=f"bim2sim_{job_id[:8]}.zip")

    elif file_type == "log":
        log_path = job_dir / "conversion.log"
        if not log_path.exists():
            return HTMLResponse("Noch kein Log vorhanden.")
        return FileResponse(log_path, media_type="text/plain")

    elif file_type == "decisions":
        dec_path = job_dir / "decisions_log.txt"
        if not dec_path.exists():
            return HTMLResponse("Keine offenen Rückfragen (Decisions) aufgetreten.")
        return FileResponse(dec_path, media_type="text/plain")

    raise HTTPException(status_code=400, detail="Ungültiger Dateityp")
