from pathlib import Path
import json, re, ifcopenshell

CAT_PATH = Path("/opt/bim2sim_workspace/bauteil_katalog.json")
TEMPLATES = {
    "massiv": ("Massivbau (Ziegel/KS)", 0.17, 1.40, ("Mauerwerk", 0.175, 0.56, 1800, 1000), ("WDVS_Daemmung", 0.035, 30, 1400)),
    "leichtbau": ("Leichtbau / Sparrendach", 0.14, 0.45, ("Holzschalung", 0.02, 0.13, 600, 1600), ("Mineralwolle", 0.035, 40, 1400)),
    "bodenplatte": ("Stahlbeton / Erdreich", 0.17, 0.56, ("Stahlbeton_BP", 0.20, 2.30, 2400, 1000), ("Perimeterdaemm", 0.035, 35, 1400)),
    "geschossdecke": ("Geschossdecke (STB+Estrich)", 0.34, 1.18, ("Stahlbeton_GD", 0.16, 2.30, 2400, 1000), ("Trittschall", 0.040, 30, 1400)),
    "innenwand": ("Innenwand (Mauerwerk 10cm)", 0.26, 2.32, ("Innenmauerwerk", 0.10, 0.56, 1400, 1000), ("Feinputz", 0.70, 1400, 1000)),
    "tuer": ("Aussentuer Holz/Metall", 0.17, 1.80, ("Tuerblatt", 0.04, 0.14, 600, 1600), ("Tuerkern", 0.04, 80, 1400)),
    "fenster": ("Verglasung / Fenster", 0.0, 1.30, None, None),
}

def load_catalog() -> dict:
    return json.loads(CAT_PATH.read_text("utf-8")) if CAT_PATH.exists() else {}

def save_catalog(data: dict) -> dict:
    cat = load_catalog(); cat.update(data)
    CAT_PATH.write_text(json.dumps(cat, indent=2, ensure_ascii=False), "utf-8")
    return cat

def _default_tpl(name: str, cls: str) -> str:
    n = name.lower()
    if cls == "IfcWindow": return "fenster"
    if cls == "IfcDoor": return "tuer"
    if "innenwand" in n: return "innenwand"
    if "geschossdecke" in n: return "geschossdecke"
    if "bodenplatte" in n: return "bodenplatte"
    if cls == "IfcRoof" or "dach" in n: return "leichtbau"
    return "bodenplatte" if cls == "IfcSlab" else "massiv"

def inspect_ifc_types(ifc_path: Path) -> list:
    cat = load_catalog(); f = ifcopenshell.open(str(ifc_path)); found = {}
    for cls in ("IfcWall", "IfcWallStandardCase", "IfcRoof", "IfcSlab", "IfcWindow", "IfcDoor"):
        for el in f.by_type(cls):
            t_lbl = None
            for rel in getattr(el, "IsTypedBy", []) or []:
                if getattr(rel, "RelatingType", None) and rel.RelatingType.Name:
                    t_lbl = rel.RelatingType.Name; break
            t_lbl = t_lbl or el.Name or cls
            c_norm = "IfcWall" if "Wall" in cls else cls
            found.setdefault(t_lbl, {"type_name": t_lbl, "ifc_class": c_norm, "count": 0})["count"] += 1
    res = []
    for k, v in sorted(found.items(), key=lambda x: (x[1]["ifc_class"], x[0])):
        tpl = _default_tpl(k, v["ifc_class"])
        saved = cat.get(k, {})
        res.append({**v, "template": saved.get("template", tpl), "u_value": float(saved.get("u_value", TEMPLATES[tpl][2])), "shgc": float(saved.get("shgc", 0.60)) if v["ifc_class"]=="IfcWindow" else None})
    return res

def _clean_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)

def calibrate_layers(t_name: str, tpl_key: str, target_u: float) -> tuple:
    tpl = TEMPLATES.get(tpl_key, TEMPLATES["massiv"])
    r_sise, _, base, ins = (0.34 if tpl_key=='geschossdecke' else (0.26 if tpl_key=='innenwand' else tpl[1])), tpl[2], tpl[3], tpl[4]
    u = max(0.05, float(target_u))
    r_tot = 1.0 / u
    r_lay = max(0.03, r_tot - r_sise)
    cid = _clean_id(t_name)
    b_nm, b_d, b_lam, b_rho, b_cp = base
    i_nm, i_lam, i_rho, i_cp = ins
    r_base = b_d / b_lam
    if r_lay > r_base + 0.015 and tpl_key != "innenwand":
        d_ins = round((r_lay - r_base) * i_lam, 4)
        layers = [(f"Mat_{cid}_1", b_nm, b_d, b_lam, b_rho, b_cp), (f"Mat_{cid}_2", i_nm, d_ins, i_lam, i_rho, i_cp)]
    else:
        lam_cal = round(b_d / r_lay, 4)
        layers = [(f"Mat_{cid}_1", b_nm, b_d, lam_cal, b_rho, b_cp)]
    return round(r_lay, 4), layers

def apply_catalog_to_idf(project_dir: Path, ifc_path: Path = None):
    for xml_f in Path(project_dir).rglob("*_gbxml.xml"): xml_f.unlink()

def apply_catalog_to_gbxml_file(xml_path: Path, project_dir: Path):
    import xml.etree.ElementTree as ET
    cands = sorted(Path(project_dir).parent.rglob("*.ifc"), key=lambda x: "space_boundaries" in x.name)
    if not cands: return
    types = {x["type_name"]: x for x in inspect_ifc_types(cands[0])}
    ET.register_namespace("", "http://www.gbxml.org/schema")
    tree = ET.parse(xml_path); root = tree.getroot()
    ns = "{http://www.gbxml.org/schema}"
    wt_l = list(root.findall(f"{ns}WindowType"))
    dh = root.find(f"{ns}DocumentHistory")
    for tag in ("Construction", "Layer", "Material", "WindowType", "DocumentHistory"):
        for el in list(root.findall(f"{ns}{tag}")): root.remove(el)
    c_l, l_l, m_l = [], [], []
    for t_nm, info in types.items():
        if info["ifc_class"] == "IfcWindow": continue
        cid, u = _clean_id(t_nm), float(info["u_value"])
        r_v, lays = calibrate_layers(t_nm, info["template"], u)
        ce = ET.Element(f"{ns}Construction", {"id": f"Cons_{cid}"})
        ET.SubElement(ce, f"{ns}Name").text = f"{t_nm} ({TEMPLATES[info['template']][0]})"
        ET.SubElement(ce, f"{ns}U-value", {"unit": "WPerSquareMeterK"}).text = f"{u:.3f}"
        ET.SubElement(ce, f"{ns}LayerId", {"layerIdRef": f"Lay_{cid}"}); c_l.append(ce)
        le = ET.Element(f"{ns}Layer", {"id": f"Lay_{cid}"})
        ET.SubElement(le, f"{ns}Name").text = f"Lay_{cid}"
        ET.SubElement(le, f"{ns}R-value", {"unit": "SquareMeterKPerW"}).text = f"{r_v:.4f}"
        for mid, mnm, md, ml, mr, mc in lays:
            ET.SubElement(le, f"{ns}MaterialId", {"materialIdRef": mid})
            me = ET.Element(f"{ns}Material", {"id": mid})
            ET.SubElement(me, f"{ns}Name").text = mnm
            ET.SubElement(me, f"{ns}Thickness", {"unit": "Meters"}).text = f"{md:.4f}"
            ET.SubElement(me, f"{ns}Conductivity", {"unit": "WPerMeterK"}).text = f"{ml:.4f}"
            ET.SubElement(me, f"{ns}Density", {"unit": "KgPerCubicMeter"}).text = f"{mr:.1f}"
            ET.SubElement(me, f"{ns}SpecificHeat", {"unit": "JPerKgK"}).text = f"{mc:.1f}"; m_l.append(me)
        l_l.append(le)
    wt_l = [wt for wt in wt_l if wt.findtext(f'{ns}Name','') in types]
    for t_nm, info in types.items():
        if info["ifc_class"] == "IfcDoor":
            cid = _clean_id(t_nm)
            wte = ET.Element(f"{ns}WindowType", {"id": f"WinType_{cid}"})
            ET.SubElement(wte, f"{ns}Name").text = f"{t_nm} (Tür)"
            ET.SubElement(wte, f"{ns}U-value", {"unit": "WPerSquareMeterK"}).text = f"{float(info['u_value']):.2f}"
            ET.SubElement(wte, f"{ns}SolarHeatGainCoeff", {"unit": "Fraction"}).text = "0.00"
            wt_l.append(wte)
    for wt in wt_l:
        wnm = wt.findtext(f"{ns}Name", "")
        if wnm in types:
            wt.find(f"{ns}U-value").text = f"{ float(types[wnm]['u_value']):.2f}"
            wt.find(f"{ns}SolarHeatGainCoeff").text = f"{ float(types[wnm].get('shgc') or 0.6):.2f}"
    for el in c_l + l_l + m_l + wt_l + ([dh] if dh is not None else []): root.append(el)
    sp_ids = [sp.attrib["id"] for sp in root.findall(f".//{ns}Space")]
    for sf in root.findall(f".//{ns}Surface"):
        if sf.attrib.get("surfaceType") in ("InteriorFloor", "InteriorWall", "Ceiling"):
            sf.set("exposedToSun", "false")

        nm = sf.findtext(f"{ns}Name", "")
        for t_nm in sorted(types, key=len, reverse=True):
            if nm.startswith(t_nm + "_") or nm == t_nm:
                sf.set("constructionIdRef", f"Cons_{_clean_id(t_nm)}"); break
        for op in sf.findall(f"{ns}Opening"):
            onm = op.findtext(f"{ns}Name", "")
            for t_nm in sorted(types, key=len, reverse=True):
                if onm.startswith(t_nm + "_") or onm == t_nm:
                    cid = _clean_id(t_nm)
                    if types[t_nm]["ifc_class"] == "IfcDoor":
                        if "evebi" in str(xml_path).lower():
                            op.attrib.pop("constructionIdRef", None)
                            op.set("windowTypeIdRef", f"WinType_{cid}")
                        else:
                            op.set("constructionIdRef", f"Cons_{cid}")
                    else:
                        op.set("windowTypeIdRef", f"WinType_{cid}")
                    break
    ET.indent(tree, space="  "); tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def fix_geometry_report_html(job_dir: Path, ht: str) -> str:
    import xml.etree.ElementTree as ET, re
    from gbxml_exporter import generate_gbxml_from_job
    proj = Path(job_dir) / "bim2sim_project" if (Path(job_dir) / "bim2sim_project").exists() else Path(job_dir)
    xp = proj / "sync_temp_gbxml.xml"
    generate_gbxml_from_job(proj, xp)
    root = ET.parse(xp).getroot()
    ns = "{http://www.gbxml.org/schema}"
    cu = {c.attrib["id"]: float(c.findtext(f"{ns}U-value","0")) for c in root.findall(f"{ns}Construction")}
    wu = {w.attrib["id"]: float(w.findtext(f"{ns}U-value","0")) for w in root.findall(f"{ns}WindowType")}
    sp_m = {sp.attrib["id"]: sp.findtext(f"{ns}Name","") for sp in root.findall(f".//{ns}Space")}
    tb = {"ExteriorWall":("Außenwand","#dbeafe","#1e40af"),"Roof":("Dachfläche","#ffedd5","#9a3412"),"SlabOnGrade":("Bodenplatte","#f1f5f9","#334155"),"InteriorWall":("Innenwand","#dcfce7","#166534"),"InteriorFloor":("Geschossdecke","#f3e8ff","#6b21a8"),"Ceiling":("Geschossdecke","#f3e8ff","#6b21a8")}
    rows, ze = [], {nm: {"w":0.0,"r":0.0,"f":0.0,"win":0.0} for nm in sp_m.values()}
    for sf in root.findall(f".//{ns}Surface"):
        st, nm = sf.attrib.get("surfaceType",""), sf.findtext(f"{ns}Name","")
        t_nm = re.sub(r"_[A-Za-z0-9_$]{8}$", "", nm)
        sids = [e.attrib.get("spaceIdRef","") for e in sf.findall(f"{ns}AdjacentSpaceId")]
        zn = sp_m.get(sids[0],"—") if sids else "—"
        rg = sf.find(f"{ns}RectangularGeometry")
        w, h, tilt, az = [float(rg.findtext(f"{ns}{k}","0")) for k in ("Width","Height","Tilt","Azimuth")]
        gr = round(w*h, 2)
        zs = [float(c.findall(f"{ns}Coordinate")[2].text) for c in sf.findall(f".//{ns}PlanarGeometry//{ns}CartesianPoint")]
        z0, z1 = (min(zs), max(zs)) if zs else (0.0, 0.0)
        uv = cu.get(sf.attrib.get("constructionIdRef",""), 0.0)
        ops = sf.findall(f"{ns}Opening")
        wa = sum(round(float(o.find(f"{ns}RectangularGeometry/{ns}Width").text)*float(o.find(f"{ns}RectangularGeometry/{ns}Height").text),2) for o in ops)
        net = round(max(0.0, gr - wa), 2)
        if zn in ze:
            if st=="ExteriorWall": ze[zn]["w"]+=net; ze[zn]["win"]+=wa
            elif st=="Roof": ze[zn]["r"]+=net
            elif st=="SlabOnGrade": ze[zn]["f"]+=net
        lb, bg, fg = tb.get(st, (st,"#f1f5f9","#334155"))
        ori = "Horizontal (Dach)" if tilt<10 else ("Horizontal (Boden)" if tilt>170 else f"{az:.0f}°")
        rows.append(f"<tr><td><span class='badge' style='background:{bg};color:{fg};'>{lb}</span></td><td style='font-family:monospace;font-size:11px;'>{sf.attrib['id'].replace('Surf_','')}</td><td><code>{t_nm}</code></td><td><b>{zn}</b></td><td>{ori}</td><td class='num'>{tilt:.1f}°</td><td class='num'>{w:.2f} &times; {h:.2f} m</td><td class='num'>{z0:.2f} &ndash; {z1:.2f} m</td><td class='num'>{gr:.2f} m²</td><td class='num'>{wa:.2f} m²</td><td class='num'><b>{net:.2f} m²</b></td><td class='num'>{uv:.2f}</td></tr>")
        for o in ops:
            ow, oh = float(o.find(f"{ns}RectangularGeometry/{ns}Width").text), float(o.find(f"{ns}RectangularGeometry/{ns}Height").text)
            is_d = (o.attrib.get("openingType") == "NonSlidingDoor")
            ref_id = o.attrib.get("constructionIdRef") if is_d else o.attrib.get("windowTypeIdRef", "")
            oa = round(ow*oh, 2)
            ou = cu.get(ref_id, 1.8) if is_d else wu.get(ref_id, 1.3)
            olbl = "↳ Tür" if is_d else "↳ Fenster"
            obg = "#fef3c7;color:#92400e" if is_d else "#e0f2fe;color:#0369a1"
            otnm = re.sub(r"^(Cons_|WinType_|Win_)", "", ref_id or "")
            rows.append(f"<tr class='win-row'><td><span class='badge' style='background:{obg};'>{olbl}</span></td><td style='font-family:monospace;font-size:11px;'>{o.attrib['id']}</td><td><code>{otnm}</code></td><td>{zn}</td><td>{ori}</td><td class='num'>{tilt:.1f}°</td><td class='num'>{ow:.2f} &times; {oh:.2f} m</td><td class='num'>—</td><td class='num'>{oa:.2f} m²</td><td class='num'>—</td><td class='num'><b>{oa:.2f} m²</b></td><td class='num'>{ou:.2f}</td></tr>")
    ht = re.sub(r"(<h2>2\. Alle Hüll- und Bauteilflächen.*?<tbody>).*?(</tbody>)", rf"\g<1>{''.join(rows)}\2", ht, flags=re.S)
    for zn, d in ze.items():
        es = d["w"]+d["r"]+d["f"]+d["win"]
        def _upd(m):
            c = re.findall(r"<td[^>]*>.*?</td>", m.group(0), flags=re.S)
            if len(c)>=10:
                c[6], c[7], c[8], c[9] = f"<td class='num'>{d['w']:.2f} m²</td>", f"<td class='num'>{d['r']:.2f} m²</td>", f"<td class='num'>{d['win']:.2f} m²</td>", f"<td class='num'><b>{es:.2f} m²</b></td>"
            return "<tr>" + "".join(c) + "</tr>"
        ht = re.sub(rf"<tr>\s*<td><b>{re.escape(zn)}</b>.*?</tr>", _upd, ht, flags=re.S)
    tot = sum(d["w"]+d["r"]+d["f"]+d["win"] for d in ze.values())
    ht = re.sub(r"(Thermische Hüllfläche \(A\)</div>\s*<div[^>]*>)[^<]+(</div>)", rf"\g<1>{tot:.2f} m²\2", ht)
    ht = re.sub(r"A/V-Verhältnis:\s*[0-9.]+", f"A/V-Verhältnis: {tot/(sum(float(sp.findtext(f'{ns}Volume', '0')) for sp in root.findall(f'.//{ns}Space')) or 107.18):.2f}", ht)
    return ht
