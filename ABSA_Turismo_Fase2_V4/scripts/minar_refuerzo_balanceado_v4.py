# -*- coding: utf-8 -*-
"""
minar_refuerzo_balanceado_v4.py
================================
Diagnostico profundo de balance de polaridades del gold V4 + minado de candidatos
de refuerzo BALANCEADO (negativo / neutro / positivo) para anotacion humana.

NO etiqueta automaticamente. `polaridad_sugerida` es solo una guia de muestreo;
la anotacion humana manda.

Salidas:
  outputs/reports/diagnostico_balance_polaridades_v4.csv   (global + por aspecto)
  outputs/reports/diagnostico_balance_polaridades_v4.md    (narrativa + planes)
  data/gold/candidatos_refuerzo_balanceado_v4.csv          (candidatos seleccionados, formato largo)
  data/gold/plantilla_refuerzo_balanceado_v4.csv           (plantilla multi-etiqueta, una fila/resena)

Uso:
  python scripts/minar_refuerzo_balanceado_v4.py
"""
import re
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuracion / metas (transparentes y editables)
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GOLD = BASE / "outputs" / "gold"
REP = BASE / "outputs" / "reports"
GOLD_DATA = DATA / "gold"
for d in (REP, GOLD_DATA):
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["negativo", "neutro", "positivo"]
POOL_NAME = "C_refuerzo_balanceado_v4"

# Meta de soporte por (aspecto x clase) que se quiere alcanzar para entrenamiento.
MIN_PER_ASPECT_CLASS = 30
# Empuje global a la clase mas escasa (negativo): cuantos candidatos negativos
# seleccionar como maximo (la anotacion confirmara o no). 295 actuales -> ~600.
NEG_SELECT_TARGET = 430
NEU_SELECT_TARGET = 210
POS_SELECT_TARGET = 130
# Diversidad de estrellas en negativos: tope de negativos provenientes de 1-2 estrellas,
# para forzar la inclusion de negativos "ocultos" en resenas de 3-5 estrellas (lexico + adversativo).
NEG_MAX_LOWSTAR = 340
# Topes anti-dominacion: maximo de candidatos por (clase, destino) y por resena.
CAP_PER_DEST = {"negativo": 45, "neutro": 35, "positivo": 18}
MAX_ASPECTS_PER_REVIEW = 3  # una resena no debe llenar demasiadas celdas

# ----------------------------------------------------------------------------
# Normalizacion y deteccion de aspectos (consistente con el NB03 / diccionario)
# ----------------------------------------------------------------------------
def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9n\s]", " ", t)).strip()


def build_aspect_patterns(dic: dict) -> dict:
    pat = {}
    for asp, kws in dic.items():
        ps = []
        for k in kws:
            base = norm(k.rstrip("*"))
            if not base:
                continue
            suffix = r"\w*" if str(k).endswith("*") else r"(?!\w)"
            ps.append(r"(?<!\w)" + re.escape(base) + suffix)
        pat[asp] = ps
    return pat


def detect_aspects(text_norm: str, pat: dict) -> list:
    return [a for a, ps in pat.items() if any(re.search(p, text_norm) for p in ps)]


# ----------------------------------------------------------------------------
# Lexicos de senal de polaridad (se normalizan -> sin tildes)
# ----------------------------------------------------------------------------
def _nset(words):
    return [norm(w) for w in words if norm(w)]


LEX_NEG = _nset([
    "caro", "costoso", "carisimo", "inseguro", "peligroso", "sucio", "suciedad",
    "mala atencion", "mal servicio", "pesima atencion", "estafa", "estafaron",
    "demora", "demoró", "lleno", "saturado", "abarrotado", "dificil acceso",
    "decepcionante", "decepcion", "no vale", "no vale la pena", "terrible",
    "pesimo", "pesima", "horrible", "robo", "robaron", "maltrato", "cola enorme",
    "expensive", "unsafe", "dirty", "rude", "crowded", "overpriced", "overcrowded",
    "disappointing", "disappointment", "scam", "bad service", "poor service",
    "filthy", "ripoff", "rip off", "waste of money", "too expensive", "long queue",
])
LEX_ADVERS = _nset([
    "pero", "aunque", "sin embargo", "no obstante", "unfortunately", "but", "however",
])
LEX_NEU = _nset([
    "estuvo bien pero", "regular", "normal", "aceptable", "depende", "informativo",
    "no es malo pero", "ni bueno ni malo", "del monton", "promedio", "mas o menos",
    "okay", "ok", "average", "decent", "nothing special", "as expected", "so so",
    "it is ok", "its ok", "fairly", "mediocre",
])
LEX_POS = _nset([
    "hermoso", "hermosa", "increible", "excelente", "maravilloso", "maravillosa",
    "espectacular", "recomendado", "recomiendo", "encanto", "bonito", "lindo",
    "imperdible", "fascinante", "amazing", "beautiful", "excellent", "wonderful",
    "great", "lovely", "stunning", "recommended", "awesome", "fantastic", "gorgeous",
])


def match_terms(text_norm: str, terms: list) -> list:
    """Devuelve los terminos encontrados (multi-palabra: substring; 1 palabra: con frontera)."""
    found = []
    for t in terms:
        if " " in t:
            if t in text_norm:
                found.append(t)
        else:
            if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text_norm):
                found.append(t)
    return found


# ----------------------------------------------------------------------------
# 1) Carga
# ----------------------------------------------------------------------------
print("Cargando gold y corpus...")
g = pd.read_csv(GOLD / "gold_consolidado_largo.csv", encoding="utf-8-sig")
corpus = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig")
dic_raw = json.load(open(DATA / "diccionario_aspectos.json", encoding="utf-8"))
DIC = {k: v for k, v in dic_raw.items() if not str(k).startswith("_") and isinstance(v, list)}
PAT = build_aspect_patterns(DIC)
ASPECTS = list(PAT.keys())

gold_tup = g[g["label"].isin(LABELS)].copy()
gold_uids = set(g["review_uid"])

# ----------------------------------------------------------------------------
# 2) DIAGNOSTICO
# ----------------------------------------------------------------------------
# --- global ---
glob = gold_tup["label"].value_counts().reindex(LABELS, fill_value=0)
total = int(glob.sum())
pct = (glob / total * 100).round(2)
ratio_base = glob["negativo"] if glob["negativo"] else 1
ratio = {l: round(glob[l] / ratio_base, 2) for l in LABELS}

# --- por aspecto ---
asp_rows = []
ct_asp = gold_tup.groupby(["aspecto", "label"]).size().unstack("label", fill_value=0).reindex(columns=LABELS, fill_value=0)
for asp in sorted(ASPECTS):
    r = ct_asp.loc[asp] if asp in ct_asp.index else pd.Series({l: 0 for l in LABELS})
    npos, nneu, nneg = int(r["positivo"]), int(r["neutro"]), int(r["negativo"])
    tot = npos + nneu + nneg
    counts = {"negativo": nneg, "neutro": nneu, "positivo": npos}
    mincls = min(counts, key=counts.get)
    maxcls = max(counts, key=counts.get)
    asp_rows.append({
        "scope": "aspecto", "aspecto": asp,
        "n_negativo": nneg, "n_neutro": nneu, "n_positivo": npos, "total": tot,
        "pct_neg": round(nneg / tot * 100, 1) if tot else 0.0,
        "pct_neu": round(nneu / tot * 100, 1) if tot else 0.0,
        "pct_pos": round(npos / tot * 100, 1) if tot else 0.0,
        "clase_minoritaria": mincls, "soporte_min_clase": counts[mincls],
        "brecha_may_min": counts[maxcls] - counts[mincls],
        "def_neg": max(0, MIN_PER_ASPECT_CLASS - nneg),
        "def_neu": max(0, MIN_PER_ASPECT_CLASS - nneu),
        "def_pos": max(0, MIN_PER_ASPECT_CLASS - npos),
    })
asp_df = pd.DataFrame(asp_rows)

glob_row = {
    "scope": "global", "aspecto": "TODOS",
    "n_negativo": int(glob["negativo"]), "n_neutro": int(glob["neutro"]), "n_positivo": int(glob["positivo"]),
    "total": total, "pct_neg": pct["negativo"], "pct_neu": pct["neutro"], "pct_pos": pct["positivo"],
    "clase_minoritaria": glob.idxmin(), "soporte_min_clase": int(glob.min()),
    "brecha_may_min": int(glob.max() - glob.min()),
    "def_neg": "", "def_neu": "", "def_pos": "",
}
diag = pd.concat([pd.DataFrame([glob_row]), asp_df], ignore_index=True)
diag.to_csv(REP / "diagnostico_balance_polaridades_v4.csv", index=False, encoding="utf-8-sig")
print("  -> diagnostico CSV (global + por aspecto) guardado")

# --- por destino ---
dest_ct = gold_tup.groupby(["destination", "label"]).size().unstack("label", fill_value=0).reindex(columns=LABELS, fill_value=0)
dest_ct["total"] = dest_ct.sum(axis=1)
dest_ct = dest_ct.sort_values("negativo")

# --- por estrellas / idioma (sobre las resenas del gold) ---
gold_meta = corpus[corpus["review_uid"].isin(gold_uids)][["review_uid", "language_review"]]
gtup_meta = gold_tup.merge(gold_meta, on="review_uid", how="left")
stars_ct = gtup_meta.groupby(["stars", "label"]).size().unstack("label", fill_value=0).reindex(columns=LABELS, fill_value=0)
lang_ct = gtup_meta.groupby(["language_review", "label"]).size().unstack("label", fill_value=0).reindex(columns=LABELS, fill_value=0)

# ----------------------------------------------------------------------------
# 3) MINADO de candidatos sobre el corpus SIN anotar
# ----------------------------------------------------------------------------
print("Minando candidatos sobre el corpus sin anotar...")
cu = corpus[~corpus["review_uid"].isin(gold_uids)].copy()
cu["text_norm"] = cu["text_clean"].map(norm)

# deficits de neutro/positivo por aspecto (para priorizar)
neu_def = {a: max(0, MIN_PER_ASPECT_CLASS - int(ct_asp.loc[a, "neutro"]) if a in ct_asp.index else MIN_PER_ASPECT_CLASS) for a in ASPECTS}
pos_def = {a: max(0, MIN_PER_ASPECT_CLASS - int(ct_asp.loc[a, "positivo"]) if a in ct_asp.index else MIN_PER_ASPECT_CLASS) for a in ASPECTS}

cands = []  # formato largo: una fila por (resena x aspecto_sugerido)
for row in cu.itertuples(index=False):
    tn = row.text_norm
    if not tn:
        continue
    asps = detect_aspects(tn, PAT)
    if not asps:
        continue
    s = int(row.stars) if not pd.isna(row.stars) else 0
    neg_lex = match_terms(tn, LEX_NEG)
    adv = match_terms(tn, LEX_ADVERS)
    neu_lex = match_terms(tn, LEX_NEU)
    pos_lex = match_terms(tn, LEX_POS)
    mixed = bool(neg_lex and pos_lex)

    neg_score = (2.0 if s <= 2 and s >= 1 else 0.0) + 1.0 * len(neg_lex) + (0.5 if adv else 0.0)
    neu_score = (2.0 if s == 3 else 0.0) + 1.0 * len(neu_lex) + (1.0 if mixed else 0.0) + (0.3 if (adv and not neg_lex) else 0.0)
    pos_score = (1.0 if s >= 4 else 0.0) + 0.7 * len(pos_lex)

    scores = {"negativo": neg_score, "neutro": neu_score, "positivo": pos_score}
    # desempate: priorizar la clase mas escasa (neg > neu > pos)
    bucket = max(LABELS, key=lambda l: (scores[l], {"negativo": 2, "neutro": 1, "positivo": 0}[l]))
    win = scores[bucket]
    if win <= 0:
        continue
    # las senales positivas solo interesan si hay aspecto con deficit positivo
    if bucket == "positivo" and not any(pos_def[a] > 0 for a in asps):
        continue
    # negativos en resenas de 3-5 estrellas: solo si la senal es FUERTE (evita ruido tipo
    # "amazing place" con un 'but' suelto). Estrellas bajas (1-2) no requieren esto.
    strong_neg = (len(neg_lex) >= 1 and bool(adv)) or len(neg_lex) >= 2
    if bucket == "negativo" and s >= 3 and not strong_neg:
        continue
    lowstar = 1 if (s in (1, 2)) else 0

    # motivo legible
    parts = [f"estrellas={s}"]
    if bucket == "negativo":
        if neg_lex:
            parts.append("lex_neg:" + ",".join(neg_lex[:4]))
        if adv:
            parts.append("conector:" + ",".join(adv[:3]))
        if s <= 2 and s >= 1:
            parts.append("estrellas_bajas")
    elif bucket == "neutro":
        if s == 3:
            parts.append("estrellas_medias")
        if neu_lex:
            parts.append("lex_neu:" + ",".join(neu_lex[:4]))
        if mixed:
            parts.append("mixto(pos+neg)")
    else:
        if pos_lex:
            parts.append("lex_pos:" + ",".join(pos_lex[:4]))
    motivo = " | ".join(parts)
    score_minado = round(min(win / 5.0, 1.0), 3)

    for a in asps[:MAX_ASPECTS_PER_REVIEW]:
        # prioridad de anotacion
        if bucket == "negativo":
            prio = "alta"
        elif bucket == "neutro":
            prio = "alta" if neu_def.get(a, 0) > 0 else "media"
        else:
            prio = "media" if pos_def.get(a, 0) > 0 else "baja"
        cands.append({
            "review_uid": row.review_uid, "destination": row.destination, "source": row.source,
            "language_review": row.language_review, "stars": s, "date": row.publishedAtDate,
            "text_clean": row.text_clean, "polaridad_sugerida": bucket, "aspecto_sugerido": a,
            "motivo_minado": motivo, "score_minado": score_minado, "prioridad_anotacion": prio,
            "_lowstar": lowstar,
        })

cand_df = pd.DataFrame(cands)
print(f"  candidatos minados (largo): {len(cand_df)} | resenas distintas: {cand_df['review_uid'].nunique() if len(cand_df) else 0}")

# ----------------------------------------------------------------------------
# 4) SELECCION BALANCEADA (deficit por aspecto + empuje global, con topes)
# ----------------------------------------------------------------------------
print("Seleccionando set balanceado con topes anti-dominacion...")
PRIO_RANK = {"alta": 0, "media": 1, "baja": 2}
cand_df["_prio_rank"] = cand_df["prioridad_anotacion"].map(PRIO_RANK)
cand_df = cand_df.sort_values(["_prio_rank", "score_minado"], ascending=[True, False]).reset_index(drop=True)

global_target = {"negativo": NEG_SELECT_TARGET, "neutro": NEU_SELECT_TARGET, "positivo": POS_SELECT_TARGET}
deficit = {(a, l): max(0, MIN_PER_ASPECT_CLASS - int(ct_asp.loc[a, l]) if a in ct_asp.index else MIN_PER_ASPECT_CLASS)
           for a in ASPECTS for l in LABELS}
sel_global = defaultdict(int)
sel_dest = defaultdict(int)          # (clase, destino)
sel_review_aspects = defaultdict(int)  # review_uid -> nro aspectos ya seleccionados
sel_neg_lowstar = 0                   # negativos de 1-2 estrellas (para diversidad)
selected_idx = []

for i, c in cand_df.iterrows():
    cls, asp, dest, ruid = c.polaridad_sugerida, c.aspecto_sugerido, c.destination, c.review_uid
    if sel_global[cls] >= global_target[cls]:
        continue
    if sel_dest[(cls, dest)] >= CAP_PER_DEST[cls]:
        continue
    if sel_review_aspects[ruid] >= MAX_ASPECTS_PER_REVIEW:
        continue
    # diversidad de estrellas en negativos: no agotar el cupo con 1-2 estrellas
    if cls == "negativo" and c._lowstar == 1 and sel_neg_lowstar >= NEG_MAX_LOWSTAR:
        continue
    # criterio de aceptacion:
    #  - positivo: SOLO si el aspecto aun tiene deficit positivo
    #  - neg/neu: si hay deficit en el aspecto O aun no se cubre la meta global de la clase
    if cls == "positivo":
        if deficit[(asp, cls)] <= 0:
            continue
    else:
        if deficit[(asp, cls)] <= 0 and sel_global[cls] >= global_target[cls]:
            continue
    selected_idx.append(i)
    sel_global[cls] += 1
    sel_dest[(cls, dest)] += 1
    sel_review_aspects[ruid] += 1
    if cls == "negativo" and c._lowstar == 1:
        sel_neg_lowstar += 1
    if deficit[(asp, cls)] > 0:
        deficit[(asp, cls)] -= 1

sel = cand_df.loc[selected_idx].drop(columns=["_prio_rank", "_lowstar"]).reset_index(drop=True)

CAND_COLS = ["review_uid", "destination", "source", "language_review", "stars", "date",
             "text_clean", "polaridad_sugerida", "aspecto_sugerido", "motivo_minado",
             "score_minado", "prioridad_anotacion"]
sel[CAND_COLS].to_csv(GOLD_DATA / "candidatos_refuerzo_balanceado_v4.csv", index=False, encoding="utf-8-sig")
print(f"  -> candidatos seleccionados: {len(sel)} (neg {sel_global['negativo']} / neu {sel_global['neutro']} / pos {sel_global['positivo']})")

# ----------------------------------------------------------------------------
# 5) PLANTILLA multi-etiqueta (una fila por resena, aspectos = 'ausente')
# ----------------------------------------------------------------------------
uniq = sel.drop_duplicates("review_uid")
plant = corpus[corpus["review_uid"].isin(set(uniq["review_uid"]))][
    ["review_uid", "destination", "source", "language_review", "stars", "publishedAtDate", "text_clean"]
].copy()
plant["pool"] = POOL_NAME
for a in ASPECTS:
    plant[a] = "ausente"
PLANT_COLS = ["review_uid", "destination", "source", "language_review", "stars",
              "publishedAtDate", "text_clean", "pool"] + ASPECTS
plant[PLANT_COLS].to_csv(GOLD_DATA / "plantilla_refuerzo_balanceado_v4.csv", index=False, encoding="utf-8-sig")
print(f"  -> plantilla (resenas a anotar): {len(plant)}")

# ----------------------------------------------------------------------------
# 6) REPORTE MARKDOWN
# ----------------------------------------------------------------------------
def md_table(df, cols=None, idx_name=None):
    df = df.copy()
    if cols:
        df = df[cols]
    head = (([idx_name] if idx_name else []) + list(df.columns))
    out = ["| " + " | ".join(map(str, head)) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for ix, r in df.iterrows():
        vals = (([ix] if idx_name else []) + list(r.values))
        out.append("| " + " | ".join(map(lambda x: str(x), vals)) + " |")
    return "\n".join(out)


sel_by_asp = sel.groupby(["aspecto_sugerido", "polaridad_sugerida"]).size().unstack("polaridad_sugerida", fill_value=0).reindex(columns=LABELS, fill_value=0)
sel_by_dest = sel.groupby(["destination", "polaridad_sugerida"]).size().unstack("polaridad_sugerida", fill_value=0).reindex(columns=LABELS, fill_value=0)
sel_by_stars = sel.groupby(["stars", "polaridad_sugerida"]).size().unstack("polaridad_sugerida", fill_value=0).reindex(columns=LABELS, fill_value=0)

# proyeccion post-anotacion (orientativa: asume que la senal se confirma)
proj = glob.copy()
for l in LABELS:
    proj[l] = int(glob[l]) + int(sel_global[l])

# limitaciones: aspectos que NO alcanzan 30/clase ni con el refuerzo disponible
limit_rows = []
for a in sorted(ASPECTS):
    cur = {l: int(ct_asp.loc[a, l]) if a in ct_asp.index else 0 for l in LABELS}
    add = {l: int(sel_by_asp.loc[a, l]) if a in sel_by_asp.index else 0 for l in LABELS}
    for l in LABELS:
        proj_al = cur[l] + add[l]
        if proj_al < MIN_PER_ASPECT_CLASS:
            limit_rows.append(f"- `{a}` / **{l}**: {cur[l]} actuales + {add[l]} candidatos = {proj_al} (< {MIN_PER_ASPECT_CLASS}; limitado por el corpus)")

md = []
md.append("# Diagnostico de balance de polaridades + plan de refuerzo — Gold V4\n")
md.append("> `polaridad_sugerida` es **solo una guia de muestreo** para priorizar la anotacion; "
          "**no es una etiqueta final**. La anotacion humana decide la polaridad real.\n")

md.append("## 1. Distribucion global actual\n")
md.append(f"- Total de tuplas con polaridad: **{total}**")
md.append(f"- Positivo: **{int(glob['positivo'])}** ({pct['positivo']}%)")
md.append(f"- Neutro: **{int(glob['neutro'])}** ({pct['neutro']}%)")
md.append(f"- Negativo: **{int(glob['negativo'])}** ({pct['negativo']}%)")
md.append(f"- Ratio positivo:neutro:negativo = **{ratio['positivo']} : {ratio['neutro']} : {ratio['negativo']}** "
          f"(normalizado a negativo=1)")
md.append(f"- Clase minoritaria global: **{glob.idxmin()}** ({int(glob.min())}). El gold esta fuertemente "
          "sesgado a positivo, coherente con un corpus turistico de resenas (94% 4-5 estrellas).\n")

md.append("## 2. Distribucion por aspecto\n")
md.append(md_table(asp_df, ["aspecto", "n_negativo", "n_neutro", "n_positivo", "total",
                            "clase_minoritaria", "soporte_min_clase", "brecha_may_min",
                            "def_neg", "def_neu", "def_pos"]))
md.append("\n*`def_*` = cuantos faltan para llegar a 30 por clase en ese aspecto.*\n")

md.append("## 3. Distribucion por destino (ordenado por # negativos)\n")
md.append(md_table(dest_ct.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["destination", "neg", "neu", "pos", "total"]))
neg_pobres = dest_ct[dest_ct["negativo"] <= 5].index.tolist()
md.append(f"\n- Destinos con <=5 negativos en el gold: {', '.join('`'+d+'`' for d in neg_pobres) if neg_pobres else 'ninguno'}.")
md.append("- Los positivos se concentran en los destinos mas famosos (atractivos casi siempre positivos); "
          "negativos y neutros estan dispersos y son escasos en casi todos.\n")

md.append("## 4. Distribucion por estrellas e idioma\n")
md.append("**Por estrellas (tuplas del gold):**\n")
md.append(md_table(stars_ct.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["stars", "neg", "neu", "pos"]))
md.append("\n**Por idioma:**\n")
md.append(md_table(lang_ct.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["language_review", "neg", "neu", "pos"]))
md.append("\n**Lectura de la senal:**")
md.append("- Las **estrellas bajas (1-2) si aportan negativos**, pero son ESCASAS: el corpus completo tiene "
          f"solo ~320 resenas con 1-2 estrellas. En el pool sin anotar quedan ~201. Por eso el negativo "
          "**no puede salir solo de estrellas bajas**; se mina tambien con lexico negativo + conectores "
          "adversativos dentro de resenas de 4-5 estrellas ('hermoso PERO carisimo y lleno').")
md.append("- Las **estrellas medias (3) aportan neutros / mixtos**, tambien escasas (~489 totales).")
md.append("- **Fuente unica** (Google Maps) -> no hay sesgo por fuente que controlar. Idioma es/en esta "
          "razonablemente balanceado (~53/47), asi que el refuerzo mantiene ambos idiomas.\n")

md.append("## 5. Plan de refuerzo balanceado (candidatos seleccionados)\n")
md.append(f"- **Negativo (prioridad maxima):** {sel_global['negativo']} candidatos sugeridos.")
md.append(f"- **Neutro:** {sel_global['neutro']} candidatos (foco en aspectos con neutro escaso).")
md.append(f"- **Positivo:** {sel_global['positivo']} candidatos (SOLO en aspectos con bajo soporte; no se sobrecarga).")
md.append(f"- **Resenas unicas a anotar:** {len(plant)} (pool `{POOL_NAME}`).\n")
md.append("**Candidatos por aspecto y polaridad sugerida:**\n")
md.append(md_table(sel_by_asp.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["aspecto_sugerido", "neg", "neu", "pos"]))
md.append("\n**Candidatos por destino (cobertura, con tope anti-dominacion):**\n")
md.append(md_table(sel_by_dest.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["destination", "neg", "neu", "pos"]))
md.append("\n**Candidatos por estrellas (diversidad: no todo de 1 estrella):**\n")
md.append(md_table(sel_by_stars.reset_index().rename(columns={"negativo": "neg", "neutro": "neu", "positivo": "pos"}),
                   ["stars", "neg", "neu", "pos"]))

md.append("\n## 6. Proyeccion orientativa post-anotacion\n")
md.append("*(asume que la senal sugerida se confirma; la anotacion real puede reclasificar)*\n")
md.append(f"- Negativo: {int(glob['negativo'])} -> ~**{int(proj['negativo'])}**")
md.append(f"- Neutro: {int(glob['neutro'])} -> ~**{int(proj['neutro'])}**")
md.append(f"- Positivo: {int(glob['positivo'])} -> ~**{int(proj['positivo'])}**\n")

md.append("## 7. Limitaciones que persisten (corpus-limitadas)\n")
if limit_rows:
    md.extend(limit_rows)
else:
    md.append("- Todos los aspectos alcanzan el minimo de 30 por clase con el refuerzo disponible.")
md.append("\n- No se busca un balance perfecto artificial: se prioriza balance **util para entrenamiento**. "
          "Donde el corpus no tiene negativos/neutros reales suficientes para un aspecto, **no se inventan**; "
          "se declara como limitacion.\n")

md.append("## 8. Como usar este refuerzo\n")
md.append(f"1. Anotar `data/gold/plantilla_refuerzo_balanceado_v4.csv` (multi-etiqueta, pool `{POOL_NAME}`), "
          "usando `candidatos_refuerzo_balanceado_v4.csv` como guia de prioridad.")
md.append("2. Consolidar con el gold existente (mismo formato largo) y re-particionar sin fuga por `review_uid`.")
md.append("3. Reentrenar XLM-R con **5 semillas** + calibracion y re-evaluar contra la spec.\n")

(REP / "diagnostico_balance_polaridades_v4.md").write_text("\n".join(md), encoding="utf-8")
print("  -> reporte MD guardado")

# ----------------------------------------------------------------------------
# Resumen en consola
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"Gold actual: neg {int(glob['negativo'])} / neu {int(glob['neutro'])} / pos {int(glob['positivo'])} (total {total})")
print(f"Seleccionados: neg {sel_global['negativo']} / neu {sel_global['neutro']} / pos {sel_global['positivo']} "
      f"({len(sel)} filas, {len(plant)} resenas)")
print(f"Proyeccion: neg ~{int(proj['negativo'])} / neu ~{int(proj['neutro'])} / pos ~{int(proj['positivo'])}")
print("Salidas:")
print("  outputs/reports/diagnostico_balance_polaridades_v4.csv")
print("  outputs/reports/diagnostico_balance_polaridades_v4.md")
print("  data/gold/candidatos_refuerzo_balanceado_v4.csv")
print("  data/gold/plantilla_refuerzo_balanceado_v4.csv")
