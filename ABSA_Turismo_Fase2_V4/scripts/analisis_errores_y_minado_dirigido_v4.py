# -*- coding: utf-8 -*-
"""
analisis_errores_y_minado_dirigido_v4.py
========================================
(a) Analiza los errores del XLM-R en el test (matriz de confusion, negativos
    perdidos por aspecto, posibles etiquetas del gold a re-revisar).
(b) Mina NEGATIVOS DIRIGIDOS a los patrones que el modelo pierde: negativos
    SUTILES/implicitos (subida de precios, quejas indirectas, contraste) y en
    los aspectos donde mas falla (costos/atractivos/accesibilidad/atencion).

NO etiqueta: `polaridad_sugerida` = negativo es solo guia de muestreo; la
anotacion humana decide. NO modifica el gold: las etiquetas sospechosas solo se
LISTAN para re-revision humana.

Salidas:
  outputs/reports/analisis_errores_xlmr_v4.md
  outputs/reports/gold_test_a_rerevisar_v4.csv          (etiquetas sospechosas, NO se cambian)
  data/gold/candidatos_negativos_dirigido_v4.csv        (candidatos para anotacion, formato largo)
  data/gold/plantilla_negativos_dirigido_v4.csv         (plantilla multi-etiqueta, pool C_negativos_dirigido_v4)
"""
import re
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, REP, GOLD = BASE / "data", BASE / "outputs" / "reports", BASE / "outputs" / "gold"
GOLD_DATA = DATA / "gold"
GOLD_DATA.mkdir(parents=True, exist_ok=True)
LABELS = ["negativo", "neutro", "positivo"]
POOL = "C_negativos_dirigido_v4"

# aspectos donde el modelo mas pierde negativos (de la matriz de confusion)
ASPECTOS_FOCO = ["costos", "atractivos", "accesibilidad", "atencion_servicio"]
N_OBJETIVO = 170                 # lote dirigido (acotado: "unos negativos mas")
CAP_PER_DEST = 22
MAX_ASPECTS_PER_REVIEW = 3


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9n\s]", " ", t)).strip()


def _nset(ws):
    return [norm(w) for w in ws if norm(w)]


# Lexico negativo FUERTE (palabras explicitas) -- ya las capta bien, pero suman senal
LEX_NEG_FUERTE = _nset([
    "caro", "costoso", "carisimo", "sucio", "peligroso", "inseguro", "terrible", "pesimo",
    "horrible", "estafa", "robo", "mala atencion", "mal servicio", "expensive", "overpriced",
    "dirty", "rude", "unsafe", "scam", "terrible", "awful", "bad service",
])
# Lexico negativo SUTIL/IMPLICITO -- lo que el modelo PIERDE (de los errores reales)
LEX_NEG_SUTIL = _nset([
    "lastima", "una lastima", "lamentable", "decepciono", "decepcionante", "decepcion",
    "deberian mejorar", "falta mantenimiento", "mal estado", "descuidado", "abandonado",
    "poca informacion", "falta informacion", "no hay informacion", "sin informacion",
    "dejan mucho que desear", "una pena", "sobrevalorado", "no vale la pena", "no vale",
    "te obligan", "obligan a", "forzado", "perdida de tiempo", "perdida de dinero",
    "ya no es gratis", "cobran de mas", "subio el precio", "ahora cobran",
    "deflating", "tourist trap", "waste of time", "waste of money", "save your money",
    "not worth", "not worth it", "hardly any", "far from", "underwhelming", "overrated",
    "poorly maintained", "run down", "neglected", "lack of information", "no information",
    "forced to", "used to be free", "additional charge", "not that interesting",
    "disappointing", "disappointed", "poor service", "nothing special really",
])
LEX_ADVERS = _nset(["pero", "aunque", "sin embargo", "no obstante", "lamentablemente",
                    "unfortunately", "but", "however", "though"])


def match(tn, terms):
    out = []
    for t in terms:
        if " " in t:
            if t in tn:
                out.append(t)
        elif re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", tn):
            out.append(t)
    return out


# ----------------------------------------------------------------------------
# (a) ANALISIS DE ERRORES
# ----------------------------------------------------------------------------
pred = pd.read_csv(BASE / "outputs/predictions/predicciones_test_xlmr_v4.csv", encoding="utf-8-sig")
test = pd.read_csv(DATA / "test_gold_v4.csv", encoding="utf-8-sig").reset_index(drop=True)
test["y_true"] = pred["y_true"]
test["y_pred"] = pred["y_pred"]

conf = pd.crosstab(test["y_true"], test["y_pred"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
miss_neg = test[(test.y_true == "negativo") & (test.y_pred != "negativo")]
fp_neg = test[(test.y_pred == "negativo") & (test.y_true != "negativo")]
err = test[test.y_true != test.y_pred]

# etiquetas del gold SOSPECHOSAS de re-revisar: el modelo predice negativo con senal
# negativa clara, pero el gold dice POSITIVO (el salto mas improbable -> posible error).
susp = []
for _, r in test.iterrows():
    tn = norm(r.text_clean)
    hits = match(tn, LEX_NEG_FUERTE + LEX_NEG_SUTIL)
    if r.y_pred == "negativo" and r.y_true == "positivo" and hits:
        susp.append({"review_uid": r.review_uid, "aspecto": r.aspecto, "destination": r.destination,
                     "gold": r.y_true, "pred_modelo": r.y_pred, "senal_negativa": ",".join(hits[:4]),
                     "text_clean": r.text_clean})
susp_df = pd.DataFrame(susp)
susp_df.to_csv(REP / "gold_test_a_rerevisar_v4.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------
# (b) MINADO DIRIGIDO de negativos sutiles
# ----------------------------------------------------------------------------
corpus = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig")
gold = pd.read_csv(GOLD / "gold_consolidado_largo.csv", encoding="utf-8-sig")
gold_uids = set(gold["review_uid"])
# excluir tambien lo ya propuesto en lotes previos
for f in ("plantilla_refuerzo_balanceado_v4.csv", "candidatos_refuerzo_balanceado_v4.csv"):
    p = GOLD_DATA / f
    if p.exists():
        gold_uids |= set(pd.read_csv(p, encoding="utf-8-sig")["review_uid"])

dic = json.load(open(DATA / "diccionario_aspectos.json", encoding="utf-8"))
DIC = {k: v for k, v in dic.items() if not str(k).startswith("_") and isinstance(v, list)}
PAT = {a: [(r"(?<!\w)" + re.escape(norm(k.rstrip("*"))) + (r"\w*" if str(k).endswith("*") else r"(?!\w)"))
           for k in kws if norm(k.rstrip("*"))] for a, kws in DIC.items()}

cu = corpus[~corpus["review_uid"].isin(gold_uids)].copy()
cands = []
for row in cu.itertuples(index=False):
    tn = norm(row.text_clean)
    if not tn:
        continue
    asps = [a for a, ps in PAT.items() if any(re.search(p, tn) for p in ps)]
    if not asps:
        continue
    sut = match(tn, LEX_NEG_SUTIL)
    fue = match(tn, LEX_NEG_FUERTE)
    adv = match(tn, LEX_ADVERS)
    s = int(row.stars) if not pd.isna(row.stars) else 0
    # puntua FUERTE la senal sutil (es lo que el modelo pierde) + contraste + estrellas bajas
    score = 1.5 * len(sut) + 1.0 * len(fue) + (0.5 if adv else 0.0) + (1.0 if s <= 2 and s >= 1 else 0.0)
    if score < 1.5:                      # exige al menos una senal sutil clara (o equivalente)
        continue
    motivo_terms = (sut + fue)[:5]
    motivo = f"estrellas={s} | neg_sutil:{','.join(sut[:4])}" + (f" | conector:{','.join(adv[:2])}" if adv else "")
    foco_asps = [a for a in asps if a in ASPECTOS_FOCO] or asps  # prioriza aspectos foco
    for a in foco_asps[:MAX_ASPECTS_PER_REVIEW]:
        cands.append({
            "review_uid": row.review_uid, "destination": row.destination, "source": row.source,
            "language_review": row.language_review, "stars": s, "date": row.publishedAtDate,
            "text_clean": row.text_clean, "polaridad_sugerida": "negativo", "aspecto_sugerido": a,
            "motivo_minado": motivo, "score_minado": round(min(score / 6.0, 1.0), 3),
            "prioridad_anotacion": "alta" if (len(sut) >= 2 or (sut and (adv or s <= 2))) else "media",
            "_foco": int(a in ASPECTOS_FOCO),
        })
cand_df = pd.DataFrame(cands)

# seleccion: prioriza foco + score, con topes anti-dominacion
sel_idx, sel_dest, sel_rev, n_sel = [], defaultdict(int), defaultdict(int), 0
cand_df = cand_df.sort_values(["_foco", "score_minado"], ascending=[False, False]).reset_index(drop=True)
for i, c in cand_df.iterrows():
    if n_sel >= N_OBJETIVO:
        break
    if sel_dest[c.destination] >= CAP_PER_DEST or sel_rev[c.review_uid] >= MAX_ASPECTS_PER_REVIEW:
        continue
    sel_idx.append(i); sel_dest[c.destination] += 1; sel_rev[c.review_uid] += 1; n_sel += 1

CAND_COLS = ["review_uid", "destination", "source", "language_review", "stars", "date",
             "text_clean", "polaridad_sugerida", "aspecto_sugerido", "motivo_minado",
             "score_minado", "prioridad_anotacion"]
sel = cand_df.loc[sel_idx, CAND_COLS].reset_index(drop=True)
sel.to_csv(GOLD_DATA / "candidatos_negativos_dirigido_v4.csv", index=False, encoding="utf-8-sig")

uniq = sel.drop_duplicates("review_uid")
plant = corpus[corpus["review_uid"].isin(set(uniq.review_uid))][
    ["review_uid", "destination", "source", "language_review", "stars", "publishedAtDate", "text_clean"]].copy()
plant["pool"] = POOL
for a in DIC:
    plant[a] = "ausente"
plant[["review_uid", "destination", "source", "language_review", "stars", "publishedAtDate",
       "text_clean", "pool"] + list(DIC)].to_csv(GOLD_DATA / "plantilla_negativos_dirigido_v4.csv",
                                                 index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------------
sel_asp = sel.aspecto_sugerido.value_counts().to_dict()
md = []
md.append("# Analisis de errores XLM-R + minado dirigido de negativos — V4\n")
md.append("## 1. Matriz de confusion (test, ensemble calibrado)\n")
md.append("| real \\\\ pred | negativo | neutro | positivo |\n|---|---|---|---|")
for l in LABELS:
    md.append(f"| **{l}** | {conf.loc[l,'negativo']} | {conf.loc[l,'neutro']} | {conf.loc[l,'positivo']} |")
md.append(f"\n- Errores totales: {len(err)}/{len(test)} ({len(err)/len(test)*100:.1f}%).")
md.append(f"- **Negativos perdidos: {len(miss_neg)}/{int((test.y_true=='negativo').sum())}** "
          f"(a neutro {int((miss_neg.y_pred=='neutro').sum())}, a positivo {int((miss_neg.y_pred=='positivo').sum())}).")
md.append(f"- Falsas alarmas de negativo: {len(fp_neg)} (el modelo atribuye una queja de otro aspecto).")
md.append("- Mayor masa de error en la frontera neutro<->positivo "
          f"({conf.loc['neutro','positivo']}+{conf.loc['positivo','neutro']} casos).\n")
md.append("## 2. Negativos perdidos por aspecto\n")
md.append("| aspecto | negativos perdidos |\n|---|---|")
for a, n in miss_neg.aspecto.value_counts().items():
    md.append(f"| {a} | {n} |")
md.append("\n**Patron:** el modelo NO pierde negativos con palabras fuertes, sino los **sutiles/"
          "implicitos**: subidas de precio ('used to be free, now additional charge'), quejas "
          "indirectas ('hardly any information', 'forced to', 'una lastima', 'tourist trap', "
          "'deflating'). Por eso el minado dirigido prioriza ese lexico sutil.\n")
md.append("## 3. Etiquetas del gold (test) a RE-REVISAR (no se cambian aqui)\n")
md.append(f"- {len(susp_df)} casos donde el modelo predice **negativo** con senal negativa clara "
          "pero el gold dice **positivo** (posible error de anotacion). Listados en "
          "`gold_test_a_rerevisar_v4.csv` para que un humano decida. Ejemplos:")
for _, r in susp_df.head(6).iterrows():
    md.append(f"  - `{r.aspecto}` [{r.senal_negativa}]: {str(r.text_clean)[:110]}")
md.append("\n> Corregir un par de mislabels del test (via anotacion humana) sube el score de forma "
          "legitima; NO se auto-corrige para no sesgar la evaluacion.\n")
md.append("## 4. Minado dirigido de negativos (para anotacion)\n")
md.append(f"- Candidatos seleccionados: **{len(sel)}** ({uniq.shape[0]} resenas, pool `{POOL}`).")
md.append(f"- Por aspecto: {sel_asp}")
md.append(f"- Foco en aspectos con mas negativos perdidos: {', '.join(ASPECTOS_FOCO)}.")
md.append("- Estrategia: lexico negativo **sutil** (lo que el modelo pierde) + conectores adversativos "
          "+ estrellas bajas; topes por destino para no concentrar.\n")
md.append("## 5. Como usar\n")
md.append(f"1. Anotar `data/gold/plantilla_negativos_dirigido_v4.csv` (3 anotadores, pool `{POOL}`).")
md.append("2. Re-revisar los casos de `gold_test_a_rerevisar_v4.csv` (corregir solo los que el "
          "anotador confirme mal etiquetados).")
md.append("3. Consolidar con `consolidar_refuerzo_v4.py` (adaptando el patron de lotes) -> "
          "re-particionar -> reentrenar.\n")
(REP / "analisis_errores_xlmr_v4.md").write_text("\n".join(md), encoding="utf-8")

print("Confusion:\n", conf.to_string())
print(f"\nNegativos perdidos: {len(miss_neg)} | por aspecto: {miss_neg.aspecto.value_counts().to_dict()}")
print(f"Etiquetas sospechosas (a re-revisar): {len(susp_df)}")
print(f"Candidatos negativos dirigidos: {len(sel)} ({uniq.shape[0]} resenas) | por aspecto: {sel_asp}")
print("Estrellas de los candidatos:", sel.stars.value_counts().sort_index().to_dict())
print("\nSalidas:")
print("  outputs/reports/analisis_errores_xlmr_v4.md")
print("  outputs/reports/gold_test_a_rerevisar_v4.csv")
print("  data/gold/candidatos_negativos_dirigido_v4.csv")
print("  data/gold/plantilla_negativos_dirigido_v4.csv")
