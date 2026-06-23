# -*- coding: utf-8 -*-
"""
consolidar_negativos_dirigido_v4.py
===================================
Consolida la anotacion dirigida de negativos (formato LARGO: lote_negativos_anotador_1/2/3.csv,
una fila por review x aspecto con `label`). Diseno de solapamiento PARCIAL:
  - items anotados por los 3  -> consenso por mayoria (>=2) + Fleiss kappa.
  - items anotados por 1       -> se acepta la anotacion individual (kappa del solapamiento alto).
Trae `stars` del corpus, deduplica contra el gold por (review_uid, aspecto), fusiona y
re-particiona sin fuga.

Salidas:
  outputs/reports/acuerdo_negativos_dirigido_v4.md
  outputs/gold/gold_consolidado_largo.csv         (ACTUALIZADO; backup *_pre_negdirigido)
  data/{train,val,test}_gold_v4.csv               (re-particion; backup *_pre_negdirigido)
"""
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent.parent
DATA, GOLD, REP = BASE / "data", BASE / "outputs" / "gold", BASE / "outputs" / "reports"
GOLD_DATA = DATA / "gold"
POL = ["negativo", "neutro", "positivo"]
CATS = ["ausente"] + POL
ASP_V4 = {"atractivos", "costos", "seguridad", "accesibilidad", "limpieza",
          "atencion_servicio", "gastronomia", "alojamiento"}
POOL = "C_negativos_dirigido_v4"
SEED = 42

# ---- carga ----
L = [pd.read_csv(GOLD_DATA / f"lote_negativos_anotador_{n}.csv", encoding="utf-8-sig") for n in (1, 2, 3)]
for d in L:
    d["label"] = d["label"].astype(str).str.lower().str.strip()
maps = [dict(zip(d["annotation_id"], d["label"])) for d in L]
meta = {}
for d in L:
    for _, r in d.iterrows():
        meta.setdefault(r["annotation_id"], r)

# ---- Fleiss kappa en el solapamiento (3 anotadores) ----
common = set(maps[0]) & set(maps[1]) & set(maps[2])
def fleiss(items):
    M = np.zeros((len(items), len(CATS)), int); ci = {c: i for i, c in enumerate(CATS)}
    for r, labs in enumerate(items):
        for l in labs: M[r, ci.get(l, 0)] += 1
    N, k = M.shape; n = M.sum(1)[0]; p = M.sum(0) / (N * n)
    P = (np.square(M).sum(1) - n) / (n * (n - 1)); Pe = np.square(p).sum()
    return (P.mean() - Pe) / (1 - Pe) if (1 - Pe) > 0 else 1.0
trips = [(maps[0][i], maps[1][i], maps[2][i]) for i in common]
kappa = fleiss(trips) if trips else float("nan")
pat = Counter("unanime" if max(Counter(t).values()) == 3 else ("mayoria" if max(Counter(t).values()) == 2 else "sin_consenso") for t in trips)

# ---- consenso ----
final, sin = [], 0
for aid in set().union(*maps):
    labs = [m[aid] for m in maps if aid in m]
    if len(labs) >= 2:
        c = Counter(labs); lab, cnt = c.most_common(1)[0]
        if cnt < 2:
            sin += 1; continue
    else:
        lab = labs[0]
    r = meta[aid]
    final.append({"review_uid": r["review_uid"], "destination": r["destination"],
                  "aspecto": r["aspecto"], "text_clean": r["text_clean"], "label": lab})
nuevo = pd.DataFrame(final)
nuevo = nuevo[nuevo["aspecto"].isin(ASP_V4)]                 # solo taxonomia V4
nuevo_pol = nuevo[nuevo["label"].isin(POL)].copy()

# ---- stars desde el corpus ----
corpus = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig")[["review_uid", "stars"]]
nuevo_pol = nuevo_pol.merge(corpus, on="review_uid", how="left")
nuevo_pol["pool"] = POOL

# ---- gold actual + backup ----
gp_path = GOLD / "gold_consolidado_largo.csv"
gold = pd.read_csv(gp_path, encoding="utf-8-sig")
bak = GOLD / "gold_consolidado_largo_pre_negdirigido.csv"
if not bak.exists(): shutil.copy2(gp_path, bak)

# ---- dedup por (review_uid, aspecto): no re-anotar pares ya presentes en el gold ----
ya = set(zip(gold["review_uid"], gold["aspecto"]))
antes = len(nuevo_pol)
nuevo_pol = nuevo_pol[~nuevo_pol.apply(lambda r: (r["review_uid"], r["aspecto"]) in ya, axis=1)]
dups = antes - len(nuevo_pol)

add = nuevo_pol[gold.columns.intersection(nuevo_pol.columns).tolist()].copy()
for c in gold.columns:
    if c not in add.columns: add[c] = np.nan
add = add[gold.columns]
gold_ref = pd.concat([gold, add], ignore_index=True)
gold_ref.to_csv(gp_path, encoding="utf-8-sig", index=False)

# ---- re-particion sin fuga ----
tup = gold_ref[gold_ref["label"].isin(POL)].copy()
tup["input_modelo"] = "aspecto: " + tup["aspecto"].astype(str) + " resena: " + tup["text_clean"].astype(str)
g1 = GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
itv, ite = next(g1.split(tup, groups=tup["review_uid"])); tv, test = tup.iloc[itv], tup.iloc[ite]
g2 = GroupShuffleSplit(1, test_size=0.1765, random_state=SEED)
itr, iva = next(g2.split(tv, groups=tv["review_uid"])); train, val = tv.iloc[itr], tv.iloc[iva]
inter = (set(train.review_uid) & set(test.review_uid)) | (set(train.review_uid) & set(val.review_uid)) | (set(val.review_uid) & set(test.review_uid))
assert not inter, "FUGA"
cols = ["review_uid", "aspecto", "destination", "text_clean", "label", "input_modelo"]
for nm, d in [("train", train), ("val", val), ("test", test)]:
    p, pb = DATA / f"{nm}_gold_v4.csv", DATA / f"{nm}_gold_v4_pre_negdirigido.csv"
    if p.exists() and not pb.exists(): shutil.copy2(p, pb)
    d[cols].to_csv(p, index=False, encoding="utf-8-sig")

# ---- resumen ----
gpf = gold_ref[gold_ref["label"].isin(POL)]
def dist(d): return {k: int((d == k).sum()) for k in POL}
def interp(k): return "casi perfecto" if k >= 0.8 else ("sustancial" if k >= 0.6 else "moderado")
md = []
md.append("# Consolidacion — negativos dirigidos V4\n")
md.append(f"- Solapamiento (3 anotadores): {len(common)} items | **Fleiss kappa = {kappa:.3f}** ({interp(kappa)})")
md.append(f"  - patron: {dict(pat)}")
md.append(f"  - NOTA: kappa = {kappa:.3f}. Un valor perfecto (1.0) es inusual en anotacion subjetiva; "
          "reportar con honestidad (idealmente con desacuerdos reales del solapamiento).")
md.append(f"- Items anotados por 1 anotador (aceptados): {len(set().union(*maps)) - len(common)}")
md.append(f"- Tuplas nuevas con polaridad (taxonomia V4, tras dedup -{dups}): **{len(add[add.label.isin(POL)])}** -> {dist(add['label'])}")
md.append(f"- Gold reforzado total: **{len(gpf)}** tuplas -> {dist(gpf['label'])}")
md.append(f"- Re-particion sin fuga: train {len(train)} / val {len(val)} / test {len(test)}")
md.append(f"  - negativos por split: train {int((train.label=='negativo').sum())} / "
          f"val {int((val.label=='negativo').sum())} / test {int((test.label=='negativo').sum())}")
(REP / "acuerdo_negativos_dirigido_v4.md").write_text("\n".join(md), encoding="utf-8")

print(f"Fleiss kappa (solapamiento {len(common)}): {kappa:.3f} | patron {dict(pat)}")
print(f"Tuplas nuevas (V4, tras dedup -{dups}): {len(add[add.label.isin(POL)])} -> {dist(add['label'])}")
print(f"GOLD reforzado: {len(gpf)} -> {dist(gpf['label'])}")
print(f"Splits: train {len(train)} / val {len(val)} / test {len(test)} | "
      f"neg: {int((train.label=='negativo').sum())}/{int((val.label=='negativo').sum())}/{int((test.label=='negativo').sum())}")
print("Por aspecto (gold reforzado):")
print(gpf.groupby(["aspecto", "label"]).size().unstack("label", fill_value=0).reindex(columns=POL, fill_value=0).to_string())
