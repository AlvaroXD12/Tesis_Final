# -*- coding: utf-8 -*-
"""
aplicar_reanotacion_blind_v4.py
===============================
Aplica la RE-ANOTACION CIEGA del lote balanceado (lote_negativos_anotador_1/2/3.csv,
formato candidato con columnas review_uid/aspecto_sugerido/label) SOBRE el gold:
reemplaza la etiqueta de los pares (review_uid, aspecto) ya presentes con el
CONSENSO por mayoria de la re-anotacion. Los anotadores etiquetaron a ciegas
(sin ver polaridad_sugerida/motivo_minado), por lo que esta anotacion supersede
la de la ronda 1 para esos pares.

Salidas:
  outputs/reports/reanotacion_blind_v4.md
  outputs/gold/gold_consolidado_largo.csv         (ACTUALIZADO; backup *_pre_reanotacion)
  data/{train,val,test}_gold_v4.csv               (re-particion; backup *_pre_reanotacion)
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
SEED = 42

# ---- re-anotacion: consenso por mayoria sobre el full overlap ----
L = [pd.read_csv(GOLD_DATA / f"lote_negativos_anotador_{n}.csv", encoding="utf-8-sig") for n in (1, 2, 3)]
for d in L:
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    d["key"] = d["review_uid"].astype(str) + "__" + d["aspecto_sugerido"].astype(str)
maps = [dict(zip(d["key"], d["label"])) for d in L]
common = sorted(set(maps[0]) & set(maps[1]) & set(maps[2]))

def fleiss(items):
    M = np.zeros((len(items), len(CATS)), int); ci = {c: i for i, c in enumerate(CATS)}
    for r, labs in enumerate(items):
        for l in labs: M[r, ci.get(l, 0)] += 1
    N, k = M.shape; n = M.sum(1)[0]; p = M.sum(0) / (N * n)
    P = (np.square(M).sum(1) - n) / (n * (n - 1)); Pe = np.square(p).sum()
    return (P.mean() - Pe) / (1 - Pe) if (1 - Pe) > 0 else 1.0
trips = [(maps[0][k], maps[1][k], maps[2][k]) for k in common]
kappa = fleiss(trips)
pat = Counter("unanime" if max(Counter(t).values()) == 3 else ("mayoria" if max(Counter(t).values()) == 2 else "sin_consenso") for t in trips)

cons, sin = {}, 0
for k in common:
    c = Counter([maps[0][k], maps[1][k], maps[2][k]]); lab, cnt = c.most_common(1)[0]
    if cnt >= 2:
        cons[k] = lab
    else:
        sin += 1

# ---- aplicar sobre el gold (reemplazo por (review_uid, aspecto)) ----
gp_path = GOLD / "gold_consolidado_largo.csv"
gold = pd.read_csv(gp_path, encoding="utf-8-sig")
bak = GOLD / "gold_consolidado_largo_pre_reanotacion.csv"
if not bak.exists(): shutil.copy2(gp_path, bak)
gold["label"] = gold["label"].astype(str).str.lower().str.strip()
gold["key"] = gold["review_uid"].astype(str) + "__" + gold["aspecto"].astype(str)

antes_dist = gold[gold.label.isin(POL)]["label"].value_counts().reindex(POL).to_dict()
n_cambiados = 0
new_labels = []
for k, l in zip(gold["key"], gold["label"]):
    if k in cons and cons[k] != l:
        new_labels.append(cons[k]); n_cambiados += 1
    else:
        new_labels.append(l)
gold["label"] = new_labels
no_match = [k for k in cons if k not in set(gold["key"])]  # deberia ser 0 (todos ya en gold)
gold.drop(columns=["key"]).to_csv(gp_path, index=False, encoding="utf-8-sig")
despues_dist = gold[gold.label.isin(POL)]["label"].value_counts().reindex(POL).to_dict()

# ---- re-particion sin fuga ----
tup = gold[gold["label"].isin(POL)].copy()
tup["input_modelo"] = "aspecto: " + tup["aspecto"].astype(str) + " resena: " + tup["text_clean"].astype(str)
g1 = GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
itv, ite = next(g1.split(tup, groups=tup["review_uid"])); tv, test = tup.iloc[itv], tup.iloc[ite]
g2 = GroupShuffleSplit(1, test_size=0.1765, random_state=SEED)
itr, iva = next(g2.split(tv, groups=tv["review_uid"])); train, val = tv.iloc[itr], tv.iloc[iva]
assert not ((set(train.review_uid) & set(test.review_uid)) | (set(train.review_uid) & set(val.review_uid)) | (set(val.review_uid) & set(test.review_uid))), "FUGA"
cols = ["review_uid", "aspecto", "destination", "text_clean", "label", "input_modelo"]
for nm, d in [("train", train), ("val", val), ("test", test)]:
    p, pb = DATA / f"{nm}_gold_v4.csv", DATA / f"{nm}_gold_v4_pre_reanotacion.csv"
    if p.exists() and not pb.exists(): shutil.copy2(p, pb)
    d[cols].to_csv(p, index=False, encoding="utf-8-sig")

md = [
    "# Re-anotacion ciega del lote balanceado aplicada al gold\n",
    f"- Solapamiento completo: {len(common)} pares (review x aspecto) | **Fleiss kappa = {kappa:.3f}** | patron {dict(pat)}",
    f"- Sin consenso (no aplicados): {sin}",
    "- Anotacion CIEGA (sin ver polaridad_sugerida/motivo_minado) -> supersede la ronda 1 para estos pares.",
    f"- Pares con etiqueta cambiada respecto al gold: **{n_cambiados}** de {len(cons)}",
    f"- Distribucion (polaridad) ANTES: {antes_dist}",
    f"- Distribucion (polaridad) DESPUES: {despues_dist}",
    f"- Re-particion sin fuga: train {len(train)} / val {len(val)} / test {len(test)} | "
    f"neg {int((train.label=='negativo').sum())}/{int((val.label=='negativo').sum())}/{int((test.label=='negativo').sum())}",
]
(REP / "reanotacion_blind_v4.md").write_text("\n".join(md), encoding="utf-8")

print(f"Fleiss kappa: {kappa:.3f} | patron {dict(pat)} | sin consenso {sin}")
print(f"Pares matcheados en gold: {len(cons)-len(no_match)}/{len(cons)} (no_match={len(no_match)})")
print(f"Etiquetas cambiadas: {n_cambiados}")
print(f"ANTES:   {antes_dist}")
print(f"DESPUES: {despues_dist}")
print(f"Splits: train {len(train)} / val {len(val)} / test {len(test)} | "
      f"neg {int((train.label=='negativo').sum())}/{int((val.label=='negativo').sum())}/{int((test.label=='negativo').sum())}")
