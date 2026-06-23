# -*- coding: utf-8 -*-
"""
consolidar_refuerzo_v4.py
=========================
Consolida la triple anotacion del refuerzo balanceado (lote_anotador_1/2/3) en
etiquetas de consenso, calcula la concordancia entre anotadores (Fleiss kappa
global y por aspecto + patron de desacuerdos), fusiona con el gold existente y
re-particiona sin fuga por review_uid.

NO usa heuristicas: la etiqueta final es el CONSENSO HUMANO (mayoria >=2 de 3).
Las celdas sin mayoria (los 3 distintos) se excluyen y se reportan.

Salidas:
  outputs/reports/acuerdo_anotadores_refuerzo_v4.csv   (kappa global + por aspecto)
  outputs/reports/acuerdo_anotadores_refuerzo_v4.md    (narrativa + patron de acuerdo)
  outputs/gold/gold_consolidado_largo.csv              (ACTUALIZADO; backup del original)
  data/{train,val,test}_gold_v4.csv                    (re-particion; backup de los originales)
"""
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GOLD = BASE / "outputs" / "gold"
REP = BASE / "outputs" / "reports"
REP.mkdir(parents=True, exist_ok=True)

ASPECTS = ["atractivos", "costos", "seguridad", "accesibilidad", "limpieza",
           "atencion_servicio", "gastronomia", "alojamiento"]
CATS = ["ausente", "negativo", "neutro", "positivo"]
POL = ["negativo", "neutro", "positivo"]
SEED = 42

# ----------------------------------------------------------------------------
# 1) Carga de los 3 lotes y validacion de alineacion
# ----------------------------------------------------------------------------
print("Cargando lotes de anotacion...")
lotes = {n: pd.read_csv(DATA / "gold" / f"lote_anotador_{n}.csv", encoding="utf-8-sig") for n in (1, 2, 3)}
for n, d in lotes.items():
    for a in ASPECTS:
        d[a] = d[a].astype(str).str.lower().str.strip()
    lotes[n] = d.set_index("review_uid")

uids = sorted(set(lotes[1].index) & set(lotes[2].index) & set(lotes[3].index))
union = set(lotes[1].index) | set(lotes[2].index) | set(lotes[3].index)
assert len(uids) == len(union), f"Los lotes no cubren las mismas resenas: comun={len(uids)} union={len(union)}"
print(f"  Resenas con triple anotacion: {len(uids)}")

meta = lotes[1].loc[uids, ["destination", "stars", "text_clean"]].copy()

# ----------------------------------------------------------------------------
# 2) Fleiss kappa (global + por aspecto) sobre las celdas review x aspecto
# ----------------------------------------------------------------------------
def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: matriz (N items x k categorias) con el # de anotadores por categoria."""
    N, k = counts.shape
    n = counts.sum(axis=1)[0]                       # anotadores por item (=3)
    p = counts.sum(axis=0) / (N * n)                # proporcion por categoria
    P = (np.square(counts).sum(axis=1) - n) / (n * (n - 1))
    Pbar = P.mean()
    Pe = np.square(p).sum()
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) > 0 else 1.0


def counts_matrix(items):
    """items: lista de tuplas (l1,l2,l3) -> matriz de conteos por categoria."""
    M = np.zeros((len(items), len(CATS)), dtype=int)
    cidx = {c: i for i, c in enumerate(CATS)}
    for r, labs in enumerate(items):
        for l in labs:
            M[r, cidx.get(l, cidx["ausente"])] += 1
    return M


# items globales y por aspecto
items_global, items_asp = [], {a: [] for a in ASPECTS}
for u in uids:
    for a in ASPECTS:
        trip = (lotes[1].at[u, a], lotes[2].at[u, a], lotes[3].at[u, a])
        items_global.append(trip)
        items_asp[a].append(trip)

kappa_global = fleiss_kappa(counts_matrix(items_global))
kappa_asp = {a: fleiss_kappa(counts_matrix(items_asp[a])) for a in ASPECTS}

# patron de acuerdo: unanime (3/3), mayoria (2/3), sin consenso (1/1/1)
pat = Counter()
for trip in items_global:
    c = Counter(trip)
    top = max(c.values())
    pat["unanime_3" if top == 3 else ("mayoria_2" if top == 2 else "sin_consenso")] += 1

print(f"  Fleiss kappa global: {kappa_global:.3f}")
print(f"  Patron: {dict(pat)}")

# ----------------------------------------------------------------------------
# 3) Consenso por mayoria (>=2). Sin mayoria -> excluido.
# ----------------------------------------------------------------------------
def consenso(trip):
    c = Counter(trip)
    lab, cnt = c.most_common(1)[0]
    return lab if cnt >= 2 else None  # None = sin consenso (3 distintos)


long_rows, sin_consenso = [], 0
for u in uids:
    for a in ASPECTS:
        trip = (lotes[1].at[u, a], lotes[2].at[u, a], lotes[3].at[u, a])
        lab = consenso(trip)
        if lab is None:
            sin_consenso += 1
            continue
        long_rows.append({
            "review_uid": u, "destination": meta.at[u, "destination"],
            "pool": "C_refuerzo_balanceado_v4", "stars": meta.at[u, "stars"],
            "text_clean": meta.at[u, "text_clean"], "aspecto": a, "label": lab,
        })
nuevo = pd.DataFrame(long_rows)
nuevo_pol = nuevo[nuevo["label"].isin(POL)]
print(f"  Celdas sin consenso (excluidas): {sin_consenso}")
print(f"  Tuplas nuevas con polaridad: {len(nuevo_pol)} "
      f"(neg {int((nuevo_pol.label=='negativo').sum())} / "
      f"neu {int((nuevo_pol.label=='neutro').sum())} / "
      f"pos {int((nuevo_pol.label=='positivo').sum())})")

# ----------------------------------------------------------------------------
# 4) Fusion con el gold existente (sin duplicar review_uid) + backup
# ----------------------------------------------------------------------------
gold_path = GOLD / "gold_consolidado_largo.csv"
gold = pd.read_csv(gold_path, encoding="utf-8-sig")
bak = GOLD / "gold_consolidado_largo_pre_refuerzo.csv"
if not bak.exists():
    shutil.copy2(gold_path, bak)
    print(f"  Backup del gold original -> {bak.name}")

overlap = set(gold["review_uid"]) & set(nuevo["review_uid"])
assert not overlap, f"Las resenas de refuerzo YA estaban en el gold ({len(overlap)}); revisar."
nuevo = nuevo[gold.columns]  # mismas columnas y orden
gold_ref = pd.concat([gold, nuevo], ignore_index=True)
gold_ref.to_csv(gold_path, encoding="utf-8-sig", index=False)

gp = gold_ref[gold_ref["label"].isin(POL)]
print(f"\n  Gold reforzado: {len(gold_ref)} filas | tuplas polaridad {len(gp)} "
      f"(neg {int((gp.label=='negativo').sum())} / neu {int((gp.label=='neutro').sum())} / pos {int((gp.label=='positivo').sum())})")

# ----------------------------------------------------------------------------
# 5) Re-particion sin fuga por review_uid (70/15/15), igual que el NB03
# ----------------------------------------------------------------------------
tup = gold_ref[gold_ref["label"].isin(POL)].copy()
tup["input_modelo"] = "aspecto: " + tup["aspecto"].astype(str) + " resena: " + tup["text_clean"].astype(str)
g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
itv, ite = next(g1.split(tup, groups=tup["review_uid"]))
tv, test = tup.iloc[itv].copy(), tup.iloc[ite].copy()
g2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)
itr, iva = next(g2.split(tv, groups=tv["review_uid"]))
train, val = tv.iloc[itr].copy(), tv.iloc[iva].copy()

inter = (set(train.review_uid) & set(test.review_uid)) | (set(train.review_uid) & set(val.review_uid)) | (set(val.review_uid) & set(test.review_uid))
assert len(inter) == 0, "FUGA entre splits"
cols = ["review_uid", "aspecto", "destination", "text_clean", "label", "input_modelo"]
for nm, d in [("train", train), ("val", val), ("test", test)]:
    p = DATA / f"{nm}_gold_v4.csv"
    pbak = DATA / f"{nm}_gold_v4_pre_refuerzo.csv"
    if p.exists() and not pbak.exists():
        shutil.copy2(p, pbak)
    d[cols].to_csv(p, index=False, encoding="utf-8-sig")
print(f"  Re-particion sin fuga -> train {len(train)} | val {len(val)} | test {len(test)}")
print("  (backup de los splits anteriores: *_gold_v4_pre_refuerzo.csv)")

# ----------------------------------------------------------------------------
# 6) Reportes de concordancia
# ----------------------------------------------------------------------------
def interp(k):
    if k < 0.20: return "pobre"
    if k < 0.40: return "debil"
    if k < 0.60: return "moderado"
    if k < 0.80: return "sustancial"
    return "casi perfecto"

kap_df = pd.DataFrame(
    [{"ambito": "GLOBAL", "fleiss_kappa": round(kappa_global, 4), "interpretacion": interp(kappa_global)}] +
    [{"ambito": a, "fleiss_kappa": round(kappa_asp[a], 4), "interpretacion": interp(kappa_asp[a])} for a in ASPECTS]
)
kap_df.to_csv(REP / "acuerdo_anotadores_refuerzo_v4.csv", index=False, encoding="utf-8-sig")

ntot = len(items_global)
md = []
md.append("# Concordancia entre anotadores — refuerzo balanceado V4\n")
md.append(f"- Resenas con triple anotacion: **{len(uids)}** | celdas (resena x aspecto): **{ntot}**")
md.append(f"- **Fleiss kappa global: {kappa_global:.3f}** ({interp(kappa_global)})\n")
md.append("## Patron de acuerdo (señal de anotacion independiente)\n")
md.append(f"- Unanime (3/3): **{pat['unanime_3']}** ({pat['unanime_3']/ntot*100:.1f}%)")
md.append(f"- Mayoria (2/3): **{pat['mayoria_2']}** ({pat['mayoria_2']/ntot*100:.1f}%)")
md.append(f"- Sin consenso (1/1/1, excluidas): **{pat['sin_consenso']}** ({pat['sin_consenso']/ntot*100:.1f}%)\n")
md.append("> Un patron sano combina unanimidad alta con una fraccion realista de mayorias y "
          "pocos casos sin consenso. Unanimidad ~100% seria sospechosa (anotacion no independiente).\n")
md.append("## Fleiss kappa por aspecto\n")
md.append("| aspecto | kappa | interpretacion |\n|---|---|---|")
for a in ASPECTS:
    md.append(f"| {a} | {kappa_asp[a]:.3f} | {interp(kappa_asp[a])} |")
md.append("\n## Consolidacion\n")
md.append(f"- Celdas sin consenso excluidas: {sin_consenso}")
md.append(f"- Tuplas nuevas con polaridad: **{len(nuevo_pol)}** "
          f"(neg {int((nuevo_pol.label=='negativo').sum())} / neu {int((nuevo_pol.label=='neutro').sum())} / pos {int((nuevo_pol.label=='positivo').sum())})")
md.append(f"- Gold reforzado total (polaridad): **{len(gp)}** "
          f"(neg {int((gp.label=='negativo').sum())} / neu {int((gp.label=='neutro').sum())} / pos {int((gp.label=='positivo').sum())})")
md.append(f"- Re-particion sin fuga: train {len(train)} / val {len(val)} / test {len(test)}\n")
(REP / "acuerdo_anotadores_refuerzo_v4.md").write_text("\n".join(md), encoding="utf-8")

print("\n=== Concordancia por aspecto ===")
for a in ASPECTS:
    print(f"  {a:18s} kappa={kappa_asp[a]:.3f} ({interp(kappa_asp[a])})")
print("\nSalidas:")
print("  outputs/reports/acuerdo_anotadores_refuerzo_v4.{csv,md}")
print("  outputs/gold/gold_consolidado_largo.csv (actualizado; backup *_pre_refuerzo)")
print("  data/{train,val,test}_gold_v4.csv (re-particion; backup *_pre_refuerzo)")
