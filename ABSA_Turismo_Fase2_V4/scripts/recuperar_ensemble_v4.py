# -*- coding: utf-8 -*-
"""
recuperar_ensemble_v4.py
========================
RECUPERA el resultado tras un crash de kernel (OOM) SIN reentrenar: usa los
modelos de semilla ya guardados en disco (models/modelo_xlmr_seed*_v4.pt),
hace solo INFERENCIA (bajo consumo de memoria), arma el ensemble + calibracion
y regenera TODOS los artefactos que esperan las celdas de reporte del NB03.

Uso (en la maquina con los modelos):
  1) Reinicia el kernel / cierra Python para liberar la GPU.
  2) python scripts/recuperar_ensemble_v4.py
  3) Re-ejecuta SOLO las celdas de reporte del notebook (no la de entrenamiento),
     o el notebook con RUN_TRAINING=False.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

import absa_common as ac

tr, va, te = ac.load_splits()
tok = ac.AutoTokenizer.from_pretrained(ac.MODELOS["xlmr"])
vl = DataLoader(ac.ABSADataset(va["input_modelo"], va["label"], tok), batch_size=ac.BATCH)
el = DataLoader(ac.ABSADataset(te["input_modelo"], te["label"], tok), batch_size=ac.BATCH)

modelos = sorted(ac.MODELS_DIR.glob("modelo_xlmr_seed*_v4.pt"))
if not modelos:
    raise SystemExit("No hay modelos guardados (models/modelo_xlmr_seed*_v4.pt). Hay que reentrenar.")
print(f"Modelos de semilla encontrados: {len(modelos)} -> {[p.stem for p in modelos]}")

rows, probs, vprobs, tt0, vt0 = [], [], [], None, None
for p in modelos:
    seed = int(p.stem.split("seed")[1].split("_")[0])
    m = ac.TextCNN(ac.MODELOS["xlmr"]).to(ac.DEVICE)
    m.load_state_dict(torch.load(p, map_location=ac.DEVICE)); m.eval()
    tp, tt, _ = ac.predict(m, el)
    vp, vtv, _ = ac.predict(m, vl)
    tt0, vt0 = tt, vtv
    probs.append(tp); vprobs.append(vp)
    mm = ac.metrics(tt, [ac.I2L[i] for i in tp.argmax(1)])
    rows.append({"seed": seed, **{k: round(v, 4) for k, v in mm.items()}})
    print(f"  seed {seed}: F1-macro={mm['f1_macro']:.4f}", flush=True)
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()

det = pd.DataFrame(rows)
ens, ensv = np.mean(probs, 0), np.mean(vprobs, 0)
bias = ac.best_bias(ensv, vt0)
preds = ac.apply_bias(ens, bias)
em = ac.metrics(tt0, preds)

# --- guardar artefactos (mismos nombres que run_modelo) ---
det.to_csv(ac.art("xlmr", "det"), index=False, encoding="utf-8-sig")
np.save(ac.art("xlmr", "bias"), bias)
pd.DataFrame({"y_true": tt0, "y_pred": preds}).to_csv(ac.art("xlmr", "preds"), index=False, encoding="utf-8-sig")
pd.DataFrame([{"modelo": "xlmr", **{f"ensemble_{k}": round(v, 4) for k, v in em.items()},
               "media_f1_macro": round(det["f1_macro"].mean(), 4), "std_f1_macro": round(det["f1_macro"].std(), 4),
               "estable_std<=0.03": bool(det["f1_macro"].std() <= 0.03)}]).to_csv(ac.art("xlmr", "resumen"), index=False, encoding="utf-8-sig")

ta = te.copy(); ta["pred"] = preds; fa = []
for asp, gg in ta.groupby("aspecto"):
    present = [l for l in ac.LABELS if l in set(gg["label"])]
    _, _, f1m, _ = precision_recall_fscore_support(gg["label"], gg["pred"], labels=present, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(gg["label"], gg["pred"], labels=ac.LABELS, average="weighted", zero_division=0)
    fa.append({"aspecto": asp, "soporte": len(gg), "n_clases": len(present), "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4)})
pd.DataFrame(fa).sort_values("f1_macro").to_csv(ac.art("xlmr", "aspecto"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 56)
print(f"ENSEMBLE RECUPERADO con {len(modelos)} semillas")
print(f"  F1-macro={em['f1_macro']:.4f} | neg F1={em['f1_negativo']:.4f} recall={em['recall_negativo']:.4f} | "
      f"neu F1={em['f1_neutro']:.4f} | media {det['f1_macro'].mean():.4f}±{det['f1_macro'].std():.4f}")
print("Artefactos regenerados: resumen/resultados/predicciones/por_aspecto/bias.")
print("NOTA: faltara historial_xlmr_v4.csv (curvas de epoca) -> esa figura se omite; el resto del reporte y la matriz funcionan.")
