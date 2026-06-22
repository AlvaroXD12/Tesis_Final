# Tesis_Final — ABSA Turismo (recomendador turístico, Perú)

Pipeline de tesis: **análisis de sentimientos por aspectos (ABSA) multilingüe** + recomendador
contextual para **15 centros turísticos** del Perú. Reseñas públicas de Google Maps → matriz
destino-aspecto-sentimiento → afinidad destino-perfil → re-ranking por clima y aforo.

## 📂 Estructura
- `ABSA_Turismo_Fase1_V3/` — ETL y **corpus limpio** (15 centros, ~13.7k reseñas, 8 aspectos).
- `ABSA_Turismo_Fase2_V4/` — **gold multi-etiqueta**, detección de aspectos (diccionario vs SBERT)
  y **entrenamiento ABSA de polaridad (XLM-R + TextCNN) + matriz**.
- *(Fase 3 y 4: recomendador y contexto — en construcción.)*

---

## 🚀 Para correr el MODELO (entrenamiento XLM-R) — colaborador con GPU

El modelo de polaridad se entrena en la **Fase 2, Notebook 03**. Necesita **GPU con CUDA**.

### 1. Entorno y dependencias
```bash
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
# torch con CUDA (ajusta a tu CUDA; ejemplo cu121):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Verifica GPU: `python -c "import torch; print(torch.cuda.is_available())"` → debe decir `True`.

### 2. Entrenar
Abre y ejecuta de arriba a abajo:
```
ABSA_Turismo_Fase2_V4/notebooks/03_entrenamiento_absa_xlmr_y_matriz.ipynb
```
- La celda de entrenamiento debe tener `RUN_TRAINING = True`.
- Entrena **5 semillas + ensemble + calibración** (pesado; usar GPU potente).
- Si quieres más rápido para probar: pon `SEEDS_OVERRIDE = [42, 7, 123]`.

### 3. Resultados (se generan solos al terminar)
- **Modelos:** `ABSA_Turismo_Fase2_V4/models/modelo_xlmr_seed*_v4.pt`
- **Reportes:** `outputs/reports/` (resumen, por semilla, por aspecto, baselines)
- **Figuras:** `outputs/visualizations/` y `outputs/figures/` (comparativa, **curvas de
  entrenamiento**, confusión, F1 por clase/aspecto, estabilidad, **heatmap de la matriz**)
- **Matriz:** `outputs/matrices/matriz_destino_aspecto_sentimiento.csv` (insumo de Fase 3)

> El **gold ya está anotado y consolidado** (`outputs/gold/`) y los **splits sin fuga** ya están
> en `data/`. No hace falta re-anotar ni re-muestrear. El NB01 es idempotente (no borra el gold).

---

## Notas
- Detección de aspectos: se eligió **diccionario** (superó a SBERT zero-shot en el gold, NB02).
- Polaridad: **XLM-R + TextCNN** (sin BERT: ya se sabe que XLM-R es mejor).
- Métrica principal: **F1-macro** (corpus desbalanceado).
