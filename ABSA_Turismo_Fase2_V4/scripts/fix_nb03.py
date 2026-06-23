import json
import os
import re

path = r'd:\Tesis\Tesis_Final\ABSA_Turismo_Fase2_V4\notebooks\03_entrenamiento_absa_xlmr_y_matriz.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'RUN_TRAINING =' in line:
                source[i] = 'RUN_TRAINING = True            # ponlo en False para solo cargar artefactos\n'
            if 'SEEDS_OVERRIDE =' in line:
                source[i] = 'SEEDS_OVERRIDE = [42]          # p.ej. [42, 7, 123] para menos semillas (más rápido); None = 5 semillas\n'

diag_cell = {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    '# Diagnostico de artefactos\n',
    'import os\n',
    'print("Verificando scripts...", os.path.exists("../scripts/absa_common.py"))\n',
    'print("Verificando carpeta outputs/reports...", os.path.exists("../outputs/reports"))\n'
   ]
}

# Insert diagnostic cell before training
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'ENTRENAMIENTO XLM-R' in ''.join(cell['source']):
        nb['cells'].insert(idx, diag_cell)
        break

new_matrix_code = """
    # Agregación destino x aspecto con todas las combinaciones
    import itertools
    destinos = corpus["destination"].unique()
    aspectos = list(PAT.keys())
    todas_combinaciones = list(itertools.product(destinos, aspectos))
    
    agregado = {}
    for (d, a), g in tuplas.groupby(["destination", "aspecto"]):
        n = len(g)
        npos = int((g.polaridad == "positivo").sum())
        nneg = int((g.polaridad == "negativo").sum())
        nneu = int((g.polaridad == "neutro").sum())
        n_reviews_unique = g["review_uid"].nunique()
        agregado[(d, a)] = (n, npos, nneg, nneu, n_reviews_unique)

    out = []
    for d, a in todas_combinaciones:
        if (d, a) in agregado:
            n, npos, nneg, nneu, n_reviews_unique = agregado[(d, a)]
            score = (npos - nneg) / n if n else 0.0
            conf_flag = int(npos > 0 and nneg > 0 and min(npos, nneg) / n >= 0.25)
            conf = min(1.0, n / 10) * (0.65 if conf_flag else 1.0)
            adj = score * conf
            
            if n == 0: ev = "sin_datos"
            elif n < 5: ev = "evidencia_insuficiente"
            elif n < 10: ev = "baja_evidencia"
            else: ev = "evidencia_suficiente"
            
            dom = max([("negativo", nneg), ("neutro", nneu), ("positivo", npos)], key=lambda x: x[1])[0]
            pct_pos = round(npos / n * 100, 2) if n else 0.0
            pct_neu = round(nneu / n * 100, 2) if n else 0.0
            pct_neg = round(nneg / n * 100, 2) if n else 0.0
        else:
            n = npos = nneg = nneu = 0
            n_reviews_unique = 0
            score = conf = adj = 0.0
            conf_flag = 0
            ev = "sin_datos"
            dom = "neutro"
            pct_pos = pct_neu = pct_neg = 0.0

        out.append({"destination": d, "aspecto": a, "n_menciones": n, "n_reviews_unique": n_reviews_unique,
                    "n_pos": npos, "n_neu": nneu, "n_neg": nneg, 
                    "pct_pos": pct_pos, "pct_neu": pct_neu, "pct_neg": pct_neg,
                    "sentiment_score": round(score, 4), "confidence": round(conf, 4), "score_ajustado": round(adj, 4),
                    "score_normalizado_0_1": round((adj + 1) / 2, 4), "evidence_status": ev,
                    "conflict_flag": conf_flag, "dominant_label": dom})
    matriz = pd.DataFrame(out)
"""

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'out = []' in ''.join(cell['source']) and 'for (d, a), g in tuplas.groupby' in ''.join(cell['source']):
        original_source = ''.join(cell['source'])
        start_idx = original_source.find('    # Agregaci')
        if start_idx == -1: start_idx = original_source.find('    out = []')
        end_idx = original_source.find('    matriz = pd.DataFrame(out)')
        if end_idx == -1: end_idx = original_source.find('    matriz.to_csv')
        else: end_idx += len('    matriz = pd.DataFrame(out)\n')
        
        prefix = original_source[:start_idx]
        suffix = original_source[end_idx:]
        
        new_source = prefix + new_matrix_code + suffix
        cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Updated NB03 successfully.')
