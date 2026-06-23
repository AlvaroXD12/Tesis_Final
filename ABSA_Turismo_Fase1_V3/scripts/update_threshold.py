import json

path = r'd:\Tesis\Tesis_Final\ABSA_Turismo_Fase1_V3\notebooks\01_preparacion_corpus_15_centros.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'MIN_WORDS_ABSA_EMPIRICO =' in line:
                source[i] = 'MIN_WORDS_ABSA_EMPIRICO = 6\n'
            elif 'MIN_WORDS_ABSA =' in line and 'MIN_WORDS_ABSA_EMPIRICO' not in line:
                source[i] = 'MIN_WORDS_ABSA = MIN_WORDS_ABSA_EMPIRICO\n'
            elif 'En esta ejecución el umbral empírico resultó ser de 14 palabras' in line:
                source[i] = line.replace('14', '6')
                
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if '14' in line and 'palabras' in line:
                source[i] = line.replace('14', '6')

# Check if the specific exact phrase is already added, if not add it in a markdown cell at the end of section or near MIN_WORDS_ABSA
phrase_added = False
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if 'El análisis empírico del corpus' in ''.join(cell['source']):
            phrase_added = True
            break

if not phrase_added:
    # Let's add the sentence right after the first code cell where MIN_WORDS_ABSA is defined
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'MIN_WORDS_ABSA_EMPIRICO = 6' in ''.join(cell['source']):
            new_md_cell = {
               'cell_type': 'markdown',
               'metadata': {},
               'source': ['> El análisis empírico del corpus de 15 centros turísticos seleccionó un umbral mínimo de 6 palabras para el corpus ABSA.']
            }
            nb['cells'].insert(idx + 1, new_md_cell)
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook Phase 1 updated to 6 words threshold.')
