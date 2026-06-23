# Diagnostico de balance de polaridades + plan de refuerzo — Gold V4

> `polaridad_sugerida` es **solo una guia de muestreo** para priorizar la anotacion; **no es una etiqueta final**. La anotacion humana decide la polaridad real.

## 1. Distribucion global actual

- Total de tuplas con polaridad: **2302**
- Positivo: **1312** (56.99%)
- Neutro: **695** (30.19%)
- Negativo: **295** (12.81%)
- Ratio positivo:neutro:negativo = **4.45 : 2.36 : 1.0** (normalizado a negativo=1)
- Clase minoritaria global: **negativo** (295). El gold esta fuertemente sesgado a positivo, coherente con un corpus turistico de resenas (94% 4-5 estrellas).

## 2. Distribucion por aspecto

| aspecto | n_negativo | n_neutro | n_positivo | total | clase_minoritaria | soporte_min_clase | brecha_may_min | def_neg | def_neu | def_pos |
|---|---|---|---|---|---|---|---|---|---|---|
| accesibilidad | 58 | 167 | 160 | 385 | negativo | 58 | 109 | 0 | 0 | 0 |
| alojamiento | 5 | 47 | 30 | 82 | negativo | 5 | 42 | 25 | 0 | 0 |
| atencion_servicio | 29 | 135 | 159 | 323 | negativo | 29 | 130 | 1 | 0 | 0 |
| atractivos | 62 | 121 | 736 | 919 | negativo | 62 | 674 | 0 | 0 | 0 |
| costos | 81 | 122 | 131 | 334 | negativo | 81 | 50 | 0 | 0 | 0 |
| gastronomia | 3 | 47 | 35 | 85 | negativo | 3 | 44 | 27 | 0 | 0 |
| limpieza | 32 | 38 | 46 | 116 | negativo | 32 | 14 | 0 | 0 | 0 |
| seguridad | 25 | 18 | 15 | 58 | positivo | 15 | 10 | 5 | 12 | 15 |

*`def_*` = cuantos faltan para llegar a 30 por clase en ese aspecto.*

## 3. Distribucion por destino (ordenado por # negativos)

| destination | neg | neu | pos | total |
|---|---|---|---|---|
| Museo Tumbas Reales del Señor de Sipán | 7 | 39 | 97 | 143 |
| Museo de Sitio Huaca Pucllana | 11 | 68 | 81 | 160 |
| Líneas y Geoglifos de Nasca y Palpa | 13 | 40 | 62 | 115 |
| Catarata de Ahuashiyacu | 14 | 24 | 100 | 138 |
| Sitio Arqueológico de Ollantaytambo | 14 | 66 | 73 | 153 |
| Huacas del Sol y de la Luna | 15 | 52 | 83 | 150 |
| Valle del Colca | 18 | 39 | 93 | 150 |
| Complejo Arqueológico Chan Chan | 19 | 47 | 89 | 155 |
| Reserva Nacional de Paracas | 21 | 48 | 99 | 168 |
| Circuito Mágico del Agua | 21 | 53 | 75 | 149 |
| Parque Nacional Huascarán | 24 | 23 | 92 | 139 |
| Santuario Arqueológico de Pachacámac | 27 | 40 | 85 | 152 |
| Santuario Histórico de Machu Picchu | 29 | 70 | 105 | 204 |
| Parque Arqueológico de Sacsayhuamán | 30 | 51 | 69 | 150 |
| Ciudadela de Kuélap | 32 | 35 | 109 | 176 |

- Destinos con <=5 negativos en el gold: ninguno.
- Los positivos se concentran en los destinos mas famosos (atractivos casi siempre positivos); negativos y neutros estan dispersos y son escasos en casi todos.

## 4. Distribucion por estrellas e idioma

**Por estrellas (tuplas del gold):**

| stars | neg | neu | pos |
|---|---|---|---|
| 1 | 64 | 53 | 40 |
| 2 | 53 | 47 | 66 |
| 3 | 57 | 126 | 176 |
| 4 | 45 | 96 | 206 |
| 5 | 76 | 373 | 824 |

**Por idioma:**

| language_review | neg | neu | pos |
|---|---|---|---|
| en | 150 | 416 | 554 |
| es | 145 | 279 | 758 |

**Lectura de la senal:**
- Las **estrellas bajas (1-2) si aportan negativos**, pero son ESCASAS: el corpus completo tiene solo ~320 resenas con 1-2 estrellas. En el pool sin anotar quedan ~201. Por eso el negativo **no puede salir solo de estrellas bajas**; se mina tambien con lexico negativo + conectores adversativos dentro de resenas de 4-5 estrellas ('hermoso PERO carisimo y lleno').
- Las **estrellas medias (3) aportan neutros / mixtos**, tambien escasas (~489 totales).
- **Fuente unica** (Google Maps) -> no hay sesgo por fuente que controlar. Idioma es/en esta razonablemente balanceado (~53/47), asi que el refuerzo mantiene ambos idiomas.

## 5. Plan de refuerzo balanceado (candidatos seleccionados)

- **Negativo (prioridad maxima):** 415 candidatos sugeridos.
- **Neutro:** 210 candidatos (foco en aspectos con neutro escaso).
- **Positivo:** 15 candidatos (SOLO en aspectos con bajo soporte; no se sobrecarga).
- **Resenas unicas a anotar:** 316 (pool `C_refuerzo_balanceado_v4`).

**Candidatos por aspecto y polaridad sugerida:**

| aspecto_sugerido | neg | neu | pos |
|---|---|---|---|
| accesibilidad | 55 | 33 | 0 |
| alojamiento | 4 | 1 | 0 |
| atencion_servicio | 48 | 27 | 0 |
| atractivos | 133 | 78 | 0 |
| costos | 112 | 40 | 0 |
| gastronomia | 18 | 5 | 0 |
| limpieza | 26 | 8 | 0 |
| seguridad | 19 | 18 | 15 |

**Candidatos por destino (cobertura, con tope anti-dominacion):**

| destination | neg | neu | pos |
|---|---|---|---|
| Catarata de Ahuashiyacu | 4 | 5 | 8 |
| Circuito Mágico del Agua | 34 | 18 | 1 |
| Ciudadela de Kuélap | 33 | 23 | 1 |
| Complejo Arqueológico Chan Chan | 16 | 21 | 0 |
| Huacas del Sol y de la Luna | 2 | 7 | 0 |
| Líneas y Geoglifos de Nasca y Palpa | 25 | 14 | 0 |
| Museo Tumbas Reales del Señor de Sipán | 38 | 35 | 0 |
| Museo de Sitio Huaca Pucllana | 45 | 35 | 1 |
| Parque Arqueológico de Sacsayhuamán | 45 | 28 | 0 |
| Parque Nacional Huascarán | 3 | 2 | 0 |
| Reserva Nacional de Paracas | 45 | 6 | 1 |
| Santuario Arqueológico de Pachacámac | 39 | 4 | 1 |
| Santuario Histórico de Machu Picchu | 45 | 4 | 2 |
| Sitio Arqueológico de Ollantaytambo | 24 | 3 | 0 |
| Valle del Colca | 17 | 5 | 0 |

**Candidatos por estrellas (diversidad: no todo de 1 estrella):**

| stars | neg | neu | pos |
|---|---|---|---|
| 1 | 155 | 0 | 0 |
| 2 | 185 | 0 | 0 |
| 3 | 3 | 202 | 0 |
| 4 | 21 | 1 | 2 |
| 5 | 51 | 7 | 13 |

## 6. Proyeccion orientativa post-anotacion

*(asume que la senal sugerida se confirma; la anotacion real puede reclasificar)*

- Negativo: 295 -> ~**710**
- Neutro: 695 -> ~**905**
- Positivo: 1312 -> ~**1327**

## 7. Limitaciones que persisten (corpus-limitadas)

- `alojamiento` / **negativo**: 5 actuales + 4 candidatos = 9 (< 30; limitado por el corpus)
- `gastronomia` / **negativo**: 3 actuales + 18 candidatos = 21 (< 30; limitado por el corpus)

- No se busca un balance perfecto artificial: se prioriza balance **util para entrenamiento**. Donde el corpus no tiene negativos/neutros reales suficientes para un aspecto, **no se inventan**; se declara como limitacion.

## 8. Como usar este refuerzo

1. Anotar `data/gold/plantilla_refuerzo_balanceado_v4.csv` (multi-etiqueta, pool `C_refuerzo_balanceado_v4`), usando `candidatos_refuerzo_balanceado_v4.csv` como guia de prioridad.
2. Consolidar con el gold existente (mismo formato largo) y re-particionar sin fuga por `review_uid`.
3. Reentrenar XLM-R con **5 semillas** + calibracion y re-evaluar contra la spec.
