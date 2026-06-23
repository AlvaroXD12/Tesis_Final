# Concordancia entre anotadores — refuerzo balanceado V4

- Resenas con triple anotacion: **316** | celdas (resena x aspecto): **2528**
- **Fleiss kappa global: 0.909** (casi perfecto)

## Patron de acuerdo (señal de anotacion independiente)

- Unanime (3/3): **2366** (93.6%)
- Mayoria (2/3): **162** (6.4%)
- Sin consenso (1/1/1, excluidas): **0** (0.0%)

> Un patron sano combina unanimidad alta con una fraccion realista de mayorias y pocos casos sin consenso. Unanimidad ~100% seria sospechosa (anotacion no independiente).

## Fleiss kappa por aspecto

| aspecto | kappa | interpretacion |
|---|---|---|
| atractivos | 0.929 | casi perfecto |
| costos | 0.895 | casi perfecto |
| seguridad | 0.957 | casi perfecto |
| accesibilidad | 0.812 | casi perfecto |
| limpieza | 0.789 | sustancial |
| atencion_servicio | 0.943 | casi perfecto |
| gastronomia | 0.704 | sustancial |
| alojamiento | 0.962 | casi perfecto |

## Consolidacion

- Celdas sin consenso excluidas: 0
- Tuplas nuevas con polaridad: **748** (neg 169 / neu 336 / pos 243)
- Gold reforzado total (polaridad): **3050** (neg 464 / neu 1031 / pos 1555)
- Re-particion sin fuga: train 2173 / val 445 / test 432
