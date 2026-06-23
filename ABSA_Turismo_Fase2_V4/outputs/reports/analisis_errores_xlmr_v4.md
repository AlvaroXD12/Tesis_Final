# Analisis de errores XLM-R + minado dirigido de negativos — V4

## 1. Matriz de confusion (test, ensemble calibrado)

| real \\ pred | negativo | neutro | positivo |
|---|---|---|---|
| **negativo** | 41 | 16 | 6 |
| **neutro** | 16 | 94 | 41 |
| **positivo** | 8 | 34 | 176 |

- Errores totales: 121/432 (28.0%).
- **Negativos perdidos: 22/63** (a neutro 16, a positivo 6).
- Falsas alarmas de negativo: 24 (el modelo atribuye una queja de otro aspecto).
- Mayor masa de error en la frontera neutro<->positivo (41+34 casos).

## 2. Negativos perdidos por aspecto

| aspecto | negativos perdidos |
|---|---|
| costos | 6 |
| atractivos | 5 |
| accesibilidad | 4 |
| atencion_servicio | 4 |
| gastronomia | 1 |
| limpieza | 1 |
| seguridad | 1 |

**Patron:** el modelo NO pierde negativos con palabras fuertes, sino los **sutiles/implicitos**: subidas de precio ('used to be free, now additional charge'), quejas indirectas ('hardly any information', 'forced to', 'una lastima', 'tourist trap', 'deflating'). Por eso el minado dirigido prioriza ese lexico sutil.

## 3. Etiquetas del gold (test) a RE-REVISAR (no se cambian aqui)

- 6 casos donde el modelo predice **negativo** con senal negativa clara pero el gold dice **positivo** (posible error de anotacion). Listados en `gold_test_a_rerevisar_v4.csv` para que un humano decida. Ejemplos:
  - `atractivos` [lamentable]: Es lamentable la atención en el Museo. Ninguno de los trabajadores sonríe ni es amable con excepción de el muc
  - `atractivos` [disappointed]: A must visit place before you take the train to Machu Picchu. To reach the Temple Hill you have to climb 300 s
  - `accesibilidad` [disappointed]: A must visit place before you take the train to Machu Picchu. To reach the Temple Hill you have to climb 300 s
  - `atencion_servicio` [disappointed]: Food excellent but service very disappointed That kind of service does not go with this nice restaurant
  - `gastronomia` [disappointed]: Food excellent but service very disappointed That kind of service does not go with this nice restaurant
  - `atractivos` [expensive,save your money,not that interesting]: Expensive and not that interesting. Save your money for Cusco and the markets around lima

> Corregir un par de mislabels del test (via anotacion humana) sube el score de forma legitima; NO se auto-corrige para no sesgar la evaluacion.

## 4. Minado dirigido de negativos (para anotacion)

- Candidatos seleccionados: **170** (83 resenas, pool `C_negativos_dirigido_v4`).
- Por aspecto: {'atractivos': 78, 'costos': 40, 'accesibilidad': 33, 'atencion_servicio': 19}
- Foco en aspectos con mas negativos perdidos: costos, atractivos, accesibilidad, atencion_servicio.
- Estrategia: lexico negativo **sutil** (lo que el modelo pierde) + conectores adversativos + estrellas bajas; topes por destino para no concentrar.

## 5. Como usar

1. Anotar `data/gold/plantilla_negativos_dirigido_v4.csv` (3 anotadores, pool `C_negativos_dirigido_v4`).
2. Re-revisar los casos de `gold_test_a_rerevisar_v4.csv` (corregir solo los que el anotador confirme mal etiquetados).
3. Consolidar con `consolidar_refuerzo_v4.py` (adaptando el patron de lotes) -> re-particionar -> reentrenar.
