# Re-anotacion ciega del lote balanceado aplicada al gold

- Solapamiento completo: 640 pares (review x aspecto) | **Fleiss kappa = 0.927** | patron {'unanime': 592, 'mayoria': 48}
- Sin consenso (no aplicados): 0
- Anotacion CIEGA (sin ver polaridad_sugerida/motivo_minado) -> supersede la ronda 1 para estos pares.
- Pares con etiqueta cambiada respecto al gold: **298** de 640
- Distribucion (polaridad) ANTES: {'negativo': 579, 'neutro': 1248, 'positivo': 1688}
- Distribucion (polaridad) DESPUES: {'negativo': 726, 'neutro': 1120, 'positivo': 1680}
- Re-particion sin fuga: train 2488 / val 508 / test 530 | neg 516/90/120