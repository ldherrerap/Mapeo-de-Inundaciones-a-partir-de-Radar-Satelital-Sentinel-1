# Esqueleto del informe (formato NeurIPS, 6–10 páginas)

**Título sugerido:** *Mapeo de inundaciones con Sentinel-1 y U-Net: supervisión débil y generalización geográfica en Sen1Floods11*

## 1. Introducción y motivación (~0.75 pág.)
- El problema: los satélites ópticos son inútiles bajo nubes, y el SAR atraviesa las nubes y opera de noche.
- Relevancia para Perú: el Niño costero (2017, 2023) trajo inundaciones en Piura y la costa norte. El mapeo rápido apoya la respuesta del INDECI.
- Contribuciones, en 3 viñetas:
  - U-Net propia.
  - Estudio de cómo combinar etiquetas doradas y débiles.
  - Evaluación en un evento no visto (Bolivia) con análisis de fallos del SAR.

## 2. Trabajos relacionados (~0.75 pág., ≥5 referencias)
- Segmentación: U-Net, Attention U-Net, UNet++.
- Inundaciones con SAR: umbral de Otsu, Sen1Floods11, Konapala et al. 2021, ml4floods.
- Aprendizaje con etiquetas ruidosas o débiles en teledetección.

## 3. Datos (~0.75 pág.)
- Tabla de chips por partición y evento, con el % de agua (EDA).
- Figura de un chip de ejemplo: VV, VH y etiqueta.
- Diferencias entre la etiqueta dorada y las débiles (Otsu-S1 vs índices-S2). Medir su acuerdo: IoU de `S1OtsuLabelHand` contra `LabelHand` en los chips dorados.

## 4. Metodología (~1.5 págs.)
- Arquitectura: diagrama de la U-Net, canales por nivel, compuertas de atención (ecuación) y número de parámetros.
- Preprocesamiento y aumentos.
- Pérdidas: ecuaciones de BCE con `pos_weight`, Dice y combinada; manejo de los píxeles sin datos.
- Estrategias de supervisión: solo dorados, solo débiles, mezcla ponderada, pre-entrenamiento + ajuste fino.
- Selección del umbral en validación y métricas: IoU, F1, omisión y comisión.

## 5. Experimentos y resultados (~2.5 págs.)
- **5.1 Ablación (a), supervisión:** tabla con e03–e07. Usar `results/summary.md`.
- **5.2 Ablación (b), pérdida y arquitectura:** tabla con e01–e03 y e08–e10.
- **5.3 Evaluación en distribución:** IoU, F1, P y R del mejor modelo, con la matriz de confusión comentada (`confusion_test.png`).
- **5.4 Generalización geográfica (Bolivia):**
  - Figura `fig_ablation_iou.png` y la brecha.
  - ≥2 hipótesis, contrastadas con evidencia de `per_chip_bolivia.csv`:
    - Cobertura del suelo distinta (sabana o pastizal inundado vs zonas agrícolas).
    - Diferente ángulo de incidencia u órbita (metadato `orbit` en el geojson).
    - Vegetación inundada, donde el doble rebote aumenta VV en lugar de bajarlo.
    - Distinta fracción de agua por chip (cambio en la distribución de clases).
- **5.5 Análisis cualitativo de fallos del SAR:** ≥3 tipos con paneles de `errors_*`.
  - Sombra de radar en relieve, que da falso positivo.
  - Doble rebote urbano, que da falso negativo.
  - Suelo húmedo o superficies lisas (pistas, arena), que dan falso positivo.
  - Viento sobre el agua, que la vuelve rugosa y da falso negativo.
- **5.6 Bono:** e11 vs e12 en los chips sin nubes.

## 6. Discusión crítica y limitaciones (~0.75 pág.)
- Solo hay 446 chips dorados y un único evento de test geográfico.
- La varianza entre semillas puede ser del orden de las diferencias entre ablaciones.
- La resolución de 10 m pierde inundaciones en calles estrechas.
- Las etiquetas débiles heredan los sesgos del umbral de Otsu.
- La aplicación a Perú requeriría validación local, por ejemplo con imágenes del Niño costero 2023 en el Bajo Piura.

## 7. Conclusión (~0.25 pág.)

## Reproducibilidad
- Enlace al repositorio, `bash run_all.sh`, semillas, hardware y tiempo por experimento.
