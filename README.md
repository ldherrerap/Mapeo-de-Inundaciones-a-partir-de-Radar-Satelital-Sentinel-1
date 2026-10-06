# Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11)

Examen parcial · Redes Neuronales y Aprendizaje Profundo · MIA UNI FIIS 2026-II · **Pregunta 2**

Segmentación píxel a píxel de agua de inundación a partir de radar SAR Sentinel-1 (bandas VV y VH), con una **U-Net implementada desde cero en PyTorch**. Se estudia cómo combinar las etiquetas dorada y débil del dataset y cómo generaliza el modelo a un evento no visto (Bolivia).

---

## Ejecución en un solo comando

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` hace tres cosas:

1. Descarga del bucket público `gs://sen1floods11/v1.1` solo las carpetas necesarias.
2. Entrena los 12 experimentos de `configs/`.
3. Genera tablas y figuras en `results/`.

Variantes útiles:

```bash
QUICK=1 bash run_all.sh                                   # 2 épocas por experimento (verifica que todo corre)
CONFIGS="configs/e03*.yaml configs/e05*.yaml" bash run_all.sh   # solo algunos experimentos
NO_WEAK=1 bash scripts/download_data.sh                   # descargar solo los chips dorados
```

**Probar sin descargar nada** (datos sintéticos con la misma estructura de carpetas):

```bash
python tests/make_fake_data.py --out data/fake
QUICK=1 DATA_ROOT=data/fake bash run_all.sh
```

### Google Colab

Abre `notebooks/colab_run.ipynb`, activa la GPU (Entorno de ejecución → T4) y ejecuta las celdas. Cada experimento completo dura aproximadamente entre 10 y 40 minutos en una T4.

---

## Estructura

```
├── run_all.sh                  # comando único (descarga → entrenamiento → análisis)
├── requirements.txt
├── configs/                    # un YAML por experimento
├── scripts/download_data.sh    # descarga selectiva de Sen1Floods11 v1.1
├── src/
│   ├── dataset.py              # lectura GeoTIFF, normalización, particiones oficiales, aumentos
│   ├── unet.py                 # U-Net propia: profundidad, saltos concat / attention / none
│   ├── losses.py               # BCE, Dice y combinada, ignorando píxeles sin datos
│   ├── metrics.py              # IoU, F1, P, R, omisión, comisión, matriz de confusión
│   ├── engine.py               # modelo, loaders, evaluación, ajuste del umbral
│   ├── train.py                # entrenamiento + evaluación en valid / test / Bolivia
│   ├── analyze.py              # tabla resumen, gráficos, paneles de error
│   └── label_agreement.py      # ruido de la etiqueta débil (Otsu) vs dorada → results/label_agreement.csv
├── tests/make_fake_data.py     # mini dataset sintético para pruebas
├── notebooks/colab_run.ipynb
└── docs/
    ├── PLAN_TRABAJO.md         # reparto de tareas y cronograma del grupo
    └── ESQUELETO_INFORME.md    # estructura del informe NeurIPS y figuras que van en cada sección
```

---

## Datos

Se usa **Sen1Floods11 v1.1** (Bonafilia et al., 2020): 11 eventos de inundación con chips de 512×512 px a 10 m.

| Conjunto | Chips | Uso en este proyecto |
|---|---|---|
| `HandLabeled` (etiqueta dorada) | 446 | train / valid / test oficiales + Bolivia |
| `WeaklyLabeled/S1OtsuLabelWeak` | 4,385 | etiqueta débil por umbral de Otsu sobre VH |
| `WeaklyLabeled/S2IndexLabelWeak` | 4,385 | etiqueta débil por índices de Sentinel-2 |

- **Valores de la etiqueta:** `-1` es sin datos (se ignora en la pérdida y en las métricas), `0` es no agua y `1` es agua.
- **Entrada:** VV y VH en dB, recortados a [-50, 1], escalados a [0, 1] y estandarizados con los estadísticos del notebook oficial.
- **Partición:** la oficial (`splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv`). Los chips de Bolivia **nunca** se usan para entrenar; también se excluyen de los chips débiles.

---

## Experimentos

| ID | Pregunta que responde | Cambio respecto a la base (`e03`) |
|---|---|---|
| e01 / e02 / e03 | ¿Qué pérdida conviene? | BCE / Dice / BCE+Dice (solo dorados) |
| e04 | ¿Cuánto valen las etiquetas débiles solas? | solo chips débiles (Otsu S1) |
| e05 | ¿Ayuda mezclar dorados + débiles? | mezcla en cada batch, peso 0.5 a los débiles |
| e06 | ¿Mejor pre-entrenar con débiles y ajustar con dorados? | 2 fases |
| e07 | ¿Qué etiqueta débil es mejor? | como e05, pero con etiquetas débiles de S2 |
| e08 | ¿Aportan las compuertas de atención? | Attention U-Net |
| e09 | ¿Qué tan importantes son las conexiones de salto? | sin saltos |
| e10 | ¿Profundidad? | 3 niveles en vez de 4 |
| e11 / e12 | **Bono:** ¿mejora la fusión SAR+óptico? | mismos chips sin nubes; solo SAR vs SAR + 6 bandas S2 |

**Configuración base:** 32 canales iniciales, 4 niveles, AdamW con lr = 1e-3 y OneCycle, 60 épocas con *early stopping* (paciencia 15), recortes de 256 px, volteos y rotaciones de 90°, y `pos_weight = 3`. El umbral de decisión se ajusta **solo en validación** y se aplica tal cual a test y Bolivia.

Cada corrida deja en `runs/<experimento>/` los siguientes archivos:

- `best.pt`
- `history.csv`
- `metrics.json` (con valid, test, Bolivia y la brecha)
- `per_chip_*.csv` (IoU por chip, útil para elegir ejemplos)

---

## Métricas

- **IoU total** de la clase agua: TP / (TP + FP + FN), acumulando todos los píxeles del conjunto. Es la métrica de referencia de los autores.
- **IoU medio por chip**, como complemento.
- **Precisión, recall y F1** de la clase agua.
- **Tasa de omisión:** FN / (TP + FN).
- **Tasa de comisión:** FP / (TP + FP).
- **Matriz de confusión** normalizada por fila (`results/<exp>/confusion_*.png`).

---

## Salidas para el informe

| Archivo | Entregable del examen |
|---|---|
| `results/summary.md`, `summary.csv` | puntos 2 y 3: tabla de ablaciones y métricas en distribución |
| `results/fig_ablation_iou.png` | punto 4: test vs Bolivia por experimento |
| `results/<exp>/confusion_test.png` | punto 3: matriz de confusión comentada |
| `results/<exp>/errors_{test,bolivia}/FP_*.png`, `FN_*.png` | punto 5: fallos característicos del SAR |
| `results/<exp>/curves.png` | apoyo: curvas de entrenamiento |
| `summary.md` filas e11 / e12 | punto 6 (bono): fusión SAR + óptico |

---

## Referencias base

1. Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops.
2. Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. MICCAI.
3. Oktay, O. et al. (2018). *Attention U-Net: Learning where to look for the pancreas*. MIDL.
4. Zhou, Z. et al. (2018). *UNet++: A nested U-Net architecture for medical image segmentation*. DLMIA.
5. Milletari, F., Navab, N., Ahmadi, S.-A. (2016). *V-Net: Fully convolutional neural networks for volumetric medical image segmentation*. 3DV (pérdida Dice).
6. Otsu, N. (1979). *A threshold selection method from gray-level histograms*. IEEE TSMC.
7. Konapala, G., Kumar, S. V., Ahmad, S. K. (2021). *Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning*. ISPRS J. Photogramm. Remote Sens.
8. Mateo-Garcia, G. et al. (2021). *Towards global flood mapping onboard low cost satellites with machine learning*. Scientific Reports.

> Antes de citar, verifiquen cada referencia (año, venue, páginas) en Google Scholar.
