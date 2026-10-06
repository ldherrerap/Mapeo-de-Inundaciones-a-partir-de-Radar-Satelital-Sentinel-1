# Plan de trabajo – Pregunta 2 (U-Net · Sen1Floods11)

**Entrega y presentación:** viernes 30 de octubre de 2026, en clase. El paquete incluye un PDF en formato NeurIPS de 6 a 10 páginas y el repositorio de GitHub.

**Congelar código:** martes 27 de octubre. Desde esa fecha solo se escribe el informe.

## Roles (5 integrantes)

| Rol | Responsable | Experimentos | Secciones del informe |
|---|---|---|---|
| **A. Datos y reproducibilidad** | | descarga, EDA (fracción de agua por evento, % sin datos), prueba final de `run_all.sh` en un entorno limpio | Datos; README |
| **B. Arquitectura** | | e08 (attention), e09 (sin saltos), e10 (profundidad); mapas de atención | Metodología – modelo |
| **C. Supervisión débil y pérdidas** | | e01–e07 | Ablación (a) y (b) |
| **D. Evaluación y generalización** | | matrices de confusión, experimento Bolivia, ≥2 hipótesis de la brecha | Resultados; Generalización geográfica |
| **E. Fallos SAR, bono y edición** | | paneles de error (≥3 tipos), e11–e12 | Análisis cualitativo; Bono; maquetación LaTeX |

Todos aportan al menos una referencia a *Trabajos relacionados* (mínimo 5 en total) y revisan el informe completo antes del 29.

## Cronograma

### Semana 1 · 5–11 oct — Puesta en marcha
- [ ] Crear el repositorio en GitHub, dar acceso a los 5 y subir esta base.
- [ ] Ejecutar `QUICK=1 DATA_ROOT=data/fake bash run_all.sh`. Todos deben verificar que les funciona.
- [ ] **A:** descargar los datos reales y subirlos a una carpeta compartida de Google Drive, para no descargar 5 veces.
- [ ] **A:** hacer el EDA: número de chips por evento y partición, % de agua, % de píxeles sin datos e histogramas VV/VH de agua vs no agua.
- [ ] **C:** correr la línea base `e03_gold_combo` completa (60 épocas) y anotar tiempo e IoU.
- [ ] Todos: buscar 2 papers y escribir de cada uno un resumen de 3 líneas.

### Semana 2 · 12–18 oct — Ablaciones de supervisión y pérdida
- [ ] **C:** correr e01, e02, e04, e05, e06 y e07. Repartir las corridas entre las cuentas de Colab del grupo (2–3 por persona).
- [ ] **B:** correr e08, e09 y e10.
- [ ] **D:** preparar el notebook de lectura de `runs/*/metrics.json` y las matrices de confusión.
- [ ] Reunión del viernes: elegir el **mejor modelo**, que será el que se analiza a fondo.

### Semana 3 · 19–25 oct — Generalización, fallos y bono
- [ ] **Recomendado:** repetir e03 y el mejor modelo con 3 semillas (`seed: 1, 2, 3`) para reportar media ± desviación.
- [ ] **D:** cuantificar la brecha test vs Bolivia, revisar `per_chip_bolivia.csv` y redactar las hipótesis.
- [ ] **E:** recorrer `results/<mejor>/errors_*` y clasificar los ejemplos (sombra de radar, doble rebote urbano, suelo húmedo, vegetación inundada, ruido speckle).
- [ ] **E:** correr el bono e11 vs e12.
- [ ] Borrador de todas las secciones para el domingo 25.

### Semana 4 · 26–30 oct — Informe y presentación
- [ ] 27: congelar el código; **A** clona el repo en limpio y corre `bash run_all.sh` de punta a punta.
- [ ] 28: informe completo en LaTeX con la plantilla NeurIPS.
- [ ] 29: revisión cruzada (cada uno revisa una sección ajena), diapositivas y ensayo.
- [ ] 30: entrega y presentación.

## Cómputo
- Colab gratuito (T4) alcanza. Cada uno corre sus experimentos y sube la carpeta `runs/<exp>/` a Drive.
- Después, `python -m src.analyze` arma la tabla con todo.
- Si Colab corta la sesión, bajar `train.epochs` o `batch_size` en el YAML y anotarlo en el informe.

## Reglas para evitar problemas de plagio
- El código de `src/` es propio. No copiar de `segmentation_models_pytorch`, `torchgeo` ni de notebooks de Kaggle o GitHub.
- Todo lo que se adapte del notebook oficial de Sen1Floods11, como las constantes de normalización, se cita en el informe.
