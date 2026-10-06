#!/usr/bin/env bash
# Comando único: descarga datos, entrena todas las configuraciones y genera tablas y figuras.
#
#   bash run_all.sh                         # experimentos completos
#   QUICK=1 bash run_all.sh                 # 2 épocas por experimento (verificar que todo corre)
#   CONFIGS="configs/e03*.yaml configs/e05*.yaml" bash run_all.sh   # solo algunos
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:-data/sen1floods11/v1.1}
CONFIGS=${CONFIGS:-configs/*.yaml}
EXTRA=()
[ "${QUICK:-0}" = "1" ] && EXTRA+=(--epochs 2)

if [ ! -d "$DATA_ROOT/data/flood_events/HandLabeled/S1Hand" ]; then
  DEST="$DATA_ROOT" bash scripts/download_data.sh
fi

for cfg in $CONFIGS; do
  name=$(basename "$cfg" .yaml)
  if [ -f "runs/$name/metrics.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo ">> $name ya existe, se omite (FORCE=1 para repetir)"; continue
  fi
  echo "================ $name ================"
  python -m src.train --config "$cfg" --data-root "$DATA_ROOT" "${EXTRA[@]}"
done

python -m src.label_agreement --data-root "$DATA_ROOT" || echo ">> (sin S1OtsuLabelHand, se omite el análisis de ruido)"
python -m src.analyze --runs runs --out results --data-root "$DATA_ROOT"
echo ">> Resultados en results/summary.md y figuras en results/"
