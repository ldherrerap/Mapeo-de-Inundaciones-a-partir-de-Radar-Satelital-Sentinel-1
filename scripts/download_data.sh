#!/usr/bin/env bash
# Descarga solo las carpetas de Sen1Floods11 v1.1 que usa el proyecto (~ unos pocos GB en vez de 14 GB).
#   bash scripts/download_data.sh            # dorados + débiles (S1) + S2 de los dorados
#   NO_WEAK=1 bash scripts/download_data.sh  # solo dorados (para empezar rápido)
set -euo pipefail

DEST=${DEST:-data/sen1floods11/v1.1}
BUCKET=gs://sen1floods11/v1.1

if ! command -v gsutil >/dev/null 2>&1; then
  echo ">> gsutil no encontrado; instalando vía pip"
  pip install -q gsutil
fi

sync () {
  mkdir -p "$DEST/$1"
  echo ">> $1"
  gsutil -m -q rsync -r "$BUCKET/$1" "$DEST/$1"
}

sync splits/flood_handlabeled
sync data/flood_events/HandLabeled/S1Hand
sync data/flood_events/HandLabeled/LabelHand
sync data/flood_events/HandLabeled/S1OtsuLabelHand   # para medir el ruido de la etiqueta débil
[ "${NO_S2:-0}" = "1" ] || sync data/flood_events/HandLabeled/S2Hand

if [ "${NO_WEAK:-0}" != "1" ]; then
  sync data/flood_events/WeaklyLabeled/S1Weak
  sync data/flood_events/WeaklyLabeled/S1OtsuLabelWeak
  sync data/flood_events/WeaklyLabeled/S2IndexLabelWeak
fi

echo ">> Listo. Conteo de archivos:"
for d in "$DEST"/data/flood_events/*/*; do
  printf "   %-60s %s\n" "$d" "$(ls "$d" | wc -l)"
done
ls "$DEST/splits/flood_handlabeled"
