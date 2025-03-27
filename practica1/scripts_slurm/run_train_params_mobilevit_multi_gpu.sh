#!/bin/bash
#SBATCH -p cola03                # ? Usar la cola correcta
#SBATCH --nodes=5               # ? 5 nodos
#SBATCH --ntasks-per-node=1     # ? 1 tarea por nodo (una por GPU)
#SBATCH --gres=gpu:1            # ? 1 GPU por nodo
#SBATCH --mem=0                 # Sin límite de memoria
#SBATCH --job-name=TrainGPUMViT
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/train_mvit_gpu_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=salvadorde.haroo@um.es

# Parámetros del script
BATCH_SIZE=16
IMG_SIZE=256

# Ruta al contenedor SIF y al config
CONTAINER=/software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif
CONFIG=/home/salvadordeharoo/IDL/practica1/scripts_python/config_multi_gpu.yaml
SCRIPT=/home/salvadordeharoo/IDL/practica1/scripts_python/mobileViT_train_params_multi_gpu.py

# Ejecutar el script Python con accelerate y Apptainer
time apptainer exec --writable-tmpfs --nv "$CONTAINER" \
bash -c "
export ACCELERATE_MACHINE_RANK=\$SLURM_PROCID && \
accelerate launch --multi_gpu --config_file $CONFIG $SCRIPT --batch_size $BATCH_SIZE --img_size $IMG_SIZE
"
