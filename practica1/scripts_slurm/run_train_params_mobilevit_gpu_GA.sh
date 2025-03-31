#!/bin/bash
#SBATCH -p cola02  # Usar la cola con GPU
#SBATCH --gres=gpu:1  # Solicitar 1 GPU (ajustar si necesitas más)
#SBATCH --mem=0     # Sin límite de memoria (usará toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una única tarea
#SBATCH --job-name=TrainGPUMViT
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/train_mvit_gpu_GA_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es

# Parámetros de configuración
BATCH_SIZE=4
IMG_SIZE=128
GRAD_ACCUM_STEPS=8

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
time apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/salvadordeharoo/IDL/practica1/scripts_python/config_gpubasebase.yaml /home/salvadordeharoo/IDL/practica1/scripts_python/mobileViT_train_params_gpu_GA.py \
--batch_size $BATCH_SIZE --img_size $IMG_SIZE --grad_accum_steps $GRAD_ACCUM_STEPS