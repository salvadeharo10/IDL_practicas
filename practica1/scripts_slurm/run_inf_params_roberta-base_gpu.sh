#!/bin/bash
#SBATCH -p cola02  # Usar la cola con GPU
#SBATCH --gres=gpu:1  # Solicitar 1 GPU (ajustar si necesitas más)
#SBATCH --mem=0     # Sin límite de memoria (usará toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una única tarea
#SBATCH --job-name=InfGPURob
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/inf_roberta_gpu_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es

# Parámetros de configuración
BATCH_SIZE=16
SEQ_LENGTH=512

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/salvadordeharoo/IDL/practica1/scripts_python/config_gpubase.yaml \
/home/salvadordeharoo/IDL/practica1/scripts_python/roberta-base_inf_params_gpu.py \
--batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH