#!/bin/bash
#SBATCH -p cola01  # Usar la cola de solo CPU
#SBATCH --mem=0     # Sin l�mite de memoria (usar� toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una �nica tarea, pero sin l�mite de n�cleos
#SBATCH --job-name=TrainCPUMViT
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/train_mvit_cpu_GA_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es

# Par�metros de configuraci�n
BATCH_SIZE=4
IMG_SIZE=128
GRAD_ACCUM_STEPS=2

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
time apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/salvadordeharoo/IDL/practica1/scripts_python/config_cpubasebase.yaml /home/salvadordeharoo/IDL/practica1/scripts_python/mobileViT_train_params_cpu_GA.py \
--batch_size $BATCH_SIZE --img_size $IMG_SIZE --grad_accum_steps $GRAD_ACCUM_STEPS