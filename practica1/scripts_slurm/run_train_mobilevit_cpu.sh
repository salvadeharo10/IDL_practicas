#!/bin/bash
#SBATCH -p cola01  # Usar la cola de solo CPU
#SBATCH --mem=0     # Sin l�mite de memoria (usar� toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una �nica tarea, pero sin l�mite de n�cleos
#SBATCH --job-name=TrainCPUMViT
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/train_mvit_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es


# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
accelerate launch --config_file /home/salvadordeharoo/IDL/practica1/scripts_python/config_cpubase.yaml /home/salvadordeharoo/IDL/practica1/scripts_python/mobileViT-train_cpu.py