#!/bin/bash
#SBATCH -p cola02  # Usar la cola con GPU
#SBATCH --gres=gpu:1  # Solicitar 1 GPU (ajustar si necesitas m�s)
#SBATCH --mem=0     # Sin l�mite de memoria (usar� toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una �nica tarea
#SBATCH --job-name=InfGPUMViT
#SBATCH --output=/home/miguelvidalg/IDL/practica1/slurm_outputs/inf_mvit_gpu_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
accelerate launch --config_file /home/miguelvidalg/IDL/practica1/scripts_python/config_gpubase.yaml \
/home/miguelvidalg/IDL/practica1/scripts_python/mobileViT-inf_gpu.py
