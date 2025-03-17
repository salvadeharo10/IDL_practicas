#!/bin/bash
#SBATCH -p cola01  # Usar la cola de solo CPU
#SBATCH --mem=0     # Sin límite de memoria (usará toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una única tarea, pero sin límite de núcleos
#SBATCH --job-name=InfCPUMViT
#SBATCH --output=/home/miguelvidalg/IDL/practica1/slurm_outputs/inf_mvit_%j.out
#SBATCH --mail-type=ALL   # Notificaciones por correo en inicio, fin y fallos
#SBATCH --mail-user=salvadorde.haroo@um.es


# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
accelerate launch --config_file /home/miguelvidalg/IDL/practica1/scripts_python/config_cpubase.yaml /home/miguelvidalg/IDL/practica1/scripts_python/mobileViT-inf_cpu.py