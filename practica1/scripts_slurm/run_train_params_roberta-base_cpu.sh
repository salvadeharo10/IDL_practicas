#!/bin/bash  
#SBATCH -p cola01  # Usar la cola de solo CPU
#SBATCH --mem=0     # Sin límite de memoria (usará toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una única tarea, pero sin límite de núcleos
#SBATCH --job-name=TrainCPURob
#SBATCH --output=/home/miguelvidalg/IDL/practica1/slurm_outputs/train_roberta_cpu_ipex%j.out

# Parámetros de configuración 
BATCH_SIZE=8
SEQ_LENGTH=256

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
time apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/miguelvidalg/IDL/practica1/scripts_python/cpu_config_ipex.yaml /home/miguelvidalg/IDL/practica1/scripts_python/roberta-base_train_params_cpu.py \
--batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH