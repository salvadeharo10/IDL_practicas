#!/bin/bash
#SBATCH -p cola01  # Usar la cola de solo CPU
#SBATCH --mem=0     # Sin l�mite de memoria (usar� toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una �nica tarea, pero sin l�mite de n�cleos
#SBATCH --job-name=TrainCPURob
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/train_roberta_cpu_%j.out

# Par�metros de configuraci�n
BATCH_SIZE=16
SEQ_LENGTH=256

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
time apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/salvadordeharoo/IDL/practica1/scripts_python/config_cpubase.yaml /home/salvadordeharoo/IDL/practica1/scripts_python/roberta-base_train_params_cpu.py \
--batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH