#!/bin/bash
#SBATCH -p cola01  # Usar la cola de solo CPU    
#SBATCH --mem=0     # Sin límite de memoria (usará toda la disponible)
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Mantener una única tarea, pero sin límite de núcleos
#SBATCH --job-name=InfCPURob
#SBATCH --output=/home/miguelvidalg/IDL/practica1/slurm_outputs/inf_roberta_cpu_mixed_%j.out
 
# Parámetros de configuración
BATCH_SIZE=8
SEQ_LENGTH=256  

# Ejecutar el script de Python dentro del contenedor Singularity (Apptainer)
time apptainer exec --writable-tmpfs /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
accelerate launch --config_file /home/miguelvidalg/IDL/practica1/scripts_python/config_cpubase.yaml /home/miguelvidalg/IDL/practica1/scripts_python/roberta-base_params_inf_cpu.py \
--batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH