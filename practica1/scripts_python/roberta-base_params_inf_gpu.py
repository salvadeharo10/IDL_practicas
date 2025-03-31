from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import RobertaModel, RobertaTokenizer
import argparse

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Inferencia Roberta en GPU con parámetros configurables.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--seq_length", type=int, default=512, help="Longitud de la secuencia")

args = parser.parse_args()

# Load RoBERTa model and tokenizer
model_name = "roberta-base"
model = RobertaModel.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Create a large batch of random long sentences
batch_size = args.batch_size  # Reduced for CPU efficiency
seq_length = args.seq_length

# Generate random token IDs within the model's vocabulary size
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length - 1))
pad_tokens = torch.full((batch_size, 1), tokenizer.pad_token_id)  # Añadir <pad> al final
input_ids = torch.cat([input_ids, pad_tokens], dim=1)
attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Máscara de atención válida

# Define profiling kwargs for CPU activities
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Profile CPU activities instead of CUDA
    record_shapes=True
)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Prepare the model for CPU execution
model = accelerator.prepare(model)

# Move inputs to CPU
input_ids = input_ids.to(accelerator.device)
attention_mask = attention_mask.to(accelerator.device)

# Profile the model execution on the CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))