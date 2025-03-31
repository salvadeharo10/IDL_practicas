import argparse
from accelerate import Accelerator, ProfileKwargs
import time
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import get_trace_handler_gpu

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Entrenar RoBERTa en CPU con parámetros configurables.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--seq_length", type=int, default=512, help="Longitud de la secuencia")

args = parser.parse_args()

# Load RoBERTa model and tokenizer
model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification example
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Create a batch of random tokenized sentences (batch size 16, sequence length 512)
batch_size = args.batch_size
seq_length = args.seq_length
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length - 1))
pad_tokens = torch.full((batch_size, 1), tokenizer.pad_token_id)  # Añadir <pad> al final
input_ids = torch.cat([input_ids, pad_tokens], dim=1)
attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Máscara de atención válida
labels = torch.randint(0, 2, (batch_size,))  # Random labels for binary classification

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

trace_handler = get_trace_handler_gpu(model_name)

# Configurar el perfilado de GPU
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Registrar actividades de GPU
    record_shapes=True,
    profile_memory=True,
    with_flops = True,
    on_trace_ready = trace_handler
)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Prepare the model, optimizer, and data loader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Move model to training mode
model.train()

device = accelerator.device

# Training loop
num_epochs = 8
start_time = time.time()
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_ids_batch, attention_mask_batch, labels_batch in dataloader:
            input_ids_batch, attention_mask_batch, labels_batch = (
                input_ids_batch.to(device),
                attention_mask_batch.to(device),
                labels_batch.to(device)
            )
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

end_time = time.time()
total_training_time = end_time - start_time

# Print profiling results
print("Training completed.")
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f"Training time without initializations: {total_training_time:.2f} seconds.")