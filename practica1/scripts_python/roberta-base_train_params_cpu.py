import argparse
import torch
import time
from accelerate import Accelerator, ProfileKwargs
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import get_trace_handler_cpu

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Entrenar RoBERTa en CPU con parámetros configurables.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--seq_length", type=int, default=512, help="Longitud de la secuencia")

args = parser.parse_args()



# Cargar el modelo y tokenizer de RoBERTa
model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Crear un lote de datos aleatorios con la longitud y tamaño de batch especificados
input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_length - 1))
pad_tokens = torch.full((args.batch_size, 1), tokenizer.pad_token_id, dtype=torch.int64)  # Añadir <pad> al final
input_ids = torch.cat([input_ids, pad_tokens], dim=1)
attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Máscara de atención válida
labels = torch.randint(0, 2, (args.batch_size,), dtype=torch.int64)  # Etiquetas binarias

# Crear el dataset y DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

trace_handler = get_trace_handler_cpu(model_name)

# Configurar profiling para CPU
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Registrar solo CPU
    record_shapes=True,
    profile_memory=True,
    with_flops = True,
    on_trace_ready = trace_handler
)

# Inicializar Accelerator para CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar modelo, optimizador y dataloader para ejecución en CPU
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Modo de entrenamiento
model.train()
device = accelerator.device

# Bucle de entrenamiento
num_epochs = 3
# Medir el tiempo total de entrenamiento
start_time = time.time()
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_ids_batch, attention_mask_batch, labels_batch in dataloader:
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

end_time = time.time()
total_training_time = end_time - start_time

# Mostrar resultados del perfilado
print("Training completed.")
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# Imprimir resultados del tiempo de entrenamiento
print(f"Training time without initializations: {total_training_time:.2f} seconds.")