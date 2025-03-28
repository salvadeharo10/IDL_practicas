import argparse
from accelerate import Accelerator, ProfileKwargs
import time
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import get_trace_handler_gpu

# Argument parser
parser = argparse.ArgumentParser(description="Entrenar RoBERTa con Gradient Accumulation.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--seq_length", type=int, default=512, help="Longitud de la secuencia")
parser.add_argument("--accum_steps", type=int, default=4, help="Número de pasos para acumular gradientes")

args = parser.parse_args()

# Cargar modelo y tokenizer
model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Definir pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Crear datos sintéticos
batch_size = args.batch_size
seq_length = args.seq_length
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length - 1))
pad_tokens = torch.full((batch_size, 1), tokenizer.pad_token_id)
input_ids = torch.cat([input_ids, pad_tokens], dim=1)
attention_mask = (input_ids != tokenizer.pad_token_id).long()
labels = torch.randint(0, 2, (batch_size,))

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

trace_handler = get_trace_handler_gpu(model_name)

# Configurar perfilado
profile_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    on_trace_ready=trace_handler
)

# Inicializar Accelerator con acumulación de gradientes
accelerator = Accelerator(
    cpu=False,
    kwargs_handlers=[profile_kwargs],
    gradient_accumulation_steps=args.accum_steps
)

# Preparar todo para ejecución con Accelerator
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Entrenamiento
model.train()
device = accelerator.device

num_epochs = 16
start_time = time.time()
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids_batch, attention_mask_batch, labels_batch = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device)
            )

            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
                loss = criterion(outputs, labels_batch)
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

end_time = time.time()
total_training_time = end_time - start_time

print("Training completed.")
print(f"Training time without initializations: {total_training_time:.2f} seconds.")
