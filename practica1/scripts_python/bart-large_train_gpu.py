from accelerate import Accelerator, ProfileKwargs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartTokenizer, BartForSequenceClassification

# Cargar el modelo BART para clasificación de texto en GPU
model_name = "facebook/bart-large"
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Cargar el tokenizer de BART
tokenizer = BartTokenizer.from_pretrained(model_name)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Crear datos de entrada simulados (batch de frases aleatorias)
batch_size = 16
seq_length = 128  # Longitud máxima de secuencia
num_samples = batch_size * 5  # Total de ejemplos en el dataset

# Simular textos de ejemplo (con sentido)
input_texts = [
    "Artificial intelligence is evolving rapidly.",
    "Neural networks have transformed deep learning.",
    "BART is a powerful model for NLP tasks.",
    "This is a sample text for testing purposes.",
    "The history of AI dates back to ancient times."
] * (num_samples // 5)  # Repetimos para llenar el dataset

# Tokenizar los textos con padding y truncado, asegurando la presencia de [EOS]
inputs = tokenizer(
    input_texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length, add_special_tokens=True
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.randint(0, 2, (num_samples,))  # Etiquetas aleatorias para clasificación binaria

# Crear dataset y dataloader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Configurar el perfilado de GPU
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Registrar uso de GPU
    record_shapes=True
)

# Inicializar `Accelerator` para entrenar en GPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Preparar modelo, optimizador y dataloader para GPU
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Mover el modelo a modo entrenamiento
model.train()
device = accelerator.device

# Bucle de entrenamiento
num_epochs = 3
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
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# Imprimir resultados del perfilado en GPU
print("Training completed.")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))