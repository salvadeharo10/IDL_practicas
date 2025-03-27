import argparse
from accelerate import Accelerator, ProfileKwargs
import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from utils import trace_handler_gpu

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Entrenamiento distribuido con MobileViT y Accelerate")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--img_size", type=int, default=256, help="Tamaño de imagen")
args = parser.parse_args()

# Cargar el modelo MobileViT
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(
    model_name, num_labels=10, ignore_mismatched_sizes=True
)

# Cargar el feature extractor
feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Configuración de datos
batch_size = args.batch_size
num_classes = 10
image_size = args.img_size
num_channels = 3

# Crear datos sintéticos
input_images = torch.rand(batch_size * 10, num_channels, image_size, image_size)
labels = torch.randint(0, num_classes, (batch_size * 10,))

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Configurar perfilado (opcional, puedes comentar si no lo necesitas)
profile_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    on_trace_ready=trace_handler_gpu
)

# Inicializar Accelerator
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Preparar modelo, optimizador y dataloader para entrenamiento distribuido
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Entrenamiento
model.train()
device = accelerator.device
num_epochs = 3
start_time = time.time()

with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images_batch, labels_batch in dataloader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images_batch).logits
            loss = criterion(outputs, labels_batch)

            # ? Usar accelerator.backward en lugar de loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

end_time = time.time()
total_training_time = end_time - start_time

# Resultados
if accelerator.is_main_process:
    print("Training completed.")
    print(f"Training time without initializations: {total_training_time:.2f} seconds.")
    # Puedes descomentar esta línea para imprimir el perfilado:
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
