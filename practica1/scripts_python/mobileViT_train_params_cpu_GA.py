# -*- coding: utf-8 -*-
import argparse
import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from accelerate import Accelerator, ProfileKwargs
from torch.profiler import ProfilerActivity  # Necesario para activities
from utils import get_trace_handler_cpu

# Argument parser para recibir parametros desde el script de ejecucion
parser = argparse.ArgumentParser(description="Entrenar MobileViT en CPU con acumulacion de gradiente.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamano del lote (batch size)")
parser.add_argument("--img_size", type=int, default=256, help="Tamano de imagen (image size)")
parser.add_argument("--grad_accum_steps", type=int, default=2, help="Numero de pasos de acumulacion de gradiente")
args = parser.parse_args()

# Cargar modelo con adaptacion al numero de clases
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(
    model_name, num_labels=10, ignore_mismatched_sizes=True
)
feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)

# Funcion de perdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Datos sinteticos de entrada
batch_size = args.batch_size
num_classes = 10
image_size = args.img_size
num_channels = 3
num_images = 128

input_images = torch.rand(num_images, num_channels, image_size, image_size)
labels = torch.randint(0, num_classes, (num_images,))
dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Perfilador
trace_handler = get_trace_handler_cpu("mobileViT")
profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    on_trace_ready=trace_handler
)

# Inicializar accelerator con perfilador y CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs], gradient_accumulation_steps=args.grad_accum_steps)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Entrenamiento
num_epochs = 3
model.train()

start_time = time.time()
with accelerator.profile():
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (images_batch, labels_batch) in dataloader:
            with accelerator.accumulate(model):
                outputs = model(images_batch).logits
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

