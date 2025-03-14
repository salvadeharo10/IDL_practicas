from accelerate import Accelerator, ProfileKwargs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

# Cargar el modelo MobileViT
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(
    model_name, num_labels=10, ignore_mismatched_sizes=True
)

# Cargar el feature extractor (preprocesador de imágenes)
feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Configuración de datos de entrenamiento
batch_size = 16
num_classes = 10
image_size = 256  # Tamaño de imagen esperado por MobileViT
num_channels = 3  # Imágenes RGB

# Crear imágenes aleatorias y etiquetas aleatorias
input_images = torch.rand(batch_size * 10, num_channels, image_size, image_size)  # 5 lotes de batch_size imágenes
labels = torch.randint(0, num_classes, (batch_size * 10,))  # Etiquetas aleatorias de 10 clases

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Configurar el perfilado de GPU
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Registrar actividades de GPU
    record_shapes=True
)

# Inicializar Accelerator para optimizar en GPU
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
        for images_batch, labels_batch in dataloader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images_batch).logits  # Obtener predicciones
            loss = criterion(outputs, labels_batch)  # Calcular pérdida

            loss.backward()
            optimizer.step()  # Actualización de pesos

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# Imprimir resultados del perfilado
print("Training completed.")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))