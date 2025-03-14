from accelerate import Accelerator, ProfileKwargs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

# Cargar el modelo con la opción para evitar errores de tamaño
'''
El modelo apple/mobilevit-small fue preentrenado con 1000 clases (como en ImageNet).
En el código, estamos intentando cargarlo con 10 clases (num_labels=10).
Poir tanto, al cargar el modelo preentrenado, Los pesos preentrenados de classifier.weight y
classifier.bias no coinciden con las dimensiones esperadas.
Lo que hacemos es ignorar el error al cargar los pesos,
permitiendo que los últimos pesos de la clasificación se
inicialicen aleatoriamente.
'''
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(
    model_name, num_labels=10, ignore_mismatched_sizes=True
)

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
input_images = torch.rand(batch_size * 10, num_channels, image_size, image_size)  # 10 lotes de batch_size imágenes
labels = torch.randint(0, num_classes, (batch_size * 10,))  # Etiquetas aleatorias de 10 clases

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Configurar el perfilado de CPU
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Registrar solo CPU
    record_shapes=True
)

# Inicializar Accelerator para optimizar en CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar modelo, optimizador y dataloader para CPU
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
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

