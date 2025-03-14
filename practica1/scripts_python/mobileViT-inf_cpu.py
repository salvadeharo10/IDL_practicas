from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

# Cargar el modelo MobileViT y el feature extractor
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(model_name)
feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)

# Configuración del tamaño del lote e imágenes aleatorias (batch size 128, 3 canales RGB, 256x256)
batch_size = 128
image_size = 256  # Tamaño de imagen esperado por MobileViT
num_channels = 3  # Imágenes RGB

# Crear imágenes aleatorias (valores normalizados entre 0 y 1)
input_images = torch.rand(batch_size, num_channels, image_size, image_size)

# Definir los kwargs para el perfilado en CPU
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Registrar actividades de CPU
    record_shapes=True
)

# Inicializar `Accelerator` para ejecutar en CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar el modelo para ejecución en CPU
model = accelerator.prepare(model)

# Mover las imágenes a la CPU
input_images = input_images.to(accelerator.device)

# Perfilado del modelo en CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

# Imprimir resultados del perfilado
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
