from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
import argparse
import time

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Entrenar MobileViT en CPU con parámetros configurables.")
parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote (batch size)")
parser.add_argument("--img_size", type=int, default=256, help="Longitud de la secuencia")

args = parser.parse_args()

# Cargar el modelo MobileViT y el feature extractor
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(model_name)
feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)

# Configuración del tamaño del lote e imágenes aleatorias (batch size 128, 3 canales RGB, 256x256)
batch_size = args.batch_size
image_size = args.img_size  # Tamaño de imagen esperado por MobileViT
num_channels = 3  # Imágenes RGB

# Crear imágenes aleatorias (valores normalizados entre 0 y 1)
input_images = torch.rand(batch_size, num_channels, image_size, image_size)

# Definir los kwargs para el perfilado en CPU
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Registrar solo CPU
    record_shapes=True,
    profile_memory=True,
    with_flops = True
)

# Inicializar `Accelerator` para ejecutar en CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar el modelo para ejecución en CPU
model = accelerator.prepare(model)

# Mover las imágenes a la CPU
input_images = input_images.to(accelerator.device)

start_time = time.time()
# Perfilado del modelo en CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

end_time = time.time()
total_time = end_time - start_time

# Imprimir resultados del perfilado
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
print(f"Tiempo puro de inferencia: {total_time:.6f} segundos")