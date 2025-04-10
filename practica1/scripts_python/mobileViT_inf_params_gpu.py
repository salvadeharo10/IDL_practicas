from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
import argparse
import time

# Argument parser para recibir parámetros desde el script de ejecución
parser = argparse.ArgumentParser(description="Entrenar MobileViT en GPU con parámetros configurables.")
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

profile_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True,
    with_flops = True
)
# Inicializar `Accelerator` para ejecutar en GPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Preparar el modelo para ejecución en GPU
model = accelerator.prepare(model)

# Mover las imágenes a la GPU
input_images = input_images.to(accelerator.device)

start_time = time.time()
# Perfilado del modelo en GPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

end_time = time.time()
total_time = end_time - start_time

# Imprimir resultados del perfilado
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f"Tiempo puro de inferencia: {total_time:.6f} segundos")