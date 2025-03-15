from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Cargar el modelo BART y el tokenizer
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Definir ejemplos de entrada para inferencia
textos = [
    "The history of artificial intelligence dates back to ancient times.",
    "Machine learning is a subset of artificial intelligence focused on data-driven learning."
]

# Tokenizar los textos de entrada
inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Definir los kwargs para el perfilado en GPU
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Registrar actividades en GPU
    record_shapes=True
)

# Inicializar `Accelerator` para ejecución en GPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Preparar el modelo para ejecución en GPU
model = accelerator.prepare(model)

# Mover los datos de entrada a la GPU
inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

# Inferencia con perfilado en GPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

# Decodificar y mostrar los resultados
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, text in enumerate(decoded_outputs):
    print(f"Generated text {i+1}: {text}")

# Imprimir resultados del perfilado
print("Inference completed.")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))