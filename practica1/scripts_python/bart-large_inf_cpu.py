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

# Definir los kwargs para el perfilado en CPU
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Registrar actividades en CPU
    record_shapes=True
)

# Inicializar `Accelerator` para ejecución en CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar el modelo para ejecución en CPU
model = accelerator.prepare(model)

# Mover los datos de entrada a la CPU
inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

# Inferencia con perfilado en CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

# Decodificar y mostrar los resultados
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, text in enumerate(decoded_outputs):
    print(f"Generated text {i+1}: {text}")

# Imprimir resultados del perfilado
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))