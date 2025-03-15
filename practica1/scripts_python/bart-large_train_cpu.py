from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import BartForSequenceClassification, BartTokenizer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load BART model and tokenizer
model_name = "facebook/bart-base"
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification example
tokenizer = BartTokenizer.from_pretrained(model_name)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Create a batch of random tokenized sentences (batch size 16, sequence length 512)
batch_size = 8
seq_length = 128
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length - 1))
eos_tokens = torch.full((batch_size, 1), tokenizer.eos_token_id)  # Añadir <eos> al final
input_ids = torch.cat([input_ids, eos_tokens], dim=1)
attention_mask = torch.ones_like(input_ids)  # Assume all tokens are attended to
labels = torch.randint(0, 2, (batch_size,))  # Random labels for binary classification

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define profiling kwargs for CPU activities
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Profile CPU activities instead of CUDA
    record_shapes=True
)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Prepare the model, optimizer, and data loader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Move model to training mode
model.train()

device = accelerator.device

# Training loop
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Print profiling results
print("Training completed.")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))