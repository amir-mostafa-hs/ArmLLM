import math

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

# Transformer implementation from scratch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # [B, H, L, L]
        weights = self.softmax(scores)
        output = torch.matmul(weights, V)  # [B, H, L, d_k]
        return output

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections + reshape to [B, H, L, d_k]
        Q = (
            self.q_linear(Q)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(K)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(V)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        out = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        # Final projection
        return self.out_proj(out)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-head attention block with residual
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward block with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x  # Shape: [batch_size, seq_len, d_model]


class TransformerEncoder(nn.Module):
    def __init__(
        self, img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.num_patches)

        # stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def patchify(self, images):
        # images shape: [batch_size, channels, height, width]
        batch_size = images.shape[0]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches  # Shape: [batch_size, num_patches, patch_dim]

    def forward(self, x):
        x = self.patchify(x)  # [B, N_patches, patch_dim]
        x = self.patch_embedding(x)  # [B, N_patches, d_model]
        x = self.positional_encoding(x)  # [B, N_patches, d_model]

        for layer in self.encoder_layers:
            x = layer(x)  # [B, N_patches, d_model]

        x = self.norm(x)  # [B, N_patches, d_model]
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)  # [B, num_classes]


# Data loading and preprocessing
def load_and_preprocess_data():
    # Load the full dataset and split it
    dataset = (
        load_dataset("microsoft/cats_vs_dogs", trust_remote_code=True, split="train")
        .shuffle()
        .take(1000)
    )
    train_dataset = dataset.train_test_split(test_size=0.1)  # 10% for validation

    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224", use_fast=True
    )

    def preprocess_batch(examples):
        """Process a batch of images, skipping corrupted ones."""
        pixel_values = []
        labels = []

        for i in range(len(examples["image"])):
            try:
                if examples["image"][i] is not None:
                    inputs = image_processor(examples["image"][i], return_tensors="pt")
                    pixel_values.append(inputs.pixel_values.squeeze(0))
                    labels.append(examples["labels"][i])
            except Exception as e:
                # Skip corrupted images
                print(f"Skipping corrupted image at index {i}: {e}")
                continue

        return {"pixel_values": pixel_values, "label": labels}

    # Process datasets with batched processing
    train_dataset["train"] = train_dataset["train"].map(
        preprocess_batch,
        remove_columns=["image", "labels"],
        batched=True,
        batch_size=100,
    )
    train_dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])

    train_dataset["test"] = train_dataset["test"].map(
        preprocess_batch,
        remove_columns=["image", "labels"],
        batched=True,
        batch_size=100,
    )
    train_dataset["test"].set_format(type="torch", columns=["pixel_values", "label"])

    return train_dataset["train"], train_dataset["test"]


def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0

    for batch in dataloader:
        inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Training loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    img_size = 224
    patch_size = 16
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    num_classes = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001

    # Load and preprocess data
    train_data, validation_data = load_and_preprocess_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )

    # Initialize model
    model = TransformerEncoder(
        img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        validate(model, validation_loader, criterion, device)


if __name__ == "__main__":
    main()
