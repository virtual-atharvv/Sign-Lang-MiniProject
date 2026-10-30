import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ISLGrammarDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=64, max_target_len=64):
        self.inputs = df["input_text"].tolist()
        self.targets = df["target_text"].tolist()
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        model_inputs = self.tokenizer(
            input_text, max_length=self.max_input_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            target_text, max_length=self.max_target_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        labels_ids = labels["input_ids"].squeeze(0)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels_ids
        }

def train_t5(csv_path: str, output_dir: str, epochs: int = 5, batch_size: int = 16, lr: float = 5e-4):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training T5-small on {device}...")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = ISLGrammarDataset(train_df, tokenizer)
    val_dataset = ISLGrammarDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/grammar_pairs/isl_grammar_pairs.csv")
    parser.add_argument("--out", default="models/grammar_correction/t5_isl_checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if os.path.exists(args.csv):
        train_t5(args.csv, args.out, epochs=args.epochs)
    else:
        print(f"CSV dataset not found at {args.csv}. Run build_grammar_pairs.py first.")
