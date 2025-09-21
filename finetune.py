import torch
import os
import time
import datetime
import tiktoken
from GPTv2 import BigramLanguageModel
from tests.LLMv2.GPTv2 import start_epoch

# ---------------- Hyperparameters ---------------- #
batch_size = 32
block_size = 128
finetune_iters = 10000
eval_interval = 500
learning_rate = 1e-5
eval_iters = 200
model_name = "HelenaGPTv3_3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# ---------------- Load Dataset ---------------- #
with open("dataset/finetuning/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------------- Utilities ---------------- #
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- Finetuning ---------------- #
if __name__ == "__main__":
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Loading checkpoint...")
    ckpt_path = "checkpoints/HelenaGPTv3_3/HelenaGPTv3_3_overwritten.pt"  # Change if needed
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {ckpt_path}")

    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/{model_name}", exist_ok=True)

    print(f"Finetuning started at {datetime.datetime.now().strftime('%H:%M')}")
    start = time.perf_counter()

    for iter in range(start_epoch, start_epoch + finetune_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"[{iter}] train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')

        with torch.amp.autocast('cuda'):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(model.state_dict(), f"models/{model_name}/{model_name}_finetuned_v2.pth")
    print(f"Finetuning complete. Final model saved.")
    print(f"Duration: {(time.perf_counter() - start) / 60:.2f} minutes")
