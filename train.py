from GPTv2 import *

# ---------------- Utilities ---------------- #
def load_ckpt(fpath, model, optim):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    return model, optim, checkpoint['epoch']

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- Training ---------------- #
if __name__ == "__main__":
    model = HelenaGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/{model_name}", exist_ok=True)

    cmd = input("Type checkpoint path to resume training, press Enter to train from scratch: ")
    if cmd:
        model, optimizer, start_epoch = load_ckpt(cmd, model, optimizer)

    print(f"Training started at {datetime.datetime.now().strftime('%H:%M')}")
    start = time.perf_counter()

    for iter in range(start_epoch, max_iters):
        if iter % 500 == 0:
            checkpoint = {
                "epoch": iter + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_path = f"checkpoints/{model_name}/{model_name}_{iter if iter % 5000 == 0 else 'overwritten'}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(model.state_dict(), f"models/{model_name}/{model_name}_{iter}.pth")

        xb, yb = get_batch('train')

        with torch.amp.autocast('cuda'):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(model.state_dict(), f"models/{model_name}/{model_name}_final.pth")
    print(f"Finished training at {datetime.datetime.now().strftime('%H:%M')}")
    print(f"Took: {(time.perf_counter() - start) * 1000:.2f} ms")

    # ---------------- Inference Loop ---------------- #
    while True:
        prompt = input("Type >>> ")
        if not prompt: break
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        output = model.generate(context, max_new_tokens=500, eos_token=encode("<|endoftext|>"))
        print(decode(output[0].tolist()))