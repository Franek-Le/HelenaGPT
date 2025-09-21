import torch
from GPTv2 import BigramLanguageModel, encode, decode

model = BigramLanguageModel()
model.load_state_dict(torch.load("models/HelenaGPTv3_3/HelenaGPTv3_3_final.pth", weights_only=True))
m = model.to('cuda')

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

while True:
    prompt = input("Type >>> ")
    if not prompt: break
    context = torch.tensor(encode(prompt), dtype=torch.long, device='cuda').unsqueeze(0)
    output = model.generate(context, max_new_tokens=500, eos_token=encode("<|endoftext|>"))
    print(decode(output[0].tolist()).replace("<|endoftext|>", ""))