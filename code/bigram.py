import torch
import torch.nn as nn
import torch.nn.functional as F

train_size_p = 0.9
block_size = 16
batch_size = 64
learning_rate = 1e-3
n_steps = 10000
eval_step = 1000
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('./data/bible.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

ctoi = {c : i for i, c in enumerate(chars)}
itoc = {i : c for i, c in enumerate(chars)}
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i.item()] if type(i) is torch.Tensor else itoc[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * train_size_p)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for eval_i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[eval_i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.embeddings(idx) # batch x block x vocab_size

        if target is None:
            loss = None
        else:
            logits = logits.view(batch_size * block_size, vocab_size)
            target = target.view(batch_size * block_size)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # batch x block x vocab_size
            logits = logits[:, -1, :] # batch x vocab_size
            probs = F.softmax(logits, dim=-1) # batch x vocab_size
            idx_next = torch.multinomial(probs, num_samples=1) # batch x 1
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(n_steps):
    if step % eval_step == 0:
        out = estimate_loss()
        print(f'Train loss {out["train"]}, Validate loss {out["val"]}')

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))