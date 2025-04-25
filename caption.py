# File: vocab.py
```python
import json
from collections import Counter

class Vocabulary:
    """
    Build a simple word2idx and idx2word mapping.
    """
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return text.lower().strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [self.stoi.get(word, self.stoi['<UNK>']) for word in tokenized]
```


# File: dataset.py
```python
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import json

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        """
        root_dir: path to Flickr images
        captions_file: path to JSON with {"image_id": ["caption1", ...]}
        vocab: Vocabulary object
        transform: torchvision transforms
        """
        self.root = root_dir
        self.vocab = vocab
        self.transform = transform
        # load captions
        with open(captions_file, 'r') as f:
            data = json.load(f)
        self.image_ids = list(data.keys())
        self.captions = data

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        captions = self.captions[img_id]
        # pick the first caption (or randomize)
        caption = captions[0]

        img_path = os.path.join(self.root, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        numericalized = [self.vocab.stoi['<START>']]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi['<END>'])
        return image, torch.tensor(numericalized)

# Collate function for padding
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions
```


# File: model.py
```python
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-3]  # up to conv4_x
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.init_h = nn.Linear(1024, hidden_size)

    def forward(self, images):
        feats = self.backbone(images)               # (B,1024,7,7)
        B, C, H, W = feats.shape
        flat = feats.view(B, C, H*W).permute(0,2,1) # (B,49,1024)
        pooled = self.pool(feats).view(B, C)        # (B,1024)
        init_hidden = self.init_h(pooled)           # (B,hidden_size)
        return flat, init_hidden

class Attention(nn.Module):
    def __init__(self, feat_dim, hid_dim, attn_dim):
        super().__init__()
        self.v_proj = nn.Linear(feat_dim, attn_dim)
        self.h_proj = nn.Linear(hid_dim, attn_dim)
        self.score  = nn.Linear(attn_dim, 1)

    def forward(self, feats, hidden):
        # feats: (B,N,feat_dim), hidden: (B,hid_dim)
        proj_feats = self.v_proj(feats)                # (B,N,attn_dim)
        proj_hidden= self.h_proj(hidden).unsqueeze(1)  # (B,1,attn_dim)
        e = torch.tanh(proj_feats + proj_hidden)       # (B,N,attn_dim)
        scores = self.score(e).squeeze(-1)             # (B,N)
        alpha  = torch.softmax(scores, dim=1)          # (B,N)
        context= (alpha.unsqueeze(-1) * feats).sum(dim=1)  # (B,feat_dim)
        return context, alpha

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, feat_dim, attn_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(feat_dim, hid_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim+feat_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, init_hidden, captions):
        B, N, D = feats.size()
        T = captions.size(1) - 1
        outputs = torch.zeros(B, T, self.fc_out.out_features).to(feats.device)
        h, c = init_hidden, torch.zeros_like(init_hidden)
        for t in range(T):
            emb = self.dropout(self.embedding(captions[:, t]))  # (B,embed_dim)
            context, _ = self.attention(feats, h)               # (B,feat_dim)
            lstm_in = torch.cat([emb, context], dim=1)
            h, c = self.lstm(lstm_in, (h, c))                   # (B,hid_dim)
            outputs[:, t, :] = self.fc_out(self.dropout(h))     # (B,vocab_size)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hid_dim=512, feat_dim=1024, attn_dim=256):
        super().__init__()
        self.encoder = EncoderCNN(hid_dim)
        self.decoder = DecoderRNN(vocab_size, embed_dim, hid_dim, feat_dim, attn_dim)

    def forward(self, images, captions):
        feats, init_hidden = self.encoder(images)
        outputs = self.decoder(feats, init_hidden, captions)
        return outputs
```


# File: train.py
```python
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from torchvision import transforms

from vocab import Vocabulary
from dataset import FlickrDataset, collate_fn
from model import ImageCaptioningModel

# Hyperparameters
BATCH_SIZE = 32
EMBED_DIM  = 512
HID_DIM    = 512
ATTN_DIM   = 256
LR         = 4e-4
NUM_EPOCHS = 20

# Paths (set appropriately)
IMG_ROOT      = '/path/to/flickr/images'
CAPTIONS_FILE = '/path/to/flickr/captions.json'

# 1) Build vocabulary from all captions
with open(CAPTIONS_FILE, 'r') as f:
    data = json.load(f)
sentences = [cap for caps in data.values() for cap in caps]
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(sentences)

# 2) Data transforms and loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
dataset = FlickrDataset(IMG_ROOT, CAPTIONS_FILE, vocab, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, collate_fn=collate_fn)

# 3) Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageCaptioningModel(len(vocab), EMBED_DIM, HID_DIM, feat_dim=1024, attn_dim=ATTN_DIM)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 4) Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions)           # (B, T, V)
        B, T, V = outputs.shape
        preds = outputs.view(-1, V)
        targs = captions[:, 1:].contiguous().view(-1)
        loss  = criterion(preds, targs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Save checkpoint
torch.save(model.state_dict(), 'caption_model_flickr.pth')