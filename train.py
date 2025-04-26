import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from vocab import Vocabulary
from dataset import FlickrDataset, collate_fn
from model import ImageCaptioningModel
import json
import pickle
from pred import generate_caption, preprocess_image

# Hyperparameters
BATCH_SIZE = 32
EMBED_DIM  = 512
HID_DIM    = 512
ATTN_DIM   = 256
LR         = 4e-4
NUM_EPOCHS = 25
max_samples = 8080
UNFREEZE_POINT = 10

# Paths (set appropriately)
IMG_ROOT      = 'C:/Users/shubh/Desktop/Workspace/Image caption generator/Flicker8k_Dataset/'
CAPTIONS_FILE = 'C:/Users/shubh/Desktop/Workspace/Image caption generator/flickr8k_captions.json'


from multiprocessing import freeze_support

def main():
    # all of your current top-level code:
    #  - build vocab
    #  - create transforms/dataset/dataloader
    #  - instantiate model, loss, optimizer
    #  - training loop
    # 1) Build vocabulary from all captions
    with open(CAPTIONS_FILE, 'r') as f:
        data = json.load(f)
    sentences = [cap for caps in data.values() for cap in caps]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(sentences)


    #Pretrained embedding model
    GLOVE_PATH = "embedding/glove.6B.300d.txt"
    EMBED_DIM  = 300

    print("Loading GloVe vectors…")
    glove = {}
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word, vec = parts[0], np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"  loaded {len(glove)} tokens")

    # 2) Build an embedding matrix aligned with vocab.stoi
    vocab_size = len(vocab)
    matrix = np.random.normal(
        scale=0.6, size=(vocab_size, EMBED_DIM)
    ).astype(np.float32)

    for token, idx in vocab.stoi.items():
        if token in glove:
            matrix[idx] = glove[token]
        # else: leave the random init

    pretrained_embeddings = torch.from_numpy(matrix)



    # 2) Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    dataset = FlickrDataset(IMG_ROOT, CAPTIONS_FILE, vocab, transform)
    #### data subset for testing flow - pick max_samples = N for the number to consider
    # dataset = Subset(dataset, list(range(max_samples)))
    n = len(dataset)
    if max_samples is not None:
        k = min(max_samples, n)
        dataset = Subset(dataset, list(range(k)))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, collate_fn=collate_fn)
    
    for images, captions in dataloader:
        print("Loaded batch:", images.shape, captions.shape)
        break

    # 3) Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ImageCaptioningModel(len(vocab), EMBED_DIM, HID_DIM, feat_dim=1024, attn_dim=ATTN_DIM)
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hid_dim=HID_DIM,
        feat_dim=1024,
        attn_dim=ATTN_DIM,
        pretrained_emb=pretrained_embeddings,
        freeze_emb=False
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4) Training loop
    for epoch in range(NUM_EPOCHS):

        #embedding unfreeze point
        if epoch == UNFREEZE_POINT + 1:
            print(f">>> Epoch {epoch}: unfreezing embedding layer")
            for param in model.decoder.embedding.parameters():
                param.requires_grad = True
            # re-create optimizer so embeddings are now included
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

        #quality check every epoch
        img_tensor = preprocess_image(r'C:\Users\shubh\Desktop\Workspace\Image caption generator\input\WhatsApp Image 2025-03-31 at 1.53.57 PM.jpeg', device)
        cap = generate_caption(model,
                                vocab,
                                img_tensor,
                                device,
                                max_len=20)
        print(f" Test caption @ epoch {epoch}: {cap}")

        if epoch%5==0:
            torch.save(model.state_dict(), f'model/caption_model_flickr_{max_samples}_ep{epoch}.pth')

    # Save checkpoint
    torch.save(model.state_dict(), f'model/caption_model_flickr_{max_samples}.pth')


    # … after training is done …
    vocab_file = "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({
            "freq_threshold": vocab.freq_threshold,
            "itos": vocab.itos,
            "stoi": vocab.stoi
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_file}")


    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Saved full Vocabulary object to vocab.pkl")



if __name__ == "__main__":
    freeze_support()           # on Windows, safe-guard for spawn
    main()
