import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from vocab import Vocabulary
from dataset import FlickrDataset, collate_fn
from model import ImageCaptioningModel


def train(args):
    # 1) Build vocabulary
    with open(args.captions_file, 'r') as f:
        data = json.load(f)
    sentences = [cap for caps in data.values() for cap in caps]
    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(sentences)

    # 2) Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    dataset = FlickrDataset(
        args.image_root,
        args.captions_file,
        vocab,
        transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # 3) Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hid_dim=args.hid_dim,
        feat_dim=1024,
        attn_dim=args.attn_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4) Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(images, captions)  # (B, T, V)
            B, T, V = outputs.shape
            preds = outputs.view(-1, V)
            targs = captions[:, 1:].contiguous().view(-1)
            loss = criterion(preds, targs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

    # 5) Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning Training')
    parser.add_argument('--image_root', type=str, required=True,
                        help='path to directory with Flickr images')
    parser.add_argument('--captions_file', type=str, required=True,
                        help='path to JSON file containing image captions')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--freq_threshold', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='caption_model_flickr.pth')
    args = parser.parse_args()

    train(args)