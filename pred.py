# predict.py

import argparse
import json
import os
from PIL import Image

import torch
from torchvision import transforms

from vocab import Vocabulary
from model import ImageCaptioningModel

def load_vocab(vocab_file):
    """
    Load vocab mappings from the JSON file you saved in training.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = Vocabulary(freq_threshold=data.get("freq_threshold", 5))
    # itos keys were saved as strings in JSON; cast back to int
    vocab.itos = {int(k):v for k, v in data["itos"].items()}
    # stoi maps token->index
    vocab.stoi = {k:int(v) for k, v in data["stoi"].items()}
    return vocab

def load_model(model_path, device, vocab_size, embed_dim, hid_dim, feat_dim, attn_dim):
    """
    Instantiate model architecture, load state_dict.
    """
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hid_dim=hid_dim,
        feat_dim=feat_dim,
        attn_dim=attn_dim
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # if you saved model.state_dict(), use model.load_state_dict(...)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(image_path, device):
    """
    Apply the same transforms you used in training.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # (1,3,224,224)
    return img

def generate_caption(model, vocab, image_tensor, device, max_len=20):
    """
    Greedy decode: start with <START>, then feed back last predicted token.
    """
    # 1. Encode image
    with torch.no_grad():
        feats, h = model.encoder(image_tensor)     # feats: (1,49,1024); h: (1, hid_dim)
        c = torch.zeros_like(h)                    # init cell state

        # 2. Start token
        start_idx = vocab.stoi["<START>"]
        end_idx   = vocab.stoi["<END>"]
        input_token = torch.tensor([start_idx], device=device)  # (1,)

        generated = []
        for _ in range(max_len):
            # embed current token
            emb = model.decoder.embedding(input_token)           # (1, embed_dim)
            # attention context
            context, _ = model.decoder.attention(feats, h)       # (1, feat_dim)
            # LSTMCell step
            lstm_input = torch.cat([emb, context], dim=1)        # (1, E+feat_dim)
            h, c = model.decoder.lstm(lstm_input, (h, c))        # each (1, hid_dim)
            # project to vocab
            logits = model.decoder.fc_out(model.decoder.dropout(h))  # (1, vocab_size)
            # greedy pick
            next_idx = logits.argmax(dim=1)                       # (1,)
            idx = next_idx.item()
            if idx == end_idx:
                break
            generated.append(vocab.itos.get(idx, "<UNK>"))
            input_token = next_idx

    return " ".join(generated)

def main():
    parser = argparse.ArgumentParser(
        description="Generate an image caption with a trained model."
    )
    parser.add_argument(
        "--image_path", #required=True,
        default=r'C:\Users\shubh\Desktop\Workspace\Image caption generator\input\20231227_082517.jpg',
        help="Path to the image file to caption."
    )
    parser.add_argument(
        "--model_path", default=r'C:\Users\shubh\Desktop\Workspace\Image caption generator\model\caption_model_flickr_8080.pth',
        help="Path to the .pth checkpoint or state_dict."
    )
    parser.add_argument(
        "--vocab_file", default=r'C:\Users\shubh\Desktop\Workspace\Image caption generator\vocab.json',
        help="Path to vocab.json produced during training."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=300,
        help="Embedding dimension (should match training)."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=512,
        help="LSTM hidden dimension (must match training)."
    )
    parser.add_argument(
        "--feat_dim", type=int, default=1024,
        help="Encoder feature dimension (ResNet conv4 channels)."
    )
    parser.add_argument(
        "--attn_dim", type=int, default=256,
        help="Attention dimension (should match training)."
    )
    parser.add_argument(
        "--max_len", type=int, default=20,
        help="Max caption length to generate."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load vocab
    vocab = load_vocab(args.vocab_file)

    # 2) Load model
    model = load_model(
        args.model_path, device,
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hid_dim=args.hid_dim,
        feat_dim=args.feat_dim,
        attn_dim=args.attn_dim
    )

    # 3) Preprocess and caption
    img_tensor = preprocess_image(args.image_path, device)
    caption = generate_caption(
        model, vocab, img_tensor, device, max_len=args.max_len
    )
    print("Generated caption:")
    print(caption)

if __name__ == "__main__":
    main()
