# webcam_ui.py

import argparse
import threading
import os

import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
from torchvision import transforms

from vocab import Vocabulary
from model import ImageCaptioningModel
# reuse the predict helpers, but we'll inline preprocess_image
from pred import generate_caption, load_vocab, load_model

def preprocess_pil_image(pil_img, device):
    """
    Apply the same transforms you used in training to a PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225]),
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)
    return tensor

def start_ui(args):
    # 1) Prepare device, vocab & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.vocab_file)
    model = load_model(
        args.model_path, device,
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hid_dim=args.hid_dim,
        feat_dim=args.feat_dim,
        attn_dim=args.attn_dim
    )

    # 2) OpenCV webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # 3) Build Tkinter window
    root = tk.Tk()
    root.title("📸 Image Captioning Webcam Demo")

    # Video display label
    video_label = tk.Label(root)
    video_label.pack()

    # Caption display
    caption_var = tk.StringVar()
    caption_label = tk.Label(root, textvariable=caption_var,
                             wraplength=600, font=("Helvetica", 14))
    caption_label.pack(pady=10)

    # Capture button
    def on_capture():
        ret, frame = cap.read()
        if not ret:
            caption_var.set("Failed to capture image")
            return

        # Convert to PIL image (RGB)
        cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb)

        # Preprocess & generate caption
        img_tensor = preprocess_pil_image(pil_img, device)
        with torch.no_grad():
            caption = generate_caption(
                model, vocab, img_tensor, device, max_len=args.max_len
            )
        caption_var.set(caption or "(no caption)")

    btn = tk.Button(root, text="Capture & Caption", command=on_capture)
    btn.pack(pady=5)

    # Video update loop
    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
        root.after(30, update_frame)  # 30 ms ≈ 33 FPS

    update_frame()
    root.mainloop()
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam UI for Image Captioning"
    )
    # parser.add_argument(
    #     "--model_path", required=True,
    #     help="Path to caption model .pth checkpoint"
    # )
    # parser.add_argument(
    #     "--vocab_file", required=True,
    #     help="Path to vocab.json from training"
    # )
    parser.add_argument(
        "--model_path", default=r'C:\Users\shubh\Desktop\Workspace\Image caption generator\caption_model_flickr_8000.pth',
        help="Path to the .pth checkpoint or state_dict."
    )
    parser.add_argument(
        "--vocab_file", default=r'C:\Users\shubh\Desktop\Workspace\Image caption generator\vocab.json',
        help="Path to vocab.json produced during training."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=512,
        help="Embedding dimension (must match training)"
    )
    parser.add_argument(
        "--hid_dim", type=int, default=512,
        help="Hidden dimension (must match training)"
    )
    parser.add_argument(
        "--feat_dim", type=int, default=1024,
        help="Encoder feature dim (ResNet conv4 channels)"
    )
    parser.add_argument(
        "--attn_dim", type=int, default=256,
        help="Attention dimension (must match training)"
    )
    parser.add_argument(
        "--max_len", type=int, default=20,
        help="Max caption length to generate"
    )
    args = parser.parse_args()
    start_ui(args)
