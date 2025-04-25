import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import json


# class FlickrDataset(Dataset):
#     def __init__(self, root_dir, captions_file, vocab, transform=None):
#         """
#         root_dir: path to Flickr images
#         captions_file: path to JSON with {"image_id": ["caption1", ...]}
#         vocab: Vocabulary object
#         transform: torchvision transforms
#         """
#         self.root = root_dir
#         self.vocab = vocab
#         self.transform = transform
#         # load captions
#         with open(captions_file, 'r') as f:
#             data = json.load(f)
#         self.image_ids = list(data.keys())
#         self.captions = data

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         img_id_raw = self.image_ids[idx]
#         # 1) Remove any "#<idx>" suffix
#         img_id = img_id_raw.split('#')[0]

#         # 2) Ensure we have a proper image extension
#         if img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
#             filename = img_id
#         else:
#             filename = img_id + '.jpg'

#         # 3) Build initial path
#         img_path = os.path.join(self.root, filename)

#         # 4) If not found, try stripping a trailing ".<digits>"
#         if not os.path.exists(img_path):
#             base, ext = os.path.splitext(filename)
#             parts = base.rsplit('.', 1)
#             if len(parts) == 2 and parts[1].isdigit():
#                 # remove the numeric suffix
#                 filename_alt = parts[0] + ext
#                 img_path_alt = os.path.join(self.root, filename_alt)
#                 if os.path.exists(img_path_alt):
#                     img_path = img_path_alt
#                 else:
#                     raise FileNotFoundError(
#                         f"Tried\n  {img_path}\nand\n  {img_path_alt}\nbut neither exists."
#                     )
#             else:
#                 raise FileNotFoundError(f"Image file not found: {img_path}")

#         # 5) Load and transform
#         image = Image.open(img_path).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)

#         # 6) Pick & numericalize a caption
#         captions = self.captions[img_id]
#         caption  = captions[0]  # or random.choice(captions)
#         tokens = [self.vocab.stoi["<START>"]]
#         tokens += self.vocab.numericalize(caption)
#         tokens.append(self.vocab.stoi["<END>"])

#         return image, torch.tensor(tokens, dtype=torch.long)


# # Collate function for padding
# from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch):
#     images = [item[0] for item in batch]
#     captions = [item[1] for item in batch]
#     images = torch.stack(images, 0)
#     captions = pad_sequence(captions, batch_first=True, padding_value=0)
#     return images, captions



import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import random

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, max_samples=None):
        """
        root_dir: path to Flickr images
        captions_file: JSON mapping image_id→[caption1,…]
        vocab: Vocabulary object
        transform: torchvision transforms
        max_samples: if set, only keep that many images
        """
        self.root      = root_dir
        self.vocab     = vocab
        self.transform = transform

        # 1) load captions JSON
        data = json.load(open(captions_file, 'r'))
        self.captions = data

        # 2) build list of image_ids *only* if the file actually exists
        valid = []
        for img_id in data.keys():
            # ensure filename ends with an image extension
            fn = img_id if img_id.lower().endswith(('.jpg','.jpeg','.png')) \
                 else img_id + '.jpg'
            path = os.path.join(self.root, fn)
            if os.path.exists(path):
                valid.append(img_id)
            else:
                print(f"Warning: missing image, skipping {path}")
        # optionally truncate for debugging
        if max_samples is not None:
            valid = valid[:max_samples]
        self.image_ids = valid

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # we’ll try up to len(self) times to find a loadable image
        n = len(self.image_ids)
        for attempt in range(n):
            img_id = self.image_ids[(idx + attempt) % n]
            # build path again
            fn = img_id if img_id.lower().endswith(('.jpg','.jpeg','.png')) \
                 else img_id + '.jpg'
            path = os.path.join(self.root, fn)

            try:
                # 3) load & transform
                img = Image.open(path).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                # 4) numericalize one of its captions
                caps = self.captions[img_id]
                cap  = random.choice(caps)   # pick randomly among the 5
                tokens = [self.vocab.stoi["<START>"]]
                tokens += self.vocab.numericalize(cap)
                tokens.append(self.vocab.stoi["<END>"])
                return img, torch.tensor(tokens, dtype=torch.long)

            except Exception as e:
                # log & skip to next
                print(f"Warning: failed to load {path}: {e}. Skipping.")
                continue

        # if we get here, none of the attempts worked
        raise RuntimeError("No valid images found in dataset.")

# Collate function for padding
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, captions = zip(*batch)
    images   = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions
