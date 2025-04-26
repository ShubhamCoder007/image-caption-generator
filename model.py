import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT') #(pretrained=True)
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
    def __init__(self, vocab_size, embed_dim, hid_dim, feat_dim, attn_dim, dropout=0.3, pretrained_emb: torch.Tensor = None,
        freeze_emb: bool = False):
        super().__init__()
        
        if pretrained_emb is not None:
            # pretrained_emb: (vocab_size, embed_dim)
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_emb, freeze=freeze_emb
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        # self.embedding = nn.Embedding(vocab_size, embed_dim)
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
    def __init__(self, vocab_size, embed_dim=512, hid_dim=512, feat_dim=1024, attn_dim=256, pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.encoder = EncoderCNN(hid_dim)
        # self.decoder = DecoderRNN(vocab_size, embed_dim, hid_dim, feat_dim, attn_dim, pretrained_emb, freeze_emb)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hid_dim=hid_dim,
            feat_dim=feat_dim,
            attn_dim=attn_dim,
            pretrained_emb=pretrained_emb,
            freeze_emb=freeze_emb
        )

    def forward(self, images, captions):
        feats, init_hidden = self.encoder(images)
        outputs = self.decoder(feats, init_hidden, captions)
        return outputs