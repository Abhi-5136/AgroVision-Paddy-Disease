import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class AgroVisionNet(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128):
        super(AgroVisionNet, self).__init__()
        
        # 1. Visual Backbone (EfficientNet-B0)
        weights = EfficientNet_B0_Weights.DEFAULT
        self.visual_backbone = efficientnet_b0(weights=weights)
        in_features = self.visual_backbone.classifier[1].in_features
        self.visual_backbone.classifier = nn.Identity()
        self.visual_proj = nn.Linear(in_features, embed_dim)

        # 2. Temporal Branch (2-Layer Stacked LSTM)
        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True
        )

        # 3. Scaled Dot-Product Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, weather_seq):
        # Visual Query: [Batch, 1, Embed_Dim]
        v_feat = self.visual_backbone(image)
        v_query = self.visual_proj(v_feat).unsqueeze(1)

        # Weather Key/Value: [Batch, 7, Embed_Dim]
        w_seq, _ = self.lstm(weather_seq)

        # Cross Attention (Query=Visual, Key=Weather, Value=Weather)
        attn_out, attn_weights = self.cross_attention(query=v_query, key=w_seq, value=w_seq)
        
        # Concatenate Spatial Visual Vector & Cross-Attended Temporal Vector
        fused = torch.cat((v_query.squeeze(1), attn_out.squeeze(1)), dim=-1)
        logits = self.classifier(fused)
        
        return logits, attn_weights