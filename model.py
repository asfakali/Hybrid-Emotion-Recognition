import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Self-attention mechanism where the query, key, and value are the same
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, y):
        # Cross-attention where the query comes from x and the key/value come from y
        attn_output, _ = self.attention(x, y, y)
        return attn_output

class AttentionModel(nn.Module):
    def __init__(self, dim=64, num_heads=8):
        super(AttentionModel, self).__init__()
        # Initialize self-attention layers for each feature set
        self.self_attn_ann = SelfAttention(dim, num_heads)
        self.self_attn_cnn = SelfAttention(dim, num_heads)
        self.self_attn_bilstm = SelfAttention(dim, num_heads)

        # Initialize cross-attention layers for each pair of feature sets
        self.cross_attn_ann_cnn = CrossAttention(dim, num_heads)
        self.cross_attn_ann_bilstm = CrossAttention(dim, num_heads)
        self.cross_attn_cnn_bilstm = CrossAttention(dim, num_heads)

    def forward(self, features_ann, features_cnn, features_bilstm):
        # Self-attention on each feature set
        attn_ann = self.self_attn_ann(features_ann)
        attn_cnn = self.self_attn_cnn(features_cnn)
        attn_bilstm = self.self_attn_bilstm(features_bilstm)

        # Cross-attention between feature sets
        cross_attn_ann_cnn = self.cross_attn_ann_cnn(features_ann, features_cnn)
        cross_attn_ann_bilstm = self.cross_attn_ann_bilstm(features_ann, features_bilstm)
        cross_attn_cnn_bilstm = self.cross_attn_cnn_bilstm(features_cnn, features_bilstm)

        # Concatenate all attention outputs
        concatenated_features = torch.cat([attn_ann, attn_cnn, attn_bilstm,
                                          cross_attn_ann_cnn, cross_attn_ann_bilstm, cross_attn_cnn_bilstm], dim=-1)
        return concatenated_features

class CombinedModel(nn.Module):
    def __init__(self, input_stat_dim, input_signal_dim, num_classes,
                 normalization=nn.BatchNorm1d, activation=nn.GELU, dropout_rate=0.3):
        super(CombinedModel, self).__init__()

        # Define normalization and activation functions
        norm_layer = normalization if normalization is not None else nn.Identity
        act_layer = activation if activation is not None else nn.Identity

        # ANN Module for statistical features
        self.ann = nn.Sequential(
            nn.Linear(input_stat_dim, 128),
            norm_layer(128),
            act_layer(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            norm_layer(64),
            act_layer(),
            nn.Dropout(dropout_rate)
        )

        # BiLSTM Module for sequential features
        self.bilstm = nn.LSTM(input_size=input_signal_dim, hidden_size=64, num_layers=2,
                              bidirectional=False, batch_first=True, dropout=dropout_rate)

        # Optional normalization layer after LSTM
        self.lstm_norm = norm_layer(64)

        # CNN Module for signal data
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            norm_layer(16),
            act_layer(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            norm_layer(32),
            act_layer(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            norm_layer(64),
            act_layer(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Fully Connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(64 * 6, 64),
            norm_layer(64),
            act_layer(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Attention module
        self.attention = AttentionModel(dim=64, num_heads=16)

    def forward(self, stat_features, signal_features):
        # Process statistical features through ANN
        ann_out = self.ann(stat_features)

        # Process sequential features through BiLSTM
        signal_features = signal_features.unsqueeze(1)
        lstm_out, _ = self.bilstm(signal_features)
        lstm_out = lstm_out[:, -1, :]  # Get last time-step output
        lstm_out = self.lstm_norm(lstm_out)  # Apply normalization

        # Process signal features through CNN
        cnn_out = self.cnn(signal_features)
        cnn_out = cnn_out.mean(dim=-1)

        # Apply attention on the combined outputs
        combined_out = self.attention(ann_out, lstm_out, cnn_out)

        # Pass through final fully connected layers
        output = self.fc(combined_out)
        return output, ann_out, lstm_out, cnn_out
