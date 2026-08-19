import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 损失函数
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            # 确保 alpha 与 target 的设备相同
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ==============================================================================
# EEG_CNN_BiLSTM 模型
# ==============================================================================

class EEG_CNN_BiLSTM(nn.Module):
    """
    结合 CNN 提取特征和 Bi-LSTM 处理时序信息的混合模型。
    用于四分类任务。
    """
    def __init__(self, channels=30, num_classes=4): # num_classes 明确设置为 4
        super(EEG_CNN_BiLSTM, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        # CNN 部分: 提取空间和频率特征
        # 输入维度: (N, C_in, H, W) -> (N, channels, step, frequency)
        self.cnn_extractor = nn.Sequential(
            # Conv1D on Channels (类似空间滤波)
            nn.Conv2d(self.channels, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 2)), # 频率维度减半

            # Conv2D on Time and Frequency
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)), # 时间和频率维度减半

            # Conv2D on Time and Frequency
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)) # 时间和频率维度减半
        )

        # 假设原始 frequency=6, step=400
        # 经过 MaxPool(1,2), MaxPool(2,2), MaxPool(2,2) 后
        # frequency: 6 -> 3 -> 1.5 (ceil to 2) -> 1
        # step: 400 -> 400 -> 200 -> 100
        # 经过 CNN 后输出形状近似: (N, 128, ~100, ~1)
        # 展平后 LSTM 的 input_dim = 128 * ~1

        self.lstm_input_dim = 128 # 假设经过 CNN 后的特征维度是 128

        # Bi-LSTM 部分: 处理时序信息
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.55
        )

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64), # 128*2 是 Bi-LSTM 的输出维度 (hidden_size * 2)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes) # 输出到 4 个类别
        )

    def forward(self, x):
        # 1. CNN 提取特征
        # x 维度: (batch, channels, step, frequency)
        x = x.permute(0, 1, 3, 2) # 调整维度为 (batch, channels, frequency, step) 以匹配 Conv2D 习惯
        x = self.cnn_extractor(x)
        # x 维度: (batch, features, new_freq, new_step)

        # 2. 准备 LSTM 输入
        # 将 features 和 new_freq 展平
        x = x.permute(0, 3, 1, 2) # 调整为 (batch, new_step, features, new_freq)

        # 展平最后的 (features * new_freq) 维度，作为 LSTM 的 input_dim
        batch_size, seq_len, features, new_freq = x.shape
        x = x.reshape(batch_size, seq_len, features * new_freq)

        # 3. Bi-LSTM 处理时序
        # x 维度: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.bilstm(x)

        # 使用最后一个时间步的 Bi-LSTM 输出（双向的拼接）
        # h_n 维度: (num_layers * num_directions, batch, hidden_size)

        # 取得最后一个时间步，最后第二层 (n-1) 的前向和后向输出
        final_forward_hn = h_n[-2, :, :]
        final_backward_hn = h_n[-1, :, :]

        # 拼接 (batch, 2 * hidden_size)
        final_output = torch.cat((final_forward_hn, final_backward_hn), dim=1)

        # 4. 分类器
        output = self.fc(final_output)

        return output



# ==============================================================================
# EEG_CNN_BiLSTM_Attention 模型 (新模型，加入了注意力机制)
# ==============================================================================

class EEG_CNN_BiLSTM_Attention(nn.Module):
    """
    结合 CNN、Bi-LSTM 和 Attention Mechanism 的模型。
    用于四分类任务。
    """
    def __init__(
        self,
        channels=8,
        num_classes=4,
        lstm_hidden_size=128,
        lstm_layers=2,
        lstm_dropout=0.5,
        classifier_dropout=0.5,
    ):
        super(EEG_CNN_BiLSTM_Attention, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_output_dim = self.lstm_hidden_size * 2 # 双向 LSTM

        # --- 1. CNN 部分: 提取空间和频率特征 ---
        self.cnn_extractor = nn.Sequential(
            # Conv1D on Channels (类似空间滤波)
            nn.Conv2d(self.channels, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 2)), # 频率维度减半 (F -> F/2)

            # Conv2D on Time and Frequency
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)), # 时间和频率维度减半 (T/2, F/2)

            # Conv2D on Time and Frequency
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)) # 时间和频率维度减半 (T/4, F/4)
        )

        self.lstm_input_dim = 128 # 假设经过 CNN 后的特征维度是 128

        # --- 2. Bi-LSTM 部分: 处理时序信息 ---
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        # --- 3. Attention 机制 ---
        # 接收 Bi-LSTM 的输出维度 (2 * hidden_size)，计算注意力权重
        self.attention_weights = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False) # 输出维度为 1，用于加权
        )

        # --- 4. 分类器 ---
        # 接收 Context Vector (维度等于 Bi-LSTM 输出维度)
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        # x 维度: (batch, channels, step, frequency)

        # 1. CNN 提取特征
        x = x.permute(0, 1, 3, 2) # 调整维度: (batch, channels, frequency, step)
        cnn_out = self.cnn_extractor(x)

        # 2. 准备 LSTM 输入
        # cnn_out 维度: (batch, features, new_freq, new_step)

        # 调整为 (batch, new_step, features, new_freq)
        cnn_out = cnn_out.permute(0, 3, 1, 2)

        # 展平特征和频率维度: (batch, seq_len, input_dim)
        batch_size, seq_len, features, new_freq = cnn_out.shape
        lstm_input = cnn_out.reshape(batch_size, seq_len, features * new_freq)

        # 3. Bi-LSTM 处理时序
        # lstm_out: (batch, seq_len, 2 * hidden_size)
        lstm_out, _ = self.bilstm(lstm_input)

        # 4. 注意力机制
        # 计算注意力得分 (energy)
        # energy 维度: (batch, seq_len, 1)
        energy = self.attention_weights(lstm_out)

        # 将 energy 展平并应用 softmax 得到权重
        # attention_weights 维度: (batch, seq_len)
        attention_weights = F.softmax(energy.squeeze(-1), dim=1)

        # 计算 Context Vector
        # context_vector = sum(lstm_out * weights)
        # 维度: (batch, 2 * hidden_size)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # 5. 分类器
        output = self.fc(context_vector)

        return output


# ==============================================================================
# EEG_CNN_Transformer
# ==============================================================================

class EEG_CNN_Transformer(nn.Module):
    """CNN 时频特征提取器 + Transformer Encoder EEG 分类器。"""

    def __init__(
        self,
        channels=8,
        num_classes=4,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        transformer_dropout=0.2,
        classifier_dropout=0.3,
        max_sequence_length=512,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.channels = channels
        self.num_classes = num_classes
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(d_model),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length + 1, d_model)
        )
        self.embedding_dropout = nn.Dropout(transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, channels, time, bands], got {tuple(x.shape)}")
        # [B,C,T,F] -> [B,C,F,T] -> [B,D,1,T'] -> [B,T',D]
        features = self.cnn_extractor(x.permute(0, 1, 3, 2))
        tokens = features.squeeze(2).transpose(1, 2)
        if tokens.size(1) + 1 > self.position_embedding.size(1):
            raise ValueError("Token sequence exceeds max_sequence_length")
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.embedding_dropout(
            tokens + self.position_embedding[:, :tokens.size(1)]
        )
        encoded = self.transformer(tokens)
        return self.classifier(encoded[:, 0])
