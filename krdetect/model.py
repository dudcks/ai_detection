import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, num_classes=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Linear(d_model, num_classes)

        self.d_model = d_model

    def forward(self, x, attention_mask=None):

        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) 

        cls_token = out[:, 0, :]
        return self.cls_head(cls_token)
    

class TransformerLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=8,
                 num_classes=2, max_len=128,
                 lstm_hidden_size=128, lstm_layers=1, bidirectional=True): # LSTM 파라미터 추가
        super().__init__()
        # ... (Embedding, PositionalEncoding, TransformerEncoder 정의) ...
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.d_model = d_model

        # --- LSTM 레이어 추가 ---
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True, # 입력: (batch, seq, feature)
                            bidirectional=bidirectional)

        # --- 최종 분류 레이어 ---
        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.final_classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x, attention_mask=None):
        # ... (Embedding, PositionalEncoding, TransformerEncoder 적용) ...
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # encoder_output Shape: (batch_size, seq_len, d_model)

        # --- LSTM 적용 ---
        # LSTM은 패딩을 직접 처리하지 않으므로, 필요시 pack_padded_sequence 사용 고려
        # 여기서는 간단히 마지막 hidden state 사용
        lstm_out, (h_n, c_n) = self.lstm(encoder_output)
        # h_n shape: (num_layers * num_directions, batch_size, lstm_hidden_size)

        # 마지막 레이어의 hidden state 사용
        if self.lstm.bidirectional:
            # 양방향이면 마지막 두 개 hidden state 연결 (forward / backward)
            final_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            final_hidden = h_n[-1,:,:]
        # final_hidden Shape: (batch_size, lstm_output_dim)

        # --- 최종 분류 ---
        logits = self.final_classifier(final_hidden)
        return logits