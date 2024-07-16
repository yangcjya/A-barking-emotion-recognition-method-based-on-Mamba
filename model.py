import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from mamba_ssm import Mamba

class AudioEmotionModel(nn.Module):
    def __init__(self, d_model=257, num_classes=5, sample_rate=16000, patch_length_ms=32, hop_length_ms=10):
        super(AudioEmotionModel, self).__init__()
        self.sample_rate = sample_rate
        self.patch_length_ms = patch_length_ms
        self.hop_length_ms = hop_length_ms

        self.mamba1 = Mamba(d_model=120, d_state=16, d_conv=4, expand=2).to('cuda')
        self.mamba2 = Mamba(d_model=120, d_state=16, d_conv=4, expand=2).to('cuda')
        self.mamba3 = Mamba(d_model=137, d_state=16, d_conv=4, expand=2).to('cuda')
        self.mamba4 = Mamba(d_model=137, d_state=16, d_conv=4, expand=2).to('cuda')
        self.mamba5 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to('cuda')
        self.mamba6 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to('cuda')


        self.classifier = nn.Linear(d_model, num_classes)
        self.layer_norm1 = nn.LayerNorm(120)
        self.layer_norm2 = nn.LayerNorm(137)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0)

        self.alpha = nn.Parameter(torch.rand(1))  # 可学习的权重参数

    def forward(self, src):
        src = self.preprocess_audio(src)
        src = self.apply_reassigned_stft(src)
        src = src.reshape(-1, src.shape[2], src.shape[3])

        src1 = src[:, :, 137:]
        src2 = src[:, :, :137]

        src1 = src1 + self.layer_norm1(self.mamba1(src1))
        src1 = src1 + self.layer_norm1(self.mamba2(src1))

        src2 = src2 + self.layer_norm2(self.mamba3(src2))
        src2 = src2 + self.layer_norm2(self.mamba4(src2))

        # 应用可学习的权重
        alpha = torch.sigmoid(self.alpha)  # 确保alpha在0到1之间
        src1_weighted = alpha * src1
        src2_weighted = (1 - alpha) * src2

        # 拼接处理后的高频和低频部分
        src_combined = torch.cat((src1_weighted, src2_weighted), dim=2)

        # # Apply gating
        # gate1 = torch.sigmoid(self.gate1(src1.mean(dim=1)))  # Average pooling over time
        # gate2 = torch.sigmoid(self.gate2(src2.mean(dim=1)))  # Average pooling over time
        #
        # # Weighted combination of src1 and src2
        # src_combined = torch.cat([gate1.unsqueeze(1) * src1, gate2.unsqueeze(1) * src2], dim=2)

        src = src_combined + self.layer_norm(self.mamba5(src_combined))
        src = src + self.layer_norm(self.mamba6(src))

        src = src.mean(dim=1)
        src = self.classifier(src)
        return src


    def preprocess_audio(self, audio_waveform):
        patch_size = int(self.sample_rate * (self.patch_length_ms / 1000.0))
        hop_length = int(self.sample_rate * (self.hop_length_ms / 1000.0))
        patches = audio_waveform.unfold(2, patch_size, hop_length)
        window = torch.hann_window(patch_size).to(audio_waveform.device)
        patches_windowed = patches * window.unsqueeze(0).unsqueeze(0)
        return patches_windowed


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_classifier = AudioEmotionModel().to(device)
    summary(audio_classifier, input_size=(16, 1, 64000))
    audio_data = torch.randn(16, 1, 64000).to(device)
    output = audio_classifier(audio_data)
    print(output.shape)
