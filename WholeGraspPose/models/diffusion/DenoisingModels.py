import torch
import torch.nn as nn
import einops
from colorama import Fore, Style

class UNet1D(nn.Module):
    def __init__(self, drop_out_p):
        super(UNet1D, self).__init__()

        # feature fusion
        self.fuse_in = nn.Linear(512*2, 512*2)
        self.fuse_mid = nn.Linear(512*2, 512*2)
        self.fuse_out = nn.Linear(512*2, 512)

        # UNet Encode
        self.down_sample_1 = nn.Linear(512, 256)
        self.down_sample_2 = nn.Linear(256, 128)
        self.down_sample_3 = nn.Linear(128, 64)
        self.down_sample_4 = nn.Linear(64, 32)

        # UNet middle
        self.middle_1 = nn.Linear(32, 32)
        self.middle_2 = nn.Linear(32, 32)
        self.middle_3 = nn.Linear(32, 32)

        # UNet Decode
        self.up_sample_1 = nn.Linear(32, 64)
        self.up_sample_2 = nn.Linear(64, 128)
        self.up_sample_3 = nn.Linear(128, 256)
        self.up_sample_4 = nn.Linear(256, 512)

        # BatchNorms, Regularization, GELU
        self.bn_1024 = nn.BatchNorm1d(1024)
        self.bn_512 = nn.BatchNorm1d(512)
        self.bn_256 = nn.BatchNorm1d(256)
        self.bn_128 = nn.BatchNorm1d(128)
        self.bn_64 = nn.BatchNorm1d(64)
        self.bn_32 = nn.BatchNorm1d(32)

        self.drop_out = nn.Dropout(drop_out_p)
        self.gelu = nn.SiLU()

        self.warned = False

    def forward(self, feature_vec, time, condition=None): # expect all three inputs to be of size [bs, 512]
        feat_with_time = feature_vec + time # [bs,512]
        if condition != None:
            if not self.warned:
                print(Fore.RED + "Warning: You are introducing condition during sampling! This is inconsistent with the SAGA pipeline!\n"
                               + Style.RESET_ALL)
                self.warned = True
            X = torch.cat((feat_with_time,condition), dim=-1) # [bs,1024]
            # feature fusion
            X = self.fuse_in(X)
            X = self.bn_1024(X)
            X = self.gelu(X)

            X = self.fuse_mid(X)
            X = self.bn_1024(X)
            X = self.gelu(X)
            X = self.drop_out(X)

            X = self.fuse_out(X)
            X = self.bn_512(X)
            X = self.gelu(X)
        else:
            X = feat_with_time
        
        # UNet Encoding
        X = self.down_sample_1(X)
        X = self.bn_256(X)
        X = self.gelu(X)
        X_at_256 = X

        X = self.down_sample_2(X)
        X = self.bn_128(X)
        X = self.gelu(X)
        X = self.drop_out(X)
        X_at_128 = X

        X = self.down_sample_3(X)
        X = self.bn_64(X)
        X = self.gelu(X)
        X_at_64 = X

        X = self.down_sample_4(X)
        X = self.bn_32(X)
        X = self.gelu(X)
        X = self.drop_out(X)
        X_at_32 = X

        # UNet middle
        X = self.middle_1(X)
        X = self.bn_32(X)
        X = self.gelu(X)
        X = self.drop_out(X)

        X = self.middle_2(X)
        X = self.bn_32(X)
        X = self.gelu(X)
        X = self.drop_out(X)

        X = self.middle_3(X)
        X = self.bn_32(X)
        X = self.gelu(X)
        X = self.drop_out(X)
        X = X + X_at_32

        # UNet Decoding
        X = self.up_sample_1(X)
        X = self.bn_64(X)
        X = self.gelu(X)
        X = X + X_at_64

        X = self.up_sample_2(X)
        X = self.bn_128(X)
        X = self.gelu(X)
        X = self.drop_out(X)
        X = X + X_at_128

        X = self.up_sample_3(X)
        X = self.bn_256(X)
        X = self.gelu(X)
        X = X + X_at_256

        X = self.up_sample_4(X)

        return X

class FlatPush(nn.Module):
    def __init__(self, depth, drop_out_p, input_dim=512*2, out_dim=512):
        super(FlatPush, self).__init__()
        self.warned = False
        # feature fusion
        self.fuse_in = nn.Linear(input_dim, input_dim)
        self.fuse_mid = nn.Linear(input_dim,input_dim)
        self.fuse_out = nn.Linear(input_dim, out_dim)

        # The FlatPush MLP
        layers = []
        layers.append(nn.Linear(input_dim, out_dim))
        # layers.append(nn.BatchNorm1d(out_dim))
        # layers.append(nn.GELU())
        layers.append(nn.Dropout(drop_out_p))

        for i in range(depth-1):
            layers.append(nn.Linear(out_dim, out_dim))
            # layers.append(nn.BatchNorm1d(out_dim))
            # layers.append(nn.GELU())
            layers.append(nn.Dropout(drop_out_p))
        layers.append(nn.Linear(out_dim, out_dim))
        self.model = nn.Sequential(*layers)

        # BatchNorms, Regularization, GELU
        self.bn_1024 = nn.BatchNorm1d(input_dim)
        self.bn_512 = nn.BatchNorm1d(out_dim)
        self.drop_out = nn.Dropout(drop_out_p)
        self.gelu = nn.GELU()

    def forward(self, feature_vec, time, condition=None): # expect all three inputs to be of size [bs, 512]
        feat_with_time = feature_vec + time # [bs,512]
        if condition != None:
            if not self.warned:
                print(Fore.RED + "Warning: You are introducing condition during sampling! This is inconsistent with the SAGA pipeline!\n"
                               + Style.RESET_ALL)
                self.warned = True
            X = torch.cat((feat_with_time,condition), dim=-1) # [bs,1024]
            # feature fusion
            X = self.fuse_in(X)
            X = self.bn_1024(X)
            X = self.gelu(X)

            X = self.fuse_mid(X)
            X = self.bn_1024(X)
            X = self.gelu(X)
            X = self.drop_out(X)

            X = self.fuse_out(X)
            X = self.bn_512(X)
            X = self.gelu(X)
        else:
            X = feat_with_time
        
        X = self.model(X)

        return X

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate):
        super(Attention, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward neural network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(x + ff_output)

        return x

class TransformerDenoising(nn.Module):
    def __init__(self, vec_dim, drop_out_p, heads, depth):
        '''
        - vec_dim: The dimension of the sequence of vectors that the Transformer Block will process
        - drop_out_p: drop out probability
        - heads: number of heads
        - depth: Number of transformer blocks stacked
        '''
        super(TransformerDenoising, self).__init__()
        self.warned = False
        # feature fusion
        self.fuse_in = nn.Linear(512*2, 512*2)
        self.fuse_mid = nn.Linear(512*2, 512*2)
        self.fuse_out = nn.Linear(512*2, 512)

        # map to D dim
        self.mapper = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=vec_dim//3, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=vec_dim//3, out_channels=2*vec_dim//3, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=2*vec_dim//3, out_channels=vec_dim, kernel_size=1)
        )

        # map back to 1 dim
        self.mapper2 = nn.Sequential(
            nn.Conv1d(in_channels=vec_dim, out_channels=2*vec_dim//3, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=2*vec_dim//3, out_channels=vec_dim//3, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=vec_dim//3, out_channels=1, kernel_size=1),
        )

        # Attention
        attention_blocks = []
        for i in range(depth):
            attention_blocks.append(Attention(input_dim=vec_dim, hidden_dim=vec_dim*2, num_heads=heads, dropout_rate=drop_out_p))
        self.model = nn.Sequential(*attention_blocks)

        # BatchNorms, Regularization, GELU
        self.bn_1024 = nn.BatchNorm1d(1024)
        self.bn_512 = nn.BatchNorm1d(512)
        self.drop_out = nn.Dropout(drop_out_p)
        self.gelu = nn.GELU()

    def forward(self, feature_vec, time, condition=None): # expect all three inputs to be of size [bs, 512]
        feat_with_time = feature_vec + time # [bs,512]
        if condition != None:
            if not self.warned:
                print(Fore.RED + "Warning: You are introducing condition during sampling! This is inconsistent with the SAGA pipeline!\n"
                               + Style.RESET_ALL)
                self.warned = True
            X = torch.cat((feat_with_time,condition), dim=-1) # [bs,1024]
            # feature fusion
            X = self.fuse_in(X)
            X = self.bn_1024(X)
            X = self.gelu(X)

            X = self.fuse_mid(X)
            X = self.bn_1024(X)
            X = self.gelu(X)
            X = self.drop_out(X)

            X = self.fuse_out(X)
            X = self.bn_512(X)
            X = self.gelu(X)
        else:
            X = feat_with_time

        X = X.unsqueeze(1)
        X = self.mapper(X) # gives [bs, vec_dim, seq_len] (bs, vec_dim, 512)
        X = einops.rearrange(X,"bs vec_dim seq_len -> bs seq_len vec_dim")
        X = self.model(X)
        X = einops.rearrange(X,"bs seq_len vec_dim -> bs vec_dim seq_len")
        X = self.mapper2(X)
        X = X.squeeze()

        return X

if __name__ == '__main__':
    model = TransformerDenoising(vec_dim=128, drop_out_p=0.3, heads=4, depth=3)
    batch_size = 32

    feature_vec = torch.rand((batch_size,512))
    time = torch.rand((batch_size,512))
    condition = torch.rand((batch_size,512))

    out = model(feature_vec,time,condition)

    print(out.shape)
