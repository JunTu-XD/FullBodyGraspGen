import torch

class UNet1D(torch.nn.Module):
    def __init__(self, p=0.3):
        super(UNet1D, self).__init__()

        # feature fusion
        self.fuse_in = torch.nn.Linear(512*2, 512*2)
        self.fuse_mid = torch.nn.Linear(512*2, 512*2)
        self.fuse_out = torch.nn.Linear(512*2, 512)

        # UNet Encode
        self.down_sample_1 = torch.nn.Linear(512, 256)
        self.down_sample_2 = torch.nn.Linear(256, 128)
        self.down_sample_3 = torch.nn.Linear(128, 64)
        self.down_sample_4 = torch.nn.Linear(64, 32)

        # UNet middle
        self.middle_1 = torch.nn.Linear(32, 32)
        self.middle_2 = torch.nn.Linear(32, 32)
        self.middle_3 = torch.nn.Linear(32, 32)

        # UNet Decode
        self.up_sample_1 = torch.nn.Linear(32, 64)
        self.up_sample_2 = torch.nn.Linear(64, 128)
        self.up_sample_3 = torch.nn.Linear(128, 256)
        self.up_sample_4 = torch.nn.Linear(256, 512)

        # BatchNorms, Regularization, GELU
        self.bn_1024 = torch.nn.BatchNorm1d(1024)
        self.bn_512 = torch.nn.BatchNorm1d(512)
        self.bn_256 = torch.nn.BatchNorm1d(256)
        self.bn_128 = torch.nn.BatchNorm1d(128)
        self.bn_64 = torch.nn.BatchNorm1d(64)
        self.bn_32 = torch.nn.BatchNorm1d(32)

        self.drop_out = torch.nn.Dropout(p)
        self.gelu = torch.nn.GELU()

    def forward(self, feature_vec, time, condition): # expect all three inputs to be of size [bs, 512]
        feat_with_time = feature_vec + time
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

        # UNet Encoding
        X = self.down_sample_1(X)
        X = self.bn_256(X)
        X = self.gelu(X)

        X = self.down_sample_2(X)
        X = self.bn_128(X)
        X = self.gelu(X)
        X = self.drop_out(X)

        X = self.down_sample_3(X)
        X = self.bn_64(X)
        X = self.gelu(X)

        X = self.down_sample_4(X)
        X = self.bn_32(X)
        X = self.gelu(X)
        X = self.drop_out(X)

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

        # UNet Decoding
        X = self.up_sample_1(X)
        X = self.bn_64(X)
        X = self.gelu(X)

        X = self.up_sample_2(X)
        X = self.bn_128(X)
        X = self.gelu(X)
        X = self.drop_out(X)

        X = self.up_sample_3(X)
        X = self.bn_256(X)
        X = self.gelu(X)

        X = self.up_sample_4(X)

        return X


if __name__ == '__main__':
    model = UNet1D()
    batch_size = 32


    feature_vec = torch.rand((batch_size,512))
    time = torch.rand((batch_size,512))
    condition = torch.rand((batch_size,512))

    out = model(feature_vec,time,condition)

    print(out.shape)
