import torch
import torch.nn as nn



class patch_embedding(nn.Module):
    def __init__(self,emb_dim=64,patch_size=4,img_size=28,in_channels=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels = in_channels,
                                    out_channels = emb_dim,
                                    kernel_size = patch_size,
                                    stride = patch_size)
        
    def forward(self,x):
        #[B,Number of dimension,Embedding size]
        x = self.projection(x) #[batch,emb_dim,height/4,width/4] -> [1,64,7,7]
        x = x.flatten(2) #[batch,emb_dim,(height*width)/4] -> [1,64,49]
        x = x.transpose(1,2) #[batch,number of dimension, embedding size] -> [1,49,64]
        return x

          
    pass

class positional_encoding(nn.Module):
    def __init__(self,patch_size=4,img_size=28,emb_dim=64,dropout=0.2,in_channels=1):
        super().__init__()

        self.emb_dim = emb_dim
        self.number_dim = img_size//patch_size

        self.cls_token = nn.Parameter(torch.rand(size=(1,in_channels,self.emb_dim)),requires_grad=True)
        self.position = nn.Parameter(torch.rand(size=(self.number_dim ** 2 + 1,self.emb_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,x):
        # x = [batch,number_of_dim, embed_dim] -> [1,49,64]
        batch_size = x.shape[0]
        cls_expand = self.cls_token.expand(batch_size,-1,-1) # [batch, 49,64]
        x = torch.cat([x,cls_expand],dim=1) # -> [batch, 49,64]
        x += self.position # -> [batch,50,64]
        x = self.dropout(x)

        return x
    

class mod(nn.Module):
    def __init__(self,emb_dim=64,num_of_heads=8,num_of_encoder_blocks=6,patch_size=4,num_classes=10,dropout_rate=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        self.num_of_heads = num_of_heads
        self.num_of_encoder_blocks = num_of_encoder_blocks

        self.patch_embedding = patch_embedding(
            emb_dim=emb_dim,patch_size=patch_size
        )
        self.positional_encoding = positional_encoding(
            patch_size=patch_size,emb_dim=emb_dim,dropout=dropout_rate
        )

        self.decoder = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_of_heads, batch_first=True)

        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.classifier = nn.Linear(emb_dim, num_classes)
    
    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        # print(f'shape of x after positional embedding : {x.shape}')
        # ones = torch.ones(x.shape[0])
        x = self.decoder(x,x)
        x = x.mean(dim=1) if self.pool is None else self.pool(x.transpose(1, 2)).squeeze(-1)  # Global pooling
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = mod()
    x = torch.randn(64,1,28,28)

    y = model(x)
    print(y.shape)