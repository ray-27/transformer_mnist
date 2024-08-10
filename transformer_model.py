import torch
import torch.nn as nn
from colorama import Fore,init
import config

init(autoreset=True)

def dash():
    print("--------------------------")

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

class multi_head_attention(nn.Module):
    """
    [!NOTE]
    embedding dimension (emb_dim) should be divisble by number of heads (num_head)
    """
    def __init__(self,
                 num_head=8,
                 emb_dim=64,
                 dropout=0.2):
        super().__init__()

        self.num_head = num_head
        self.emb_dim = emb_dim

        # key,query,value
        self.key = nn.Linear(emb_dim,emb_dim) # [64,64]
        self.query = nn.Linear(emb_dim,emb_dim) # [64,64]
        self.value = nn.Linear(emb_dim,emb_dim) # [64,64]

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([emb_dim//num_head])),requires_grad=False)

        self.fc_out = nn.Linear(emb_dim,emb_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # x = [batch, number_of_dim, embed_dim] -> [1,50,64]

        Q = self.query(x) # Q = [batch,number_of_dim, embed_dim] -> [1,50,64]
        K = self.key(x) # K = [batch,number_of_dim, embed_dim] -> [1,50,64]
        V = self.value(x) # V = [batch,number_of_dim, embed_dim] -> [1,50,64]

        # cange the shape to something like [batch,number_of_dim, number_of_heads, emb_dim//number_of_heads] -> [1,50,8,8]
        Q = Q.view(x.shape[0], -1, self.num_head, self.emb_dim//self.num_head)
        K = K.view(x.shape[0], -1, self.num_head, self.emb_dim//self.num_head)
        V = V.view(x.shape[0], -1, self.num_head, self.emb_dim//self.num_head)

        # matrix multiplication of the key and query
        QK = torch.einsum("nqhd,nkhd->nhqk",[Q,K]) #shape = [batch_size, num_heads, num_queries, num_keys]-> [4, 8, 50, 50]

        attention = torch.softmax(QK/self.scale, dim=-1) #shape of -> [4, 8, 50, 50]
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,V])
        out = out.contiguous().reshape(x.shape[0],-1,self.emb_dim) # give back the input shape as x [1,50,64]

        out = self.fc_out(out)
        out = self.dropout(out)
        return out

class Transformer_block(nn.Module):
    def __init__(self,
                 emb_dim=64,
                 num_head=8,
                 img_size=28,
                 patch_size=4,
                 forward_expansion=4,
                 dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim
        self.number_dim = img_size//patch_size

        self.multi_head_attention_1 = multi_head_attention(num_head=num_head,
                                                            emb_dim=emb_dim,
                                                            dropout=dropout)
        self.multi_head_attention_2 = multi_head_attention(num_head=num_head,
                                                            emb_dim=emb_dim,
                                                            dropout=dropout)

        self.norm_layer_1 = nn.LayerNorm(self.emb_dim)
        self.norm_layer_2 = nn.LayerNorm(self.emb_dim)
        self.norm_layer_3 = nn.LayerNorm(self.emb_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim,forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_dim, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)


    
    def forward(self,x):
        # x = [batch, number_of_dimensions, emb_dim] -> [1,50,64]
        attention = self.multi_head_attention_1(x)
        x = self.dropout(self.norm_layer_1(attention + x))
        
        attention = self.multi_head_attention_2(x)
        x = self.dropout(self.norm_layer_2(attention + x))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm_layer_3(forward + x))
        return out


class vision_transform(nn.Module):
    def __init__(self,emb_dim=64,num_of_heads=8,patch_size=4,dropout_rate=0.2,num_of_encoder_blocks=4,num_classes=10,show_params=True):
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

        self.encoder_block = nn.Sequential(*[Transformer_block(emb_dim=emb_dim,
                                                                num_head=num_of_heads,
                                                                patch_size=patch_size,
                                                                forward_expansion=4,
                                                                dropout=dropout_rate) for _ in range(num_of_encoder_blocks)])

        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.classifier = nn.Linear(emb_dim, num_classes)
        
        if show_params:
            self.show_params()


    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder_block(x)
        x = x.mean(dim=1) if self.pool is None else self.pool(x.transpose(1, 2)).squeeze(-1)  # Global pooling
        x = self.classifier(x)

        return x

    def show_params(self):
        #showing the passes parameters
            emb_dim = self.patch_embedding.emb_dim
            patch_size = self.patch_embedding.patch_size
            print(Fore.BLUE + "Patch Embedding parameters")
            print(f"emb_dim: \t {emb_dim}")
            print(f"patch_size: \t {patch_size}")
            dash()

            emb_dim = self.positional_encoding.emb_dim
            number_dim = self.positional_encoding.number_dim
            cls_tok = self.positional_encoding.cls_token 
            position = self.positional_encoding.position
            print(Fore.CYAN + "Positional Encoding parameters")
            print(f'emb_dim : \t {emb_dim}')
            print(f'number of dimensions : \t {number_dim}')
            print(f'cls token : \t {cls_tok.shape}')
            print(f'position token : \t {position.shape}') 
            dash()

            print(Fore.GREEN + "Vision Transformer paramerters")
            print(f"emb_dim \t {self.emb_dim}\n num_of_head \t {self.num_of_heads}\n patch_size \t {self.patch_size}\n number of \nencoder blocks \t {self.num_of_encoder_blocks}")
            #emb_dim=64,num_of_heads=8,patch_size=4,dropout_rate=0.2,num_of_encoder_blocks=4,num_classes=10

    pass

if __name__ == "__main__":
    model = vision_transform(emb_dim=config.emb_dim,
                             num_of_heads=config.num_of_heads,
                             patch_size=config.patch_size,
                             dropout_rate=config.dropout_rate,
                             num_of_encoder_blocks=config.num_of_encoder_blocks,
                             num_classes=config.num_classes,
                             show_params=config.show_params)

    x = torch.randn(64,1,28,28)
    y = model(x)


    print(Fore.GREEN + "Model output")
    print(y.shape)
    print("-----------------")