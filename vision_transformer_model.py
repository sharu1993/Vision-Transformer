import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class PatchEmbedding(nn.Module):
    #how does this applying patching??? Is it just a regular CNN layer? 
    # Is there not specific direction or algorithm to how you patch images?
    #there are other patching techniques that help improve on the OG ViT
    def __init__(self,img_size=32,patch_size=4,in_channels=4,embed_dim=128):
        super().__init__()
        self.num_patches = (img_size//patch_size)**2
        self.proj=nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self,x):
        x=self.proj(x)  
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim=128,num_heads=4,mlp_dim=254,dropout=0.1):
        super().__init__()
        #Transformer encoder block Layer Norm -> Attention Head(s) -> Residual -> Layer Norm -> MLP -> Residual
        self.norm1=nn.LayerNorm(embed_dim)
        self.attention=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2=nn.LayerNorm(embed_dim) #because attention does not change dimensions, only content
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        attention_in=self.norm1(x)
        attention_out,_=self.attention(attention_in,attention_in,attention_in) #arguments are query,key,value
        x=x+attention_out
        mlp_in=self.norm2(x)
        x=x+self.mlp(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=128, #how do you set a good value for embeded dimensions
            depth=6, #Depth: how deep is the network.
            num_heads=4, #how to decide the number of attention heads
            mlp_dim =256, #maybe this should be a multiper that is applied to embed_dim??
            dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches=self.patch_embed.num_patches
        #need explanation on these lines
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim)) #Each token is initialized with zeros and is a learned parameter?
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        
        self.dropout=nn.Dropout(dropout)

        self.encoder=nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout
                )
                for _ in range(depth) #depth defines how deep the transformer network is. For each depth, we add an encoder block
            ]
        )

        #what are these?
        self.norm = nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        B=x.shape[0] #what is B???
        
        #for transformer, applying patching first
        x=self.patch_embed(x)

        #next cls token assignment
        cls_tokens=self.cls_token.expand(B,-1,-1)
        
        x=torch.cat((cls_tokens,x),dim=1) #concatenate tokens and input?

        x=x+self.pos_embed #residual after embeding

        x=self.dropout(x)

        x=self.encoder(x)

        cls_output=x[:,0]
        cls_output=self.norm(cls_output)

        return self.head(cls_output)