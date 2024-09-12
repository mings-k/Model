import torch
import torch.nn as nn
import torch.nn.init as init
import timm

class PromptInput(nn.Module):
    def __init__(self, num_prompts, embed_dim = 768, num_layers = 12):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Initialize prompt embeddings
        self.prompts = nn.Parameter(torch.zeros(num_layers, num_prompts, embed_dim))

        init.kaiming_uniform_(self.prompts)

    def prepend_prompt(self, x, layer_idx):

        batch_size = x.shape[0]

        prompt_tokens = self.prompts[layer_idx,:,:].expand(batch_size,-1,-1)

        if layer_idx == 0:
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:,1:,:]), dim = 1) # => [batch_size, cls_token + prompt_tokens + seq_len, embed_dim]

        else:
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:, (1+self.num_prompts):, :]), dim=1)

        return x 
    

# prompt 추가 vit

class Vpt_ViT(nn.Module):
    def __init__(self, pretrained_model= 'vit_base_patch16_224',img_size=32, patch_size=4, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.prompt_embedding = PromptInput(num_prompts=100, embed_dim= 768, num_layers= depth)

        #timm을 이용한 pretrained_model 적용
        self.model = timm.create_model(pretrained_model, pretrained = True, img_size = img_size, patch_size = patch_size, num_classes = num_classes)
    
    def forward(self, x):
        x = self.model.patch_embed(x)
        cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)  # 클래스 토큰 추가
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed 
        x = self.model.pos_drop(x)

        for idx, block in enumerate(self.model.blocks):
            x = self.prompt_embedding.prepend_prompt(x, idx)
            x = block(x)

        x = self.model.norm(x)  # 최종 레이어 정규화
        x = self.model.forward_head(x)
        return x  # 분류 헤드를 통한 출력