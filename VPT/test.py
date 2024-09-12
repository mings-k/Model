from model.vpt_vit import Vpt_ViT
import timm

model = timm.models.vit_base_patch16_224(pretrained=True)

for name, param in model.named_parameters():
    print(name)