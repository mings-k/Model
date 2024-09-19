# ViT 내용

- NLP 분야에서 사용한 Transformer를 Computer Vision 분야에 적용시킴

- 논문(https://arxiv.org/abs/2010.11929)

![alt text](image.png)

- 사진과 같이 Transformer Encoder를 사용하며, NLP에 사용한 transformer의 input과 비슷한 형태를 적용하기 위해 이미지를 Patch로 분리하여 적용함.
- (Patch를 만들 때 conv를 사용하면 성능이 좀 더 상승한다는 말이 있음)
- Patch를 flatten 및 Linear projection하고, cls 토큰과 Positional embadding을 추가함