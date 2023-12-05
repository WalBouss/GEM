from PIL import Image
from gem import create_gem_model, get_gem_img_transform, visualize, available_models
import torch
import requests


print(available_models())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = 'ViT-B-16-quickgelu'
pretrained = 'metaclip_400m'
gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=device)
gem_model.eval()

###########################
# Single Image
###########################

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # cat & remote control
text = ['remote control', 'cat']
# image_path = 'path/to/image'  #,  <-- uncomment to use path

image_pil = Image.open(requests.get(url, stream=True).raw)
# image_pil = Image.open(image_path)  # <-- uncomment to use path

gem_img_transform = get_gem_img_transform()
image = gem_img_transform(image_pil).unsqueeze(0).to(device)

with torch.no_grad():
    logits = gem_model(image, text)
    visualize(image, text, logits)
    print(logits.shape)  # torch.Size([1, 2, 448, 448])
    # visualize(image_pil, text, logits)  # <-- works with torch.Tensor and PIL.Image

###########################
# Batch of Images
###########################
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://cdn.vietnambiz.vn/171464876016439296/2021/7/11/headshots16170695297430-1626006880779826347793.jpg",
    "https://preview.redd.it/do-you-think-joker-should-be-unpredictable-enough-to-put-up-v0-6a2ax4ngtlaa1.jpg?auto=webp&s=f8762e6a1b40642bcae5900bac184fc597131503",
    ]
texts = [
    ['remote control', 'cat'],
    ['elon musk', 'mark zuckerberg', 'jeff bezos', 'bill gates'],
    ['batman', 'joker', 'shoe', 'belt', 'purple suit'],
    ]  # note that the number of prompt per image can be different

# download images + convert to PIL.Image
images_pil = [Image.open(requests.get(url, stream=True).raw) for url in urls]
images = torch.stack([gem_img_transform(img) for img in images_pil]).to(device)

with torch.no_grad():
    # return list with logits of size [1, num_prompt, W, H]
    logits_list = gem_model.batched_forward(images, texts)
    print(logits_list[0].shape)  # torch.Size([2, 448, 448])
    print(logits_list[1].shape)  # torch.Size([4, 448, 448])
    print(logits_list[2].shape)  # torch.Size([5, 448, 448])
    for i, _logits in enumerate(logits_list):
        visualize(images[i], texts[i], _logits)  # (optional visualization)
