# GEM
### [Grounding Everything: Emerging Localization Properties in Vision-Language Transformers](https://arxiv.org/abs/2312.00878)
_[Walid Bousselham](http://walidbousselham.com/), [Felix Petersen](https://petersen.ai/), [Vittorio Ferrari](https://sites.google.com/view/vittoferrari), [Hilde Kuehne](https://hildekuehne.github.io/)_

Vision-Language foundation models have shown remarkable performance in various zero-shot settings such as image retrieval, classification, or captioning. But so far, those models seem to fall behind when it comes to zero-shot localization of referential expressions and objects in images.

GEM allows a training-free adaptation of Vision-Language models (_e.i., CLIP ..._) to perform zero-shot open-vocabulary segmentation. The training-free adaptation allows to fully conserve the vocabulary learned by the Vision-Language model during its pretraing, thus allowing the segmentation of uncommon classes (_e.g._ Elon Musk/Mark Zuckerberg /Jeff Besos).

<div align="center">
<img src="assets/Animation_GEM.gif" width="70%">
</div>

## :hammer: Installation
`gem` library can be simply installed via pip: 
```bash
$ pip install gem_torch
```

## Demo
- Try out our web demo on [HuggingFace Spaces](https://huggingface.co/spaces) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/WalidBouss/GEM)
- Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f9aUbIpQIfEB8ZTUh3Krco8bIPqH3Pn3?usp=sharing)
- Run [`test_examples.py`](./test_examples.py) for a usage example.

To run the gradio app locally, first install gradio and then run [`app.py`](./app.py):
```bash
$ pip install gradio
$ python app.py
```
## Usage
To see which pretrained models is available use the following code snippet:
```python
import gem
gem.available_models()
```

### Single Image
To process a single image and multiple text prompts use the following code snippet:
```python
import torch
import gem
import requests
from PIL import Image

model_name = 'ViT-B/16'  # 'ViT-B-16-quickgelu'
pretrained = 'openai'  # 'metaclip_400m'
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and image transform
preprocess = gem.get_gem_img_transform()
gem_model = gem.create_gem_model(model_name=model_name,
                                 pretrained=pretrained, 
                                 device=device)
# load image and text
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(
    Image.open(requests.get(url, stream=True).raw)
               ).unsqueeze(0).to(device)
text = ['cat', 'remote control']

with torch.no_grad():
    logits = gem_model(image, text)  # [B, num_prompt, W, H]
    gem_model.visualize(image, text, logits)  # (optional visualization)
```

### Batched Inference
To process a batch of images with **different** number of prompts per image, one must use the `batched_forward()` function of `gem_model`:

```python
import torch
import gem
import requests
from PIL import Image

model_name = 'ViT-B/16'  # 'ViT-B-16-quickgelu'
pretrained = 'openai'  # 'metaclip_400m'
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and image transform
gem_model = gem.create_gem_model(model_name=model_name,
                                 pretrained=pretrained, 
                                 device=device)
preprocess = gem.get_gem_img_transform()

# load image and text
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
images = torch.stack([preprocess(img) for img in images_pil]).to(device)

with torch.no_grad():
    # return list with logits of size [1, num_prompt, W, H]
    logits_list = gem_model.batched_forward(images, texts)
    
    for i, logits in enumerate(logits_list):  # (optional visualization)
        gem_model.visualize(images[i], texts[i], logits)
```

### API
The library provides the following methods:
- `gem.create_gem_model(model_name, pretrained, device, ...)`: 
  - Returns `model_name` Vision Language model with `pretrained` weights loaded and GEM applied. One can also specify `gem_depth`, `ss_attn_iter` and `ss_attn_temp` parameters to respectively control GEM's depth, self-self attention number of iteration and temperature (see paper for more details).  
- `gem.get_gem_img_transform(img_size)`: 
  - takes in a PIL.Image and returns a torch.Tensor. This can be used as input to the model.
- `gem.visualize(image, prompts, logits, alpha=0.6, save_path=None)`:
  - Takes in a PIL.Image **or** a torch.Tensor, as well as the list of text prompt and the logits outputed by gem and plot the gem's heatmaps for each prompt. Alternatively, the heatmaps cam be saved by specifying the saving path `save_path`. One can also change the transparence of the heatmps via the `aplha=0.6` argument. 

By default, the models loaded by `gem.create_gem_model()` returns logits outputed by GEM, but can also return the logits of the original Vision Language model (it can be useful for visualization). To do so, set `return_ori=True`.

## More Examples
### Semantic Segmentation
For the semantic segmentation task, given a list of foreground class names, one must predict a 2D map where each location is the id of the predicted class. Depending on the dataset, we may also want to predict a `background` class. However, the textual description `"a photo of a background"` is not descriptive of what the background is composed of. Hence, we propose to use the following code method:

```python
import torch
import gem
import requests
from PIL import Image

model_name = 'ViT-B/16'  # 'ViT-B-16-quickgelu'
pretrained = 'openai'  # 'metaclip_400m'
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and image transform
gem_model = gem.create_gem_model(model_name=model_name,
                                 pretrained=pretrained, 
                                 device=device)
preprocess = gem.get_gem_img_transform()

predict_background = True  # whether the background is predicted
if predict_background:
    threshold = 0.85  # the threshold depends on the number of classes

# load image and text
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
class_names = ['airplane', 'cat', 'dog', '...']  # foreground class names

with torch.no_grad():
    logits = gem_model(image, class_names)  # [1, num_class, W, H]

pred = logits.argmax(dim=1)
if predict_background:
    pred = pred + 1  # we assume the background's index is 0
    probs = logits.softmax(dim=1)
    max_prob = probs.max(dim=1)[0]
    pred[probs < threshold] = 0  # if the max prob is lower than the threshold the background is predicted
```
Note that `threshold` depends on the number of classes and should be determined via a hyperparameter sweep. 

### Dataset 
`gem` can also be used with regular pytorch dataset.
```python
import torch
import gem
from PIL import Image
from torchvision.datasets import VOCSegmentation

model_name = 'ViT-B/16'  # 'ViT-B-16-quickgelu'
pretrained = 'openai'  # 'metaclip_400m'
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and image transform
gem_model = gem.create_gem_model(model_name=model_name,
                                 pretrained=pretrained, 
                                 device=device)
preprocess = gem.get_gem_img_transform()

predict_background = True  # whether the background is predicted
if predict_background:
    threshold = 0.85  # the threshold depends on the number of classes

# load dataset
root = './data'  # path to save the dataset
dataset = VOCSegmentation(root=root, image_set='val', download=True, transform=preprocess)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
class_names = ['airplane', 'cat', 'dog', '...']  # foreground class names

with torch.no_grad():
    for (image, _) in dataloader:
        logits = gem_model(image, class_names)  # [1, num_class, W, H]
    
        pred = logits.argmax(dim=1)
        if predict_background:
            pred = pred + 1  # we assume the background's index is 0
            probs = logits.softmax(dim=1)
            max_prob = probs.max(dim=1)[0]
            pred[probs < threshold] = 0  # if the max prob is lower than the threshold the background is predicted
```


# :star: Acknowledgement
This code is build as wrapper around [OpenCLIP](https://github.com/mlfoundations/open_clip) library from [LAION](https://laion.ai/), visit their repo for more vision-language models.
This project takes inspiration from [CLIP](https://github.com/openai/CLIP) and [CLIPSurgery](https://github.com/xmed-lab/CLIP_Surgery), please visit their repository.
This repo also uses [einops](https://github.com/arogozhnikov/einops) as well and take inspiration from [CLIP](https://github.com/openai/CLIP) and [CLIPSurgery]() repository.

# :books: Citation
If you find this repository useful, please consider citing our work :pencil: and giving a star :star2: :
```
@article{bousselham2023gem,
  title={Grounding Everything: Emerging Localization Properties in Vision-Language Transformers},
  author={Walid Bousselham, Felix Petersen, Vittorio Ferrari, Hilde Kuehne},
  journal={arXiv preprint arXiv:2312.00878},
  year={2023}
}