from PIL import Image
import numpy as np
import cv2 as cv2
import torch
import requests

import gradio as gr

import gem


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# OpenCLIP
model_name = 'ViT-B-16-quickgelu'
pretrained = 'metaclip_400m'
preprocess = gem.get_gem_img_transform()
# global gem_model
gem_model = gem.create_gem_model(model_name=model_name, pretrained=pretrained, device=device)
image_source = "image"
_MODELS = {
    "OpenAI": ('ViT-B-16', 'openai'),
    "MetaCLIP": ('ViT-B-16-quickgelu', 'metaclip_400m'),
    "OpenCLIP": ('ViT-B-16', 'laion400m_e32')
}

def change_weights(pretrained_weights):
    """ Handle changing model's weights triggered by a Dropdown module change."""
    curr_model = pretrained_weights
    _new_model = _MODELS[pretrained_weights]
    print(_new_model)
    global gem_model
    gem_model = gem.create_gem_model(model_name=_new_model[0], pretrained=_new_model[1], device=device)

def change_to_url(url):
    img_pil = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return img_pil

def viz_func(url, image, text, model_weights):
    image_torch = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = gem_model(image_torch, [text])
    logits = logits[0].detach().cpu().numpy()

    img_cv = cv2.cvtColor(np.array(image.resize((448, 448))), cv2.COLOR_RGB2BGR)
    logit_cs_viz = (logits * 255).astype('uint8')
    heat_maps_cs = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logit_cs_viz]

    vizs = [0.4 * img_cv + 0.6 * heat_map for heat_map in heat_maps_cs]
    vizs = [cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB) for viz in vizs]
    return vizs[0]

inputs = [
    gr.Textbox(label="url to the image", ),
    gr.Image(type="pil"),
    gr.Textbox(label="Text Prompt"),
    gr.Dropdown(["OpenAI", "MetaCLIP", "OpenCLIP"], label="Pretrained Weights", value="MetaCLIP",
                info='It can take a few second for the model to be updated.'),
    ]

with gr.Blocks() as demo:
    inputs[-1].change(fn=change_weights, inputs=[inputs[-1]])
    inputs[0].change(fn=change_to_url, outputs=inputs[1], inputs=inputs[0])

    interact = gr.Interface(
        title="GEM: Grounding Everything Module (link to paper/code)",
        description="Grounding Everything: Emerging Localization Properties in Vision-Language Transformers",
        fn=viz_func,
        inputs=inputs,
        outputs=["image"],
    )

    gr.Examples(
        [
            ["assets/cats_remote_control.jpeg", "cat"],
            ["assets/cats_remote_control.jpeg", "remote control"],
            ["assets/elon_jeff_mark.jpeg", "elon musk"],
            ["assets/elon_jeff_mark.jpeg", "mark zuckerberg"],
            ["assets/elon_jeff_mark.jpeg", "jeff bezos"],
        ],
        [inputs[1], inputs[2]]
    )

# demo.launch(server_port=5152)
demo.launch()
