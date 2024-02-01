import time
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import numpy as np
from clip import clip
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import json


def sorted_os_listdir(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

if __name__ == "__main__":

    root='/scratch/work/firoozh1/nonar/data/'
    dataset = 'ahmed_et_al'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelclip, preprocess = clip.load('ViT-L/14', device=device)
    modelclip.eval()

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)

    max_length = 77

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    #model.to(device)
    img_cap = {}
    all_embeddings = None
    prev_class =  None
    for i, filename in enumerate(sorted_os_listdir(os.path.join(root, dataset,dataset.split("_")[0]+"_images"))):
        print(str(i+1)+":"+filename)
        
        curr_class = filename.split("_")[0]
        if prev_class is None:
            prev_class = curr_class
        elif prev_class != curr_class:
            with open(os.path.join(root, dataset, f"{dataset}_{prev_class}_blip_captions.json"), "w") as f:
                json.dump(img_cap, f)
            img_cap = {}
            prev_class = curr_class

        image_path = os.path.join(root, dataset,dataset.split("_")[0]+"_images", filename)
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        img_cap[filename] = generated_text

        # with torch.no_grad():

        #     batch_encoding = tokenizer(generated_text, truncation=True, max_length=max_length, return_length=True,
        #                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        #     tokens = batch_encoding["input_ids"]  # .to(self.device)
        #     outputs = transformer(input_ids=tokens)

        #     z = outputs.last_hidden_state.squeeze(0)

        # z=z.cpu()
        # print(z.shape)
        # if all_embeddings is None:
        #     all_embeddings = z.detach().cpu().numpy()
        # else:
        #     all_embeddings = np.vstack((all_embeddings, z.detach().cpu().numpy()))

    # np.save(os.path.join(root, dataset, f"{dataset}_blip_clip_embeddings.npy"), all_embeddings)
    with open(os.path.join(root, dataset, f"{dataset}_{curr_class}_blip_captions.json"), "w") as f:
        json.dump(img_cap, f)
    with open(os.path.join(root, dataset, f"{dataset}_img_orders.json"), "w") as f:
        json.dump(sorted_os_listdir(os.path.join(root, dataset, dataset.split("_")[0]+"_images")), f)