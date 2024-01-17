from PIL import Image
import os
import pickle as pkl
import gc

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def sorted_os_listdir(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_type = 'half'


    model_id = "llava-hf/llava-1.5-7b-hf"
    root_path = "/scratch/work/firoozh1/nonar/data/ahmed_et_al/ahmed_images/"
    prompt = "USER: <image>\nA brief caption for this image\nASSISTANT:"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if tensor_type == 'half' else torch.float32, 
        low_cpu_mem_usage=True, 
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # just to pass the error!
    # with open("/scratch/work/firoozh1/nonar/data/ahmed_et_al/ahmed_img_caps_1_1745.pkl", 'rb') as f:
    #     saved_captions = pkl.load(f)

    # saved_captions = [saved_captions[i]['img'] for i in range(len(saved_captions))]

    img_cap = []
    cnt = 1
    start = 3300
    end = start
    print(len(os.listdir(root_path)))
    for filename in sorted_os_listdir(root_path):
        image_file = os.path.join(root_path, filename)

        # if filename not in saved_captions:
        if cnt >= start:
            
            print(filename)
            raw_image = Image.open(image_file)
            inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16 if tensor_type == 'half' else torch.float32)

            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            caption = processor.decode(output[0][2:], skip_special_tokens=True)
            caption = caption.split("\n")[2][11:]

            img_cap.append({'img': filename, 'caption': caption})
            print(f"image {cnt} done!")

            del raw_image, inputs, output, caption
            gc.collect()
            end += 1
            if end - start >= 1555:
                with open(os.path.join("/scratch/work/firoozh1/nonar/data/ahmed_et_al/", f"ahmed_img_caps_{start}_{end}.pkl"), 'wb') as f:
                    pkl.dump(img_cap, f)
                start = end
                img_cap = []
                
        cnt += 1
        # if cnt > :
        #     break

    with open(os.path.join("/scratch/work/firoozh1/nonar/data/ahmed_et_al/", f"ahmed_img_caps_{start}_{end}.pkl"), 'wb') as f:
        pkl.dump(img_cap, f)
