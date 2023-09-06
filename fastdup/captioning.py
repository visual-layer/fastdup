from fastdup.sentry import fastdup_capture_exception
from fastdup.definitions import MISSING_LABEL
from fastdup.galleries import fastdup_imread

def deps_import(filenames):
    try:
        from transformers import pipeline
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the transformers package using `pip install transformers`")
        return [MISSING_LABEL]*len(filenames)
    try:
        import torch
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the pytorch package using `pip install torch`")
        return [MISSING_LABEL]*len(filenames)
    try:
        from PIL import Image
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the pillow package using `pip install pillow`")
        return [MISSING_LABEL]*len(filenames)
    try:
        from tqdm import tqdm
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the tqdm package using `pip install tqdm`")
        return [MISSING_LABEL]*len(filenames)
    try:
        import cv2
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the openCV package using `pip install opencv-python`")
        return [MISSING_LABEL]*len(filenames)

def generate_labels_vitgpt2(filenames, kwargs):
    try:
        deps_import(filenames)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=device)
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        images = []
        for image_path in tqdm(filenames):
            i_image = fastdup_imread(image_path, None, kwargs=kwargs)
            if i_image is not None:
                i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(i_image)
                images.append(im_pil)
            else:
                images.append(None)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    except Exception as e:
        fastdup_capture_exception("Auto caption image", e)
        return [MISSING_LABEL]*len(filenames)
    




with open('gts.json', 'r') as file:
    gts = json.load(file)
dataset = load_dataset("imagefolder", data_dir="coco100")

captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", batch_size=2)
for out in captioner(KeyDataset(dataset["test"], "image")):
    outputstr = ' '.join(d['generated_text'] for d in out)
    print(outputstr)









captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_names = list(glob.glob('/Users/guysinger/Desktop/Users Datasets/meesho_good_images_10pct/*.jpg'))
ims_list = []
captions = []
#ims_list = []

for i in range(0, 20):
    img = Image.open(img_names[i])
    pred = captioner(img)
    caption = pred[0]['generated_text']
    print("pred: ", caption)
    captions.append(caption)
    ims_list.append(img_names[i])


results_df = pd.DataFrame({'from':ims_list, 'to':ims_list, 'label':captions, 'distance':0*len(img_names)})
fastdup.create_outliers_gallery(results_df, save_path='.', num_images=20)
HTML('./outliers.html')
    




import glob
import torch
import fastdup
from PIL import Image
from IPython.display import HTML
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_names = list(glob.glob('/Users/guysinger/Desktop/Users Datasets/meesho_good_images_10pct/*.jpg'))
ims_list = []
captions = []
#ims_list = []

for i in range(0, 20):
    img = Image.open(img_names[i])
    pred = captioner(img)
    caption = pred[0]['generated_text']
    print("pred: ", caption)
    captions.append(caption)
    ims_list.append(img_names[i])


results_df = pd.DataFrame({'from':ims_list, 'to':ims_list, 'label':captions, 'distance':0*len(img_names)})
fastdup.create_outliers_gallery(results_df, save_path='.', num_images=20)
HTML('./outliers.html')