import torch
from fastdup.sentry import fastdup_capture_exception
from fastdup.definitions import MISSING_LABEL
from fastdup.galleries import fastdup_imread
import cv2
from tqdm import tqdm

device_to_captioner = {}

def init_captioning(model_name='automatic', device='cpu', batch_size=8, max_new_tokens=20,
                        use_float_16=True):

    '''
    This function generates captions for a given set of images, and takes the following arguments:
        - filenames: the list of images passed to the function
        - modelname: the captioning model to be used (default: vitgpt2)
            currently available models are:
            - ViT-GPT2 : 'vitgpt2'
            - BLIP-2: 'blip2'
            - BLIP: 'blip'
        - batch_size: the size of image batches to caption (default: 8)
        - device: whether to use a GPU (default: -1, CPU only ; set to 0 for GPU)
        - max_bew_tokens: set the number of allowed tokens
    '''

    global device_to_captioner
    # use GPU if device is specified
    if device == 'gpu':
        device = 0
    elif device == 'cpu':
        device = -1
        use_float_16 = False
    else:
        assert False, "Incompatible device name entered {device}. Available device names are gpu and cpu."

    # confirm necessary dependencies are installed, and import them
    try:
        from transformers import pipeline
        from transformers.utils import logging
        logging.set_verbosity(50)

    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the following libraries:\n")
        print("   huggingface transformers\n   pytorch\n")
        print("   to install, use `pip3 install transformers torch`")
        raise

    # dictionary of captioning models
    models = {
        'automatic': "nlpconnect/vit-gpt2-image-captioning",
        'vitgpt2': "nlpconnect/vit-gpt2-image-captioning",
        'blip-2': "Salesforce/blip2-opt-2.7b",
        'blip': "Salesforce/blip-image-captioning-large"
    }
    assert model_name in models.keys(), f"Unknown captioning model {model_name} allowed models are {models.keys()}"
    model = models[model_name]
    has_gpu = torch.cuda.is_available()
    captioner = pipeline("image-to-text", model=model, device=device if has_gpu else "cpu", max_new_tokens=max_new_tokens,
                         torch_dtype=torch.float16 if use_float_16 else torch.float32)
    device_to_captioner[device] = captioner

    return captioner

def generate_labels(filenames, model_name='automatic', device = 'cpu', batch_size=8, max_new_tokens=20, use_float_16=True):
    global device_to_captioner
    if device not in device_to_captioner:
        captioner = init_captioning(model_name, device, batch_size, max_new_tokens, use_float_16)
    else:
        captioner = device_to_captioner[device]

    captions = []
    # generate captions
    try:
        for i in tqdm(range(0, len(filenames), batch_size)):
            chunk = filenames[i:i + batch_size]
            try:
                for pred in captioner(chunk, batch_size=batch_size):
                    charstring = '' if model_name != 'blip' else ' '
                    caption = charstring.join([d['generated_text'] for d in pred])
                    # Split the sentence into words
                    words = caption.split()
                    # Filter out words containing '#'
                    filtered_words = [word for word in words if '#' not in word]
                    # Join the filtered words back into a sentence
                    caption = ' '.join(filtered_words)
                    caption = caption.strip()
                    captions.append(caption)
            except Exception as ex:
                print("Failed to caption chunk", chunk[:5], ex)
                captions.extend([MISSING_LABEL] * len(chunk))

    except Exception as e:
        fastdup_capture_exception("Auto caption image", e)
        return [MISSING_LABEL] * len(filenames)

    return captions


def generate_vqa_labels(filenames, text, kwargs):
    # confirm necessary dependencies are installed, and import them
    try:
        from transformers import ViltProcessor, ViltForQuestionAnswering
        from transformers.utils import logging
        logging.set_verbosity_info()
        import torch
        from PIL import Image
        from tqdm import tqdm
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the following libraries:\n")
        print("   huggingface transformers\n   pytorch\n   pillow\n   tqdm\n")
        print("to install, use `pip install transformers torch pillow tqdm`")
        return [MISSING_LABEL] * len(filenames)

    try:
        preds = []
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        for image_path in tqdm(filenames):
            i_image = fastdup_imread(image_path, None, kwargs=kwargs)
            if i_image is not None:
                i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(i_image)
                encoding = processor(im_pil, text, return_tensors="pt")

                # forward pass
                outputs = model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                preds.append(model.config.id2label[idx])
            else:
                preds.append(MISSING_LABEL)

        return preds

    except Exception as e:
        fastdup_capture_exception("Auto caption image vqa", e)
        return [MISSING_LABEL] * len(filenames)


def generate_age_labels(filenames, kwargs):
    # confirm necessary dependencies are installed, and import them
    try:
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from transformers.utils import logging
        logging.set_verbosity_info()
        import torch
        from PIL import Image
        from tqdm import tqdm
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the following libraries:\n")
        print("   huggingface transformers\n   pytorch\n   pillow\n   tqdm\n")
        print("to install, use `pip install transformers torch pillow tqdm`")
        return [MISSING_LABEL] * len(filenames)

    try:
        model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
        preds = []
        # Get example image from official fairface repo + read it in as an image
        for image_path in tqdm(filenames):
            i_image = fastdup_imread(image_path, None, kwargs=kwargs)
            # Init model, transforms

            # Transform our image and pass it through the model
            inputs = transforms(i_image, return_tensors='pt')
            output = model(**inputs)

            # Predicted Class probabilities
            proba = output.logits.softmax(1)

            # Predicted Classes
            pred = int(proba.argmax(1)[0].int())
            preds.append(model.config.id2label[pred])
        return preds
    except Exception as e:
        fastdup_capture_exception("Age label", e)
        return [MISSING_LABEL] * len(filenames)

if __name__ == "__main__":
    import fastdup
    from fastdup.captions import generate_labels
    file = "/Users/dannybickson/visual_database/cxx/unittests/two_images/"
    import os
    files = os.listdir(file)
    files = [os.path.join(file, f) for f in files]
    ret = generate_labels(files, model_name='blip')
    assert(len(ret) == 2)
    print(ret)
    for r in ret:
        assert "shelf" in r or "shelves" in r or "store" in r