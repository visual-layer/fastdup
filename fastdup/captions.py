from fastdup.sentry import fastdup_capture_exception
from fastdup.definitions import MISSING_LABEL
from fastdup.galleries import fastdup_imread
import cv2


def generate_labels(filenames, modelname='automatic', batch_size=8):
    '''
    This function generates captions for a given set of images, and takes the following arguments:
        - filenames: the list of images passed to the function
        - modelname: the captioning model to be used (default: vitgpt2)
            currently available models are:
            - ViT-GPT2 : 'vitgpt2'
            - BLIP-2: 'blip2'
            - BLIP: 'blip'
        - batch_size: the size of image batches to caption (default: 8)
    '''

    # confirm necessary dependencies are installed, and import them
    try:
        from transformers import pipeline
        import torch
        from PIL import Image
        from tqdm import tqdm
    except Exception as e:
        fastdup_capture_exception("Auto generate labels", e)
        print("Auto captioning requires an installation of the following libraries:\n")
        print("   huggingface transformers\n   pytorch\n   pillow\n   tqdm\n")
        print("to install, use `pip install transformers torch pillow tqdm`")
        return [MISSING_LABEL] * len(filenames)

    # dictionary of captioning models
    models = {
        'automatic': "nlpconnect/vit-gpt2-image-captioning",
        'vitgpt2': "nlpconnect/vit-gpt2-image-captioning",
        'blip2': "Salesforce/blip2-opt-2.7b",
        'blip': "Salesforce/blip-image-captioning-large"
    }

    model = models[modelname]

    # generate captions
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        captioner = pipeline("image-to-text", model=model, device=device, batch_size=batch_size)

        captions = []
        for image_path in tqdm(filenames):
            img = Image.open(image_path)
            pred = captioner(img)
            caption = pred[0]['generated_text']
            captions.append(caption)
        return captions


    except Exception as e:
        fastdup_capture_exception("Auto caption image", e)
        return [MISSING_LABEL] * len(filenames)


def generate_vqa_labels(filenames, text, kwargs):
    # confirm necessary dependencies are installed, and import them
    try:
        from transformers import ViltProcessor, ViltForQuestionAnswering
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

