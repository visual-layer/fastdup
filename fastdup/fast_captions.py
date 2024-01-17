import torch
from fastdup.sentry import fastdup_capture_exception
from fastdup.definitions import MISSING_LABEL
from tqdm import tqdm
import nltk
nltk.download('stopwords')
device_to_captioner = {}
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def format_captions(captions):

    # Remove stop words
    filtered_text = ' '.join(word for word in captions.split()[:8] if word.lower() not in stop_words)
    # Split the text into words and count their occurrences
    return filtered_text

def init_captioning(model_name='automatic', device='cpu', batch_size=8, max_new_tokens=20,
                        use_float_16=True):

    global device_to_captioner
    # use GPU if device is specified
    if isinstance(device, str):
        if device == 'gpu':
            device = 0
        elif device == 'cpu':
            device = -1
            use_float_16 = False

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


    model = "Salesforce/blip-image-captioning-large"
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
                    charstring = ' '
                    caption = charstring.join([d['generated_text'] for d in pred])
                    # Split the sentence into words
                    words = caption.split()
                    # Filter out words containing '#'
                    filtered_words = [word for word in words if '#' not in word]
                    # Join the filtered words back into a sentence
                    caption = ' '.join(filtered_words)
                    caption = caption.strip()
                    caption = format_captions(caption)
                    captions.append(caption)
            except Exception as ex:
                print("Failed to caption chunk", chunk[:5], ex)
                captions.extend([MISSING_LABEL] * len(chunk))

    except Exception as e:
        fastdup_capture_exception("Auto caption image", e)
        return [MISSING_LABEL] * len(filenames)

    return captions

if __name__ == "__main__":
    import fastdup
    from fastdup.fast_captions import generate_labels
    file = "/Users/dannybickson/visual_database/cxx/unittests/two_images/"
    import os
    files = os.listdir(file)
    files = [os.path.join(file, f) for f in files]
    ret = generate_labels(files, model_name='blip')
    assert(len(ret) == 2)
    print(ret)
    for r in ret:
        assert "shelf" in r or "shelves" in r or "various" in r