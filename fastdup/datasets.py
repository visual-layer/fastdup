from datasets import load_dataset, Dataset
import os

class FastdupHFDataset(Dataset):
    def __init__(self, dataset_name, split='train', cache_dir=None, img_key='image', label_key='label', **kwargs):

        self.img_key = img_key
        self.label_key = label_key
        
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            from datasets.config import HF_DATASETS_CACHE
            self.cache_dir=HF_DATASETS_CACHE

        self.hf_dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir, **kwargs)

        super().__init__(self.hf_dataset.data, self.hf_dataset.info, self.hf_dataset.split)

        self.save_as_image_files()
    
    @property
    def images_dir(self):
        return os.path.join(self.cache_dir , self.hf_dataset.info.dataset_name)

    def save_as_image_files(self):
        from tqdm.auto import tqdm
        for idx, item in tqdm(enumerate(self.hf_dataset), total=len(self.hf_dataset), desc="Converting to images:"):        
            # extract the image and label
            image = item[self.img_key]

            # label = i2d[dataset.features['label'].int2str(item['label'])]
            label = item[self.label_key]

            # remove apostrophes
            # label = label.replace("'", "")

            # replace commas with underscores
            # label = label.replace(", ", "_")

            # replace spaces with dashes
            # label = label.replace(" ", "-")

            # create a directory for the class if it doesn't exist
            label_dir = os.path.join(os.path.join(f'{self.cache_dir}',f'{self.hf_dataset.info.dataset_name}', 'images'), str(label))
            os.makedirs(label_dir, exist_ok=True)

            # save the image to the appropriate directory
            image.save(os.path.join(label_dir, f'{idx}.jpg'))

    @property
    def annotations(self):
        """Returns a Pandas DataFrame with filename, label column"""
        import os
        import pandas as pd

        path = os.path.join(self.images_dir, 'images')

        filenames = []
        labels = []

        for label in os.listdir(path):
            label_path = os.path.join(path, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    filenames.append(os.path.join(label_path, filename))
                    labels.append(label)

        df = pd.DataFrame({
            'filename': filenames,
            'label': labels
        })

        return df

