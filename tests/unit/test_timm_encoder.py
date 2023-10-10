import os
import shutil
from fastdup.embeddings.timm import TimmEncoder 
def test_initialization():
    timm_model = TimmEncoder(model_name='mobilenetv2_050')
    assert timm_model.model_name == 'mobilenetv2_050'
    assert timm_model.pretrained == True
    assert timm_model.num_classes == 0

def test_compute_embeddings():
    timm_model = TimmEncoder(model_name='mobilenetv2_050')
    timm_model.compute_embeddings('tests/sample_images_for_tests')
    
    assert timm_model.embeddings is not None
    assert timm_model.file_paths is not None
    assert timm_model.embeddings.shape == (10, 1280)
    assert len(timm_model.embeddings) == len(timm_model.file_paths)

    # Remove created files during test to avoid clutter
    os.remove("mobilenetv2_050_embeddings.npy")
    os.remove("mobilenetv2_050_file_paths.txt")
    

def test_save_files_embeddings():
    timm_model = TimmEncoder(model_name='mobilenetv2_050')
    save_dir = "saved_embeddings_files"

    os.makedirs(save_dir, exist_ok=True)

    timm_model.compute_embeddings('tests/sample_images_for_tests', save_dir=save_dir) 

    embeddings_file = os.path.join(save_dir, 'mobilenetv2_050_embeddings.npy')
    file_paths_file = os.path.join(save_dir, 'mobilenetv2_050_file_paths.txt')

    assert os.path.exists(embeddings_file)
    assert os.path.exists(file_paths_file)

     # Remove created files during test to avoid clutter
    shutil.rmtree(save_dir)