import os
from fastdup.embeddings import FastdupTimmWrapper 
def test_initialization():
    wrapper = FastdupTimmWrapper(model_name='mobilenetv2_050')
    assert wrapper.model_name == 'mobilenetv2_050'
    assert wrapper.pretrained == True
    assert wrapper.num_classes == 0

def test_compute_embeddings():
    wrapper = FastdupTimmWrapper(model_name='mobilenetv2_050')
    wrapper.compute_embeddings('tests/sample_images_for_tests')
    
    assert wrapper.embeddings is not None
    assert wrapper.file_paths is not None
    assert wrapper.embeddings.shape == (74, 1280)
    assert len(wrapper.embeddings) == len(wrapper.file_paths)

    # Remove created files during test to avoid clutter
    os.remove("mobilenetv2_050_embeddings.npy")
    os.remove("mobilenetv2_050_file_paths.txt")
    

def test_save_files_embeddings():
    wrapper = FastdupTimmWrapper(model_name='mobilenetv2_050')
    save_dir = "saved_embeddings_files"

    wrapper.compute_embeddings('tests/sample_images_for_tests', save_dir=save_dir) 

    embeddings_file = os.path.join(save_dir, 'mobilenetv2_050_embeddings.npy')
    file_paths_file = os.path.join(save_dir, 'mobilenetv2_050_file_paths.txt')

    assert os.path.exists(embeddings_file)
    assert os.path.exists(file_paths_file)

     # Remove created files during test to avoid clutter
    os.remove(embeddings_file)
    os.remove(file_paths_file)