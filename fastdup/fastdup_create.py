from pathlib import Path
from fastdup.sentry import v1_sentry_handler
from fastdup.engine import Fastdup
from typing import Union
import fastdup.fastdup_controller as FD


@v1_sentry_handler
def create(work_dir: Union[str, Path] = None, input_dir: Union[str, Path, list] = None) -> Fastdup:
    """
    Create fastdup analyzer instance.
    Usage example
    =============
    ```
    import pandas as pd
    import fastsup

    annotation_csv = '/path/to/annotation.csv'
    data_dir = '/path/to/images/'
    output_dir = '/path/to/fastdup_analysis'

    import fastdup
    fd = fastdup.create(work_dir='work_dir', input_dir='images')
    fd.run(annotations=pd.read_csv(annotation_csv))


    df_sim  = fdp.similarity(data=False)
    im1_id, im2_id, sim = df_sim.iloc[0]
    annot_im1, annot_im2 = fdp[im1_id], fdp[im2_id]

    df_cc, cc_info = fd.connected_components()


    fd = Fastdup(work_dir=work_dir, input_dir=input_dir)
    return fd
    ```
    """
    fd = Fastdup(work_dir=work_dir, input_dir=input_dir)
    return fd
