import os
import zipfile
from typing import Optional

import requests
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p
from tqdm import tqdm
from HD_BET.paths import folder_with_parameter_files, ZENODO_DOWNLOAD_URL


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_with_parameter_files)


def download_file(url: str, local_filename: str, chunk_size: Optional[int] = 8192 * 16) -> str:
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(local_filename, 'wb'), "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename


def maybe_download_parameters():
    if not isfile(join(folder_with_parameter_files, 'fold_all', 'checkpoint_final.pth')):
        maybe_mkdir_p(folder_with_parameter_files)
        fname = download_file(ZENODO_DOWNLOAD_URL, join(folder_with_parameter_files, os.pardir, 'tmp_download.zip'))
        install_model_from_zip_file(fname)
        os.remove(fname)
