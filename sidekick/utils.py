from typing import IO
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm import tqdm


def download_data(file: IO[bytes], url: str, progress: bool = True,
                  chunk_size: int = 1024) -> None:
    """Streams data download.

    Args:
        file: File object where downloaded data will be written.
        url: Download url.
        progress: Print progress.
        chunk_size: Number of bytes read into memory at once.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    content_size = int(response.headers['content-length'])
    status_bar = tqdm(total=content_size // chunk_size, disable=not progress)
    for chunk in response.iter_content(chunk_size):
        file.write(chunk)
        status_bar.update()


def extract_zip(file: IO[bytes], directory: str) -> None:
    with ZipFile(file) as zip_handle:
        zip_handle.extractall(directory)


def balance_dataset(dataset: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Balances dataset by oversampling under-represented classes."""
    class_count = dataset[target_column].value_counts().to_dict()
    largest_class = max(class_count, key=class_count.get)
    counts_largest_class = class_count[largest_class]

    for class_name in class_count:
        samples = counts_largest_class - class_count[class_name]
        sampling = dataset[dataset[target_column] == class_name].sample(
            samples, replace=True)
        dataset = pd.concat([dataset, sampling], ignore_index=True)

    return dataset.sample(frac=1).reset_index(drop=True)
