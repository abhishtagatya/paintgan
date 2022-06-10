import os
import json
import gdown
import tarfile
import argparse
from zipfile import ZipFile


def stage_path(data_dir: str, name: str):
    """
    Create Data Directory

    :param data_dir: Directory Name
    :param name: Folder Name
    :return: Full Path
    """
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url: str, dst: str, remove=True):
    """
    Download and Extract from Google Drive

    :param url: Google Drive URL
    :param dst: Folder Destination
    :param remove: Remove Compress (zip, tar, tar.gz)
    :return:
    """

    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_from_drive(data_dir: str, dataset_type: str, dataset_url=None):
    """
    Download Dataset from Google Drive

    :param data_dir: Data Directory
    :param dataset_type: Dataset Type (10K or 80K)
    :param dataset_url: Google Drive URL
    :return:
    """

    dataset_drive = {
        '10K': {
            'url': 'https://drive.google.com/u/0/uc?id=1O7PtBLs9Ljc8SSHrVJuu44GjWiCm_p9n',
            'name': 'paintgan-10k.zip'
        },
        '80K': {
            'url': 'https://drive.google.com/u/0/uc?id=1JO6TPAcM1g3YZPxSCKecnoYaqEYG-laS',
            'name': 'paintgan-80k.zip'
        }
    }

    if dataset_type not in dataset_drive.keys():
        gdown.download(dataset_url, data_dir)
        return

    download_and_extract(
        dataset_drive[dataset_type]['url'],
        os.path.join(
            data_dir, dataset_drive[dataset_type]['name']
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Dataset")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    download_from_drive(args.data_dir, args.dataset)



