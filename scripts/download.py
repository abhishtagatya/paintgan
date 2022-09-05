import os
import json
import gdown
import tarfile
import argparse
from zipfile import ZipFile

import kaggle


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
            'url': 'https://drive.google.com/u/0/uc?id=1--E_p-WM2SnjVIa6iQJ1IuSSFmz8x3lk',
            'name': 'paintgan-80k.zip'
        },
        'EVAL': {
            'url': 'https://drive.google.com/u/0/uc?id=1LUOU2WkITkC9x7tBsAnJtCW3GmKlbnid',
            'name': 'paintgan-eval.zip'
        },
        'EVAL_SET': {
            'url': 'https://drive.google.com/u/0/uc?id=1GS3bbN-fxY8fk9i0d6_moNyjSv7krhFy',
            'name': 'eval_set.hdf5'
        },
        'EVAL_MODEL': {
            'url': 'https://drive.google.com/u/0/uc?id=1xm01AmH_OX_qGZJNSYmCTysGDaW4j2Aa',
            'name': 'deception_model.h5'
        }
    }

    if dataset_type not in dataset_drive.keys():
        gdown.download(dataset_url, data_dir)
        return

    if dataset_type == 'EVAL_SET':
        gdown.download(
            dataset_drive['EVAL_SET']['url'],
            dataset_drive['EVAL_SET']['name']
        )
        return

    if dataset_type == 'EVAL_MODEL':
        gdown.download(
            dataset_drive['EVAL_MODEL']['url'],
            dataset_drive['EVAL_MODEL']['name']
        )
        return

    download_and_extract(
        dataset_drive[dataset_type]['url'],
        os.path.join(
            data_dir, dataset_drive[dataset_type]['name']
        )
    )


def download_from_kaggle(data_dir: str, dataset: str):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, data_dir, unzip=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Dataset")
    parser.add_argument('--source', help='source of download', type=str, required=True)
    parser.add_argument('--data-dir', help='download to directory', type=str, required=False)
    parser.add_argument('--dataset', help='dataset name', type=str, required=False)
    args = parser.parse_args()

    if args.source == 'DRIVE':
        download_from_drive(args.data_dir, args.dataset)

    if args.source == 'KAGGLE':
        download_from_kaggle(args.data_dir, args.dataset)




