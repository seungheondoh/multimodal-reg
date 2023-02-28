import os
import tarfile
from tqdm import tqdm
file_root = "../dataset/ml-20m/content/targz"
dest_dir = "../"
os.makedirs(dest_dir, exist_ok=True)
fnames = os.listdir(file_root)
for fname in tqdm(fnames):
    fdirs = os.path.join(file_root, fname)
    try:
        tar = tarfile.open(fdirs, "r:gz")
        for tarinfo in tar:
            tar.extract(tarinfo, dest_dir)
    except:
        print(fdirs)