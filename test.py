import h5py
import openslide
from tqdm import tqdm

slide = openslide.OpenSlide("/nfs/data371/gyf/data/NSCLC/Xinan/svs/B202108884-3.svs")

with h5py.File("/nfs/data371/gyf/data/NSCLC/Xinan/results-40-1024/patches/B202108884-3.h5", 'r') as f:
    for i in tqdm(range(len(f['coords']))):
        corrd = f['coords'][i]
        patch = slide.read_region(corrd, 0, (1024, 1024))
        patch.save(f"/nfs/data371/gyf/data/NSCLC/Xinan/results-40-1024/patches/B202108884-3/{corrd[0]}_{corrd[1]}.png")