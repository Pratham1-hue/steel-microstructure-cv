import pathlib

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
SIZE = 256
N_IMAGES = 200
SEED = 12345


def labels_for(n_seeds, size, rng):
    pts = rng.uniform(0, size, size=(n_seeds, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    _, idx = cKDTree(pts).query(coords, k=1)
    return idx.reshape(size, size)


def render(labels, rng, noise=0.0, blur=0.0):
    n = int(labels.max()) + 1
    bright = rng.uniform(0.5, 1.0, size=n)
    img = bright[labels]
    img[find_boundaries(labels, mode="inner")] = 0.0
    if blur > 0:
        img = ndimage.gaussian_filter(img, blur)
    if noise > 0:
        img = img + rng.normal(0, noise, size=img.shape)
    return np.clip(img, 0, 1)


def true_diameter(labels):
    areas = np.bincount(labels.ravel())
    areas = areas[areas > 0]
    return float((2 * np.sqrt(areas / np.pi)).mean())


def generate(n=N_IMAGES, size=SIZE, noise=0.05, blur=0.5, seed=SEED):
    imgs = np.zeros((n, size, size), dtype=np.uint8)
    labs = np.zeros((n, size, size), dtype=np.int32)
    rows = []
    for i in range(n):
        r = np.random.default_rng(seed + i)
        n_seeds = int(r.integers(20, 300))
        lab = labels_for(n_seeds, size, r)
        img = render(lab, r, noise=noise, blur=blur)
        imgs[i] = (img * 255).round().astype(np.uint8)
        labs[i] = lab
        rows.append({"id": i, "n_seeds": n_seeds, "true_diameter": true_diameter(lab),
                     "noise": noise, "blur": blur})
    return imgs, labs, pd.DataFrame(rows)


def main():
    DATA.mkdir(exist_ok=True)
    imgs, labs, meta = generate()
    np.save(DATA / "images.npy", imgs)
    np.save(DATA / "labels.npy", labs)
    meta.to_csv(DATA / "meta.csv", index=False)
    print(f"generated {len(meta)} images, mean true diameter {meta['true_diameter'].mean():.1f} px")
    print(f"saved {DATA}")


if __name__ == "__main__":
    main()
