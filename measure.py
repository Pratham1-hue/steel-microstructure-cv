import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import feature, filters, segmentation
from skimage.segmentation import find_boundaries
from sklearn.ensemble import RandomForestRegressor

from gen import render

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

PX_PER_UM = 0.1  # 1 px = 0.1 um
SIGMA0 = 100.0  # MPa, assumed
K_HP = 0.5  # MPa m^0.5, assumed
NOISE_STD = 10.0
MIN_DIST = 15
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]
BLUR_LEVELS = [0.0, 0.5, 1.0, 2.0]


def segment_grains(img):
    sm = filters.gaussian(img, sigma=1.0, preserve_range=True)
    thr = filters.threshold_otsu(sm)
    binary = sm > thr

    lab, n = ndi.label(binary)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    keep = sizes > 20
    keep[0] = False
    binary = keep[lab]
    if binary.sum() < 0.05 * binary.size or binary.all():
        return None

    dist = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(dist, min_distance=MIN_DIST, threshold_abs=2, labels=binary)
    if len(coords) == 0:
        return None
    markers = np.zeros(img.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return segmentation.watershed(-dist, markers, mask=binary)


def measure_diameter(lab):
    if lab is None:
        return np.nan
    areas = np.bincount(lab.ravel())[1:]
    areas = areas[areas > 0]
    if len(areas) == 0:
        return np.nan
    return float((2 * np.sqrt(areas / np.pi)).mean())


def measure_all(imgs):
    return np.array([measure_diameter(segment_grains(im)) for im in imgs])


def degradation(labels, true, n=100):
    rows = []
    for noise in NOISE_LEVELS:
        for blur in BLUR_LEVELS:
            errs = []
            for i in range(n):
                r = np.random.default_rng(9000 + i)
                img = render(labels[i], r, noise=noise, blur=blur)
                m = measure_diameter(segment_grains(img))
                if not np.isnan(m):
                    errs.append(abs(m - true[i]))
            rows.append({"noise": noise, "blur": blur, "mae": np.mean(errs)})
    return pd.DataFrame(rows)


def hall_petch(true, meas, seed=0):
    ok = ~np.isnan(meas)
    d_true = true[ok] * PX_PER_UM * 1e-6
    d_meas = meas[ok] * PX_PER_UM * 1e-6
    rng = np.random.default_rng(seed)
    y = SIGMA0 + K_HP / np.sqrt(d_true) + rng.normal(0, NOISE_STD, size=int(ok.sum()))

    x = 1 / np.sqrt(d_meas)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sigma0_hat, k_hat = coef

    x_true = 1 / np.sqrt(d_true)
    A_true = np.column_stack([np.ones_like(x_true), x_true])
    coef_true, *_ = np.linalg.lstsq(A_true, y, rcond=None)

    n = len(y)
    boot = np.empty((1000, 2))
    for b in range(1000):
        idx = rng.integers(0, n, n)
        boot[b] = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0]
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    print(f"hall-petch: sigma0={sigma0_hat:.1f} MPa (true {SIGMA0}), "
          f"k={k_hat:.3f} MPa m^0.5 (true {K_HP})")
    print(f"hall-petch 95% CI: sigma0 [{lo[0]:.1f}, {hi[0]:.1f}], k [{lo[1]:.3f}, {hi[1]:.3f}]")
    print(f"true values inside CI: sigma0 {lo[0] <= SIGMA0 <= hi[0]}, k {lo[1] <= K_HP <= hi[1]}")
    print(f"hall-petch fit to true sizes: sigma0={coef_true[0]:.1f}, k={coef_true[1]:.3f}")
    return x, y, coef, lo, hi


def image_features(img):
    edges = filters.sobel(img)
    edge_density = float((edges > 0.1).mean())
    f = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    cy, cx = np.array(img.shape) // 2
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, r.max(), 9)
    ri = np.digitize(r.ravel(), bins)
    prof = np.bincount(ri, weights=f.ravel()) / np.maximum(np.bincount(ri), 1)
    lv = ndi.uniform_filter(img ** 2, 9) - ndi.uniform_filter(img, 9) ** 2
    return [edge_density, img.mean(), img.std(), float(lv.mean()), *prof]


def uncertainty(imgs, true, seed=0):
    X = np.array([image_features(im) for im in imgs])
    y = true
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]

    preds = []
    for k in range(5):
        rb = np.random.default_rng(100 + k)
        b = rb.integers(0, len(tr), len(tr))
        m = RandomForestRegressor(n_estimators=100, random_state=k, n_jobs=-1)
        m.fit(X[tr][b], y[tr][b])
        preds.append(m.predict(X[te]))
    preds = np.array(preds)
    mu = preds.mean(axis=0)
    sd = preds.std(axis=0)
    err = np.abs(mu - y[te])
    rho = spearmanr(sd, err).statistic
    print(f"uncertainty: test n={len(te)}, MAE={err.mean():.2f} px, "
          f"spearman(std, |err|)={rho:.3f}")
    return rho


def save_sample(imgs):
    lab = segment_grains(imgs[0])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(imgs[0], cmap="gray")
    if lab is not None:
        ax.contour(find_boundaries(lab, mode="inner"), colors="red", linewidths=0.5)
    ax.set_title("sample segmentation")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_segmentation.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    imgs = np.load(DATA / "images.npy")
    labels = np.load(DATA / "labels.npy")
    meta = pd.read_csv(DATA / "meta.csv")
    true = meta["true_diameter"].to_numpy()

    meas = measure_all(imgs)
    ok = ~np.isnan(meas)
    mae = np.abs(meas[ok] - true[ok]).mean()
    r2 = 1 - np.sum((meas[ok] - true[ok]) ** 2) / np.sum((true[ok] - true[ok].mean()) ** 2)
    print(f"measured vs true: n={ok.sum()}/{len(true)}, MAE={mae:.2f} px, R2={r2:.3f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(true[ok], meas[ok], s=12, alpha=0.6)
    lim = [0, max(true.max(), np.nanmax(meas)) * 1.05]
    plt.plot(lim, lim, "k--", label="y = x")
    plt.xlabel("true mean grain diameter (px)")
    plt.ylabel("measured mean grain diameter (px)")
    plt.title(f"MAE {mae:.2f} px, R2 {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "measured_vs_true.png", dpi=150)
    plt.close()
    print("saved results/measured_vs_true.png")

    deg = degradation(labels, true)
    plt.figure(figsize=(6, 5))
    for blur in BLUR_LEVELS:
        sub = deg[deg["blur"] == blur]
        plt.plot(sub["noise"], sub["mae"], "o-", label=f"blur {blur}")
    plt.xlabel("gaussian noise (std)")
    plt.ylabel("mean absolute error (px)")
    plt.title("segmentation error vs degradation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "degradation.png", dpi=150)
    plt.close()
    print("saved results/degradation.png")
    print(deg.pivot(index="noise", columns="blur", values="mae").round(2).to_string())

    x, y, coef, lo, hi = hall_petch(true, meas)
    xs = np.linspace(x.min(), x.max(), 100)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=12, alpha=0.5, label="synthetic data")
    plt.plot(xs, SIGMA0 + K_HP * xs, "k--", label="true Hall-Petch")
    plt.plot(xs, coef[0] + coef[1] * xs, "r-", label="fit to measured sizes")
    plt.xlabel("1/sqrt(d) (m^-0.5)")
    plt.ylabel("yield strength (MPa)")
    plt.title(f"sigma0 {coef[0]:.1f} (true {SIGMA0}), k {coef[1]:.3f} (true {K_HP})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "hall_petch.png", dpi=150)
    plt.close()
    print("saved results/hall_petch.png")

    uncertainty(imgs, true)
    save_sample(imgs)
    print("saved results/sample_segmentation.png")


if __name__ == "__main__":
    main()
