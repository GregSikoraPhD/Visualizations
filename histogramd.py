import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def transform_to_prc(n: int, v: np.array) -> np.array:
    return 100*v/n


def get_bins(v: np.array, limit_prc: float) -> np.array:
    return np.transpose(np.array(np.where(v > limit_prc)))


def get_bins_dsc(v: np.array, w: np.array) -> np.array:
    return v[[np.where(w == i)[0][0] for i in np.sort(w)[::-1]]]


def compute_edges_lines(v: np.array, d: int, w: np.array) -> tuple:
    y1 = [v[i][j] for i, j in zip(range(0, d), w)]
    y2 = [v[i][j+1] for i, j in zip(range(0, d), w)]
    return y1, y2


def plot_regions(y: tuple, min_data: float, max_data: float, f: float) -> None:
    plt.plot(y[0], 'k')
    plt.plot(y[1], 'k')
    plt.ylim([min_data-0.1, max_data+0.1])
    plt.title(label=f'Region of {int(f)}% of vectors')
    plt.show()


def histogram_d(data: pd.DataFrame, bins: tuple, limit_prc: float) -> None:
    h, edges = np.histogramdd(data.values, bins=bins)
    h_prc = transform_to_prc(data.shape[0], h)
    m = get_bins(h_prc, limit_prc)
    h_group = h_prc[h_prc > limit_prc]
    m = get_bins_dsc(m, h_group)
    h_group = np.sort(h_group)[::-1]
    if len(m) > 0:
        print(f'Data are grouped in {len(m)} groups: '+', '.join(str(int(i))+'%' for i in h_group) +
              f'. Rest of vectors ({int(100-np.sum(h_group))}%) are not concentrated in at least {limit_prc}% groups.')
        for i, val in enumerate(m):
            y = compute_edges_lines(edges, data.shape[1], val)
            plot_regions(y, np.min(data.values), np.max(data.values), h_group[i])
    else:
        print(f'There are no classes with at least {limit_prc}% of vectors. Try to lower number of bins.')
