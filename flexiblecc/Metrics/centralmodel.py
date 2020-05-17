import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import uuid
from PIL import Image
import glob
import shutil


def make_cm_vec_plot(cm, image_dimensions_step):
    w, h = (cm.image_width, cm.image_height)

    w_cor = image_dimensions_step if w % image_dimensions_step == 0 else 0
    h_cor = image_dimensions_step if h % image_dimensions_step == 0 else 0

    points = np.mgrid[0:w + w_cor:image_dimensions_step, 0:h + h_cor:image_dimensions_step].T.reshape(-1, 2)

    sample_vec = lambda el: cm.sample(el[0], el[1])

    vecs = np.array(list(map(sample_vec, points)))

    X, Y = points.T
    Z = np.zeros_like(X)

    U, V, W = vecs.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, np.max(W) * 2)
    plt.tight_layout()
    
    return ax


def make_cm_vec_plot_gif(cm, image_dimensions_step, gif_file_out, azim_start=0, azim_end=360, azim_step=5, elev_start=0,
                         elev_end=90, elev_step=10, duration=150, loop=0, dpi=None, show=False):
    elev_cor = elev_step if elev_step != 360 else 0
    azim_cor = azim_step if azim_step != 360 else 0

    img_idx = 0
    runId = str(uuid.uuid4())
    os.makedirs(runId)

    ax = make_cm_vec_plot(cm, image_dimensions_step)

    for elev in range(elev_start, elev_end + elev_cor, elev_step):
        for azim in range(azim_start, azim_end + azim_cor, azim_step):
            ax.view_init(elev, azim)
            plt.draw()

            plt.savefig(os.path.join(runId, "image_{}.jpg".format(img_idx)), dpi=dpi)
            img_idx += 1

            if show:
                plt.pause(.001)
    if not show:
        plt.close()

    fp_in = os.path.join(runId, "image_*.jpg")
    fp_out = gif_file_out

    files = np.array(sorted(glob.glob(fp_in)))
    sort_idxs = np.argsort([int(f.split(".")[0].split("_")[-1]) for f in files])
    files = files[sort_idxs]

    img, *imgs = [Image.open(f) for f in files]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=loop)

    shutil.rmtree(runId)
