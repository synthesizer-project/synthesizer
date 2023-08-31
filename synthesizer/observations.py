import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr


class IFU:
    def __init__(self, ifu, xedges, yedges):
        return


class Image:
    def __init__(self, img, xedges, yedges):
        self.img = img
        self.xedges = xedges
        self.yedges = yedges

    def img_panel(self, ax, im, vmin=None, vmax=None, scaling=False, cmap=cmr.neutral):
        if not vmin:
            vmin = np.min(im)
        if not vmax:
            vmax = np.max(im)

        if scaling:
            im = scaling(im)

        ax.set_axis_off()
        ax.imshow(
            im, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower"
        )  # choose better scaling

        return ax

    def make_image_plot(
        self, vmin=None, vmax=None, scaling=False, cmap=cmr.neutral, show=False
    ):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plt.subplots_adjust(left=0, top=1, bottom=0, right=1, wspace=0.01, hspace=0.0)

        ax = self.img_panel(
            ax, self.img, vmin=vmin, vmax=vmax, scaling=scaling, cmap=cmap
        )

        if show:
            plt.show()

        return fig, ax
