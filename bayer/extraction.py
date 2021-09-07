import functools
import math
import numpy as np
import functools

from astropy.stats import sigma_clipped_stats

from bayer.to_rgb import rawpy_to_rgb


class FastExtraction:

    def __init__(self, rgb_layers, sigma=3, clipping=10):
        """
        :param rgb_layers: None or a three-dimensional stack of images -- the first index is the image number
        :param sigma: None or used for sigma clipping of the image background

        Either bayer or layers has to be defined
        """

        assert rgb_layers is not None and np.ndim(rgb_layers) == 3

        self.rgb = np.asarray(rgb_layers)
        self.sigma = sigma
        self.clipping = clipping

    @property
    @functools.lru_cache(maxsize=None)
    def clipped_rgb(self):
        return self._sigma_clip_image(self.rgb, self.background_mean + self.background_stddev * self.clipping)

    @property
    @functools.lru_cache(maxsize=None)
    def _background_stats(self):
        return sigma_clipped_stats(self.rgb, sigma=self.sigma, cenfunc='mean', axis=(1, 2))

    @property
    def background_mean(self):
        return self._background_stats[0]

    @property
    def background_median(self):
        return self._background_stats[1]

    @property
    def background_stddev(self):
        return self._background_stats[2]

    @classmethod
    def _sigma_clip_image(cls, image, threshold):

        clipped = np.copy(image)
        clipped[np.isinf(clipped)] = np.nan

        n, __, __ = np.shape(image)

        threshold = np.reshape(threshold, (n, 1, 1))
        clipped[clipped < threshold] = np.nan

        return clipped

    @property
    def de_rotation_angles_deg(self):
        return np.rad2deg(self.de_rotation_angles_rad)

    @property
    @functools.lru_cache(maxsize=None)
    def de_rotation_angles_rad(self):
        return np.array([self._calculate_de_rotation_angle(layer) for layer in self.clipped_rgb])

    @property
    @functools.lru_cache(maxsize=None)
    def de_rotated_rgb(self):
        from scipy.ndimage.interpolation import rotate

        angle_deg = np.mean(self.de_rotation_angles_deg)
        return rotate(self.rgb, angle_deg, axes=(1, 2), mode='constant', cval=np.nan)

    @property
    @functools.lru_cache(maxsize=None)
    def clipped_de_rotated_rgb(self):
        mean, median, stddev = self._background_stats
        return self._sigma_clip_image(self.de_rotated_rgb, mean + stddev * self.clipping)

    @classmethod
    def _calculate_de_rotation_angle(cls, image):
        assert image.ndim == 2

        # 1nd binarize image
        mean = np.nanmean(image)
        if np.isnan(mean):
            raise ValueError('mean does not exist -- check the image and evtl. try a lower sigma')

        binary = image >= mean

        # 2rd create index array where binary > 0
        [indices_y, indices_x] = np.indices(binary.shape)
        invalid = np.full(binary.shape, fill_value=-1)
        y = np.where(binary, indices_y, invalid)
        x = np.where(binary, indices_x, invalid)
        y = y[y[:, :] != -1]
        x = x[x[:, :] != -1]
        assert y.shape == x.shape
        yx = np.moveaxis(np.array((y, x)), 0, 1)

        # 3th get rotation angle from svd
        yx = yx - np.mean(yx, axis=(0,))
        u, s, v = np.linalg.svd(yx, full_matrices=False)
        rot_svd = np.arctan2(v[0, 0], v[0, 1])

        while rot_svd > math.pi / 2:
            rot_svd -= math.pi

        while rot_svd < -math.pi / 2:
            rot_svd += math.pi

        return rot_svd

    @classmethod
    def center_of_gravity(cls, image):

        if image.ndim > 2:
            return np.array([cls.center_of_gravity(layer) for layer in image])

        indices = np.indices(image.shape)

        center = np.nansum(image * indices, axis=(1, 2)) / np.nansum(image)
        return center
