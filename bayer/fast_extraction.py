import functools
import math
import numpy as np
import functools

from astropy.stats import sigma_clipped_stats

from bayer.utils import rawpy_to_rgb


class Fast:

    def __init__(self, layers=None, sigma=3, clipping=10):
        """
        :param bayer: None or a two-dimensional bayer matrix assumed to be RGBG
        :param layers: None or a three-dimensional stack of images -- the first index is the image number
        :param sigma: None or used for sigma clipping of the image background

        Either bayer or layers has to be defined
        """

        assert layers is None or np.asarray(layers).ndim == 3

        self._rgb = layers
        self.sigma = sigma
        self.clipping = clipping

    @property
    @functools.lru_cache(maxsize=None)
    def rgb(self):
        return self._rgb

    @property
    @functools.lru_cache(maxsize=None)
    def clipped_rgb(self):
        mean, median, stddev = self._background
        return self._sigma_clip_image(self.rgb, mean + stddev * self.clipping)

    @property
    @functools.lru_cache(maxsize=None)
    def _background(self):
        return sigma_clipped_stats(self.rgb, sigma=self.sigma, cenfunc='mean', axis=(1, 2))

    @property
    def background_mean(self):
        return self._background[0]

    @property
    def background_median(self):
        return self._background[1]

    @property
    def background_stddev(self):
        return self._background[2]

    @classmethod
    def _sigma_clip_image(cls, image, threshold):

        clipped = np.ma.array(image, fill_value=np.nan)
        clipped[np.isnan(clipped)] = np.ma.masked
        clipped[np.isinf(clipped)] = np.ma.masked

        with np.errstate(invalid='ignore'):
            clipped.mask |= clipped < np.reshape(threshold, (np.shape(image)[0], 1, 1))

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
        mean, median, stddev = self._background
        return self._sigma_clip_image(self.de_rotated_rgb, mean + stddev * self.clipping)

    @classmethod
    def _calculate_de_rotation_angle(cls, image):
        assert image.ndim == 2

        # 1nd binarize image
        mean = np.mean(image)
        if mean == np.ma.masked:
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
