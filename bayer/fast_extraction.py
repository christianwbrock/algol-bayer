import numpy as np
from astropy.stats import sigma_clipped_stats

from bayer.utils import bayer_to_rgb


class Fast:

    def __init__(self, bayer=None, layers=None, sigma=10):
        """
        :param bayer: None or a two-dimensional bayer matrix assumed to be RGBG
        :param layers: None or a three-dimensional stack of images -- the first index is the image number
        :param sigma: None or used for sigma clipping of the image background

        Either bayer or layers has to be defined
        """

        if bayer is None and layers is None or bayer is not None and layers is not None:
            raise ValueError('either bayer or rgb must be not None')

        assert bayer is None or np.asarray(bayer).ndim == 2
        assert layers is None or np.asarray(layers).ndim == 3

        self.bayer = bayer
        self._rgb = layers
        self.sigma = sigma

        # everything else is calculated lazily
        self._clipped = None
        self._de_rotation_angles_rad = None
        self._de_rotated_rgb = None
        self._clipped_de_rotated_rgb = None

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = bayer_to_rgb(self.bayer)

        return self._rgb

    @property
    def clipped_rgb(self):
        if self._clipped is None:
            self._clipped = self._sigma_clip_image(self.rgb, self.sigma)

        return self._clipped

    @classmethod
    def _sigma_clip_image(cls, image, sigma):
        if sigma is None:
            return image

        clipped = np.ma.array(image, fill_value=np.nan)
        clipped[np.isnan(clipped)] = np.ma.masked
        clipped[np.isinf(clipped)] = np.ma.masked
        for i in range(clipped.shape[0]):
            (mean, median, stddev) = sigma_clipped_stats(clipped[i], sigma=sigma, cenfunc='mean')

            with np.errstate(invalid='ignore'):
                clipped.mask[i] |= clipped[i] < (mean + stddev * sigma)
        return clipped

    @property
    def de_rotation_angles_deg(self):
        return np.rad2deg(self.de_rotation_angles_rad)

    @property
    def de_rotation_angles_rad(self):
        if self._de_rotation_angles_rad is None:
            self._de_rotation_angles_rad = np.array([self._calculate_de_rotation_angle(layer) for layer in self.clipped_rgb])

        return self._de_rotation_angles_rad

    @property
    def de_rotated_rgb(self):
        if self._de_rotated_rgb is None:
            from scipy.ndimage.interpolation import rotate

            angle_deg = np.mean(self.de_rotation_angles_deg)
            self._de_rotated_rgb = rotate(self.rgb, angle_deg, axes=(1, 2), mode='constant', cval=np.nan)

        return self._de_rotated_rgb

    @property
    def clipped_de_rotated_rgb(self):
        if self._clipped_de_rotated_rgb is None:
            self._clipped_de_rotated_rgb = self._sigma_clip_image(self.de_rotated_rgb, self.sigma)

        return self._clipped_de_rotated_rgb

    @classmethod
    def _calculate_de_rotation_angle(cls, image):
        assert image.ndim == 2

        # 1nd binarize image
        binary = image >= np.mean(image)

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
        yx = yx - np.mean(yx, axis=(0))
        u, s, v = np.linalg.svd(yx)
        rot_svd = np.arctan2(v[0,0], v[0,1])

        return rot_svd

    @classmethod
    def center_of_gravity(cls, image):

        if image.ndim > 2:
            return np.array([cls.center_of_gravity(layer) for layer in image])

        indices = np.indices(image.shape)

        center = np.nansum(image * indices, axis=(1, 2)) / np.nansum(image)
        return center
