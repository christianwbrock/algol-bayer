import math
import warnings
from functools import cached_property

import numpy as np
from astropy.stats import sigma_clipped_stats


class FastExtraction:

    def __init__(self, image_layers, sigma=3, clipping=10):
        """\
        Parameters
        ----------
        image_layers: array_like of shape (num_layers, rows, columns)
        sigma: number
            Used for sigma clipping of the image background
        clipping: number
            After sigma clipping the layers are cut at mean + clipping * stddev
        """

        assert image_layers is not None and np.ndim(image_layers) == 3
        assert sigma > 0
        assert clipping > 0

        self.layers = np.asarray(image_layers)
        self.sigma = sigma
        self.clipping = clipping

    @cached_property
    def clipped_layers(self):
        return self._clip_image(self.layers, self.background_mean + self.background_stddev * self.clipping)

    @cached_property
    def _background_stats(self):
        return sigma_clipped_stats(self.layers, sigma_upper=self.sigma, sigma_lower=1000, cenfunc='mean', axis=(1, 2))

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
    def _clip_image(cls, image, threshold):

        clipped = np.copy(image)
        clipped[np.isinf(clipped)] = np.nan

        n, __, __ = np.shape(image)

        threshold = np.reshape(threshold, (n, 1, 1))
        clipped[clipped < threshold] = np.nan

        return clipped

    @property
    def de_rotation_angles_deg(self):
        return np.rad2deg(self.de_rotation_angles_rad)

    @cached_property
    def de_rotation_angles_rad(self):
        return np.array([self._calculate_de_rotation_angle(layer) for layer in self.clipped_layers])

    @cached_property
    def de_rotated_layers(self):
        from scipy.ndimage import rotate

        angle_deg = np.mean(self.de_rotation_angles_deg)
        return rotate(self.layers, angle_deg, axes=(1, 2), mode='constant', cval=np.nan)

    @cached_property
    def clipped_de_rotated_layers(self):
        mean, median, stddev = self._background_stats
        return self._clip_image(self.de_rotated_layers, mean + stddev * self.clipping)

    @classmethod
    def _calculate_de_rotation_angle(cls, image):
        """\
        Calculate image orientation using image moments as described in
        <https://en.wikipedia.org/wiki/Image_moment#Examples_2>.

        """
        assert image.ndim == 2

        indices_y, indices_x = np.indices(image.shape)

        m00 = np.nansum(image)
        m10 = np.nansum(image * indices_x)
        m01 = np.nansum(image * indices_y)
        m11 = np.nansum(image * indices_x * indices_y)
        m20 = np.nansum(image * indices_x * indices_x)
        m02 = np.nansum(image * indices_y * indices_y)

        avg_x = m10 / m00
        avg_y = m01 / m00
        mu_11_ = m11 / m00 - avg_x * avg_y
        mu_20_ = m20 / m00 - avg_x ** 2
        mu_02_ = m02 / m00 - avg_y ** 2

        angle = 0.5 * np.arctan2(2 * mu_11_, mu_20_ - mu_02_)

        while angle > math.pi / 2:
            angle -= math.pi

        while angle < -math.pi / 2:
            angle += math.pi

        return angle


def find_slit_in_images(rgb, background_mean, scale=1.5):

    __, size_y, __ = rgb.shape
    wo_background = rgb - np.reshape(background_mean, (-1, 1, 1))

    # In some rows all values maybe nan, this can be ignored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        slit_function = np.nanmean(wo_background, axis=(0, 2))
    miny, maxy = _find_smallest_interval(slit_function)

    if scale != 1.0:
        center = (miny + maxy) / 2
        width = (maxy - miny) / 2

        miny = center - scale * width
        maxy = center + scale * width

    miny = math.floor(miny)
    maxy = math.ceil(maxy)

    miny = np.clip(miny, 0, size_y - 1)
    maxy = np.clip(maxy, 0, size_y - 1)

    return miny, maxy


def _find_smallest_interval(data, area_percentage=0.95):
    """
    For a given array, find the smallest interval containing more than 95% of the data.
    """
    assert np.ndim(data) == 1

    sum_data = np.nansum(data)
    target_sum = sum_data * area_percentage
    assert 0 < target_sum < sum_data

    # simplifies the area calculation below
    cumulative_data_sum = np.nancumsum(data)

    smallest_interval_size = len(data)
    smallest_interval = None, None

    a = 0
    b = 1
    while a < b < len(data):
        interval_sum = cumulative_data_sum[b] - cumulative_data_sum[a]

        if interval_sum < target_sum:
            b += 1
        else:  # interval_sum >= target_sum
            if b - a < smallest_interval_size:
                smallest_interval_size = b - a
                smallest_interval = a, b
            a += 1

    return smallest_interval
