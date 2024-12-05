import functools
import math

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

    @property
    @functools.lru_cache(maxsize=None)
    def clipped_layers(self):
        return self._clip_image(self.layers, self.background_mean + self.background_stddev * self.clipping)

    @property
    @functools.lru_cache(maxsize=None)
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

    @property
    @functools.lru_cache(maxsize=None)
    def de_rotation_angles_rad(self):
        return np.array([self._calculate_de_rotation_angle(layer) for layer in self.clipped_layers])

    @property
    @functools.lru_cache(maxsize=None)
    def de_rotated_layers(self):
        from scipy.ndimage import rotate

        angle_deg = np.mean(self.de_rotation_angles_deg)
        return rotate(self.layers, angle_deg, axes=(1, 2), mode='constant', cval=np.nan)

    @property
    @functools.lru_cache(maxsize=None)
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


def find_slit_in_images(rgb, background_mean, scale=1.0):

    __, size_y, __ = rgb.shape
    wo_background = rgb - np.reshape(background_mean, (-1, 1, 1))

    slit_function = np.nanmean(wo_background, axis=(0, 2))
    slit_center, slit_size = _center_of_gravity(slit_function)

    miny = math.floor(slit_center - scale * slit_size)
    maxy = math.ceil(slit_center + scale * slit_size)

    miny = np.clip(miny, 0, size_y - 1)
    maxy = np.clip(maxy, 0, size_y - 1)

    return miny, maxy


def find_spectra_in_layers(rgb, background_mean, scale=1.5):

    __, __, size_x = rgb.shape
    wo_background = rgb - np.reshape(background_mean, (-1, 1, 1))

    spectra = np.nanmean(wo_background, axis=1)

    spectrum_locations, spectrum_widths = _center_of_gravity(spectra)

    left_most_spectrum = np.argmin(spectrum_locations)
    right_most_spectrum = np.argmax(spectrum_locations)

    minx = math.floor(spectrum_locations[left_most_spectrum] - scale * spectrum_widths[left_most_spectrum])
    maxx = math.ceil(spectrum_locations[right_most_spectrum] + scale * spectrum_widths[right_most_spectrum])

    minx = np.clip(minx, 0, size_x - 1)
    maxx = np.clip(maxx, 0, size_x - 1)

    return minx, maxx


def _center_of_gravity(data):
    """\
    Return the weighted mean and standard deviation using moments.

    Mean is the first raw moment while variance is the second central moment of the data.
    For more details have a look at https://en.wikipedia.org/wiki/Moment_(mathematics) or
    https://en.wikipedia.org/wiki/Image_moment.
    .
    """

    if data.ndim > 1:
        result = [_center_of_gravity(data[i]) for i in range(data.shape[0])]
        result = np.transpose(result)
        return result[0], result[1]

    assert data.ndim == 1
    valid_indices = np.isfinite(data)

    indices = np.arange(data.shape[-1])
    indices = indices[valid_indices]
    data = data[valid_indices]

    mean = np.nansum(indices * data) / np.nansum(data)
    variance = np.nansum((indices - mean) ** 2 * data) / np.nansum(data)

    if variance < 0:
        # this may happen for negative data?
        variance = -variance

    return mean, math.sqrt(variance)
