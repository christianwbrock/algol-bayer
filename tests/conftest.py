import contextlib
import os


def pytest_addoption(parser):
    parser.addoption(
        "--dont-plot",
        action='store_true',
        default=False,
        help="Disable plotting in tests"
    )


def pytest_configure(config):
    """
    Disable plotting if the parameter --no-plot has been used.

    Set the matplotlib backend to 'agg' for tests that do not require a GUI.
    This is useful for running tests in environments without a display.
    """
    if config.getoption("--dont-plot"):
        import matplotlib
        matplotlib.use('agg')


@contextlib.contextmanager
def changed_environ(**changes):
    """\
    When changing the environ for a single test, all future tests will also see the changes.
    This context manager resets the environment variable after leaving the context.
    """

    old_environ = dict(os.environ)
    os.environ.update(changes)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


# Implement the pytest_generate_tests hook
def pytest_generate_tests(metafunc):
    if 'raw_filename' in metafunc.fixturenames:
        metafunc.parametrize("raw_filename", _get_raw_filenames())
    elif 'fits_filename' in metafunc.fixturenames:
        metafunc.parametrize("fits_filename", _get_fits_filenames())
    elif 'fits_bayer_filename' in metafunc.fixturenames:
        metafunc.parametrize("fits_bayer_filename", _get_fits_bayer_filenames())
    elif 'image_filename' in metafunc.fixturenames:
        metafunc.parametrize("image_filename",
                             _get_fits_bayer_filenames() + _get_raw_filenames() + _get_fits_filenames())


def _data_dir():
    target = 'data'

    here = os.path.dirname(__file__)

    while True:
        join = os.path.join(here, target)
        parent, _ = os.path.split(here)

        if os.path.exists(join):
            return join

        elif parent and parent != here:
            here = parent

        else:
            return None


def _get_raw_filenames():
    datadir = _data_dir()

    filenames = [
        'betori.CR2',
        'alpori.CR2',
        'alpcma.NEF'
    ]

    return [os.path.join(datadir, fn) for fn in filenames]


def _get_fits_filenames():
    datadir = _data_dir()

    filenames = [
        'alpleo.FIT',
        'gamOri_02.fit',
        'alpOri_01.fit',
    ]

    return [os.path.join(datadir, fn) for fn in filenames]


def _get_fits_bayer_filenames():
    datadir = _data_dir()

    filenames = [
        'minus20degC_Dark_60_secs_100.fits',
    ]

    return [os.path.join(datadir, fn) for fn in filenames]
