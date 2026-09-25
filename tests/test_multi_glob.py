import logging

import pytest

from bayer.utils import multi_glob


def test_multi_glob_returns_matches(tmp_path):
    path = tmp_path / 'a.fits'
    path.write_bytes(b'')

    result = multi_glob([str(tmp_path / '*.fits')])

    assert result == [str(path)]


def test_multi_glob_warns_on_unmatched_pattern_but_keeps_matches(
        tmp_path, caplog):
    path = tmp_path / 'a.fits'
    path.write_bytes(b'')
    missing = str(tmp_path / 'missing*.fits')
    found = str(tmp_path / '*.fits')

    with caplog.at_level(logging.WARNING, logger='bayer.utils'):
        result = multi_glob([missing, found])

    assert result == [str(path)]
    assert f'no files for pattern "{missing}"' in caplog.text


def test_multi_glob_exits_when_nothing_matches(tmp_path, caplog):
    pattern = str(tmp_path / 'nope*.fits')

    with caplog.at_level(logging.WARNING, logger='bayer.utils'):
        with pytest.raises(SystemExit, match='no files for any pattern'):
            multi_glob([pattern])

    assert f'no files for pattern "{pattern}"' in caplog.text


def test_multi_glob_accepts_a_string_pattern(tmp_path):
    path = tmp_path / 'a.fits'
    path.write_bytes(b'')

    result = multi_glob(str(tmp_path / '*.fits'))

    assert result == [str(path)]


def test_multi_glob_recursive(tmp_path):
    nested = tmp_path / 'sub'
    nested.mkdir()
    path = nested / 'a.fits'
    path.write_bytes(b'')

    result = multi_glob([str(tmp_path / '**' / '*.fits')])

    assert result == [str(path)]
