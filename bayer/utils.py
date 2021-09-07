import glob


def multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result
