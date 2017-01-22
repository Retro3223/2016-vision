import numpy 


class NumpyPool:
    def __init__(self, shape=(240, 320)):
        self.raws = _NumpyPool(shape=shape, dtype='uint16')
        self.grays = _NumpyPool(shape=shape, dtype='uint8')
        self.colors = _NumpyPool(shape=(shape[0], shape[1], 3), dtype='uint8')
        self.xyzs = _NumpyPool(shape=(3, shape[0], shape[1]), dtype='float32')

    def get_raw(self):
        return self.raws.get_array()

    def release_raw(self, raw):
        return self.raws.release_array(raw)

    def get_gray(self):
        return self.grays.get_array()

    def release_gray(self, gray):
        return self.grays.release_array(gray)

    def get_color(self):
        return self.colors.get_array()

    def release_color(self, color):
        return self.colors.release_array(color)

    def get_xyz(self):
        return self.xyzs.get_array()

    def release_xyz(self, xyz):
        return self.xyzs.release_array(xyz)


class _NumpyPool:
    def __init__(self, dtype, shape):
        self.pool = []
        self.shape = shape
        self.dtype = dtype

    def get_array(self):
        if len(self.pool) != 0:
            return self.pool.pop()
        result = numpy.zeros(shape=self.shape, dtype=self.dtype)
        return result

    def release_array(self, raw):
        assert raw.shape == self.shape, "%s %s" % (raw.shape, self.shape)
        assert raw.dtype == self.dtype
        self.pool.append(raw)

