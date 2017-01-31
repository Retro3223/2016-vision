from utils import rgbhex2bgr
from unittest import TestCase


class BGRTests(TestCase):
    def test_convert_to_bgr(self):
        assert rgbhex2bgr(0xf7ff1e) == (0x1e, 0xff, 0xf7)
