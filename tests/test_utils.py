import unittest

import numpy as np

from pyimgy.utils import *


class UtilsTest(unittest.TestCase):

    def test_encode_decode_channels(self):
        arr = np.random.randint(256, size=(15, 10, 3), dtype=np.uint8)
        arr_enc = encode_array_channels(arr, ch_dim=2)

        self.assertEqual((15, 10), arr_enc.shape)
        self.assertEqual(np.uint32, arr_enc.dtype)

        arr_dec = decode_array_channels(arr_enc, num_ch=3)

        self.assertEqual(arr.shape, arr_dec.shape)
        self.assertEqual(np.uint8, arr_dec.dtype)
        self.assertTrue(np.all(arr == arr_dec))


if __name__ == '__main__':
    unittest.main()
