import unittest

from pyimgy.core import *


class ImageConverterTest(unittest.TestCase):

    def _assert_error(self, func, *args, **kwargs):
        with self.assertRaises(AssertionError):
            func(*args, **kwargs)

    def test_get_array_shape(self):
        self.assertEqual(get_array_shape(np.zeros((10, 8))), 'WH')
        self.assertEqual(get_array_shape(np.zeros((3, 10, 8))), '3WH')
        self.assertEqual(get_array_shape(np.zeros((1, 10, 8))), '1WH')
        self.assertEqual(get_array_shape(np.zeros((10, 8, 3))), 'WH3')
        self.assertEqual(get_array_shape(np.zeros((10, 8, 1))), 'WH1')
        self.assertEqual(get_array_shape(np.zeros((10, 8, 4))), 'WHC')
        self.assertEqual(get_array_shape(np.zeros((4, 10, 8))), 'CWH')
        self.assertEqual(get_array_shape(np.zeros((1, 3, 10, 8))), '13WH')
        self.assertEqual(get_array_shape(np.zeros((1, 10, 8, 3))), '1WH3')
        self.assertEqual(get_array_shape(np.zeros((1, 1, 10, 8))), '11WH')
        self.assertEqual(get_array_shape(np.zeros((1, 4, 10, 8))), '1CWH')

    def test_get_array_shape_invalid(self):
        self._assert_error(get_array_shape, np.zeros(1))
        self._assert_error(get_array_shape, np.zeros((1, 2, 3, 4, 5)))
        self._assert_error(get_array_shape, np.zeros((1, 5, 1)))

    def test_convert_shape_WH3(self):
        arr = np.arange(240).reshape((10, 8, 3))

        self.assertEqual(convert_image(arr, shape='3WH').shape, (3, 10, 8))
        self.assertEqual(convert_image(arr, shape='WH3').shape, (10, 8, 3))
        self.assertEqual(convert_image(arr, shape='13WH').shape, (1, 3, 10, 8))
        self.assertEqual(convert_image(arr, shape='1WH3').shape, (1, 10, 8, 3))

        self._assert_error(convert_image, arr, shape='1WH')
        self._assert_error(convert_image, arr, shape='WH1')

    def test_convert_shape_WH1(self):
        arr = np.arange(80).reshape((10, 8))

        self.assertEqual(convert_image(arr, shape='1WH').shape, (1, 10, 8))
        self.assertEqual(convert_image(arr, shape='WH1').shape, (10, 8, 1))
        self.assertEqual(convert_image(arr, shape='3WH').shape, (3, 10, 8))
        self.assertEqual(convert_image(arr, shape='WH3').shape, (10, 8, 3))
        self.assertEqual(convert_image(arr, shape='13WH').shape, (1, 3, 10, 8))
        self.assertEqual(convert_image(arr, shape='1WH3').shape, (1, 10, 8, 3))


if __name__ == '__main__':
    unittest.main()
