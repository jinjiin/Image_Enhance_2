import numpy as np
from scipy import misc
from PIL import Image
import utils
I = np.asarray(misc.imread("C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg"))
print(type(I))
i2 = np.asarray(utils.FFT("C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg"))
print(type(i2))
