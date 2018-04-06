import numpy as np
from scipy import misc
from PIL import Image
import utils
from numpy.fft import fft, ifft

"""I = np.asarray(misc.imread("C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg"))
print(type(I))
i2 = np.asarray(utils.FFT("C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg"))
print(type(i2))"""


def FFT(img):
    # 打开图像文件并获取数据
    print(np.asarray(misc.imread(img)).shape) # shape 为 [1944, 2592, 3]
    srcIm = np.reshape(np.asarray(misc.imread(img)), 1944 * 2592 * 3)
    print("srcIm")
    print(srcIm)
    result = fft(srcIm)  # result.shape=(1944, 2592, 3)
    result = ifft(result)  # result.shape=(1944, 2592, 3)
    result = np.reshape(np.int8(np.real(result)), [2592, 1944, 3])
    im = Image.frombytes('RGB', (2592, 1944), result)
    #im.save('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_im.save.jpg')
    misc.imsave('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_imsave.jpg', im)


if __name__ == '__main__':
    FFT('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg')
    # compare('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg', 'C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_2.jpg')

