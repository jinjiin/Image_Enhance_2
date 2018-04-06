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
    #srcIm = np.reshape(np.asarray(misc.imread(img)), 1944 * 2592 * 3)
    #srcIm = misc.imread(img)
    srcIm = Image.open(img)
    print("srcIm")
    print(srcIm)
    result = fft(srcIm)  # result.shape=(1944, 2592, 3)
    print(type(result))
    result = ifft(result)  # result.shape=(1944, 2592, 3)
    result = np.int8(np.real(result))
    print(type(result))
    print(result.shape)

    im = Image.frombytes("RGB", (1944, 2592), result)
    im.save('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_im.save.jpg')
    #misc.imsave('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_imsve.jpg',im)


if __name__ == '__main__':
    FFT('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg')
    # compare('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg', 'C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_2.jpg')

