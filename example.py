import numpy as np
import iio
from my_interface import denoise 


        ### Test on a RGB image ###
a = iio.read("tractor.png")
out = denoise(a, 10)
iio.write("denoise_s10_color.tiff", out)

        ### Test on a grayscale image ###

a = np.expand_dims(np.mean(a,2), -1)
out = denoise(a, 10)
iio.write("denoise_s10_grayscale.tiff", out)

