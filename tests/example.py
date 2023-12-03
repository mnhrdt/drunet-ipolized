import iio
import drunet

### Test on a RGB image
a = iio.read("tractor_g20.png")
out = drunet.denoise(a, 20)
iio.write("denoised_tractor_g20.png", out)

### Test on a grayscale image
a = iio.read("gtractor_g20.png")
out = drunet.denoise(a, 20)
iio.write("gdenoised_tractor_g20.png", out)
