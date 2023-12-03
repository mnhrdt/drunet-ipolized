
def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
	'''
	model:
	L: input Low-quality image
	refield: effective receptive filed of the network, 32 is enough
	min_size: min_sizeXmin_size image, e.g., 256X256 image
	sf: scale factor for super-resolution, otherwise 1
	modulo: 1 if split
	'''
	import torch
	h, w = L.size()[-2:]

	top = slice(0, (h//2//refield+1)*refield)
	bottom = slice(h - (h//2//refield+1)*refield, h)
	left = slice(0, (w//2//refield+1)*refield)
	right = slice(w - (w//2//refield+1)*refield, w)
	Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
	Es = [model(Ls[i]) for i in range(4)]
	b, c = Es[0].size()[:2]
	E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
	E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
	E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
	E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
	E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
	return E


def denoise(img, sigma):
	"""
	img: numpy array (H,W,1) or (H,W,3) in [0,255] range
	sigma: estimated std value (for image in [0,255] range)
	return: numpy array (H,W,1) or (H,W,3). Range is [0,255]
	"""

	import torch
	from models.network_unet import UNetRes

	# select device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# normalize data into [0,1]
	img = img / 255

	# look if data are color or grayscale
	H,W,C = img.shape
	if C == 3 :
		model_name = 'drunet_color'
	else:
		model_name = 'drunet_gray'

	# absolute path to the model file
	from os.path import abspath, dirname
	M = f"{abspath(dirname(abspath(__file__)))}/model_zoo/{model_name}.pth"

	# load a pre-trained model
	model = UNetRes(in_nc=C+1, out_nc=C, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
	model.load_state_dict(torch.load(M), strict=True)
	model.to(device)
	model.eval()

	# freeze netwok parameters
	# todo, done in their file script, but I really think this is
	# unnecessary if we already use torch.no_grad
	for k, v in model.named_parameters():
		v.requires_grad = False

	noisy = img.copy()

	# put noisy as a torch tensor of shape B=1,C,H,W
	noisy = torch.Tensor(noisy.transpose(2,0,1)).unsqueeze(0)

	# concatenate a scalar noise map and put into device
	noisy_input = torch.cat((noisy, torch.Tensor([sigma/255]).repeat(1, 1, H, W)), dim=1).to(device)

	# evaluate the network
	with torch.no_grad():
		if H//8 == 0 and W//8==0:
			out = model(noisy_input)
		else:
			out = test_onesplit(model, noisy_input, refield=64)

	# convert out into a numpy array (H,W) for grayscale or (H,W,3) for RGB
	out = out.detach().cpu().numpy().squeeze()
	if C == 3:
		out = out.transpose(1,2,0)

	return 255 * out


# extract a named option from the command line arguments
def pick_option(
		o,  # option name, including hyphens
		d   # default value
		):
	from sys import argv as v
	return type(d)(v[v.index(o)+1]) if o in v else d


# main function
if __name__ == "__main__":
	i  = pick_option("-i", "in.png")   # input filename
	o  = pick_option("-o", "out.png")  # output filename
	σ  = pick_option("-s", 10)         # denoiser sigma

	import iio
	x = iio.read(i)
	y = denoise(x, σ)
	iio.write(o, y)
