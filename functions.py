import torch
from torch.autograd import Function, Variable


def concatenate_input_noise_map(input, noise_sigma):
    r"""Implements the first layer of FFDNet. This function returns a
	torch.autograd.Variable composed of the concatenation of the downsampled
	input image and the noise map. Each image of the batch of size CxHxW gets
	converted to an array of size 4*CxH/2xW/2. Each of the pixels of the
	non-overlapped 2x2 patches of the input image are placed in the new array
	along the first dimension.

	Args:
		input: batch containing CxHxW images
		noise_sigma: the value of the pixels of the CxH/2xW/2 noise map
	"""
    # noise_sigma is a list of length batch_size
    #  Extract input dimensions
    N, C, H, W = input.size()
    # Determine data type
    dtype = input.type()
    # Set scale factors
    sca = 2
    sca2 = sca * sca
    # Calculate output dimensions
    Cout = sca2 * C
    Hout = torch.div(H, sca, rounding_mode='floor')
    Wout = torch.div(W, sca, rounding_mode='floor')
    # Define indices for downscaling
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Fill the downsampled image with zeros
    # Initialize downsampled features tensor filled with zeros
    if 'cuda' in dtype:
        downsampledfeatures = torch.cuda.FloatTensor(N, Cout, Hout, Wout).fill_(0)
    else:
        downsampledfeatures = torch.FloatTensor(N, Cout, Hout, Wout).fill_(0)

    # Build the CxH/2xW/2 noise map # Create noise map based on input noise_sigma
    noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)

    #  Populate downsampled features
    for idx in range(sca2):
        downsampledfeatures[:, idx:Cout:sca2, :, :] = \
            input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

    # Concatenate noise map with downsampled features
    return torch.cat((noise_map, downsampledfeatures), 1)


class UpSampleFeaturesFunction(Function):
    r"""Extends PyTorch's modules by implementing a torch.autograd.Function.
	This class implements the forward and backward methods of the last layer
	of FFDNet. It basically performs the inverse of
	concatenate_input_noise_map(): it converts each of the images of a
	batch of size CxH/2xW/2 to images of size C/4xHxW
	"""

    @staticmethod
    def forward(ctx, input):
        # Extract input dimensions
        N, Cin, Hin, Win = input.size()
        # Determine data type
        dtype = input.type()
        # Set scale factor
        sca = 2
        sca2 = sca * sca
        # Calculate output dimensions
        Cout = torch.div(Cin, sca2, rounding_mode='floor')
        Hout = Hin * sca
        Wout = Win * sca
        # Define indices for downscaling
        idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize result tensor filled with zeros
        result = torch.zeros((N, Cout, Hout, Wout)).type(dtype)
        # Populate result tensor with down sampled values
        for idx in range(sca2):
            result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = \
                input[:, idx:Cin:sca2, :, :]

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Extracting dimensions from the size of grad_output
        N, Cg_out, Hg_out, Wg_out = grad_output.size()
        # Determining data type
        dtype = grad_output.data.type()
        # Scaling factor
        sca = 2
        # Calculating input dimensions
        sca2 = sca * sca
        Cg_in = sca2 * Cg_out
        Hg_in = Hg_out // sca
        Wg_in = Wg_out // sca
        # Defining indices for extraction
        idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize grad_input tensor with zeros
        grad_input = torch.zeros((N, Cg_in, Hg_in, Wg_in)).type(dtype)
        # Populate grad_input tensor by extracting values from grad_output tensor
        for idx in range(sca2):
            grad_input[:, idx:Cg_in:sca2, :, :] = \
                grad_output.data[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]
        # Return grad_input tensor as a Variable
        return Variable(grad_input)

upsamplefeatures = UpSampleFeaturesFunction.apply
