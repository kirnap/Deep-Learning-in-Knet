# The implementation of Neural Turing Machine (NTM)

# === Memory ===
# In the original paper the memory matrix is defined as <N X M>
# in Knet the result is the same but we took the tranpose of memory matrix, i.e. Mem,
# which is <M X N> and the weighting is still <N X 1>, to orient the operations in Knet


# === Controller ===
# Takes the input x and r at any given time
# Feeds the head layer via its hidden layer
# Outputs from its hidden layer via softmax layer

# In the cosine similarity operation we need to compare the key vector of length M, i.e key, with the column of the memory matrix
# Shift range also be provided to create s vector at a given time

# ==== Head ====
# Every Head returns:
# Key
# beta
# interpolation gate, i.e. interpol
# circular shifter, i.e. shift
# read head has additional output of read vector, i.e. read
# write head has additional outputs: erase and add vectors
# erase vector has a dimension of <M x 1>, i.e erase
# add vector also has a dimension of <M x1>, i.e add
# feeded by the hidden layer of controller


using Knet

@knet function controller(x,r; hidden=0, init=Xaiver())
	return wbf2(x,r; out=hidden, winit=init, f=:soft, o...)
end

# Generic head
@knet function head(h,Mem; memory=init=Xaiver(),shift_range=3, write=false)
	key = wbf(h; out=size(Mem,1), f=:sigm)
	beta = wbf(h; out=1, f=:sigm)
	interpol = wbf(h;out=1, f=:sigm)
	shift = wbf(h;out=shift_range, f=:soft)
	gamma = wbf(h;out=1, f=:relu) .+ 1 # TODO gamma has to be bigger than 1 any better?

	if write
		erase = wbf(h; out=size(Mem,1), f=:sigm)
		add = wbf(h; out=size(Mem,1), f=:tanh)
		return key, beta, interpol, shift, gamma, erase, add
	else
		read = wbf(h; out=size(Mem,1), f=:sigm)
		return key, beta, interpol, shift, gamma, read
	end

end

@knet function ntm(x;hidden=0, msize=(20,128), nrheads=1, nwheads=1, shift_range=3, init=Xaiver())
	Mem = par(init=Gaussian(0,0.01), dims=mszie)
	h = controller(x,r;hidden=hidden)

end
