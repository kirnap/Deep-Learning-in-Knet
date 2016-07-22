@doc """ 
	Training procedure:
	Suppose an input sequence of x1-x2-x3, 
	The backward lstm gets input as 0-x3-x2,
	The forward lstm gets input as 0-x1-x2
"""
# TODO: for now lets fix the sequence length as 20 so that forward and backward lstm networks goes 20 consecutive steps forward. It should be changed to sentence length

""" Sequentially minibatches the data """
function seqbatch(seq, mdict, batchsize; ret_sparse=true)
	data = Any[]
	T = div(length(seq), batchsize) #find the number of batches
	for t=1:T
		d = zeros(Float32, length(mdict), batchsize) # data skeleton in each batch
		for b=1:batchsize
			word_code = mdict[seq[t + (b-1) * T]] # choose the desired words
	 		d[word_code, b] = 1
		end
		if (ret_sparse)
			d_sparse = sparse(d)
		else
  			d_sparse = d
		end
		push!(data, d_sparse)
  	end
	return data
end


""" Go to forward in lstm networks either forward network or backward networks for a given sequence contains minibatches of length sequence """
function lstmforward(net, sequence; forwardlstm=true)
	hiddenstack = Any[]
	beginning = zeros(sequence[1])
	if !forwardlstm
		reverse!(sequence)
	end
	beginresult = sforw(net, beginning)
	push!(hiddenstack, copy(beginresult))

	for i=1:(length(sequence)-1)
		result = sforw(net, sequence[i])
		push!(hiddenstack, copy(result))
	end
	
	if forwardlstm
		reverse!(hiddenstack)
	end
	@assert length(hiddenstack) == length(sequence)
	return hiddenstack
end


""" Feed the merge layer until the hidden states are consumed  ypreds is a stack which has the end-sequence value at top"""
function mergeforw(net, forwhiddenstack, backhiddenstack)
	@assert length(forwhiddenstack) == length(backhiddenstack)
	ypreds = Any[]
	while !isempty(forwhiddenstack)
		forwhidden = pop!(forwhiddenstack)
		backhidden = pop!(backhiddenstack)
		ypred = sforw(net, forwhidden, backhidden)
		push!(ypreds, copy(ypred))
	end
	return ypreds
end


""" Comes back from t=final to t=beginning """
function mergeback(net, ygolds)
	gfs = Any[]
	gbs = Any[]
	for i=length(ygolds):-1:1
		(gf, gb) = sback(net, ygolds[i], softloss; getdx=true)
		push!(gfs, copy(gf))
		push!(gbs, copy(gb))
	end
	reverse!(gfs) # to provide pop! property for back functions of forward network
	return (gfs, gbs)
end


""" Gets the gradients in desired order and computes the gradients of lstm networks """
function lstmback(net, grads)
	while !isempty(grads)
		g = pop!(grads)
		sback(net, g)
	end
end
