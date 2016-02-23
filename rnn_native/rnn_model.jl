#=
Vanilla Implementation of RNN using Knet
For more explanations please refer : http://knet.readthedocs.org/en/latest/
=#

using Knet

# Minibatching for sequential data
# TODO Which one is efficient in splitting data?
function seqbatch(seq, dict, batchsize)
train_data = Any[]
test_data = Any[]
T = div(length(seq), batchsize) #find the number of batches
  for t=1:T
    d = zeros(Float32, length(dict), batchsize) # data skeleton in each batch
    for b=1:batchsize
      character_code = dict[seq[t + (b-1) * T]] # choose the desired char
      d[character_code, b] = 1
    end
    if length(train_data) < 0.8 * T
      push!(train_data, d)
    else
      push!(test_data, d)
    end
  end
  return train_data, test_data
end

# Define your RNN as a Knet function
# Model outputs in every timestep
@knet function vanillaRNN(x; hsize=100, xsize=vocab_size)
  Wxh = par(init=Xavier(), dims=(hsize, xsize))
  Whh = par(init=Xavier(), dims=(hsize, hsize))
  Who = par(init=Xavier(), dims=(xsize, hsize))
  bxh = par(init=Constant(0), dims=(hsize, 1))
  bho = par(init=Constant(0), dims=(xsize, 1))
  h = tanh(Wxh * x .+ Whh * h .+ bxh)
  return soft(bho .+ Who * h)
end
