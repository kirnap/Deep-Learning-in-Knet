#=
Vanilla Implementation of RNN using Knet
For more explanations please refer : http://knet.readthedocs.org/en/latest/
=#

using Knet

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
