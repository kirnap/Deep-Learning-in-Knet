# 2 levels of lstm layers
# Both of them are 35 unrolled steps and two layers
# Medium lstm:
# 650 units per layer and [-0.05, 0.05] initialization. 50% dropout rate.
#
using Knet

# Define the LSTM
@knet function mlstm(x;fbias=1,winit=Uniform(-0.05, 0.05),o...)
  input = wbf2(x,h;winit=winit,  o..., f=:sigm)
  forget = wbf2(x,h;winit=winit, o..., f=:sigm, binit=Constant(fbias))
  output = wbf2(x,h;winit=winit, o..., f=:sigm)
  newmem = wbf2(x,h; o..., f=:tanh)
  cell = input .* newmem + cell .* forget
  h = tanh(cell) .* output
  return h

end

# Define the dropout layer
@knet function mlstmdrop(a;pdrop=0.5, hidden=650)
  b = mlstm(a; out=hidden)
  return drop(b; pdrop=pdrop)
end

# desired LSTM structures representing the medium and large LSTMs
@knet function genlstm(x; nlayer=0, embedding=0, hidden=0, pdrop=0, nchar=0)
  embedding_layer = wdot(x; out=embedding)
  c = repeat(embedding_layer; frepeat=:mlstmdrop, nrepeat=nlayer, hidden=hidden, pdrop=pdrop)
  return wbf(c; out=nchar, f=:soft)
end
