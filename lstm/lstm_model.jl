#=
This module contains the karpathy blog implementation of lstm in character sequence
data
=#

using Knet

# Here I import my predefined train and test functions included in the repo 
include("/Users/omer/Documents/Developer/kdeep/rnn_native/trainer.jl")

# Define the lstm model
@knet function clsclstm(x; fbias=1, o...)
  input = wbf2(x,h; o..., f:=sigm)
  forget = wbf2(x,h; o...,f:=sigm, binit=Constant(fbias))
  output = wbf2(x,h; o..., f:=sigm)
  newmem = wbf2(x,h; o..., f:=tanh)
  cell = input .* newmem + forget .* cell
  h = output .* tanh(cell)
  return h
end
