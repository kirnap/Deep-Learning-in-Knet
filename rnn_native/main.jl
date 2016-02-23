include("rnn_model.jl")

function main()
  # Dataset preparation
  text = readall("../tinyshekaspare.txt")
  char2int = Dict()
  batchsize = 128
  for c in text; get!(char2int, c, 1+length(char2int));end
  vocab_size = length(char2int) # Each char will be represented as one-hot vector in size of vocab_size


  # Compile model
  naiiveRNN = compile(:vanillaRNN, xsize=vocab_size)

  # Set the learning rate
  setp(naiiveRNN; lr=0.0001)

  trn, tst = seqbatch(text, char2int, batchsize)

  for epoch=1:100
    train(naiiveRNN, trn, softloss)
    @printf("epoch number %d || train error: %g || test error: %g\n", epoch, test(naiiveRNN, trn, softloss), test(naiiveRNN, tst, softloss))
  end
end

main()
