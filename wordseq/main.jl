include("trainer.jl")
include("lstm_model.jl")


function main()
  # get ready the data
  text = readall("ptb.train.txt")
  words = split(text)
  vocabulary = Dict()
  batchsize = 20 # Batchsize given in the papers
  for word in words; get!(vocabulary, word, 1+length(vocabulary));end
  trn, tst = seqbatch(words, vocabulary, batchsize)
  learning_rate = 1

  # Create medium lstm
  info("Compiling the model...")
  mediumLSTM = compile(:genlstm; nlayer=2, hidden=650, pdrop=0.5, nchar=length(vocabulary))
  setp(mediumLSTM, lr=learning_rate)

  info("Training starting...")
  for epoch=1:39
    learning_rate = epoch > 6 ? learning_rate - 0.5*learning_rate : learning_rate
    train(mediumLSTM, trn, softloss;gclip=5)
  end
end
main()
