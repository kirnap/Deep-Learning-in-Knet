# Training script for multilayer lstm
include("lstm_model.jl")
# Here I import my predefined train and test functions included in the repo
include("/Users/omer/Documents/Developer/kdeep/rnn_native/trainer.jl")

function main()

  # Create the dataset
  text = readall("../tinyshekaspare.txt")
  char2int = Dict()
  batchsize = 128

  for c in text; get!(char2int, c, 1+length(char2int)); end
  vocab_size = length(char2int)

  # Compile 3 layers LSTM
  info("Compiling the model...")
  mlstm = compile(:charlm; nlayers=3, embedding=256, hidden=512, pdrop=0.2, nchar=vocab_size)
  setp(mlstm; lr=1.0)

  # Create train and test data
  trn, tst = seqbatch(text, char2int, batchsize)

  info("Training starting...")
  for epoch=1:20
    train(mlstm, trn, softloss; gclip=5)
    println("Epoch number: ", epoch, "||", "Train Error: ", test(mlstm, trn, softloss),"||",
            "Test Error: ", test(mlstm, tst, softloss))
  end
end

main()
