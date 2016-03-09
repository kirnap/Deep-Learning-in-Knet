# Training script for multilayer lstm
include("lstm_model.jl")

include("trainer.jl")

function main()

  # define the learning rate
  learning_rate = 1.2

  # Create the train dataset
  text = split(readall("ptb.char.train.txt"))
  char2int = Dict()
  batchsize = 128

  for c in text; get!(char2int, c, 1+length(char2int)); end
  vocab_size = length(char2int)

  # Create validation set
  val_text = split(readall("ptb.char.valid.txt"))
  val_char2int = Dict()
  for c in val_text; get!(val_char2int, c, 1+length(val_char2int)); end

  # Compile 3 layers LSTM
  info("Compiling the model...")
  mlstm = compile(:charlm; nlayers=2, embedding=256, hidden=512, pdrop=0.2, nchar=vocab_size)
  setp(mlstm; lr=learning_rate) # set it to 1.2 to make it 1 in the 1st training case.

  # Create train and valid data
  trn = seqbatch(text, char2int, batchsize)
  vld = seqbatch(val_text, val_char2int, batchsize)


  prev_tst_err = 0
  info("Training starting...")
  for epoch=1:101
    train(mlstm, trn, softloss; gclip=5)
    tst_err = test(mlstm, vld, softloss)
    if tst_err > prev_tst_err
      learning_rate /= 1.2
    end
    setp(mlstm; lr=learning_rate)
    println("Epoch number: $epoch ||", "Train Error: ", test(mlstm, trn, softloss),"||",
           "Test Error: $tst_err")
    prev_tst_err = tst_err
  end
end

main()
