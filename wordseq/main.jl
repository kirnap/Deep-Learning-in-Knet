include("trainer.jl")
include("lstm_model.jl")


function main()
  # get ready the data
  text = split(readall("ptb.train.txt"))
  vocabulary = Dict()
  batchsize = 20 # Batchsize given in the papers
  for word in text; get!(vocabulary, word, 1+length(vocabulary));end
  learning_rate = 1

  # create validation set
  val_text = split(readall("ptb.valid.txt"))

  # train and validation data
  trn = seqbatch(text, vocabulary, batchsize)
  vld = seqbatch(val_text, vocabulary, batchsize)

  # Create medium lstm
  # info("Compiling the model...")
  # mediumLSTM = compile(:genlstm; nlayer=2, embedding=512, hidden=650, pdrop=0.5, nchar=length(vocabulary))
  # setp(mediumLSTM, lr=learning_rate)

  # Create large lstm
  info("Compiling the model...")
  largeLSTM = compile(:genlstm; nlayer=2, embedding=2000, hidden=1500, pdrop=0.65, nchar=length(vocabulary))
  setp(largeLSTM, lr=learning_rate)


  # prev_tst_err = 10
  info("Training starting...")
   for epoch=1:55
    train(largeLSTM, trn, softloss;gclip=10)
    # tst_err = test(largeLSTM, vld, softloss)
    # if tst_err > prev_tst_err
    #   learning_rate /= 1.2
    # end
    learning_rate = epoch > 14 ? learning_rate/1.15 : learning_rate
    setp(largeLSTM; lr=learning_rate)
    println("Epoch number: $epoch ||", "Train Error: ", test(largeLSTM, trn, softloss),"||",
           "Test Error: ", test(largeLSTM,vld, softloss))
    # prev_tst_err = tst_err
  end
end
main()
