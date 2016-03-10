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
  val_vocab = Dict();
  for c in val_text; get!(val_vocab, c, 1+length(val_vocab)); end

  # create train and validation data
  trn = seqbatch(text, vocabulary, batchsize)
  vld = seqbatch(val_text, vocabulary, batchsize)

  # Create medium lstm
  info("Compiling the model...")
  mediumLSTM = compile(:genlstm; nlayer=2, embedding=512, hidden=650, pdrop=0.5, nchar=length(vocabulary))
  setp(mediumLSTM, lr=learning_rate)

  prev_tst_err = 0
  info("Training starting...")
  for epoch=1:39
    train(mediumLSTM, trn, softloss;gclip=5)
    tst_err = test(mediumLSTM, vld, softloss)
    if tst_err > prev_tst_err
      learning_rate /= 1.2
    end
    setp(mediumLSTM; lr=learning_rate)
    println("Epoch number: $epoch ||", "Train Error: ", test(mediumLSTM, trn, softloss),"||",
           "Test Error: $tst_err")
    prev_tst_err = tst_err
  end
end
main()
