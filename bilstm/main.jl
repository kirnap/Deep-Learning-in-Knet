include("model.jl")
include("train.jl")


function train(forwnetwork, backnetwork, mergenetwork, data, SEQUENCE_LENGTH)
	for i=1:(length(data)-SEQUENCE_LENGTH)
		sequence = data[i:(i+SEQUENCE_LENGTH)]
		
		forwhiddenstack = lstmforward(forwnetwork, sequence; forwardlstm=true)
		backhiddenstack = lstmforward(backnetwork, sequence; forwardlstm=false)
		ypreds = mergeforw(mergenetwork, forwhiddenstack, backhiddenstack)
		
		(gfs, gbs) = mergeback(mergenetwork, sequence)
		lstmback(forwnetwork, gfs)
		lstmback(backnetwork, gbs)

		update!(mergenetwork)
		update!(forwnetwork)
		update!(backnetwork)

		reset!(mergenetwork; keepstate=true)
		reset!(forwnetwork; keepstate=true)
		reset!(backnetwork; keepstate=true)
	end
end


function test(forwnetwork, backnetwork, mergenet, data, loss, SEQUENCE_LENGTH)
	sumloss = numloss = 0
	for i=1:(length(data)-SEQUENCE_LENGTH)
		sequence = data[i:(i+SEQUENCE_LENGTH)]
		numloss += length(sequence)
		forwhiddenstack = lstmforward(forwnetwork, sequence; forwardlstm=true)
		backhiddenstack = lstmforward(backnetwork, sequence; forwardlstm=false)
		ypreds = mergeforw(mergenet, forwhiddenstack, backhiddenstack)
		for k=length(sequence):-1:1
			ypred = pop!(ypreds)
			sumloss += loss(ypred, sequence[k])
		end
	end
	return sumloss / numloss
end

function main()
	SEQUENCE_LENGTH = 19
	learning_rate = 1
	text = split(readall("ptb.train.txt"))
	devtext = split(readall("ptb.valid.txt"))

	vocabulary = Dict()
	batchsize = 20
	for word in text; get!(vocabulary, word, 1+length(vocabulary));end

	info("Compiling the models")
	forwnetwork = compile(:stackedlstm; nlayer=1, embedding=512, hidden=100)
	backnetwork = compile(:stackedlstm; nlayer=1, embedding=512, hidden=100)
	mergenetwork = compile(:mergelayer, vocabsize=length(vocabulary))
	
	setp(forwnetwork, lr=learning_rate)
	setp(backnetwork, lr=learning_rate)
	setp(mergenetwork, lr=learning_rate)

	info("Getting data ready")
	data = seqbatch(text, vocabulary, batchsize; ret_sparse=true)
	dev = seqbatch(devtext, vocabulary, batchsize; ret_sparse=true)

	dev_err = test(forwnetwork, backnetwork, mergenetwork, dev, softloss, SEQUENCE_LENGTH)
	println("Before we start: $dev_err")
	info("Training started...")

	for epoch=1:15
		train(forwnetwork, backnetwork, mergenetwork, data, SEQUENCE_LENGTH)
		dev_err = test(forwnetwork, backnetwork, mergenetwork, dev, softloss)
		println("Dev error after epoch : $dev_err")
	end
end
main()