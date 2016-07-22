using Knet

@doc """ Bi-directional LSTM for language modelling """

@knet function stackedlstm(word; embedding=0, nlayer=0, hidden=0, o...)
	wordvec = wdot(word, out=embedding)
	return repeat(wordvec; frepeat=:lstm, nrepeat=nlayer, out=hidden, o...)
end

@knet function mergelayer(forward, backward; vocabsize=0, o...)
	return wbf2(forward, backward; out=vocabsize, f=:soft)
end