function seqbatch(seq, mdict, batchsize; ret_sparse=true)

  data = Any[]

  T = div(length(seq), batchsize) #find the number of batches
  for t=1:T
    d = zeros(Float32, length(mdict), batchsize) # data skeleton in each batch
    for b=1:batchsize
      character_code = mdict[seq[t + (b-1) * T]] # choose the desired char
      d[character_code, b] = 1
    end
    if (ret_sparse)
      d_sparse = sparse(d)
    else
      d_sparse = d
    end
    push!(data, d_sparse)
  end
  return data
end

function train(knetf, batcharray, loss; nforw=35, gclip=0)
  reset!(knetf)
  ystack = Any[]
  T = length(batcharray) - 1
  for t = 1:T
    x = batcharray[t] # set the time step dataset from batch array
    y = batcharray[t+1] # set the desired output for that time step
    sforw(knetf, x; dropout=true)
    push!(ystack, y)
    if (t % nforw == 0 || t==T)
      while !isempty(ystack)
        ygold = pop!(ystack)
        sback(knetf, ygold, loss)
      end
      update!(knetf; gclip=gclip)
      reset!(knetf; keepstate=true)
    end
  end
end

# Test happens in single step
function test(knetf, data, loss)
  sumloss = numloss = 0
  for t=1:length(data)-1
    x = data[t]
    ygold = data[t+1]
    ypred = forw(knetf, x) # Use forw in test time
    sumloss += loss(ypred, ygold)
    numloss += 1
  end
  return sumloss / numloss
end
