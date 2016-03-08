#=
Contains the splitter, trainer and test functions for RNNs

=#
# Minibatching for sequential data
# TODO Which one is efficient in splitting data?
function seqbatch(seq, dict, batchsize)
data = Any[]
T = div(length(seq), batchsize) #find the number of batches
  for t=1:T
    d = zeros(Float32, length(dict), batchsize) # data skeleton in each batch
    for b=1:batchsize
      character_code = dict[seq[t + (b-1) * T]] # choose the desired char
      d[character_code, b] = 1
    end
      push!(data, d)
  end
  return data
end


# Define the training for RNN
function train(knetf, batcharray, loss; nforw=100, gclip=0)
  reset!(knetf)
  ystack = Any[]
  T = length(batcharray) - 1
  for t = 1:T
    x = batcharray[t] # set the time step dataset from batch array
    y = batcharray[t+1] # set the desired output for that time step
    sforw(knetf,x; dropout=true)
    push!(ystack, y)
    if (t % nforw == 0 || t==T)
      while !isempty(ystack)
        ygold = pop!(ystack)
        sback(knetf, ygold, loss)
      end
      update!(knetf;gclip=gclip)
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
