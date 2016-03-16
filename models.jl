
# NTM implementation in Knet

using Knet

# Define the controller network
# Controller network is a typical Feed forward deep neural network
@knet function controller(x;hidden=100, f=:relu)
  h1 = wbf(x; out=hidden, f=f
  return wbf(h1;f=:soft, out=10) # TODO decide the output numbers
end
 
