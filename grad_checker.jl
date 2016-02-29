function main()
include(Pkg.dir("Knet/examples/mnist.jl"))
ninputs = 784;
noutputs =10;
xtrn = zeros(ninputs, 60000)
for i=1:size(MNIST.xtrn,4); xtrn[:,i] = vec(MNIST.xtrn[:,:,1,i]);end
xtst = zeros(ninputs, 10000)
for i=1:size(MNIST.xtst,4); xtst[:,i] = vec(MNIST.xtst[:,:,1,i]);end
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w = randn(noutputs, ninputs) * 0.001;
B = zeros(noutputs,1)

X = xtrn[:,1:100]
Y = ytrn[:,1:100]
(gw, gb) = numeric_grad(w,B,X,Y);
(gradW, gradB) = forw(w, B, X, Y)[2:end];

diff = sqrt(sum((gradW - gw) .^ 2) + sum((gradB - gb) .^ 2))
println("Diff: $diff")
if diff < 1e-7
  println("Gradient Checking Passed")
else
  println("Gradient Checking Failed!")
  println("Diff must be < 1e-7")
end

return (gw, gb, gradW, gradB)
end

function forw(W, b, X, Y)

  ypred = _forw(W,b,X)
  soft_loss = (-sum(Y.*log(ypred)))
  derror = (ypred-Y) # error w.r.t unnormalized probabilities.
  db = sum(derror,2);
  dW = derror * X' # error w.r.t weights
  return (soft_loss, dW, db)

end

function _forw(W, b, X)

  diverprob = W * X .+ b # this are the inputs of softmax

  return exp(diverprob) ./ sum(exp(diverprob),1)

end

function numeric_grad(W,b,X,Y)
  epsilon = 0.0001

  gw = zeros(size(W))
  gb = zeros(size(b))

  for i=1:size(gw,1)
    b_plus = copy(b);
    b_minus = copy(b);
    b_plus[i,1] = b[i,1] + epsilon
    b_minus[i,1]= b[i,1] - epsilon
    J_plus = forw(W, b_plus, X, Y)[1]
    J_minus = forw(W, b_minus, X, Y)[1]
    gb[i,1] = ((J_plus - J_minus)/(2*epsilon))

    for j=1:size(gw,2)
      W_plus = copy(W);
      W_minus = copy(W);
      W_plus[i,j] = W[i,j]+epsilon
      W_minus[i,j] = W[i,j]-epsilon
      J_plusW = forw(W_plus, b, X, Y)[1]
      J_minusW = forw(W_minus,b,X,Y)[1]
      gw[i,j] = ((J_plusW - J_minusW)/(2*epsilon))
    end
  end

  return (gw, gb)
end
