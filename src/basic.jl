# gradient 计算函数f(x)微分 一元微分直接在gradient函数后取第一个元组数据
function df1()
    f(x) = 3x^2 + 2x + 1
    df(x) = gradient(f, x)[1]
    df(2)
end

# 如果f(x)的参数是矩阵 gradient调用的函数需要包含在sum()函数里面 f(x)函数运算符前要加.作为广播运算符 计算结果为一个矩阵
function df_arg_matrix()
    f(x) = 3 .* x .^2 + 2 .*x .+ 1
    sf(x) = sum(f(x))
    dsf(x) = gradient(sf, x)[1]
    dsf([2,1])
end

# gradient 计算多元微分 分别计算各元的偏微分
function df_m()
    g(x, y) = (x - y)^2
    dg(x, y) = gradient(g, x, y)
    dg(2,1)
end

# 简单模型
function simple_models()
    W = rand(2, 5)
    b = rand(2)

    predict(x) = W*x .+ b

    function loss(x, y)
        ŷ = predict(x)
        sum((y .- ŷ).^2)
    end

    x, y = rand(5), rand(2)
    bf = loss(x, y)
    gs = gradient(() -> loss(x, y), params(W, b))
    W̄ = gs[W]
    W .-= 0.1 .* W̄
    af = loss(x, y)
    bf, af
end

# 建立分成模型
function build_layers()
    W1 = rand(3, 5)
    b1 = rand(3)
    layer1(x) = W1 * x .+ b1

    W2 = rand(2, 3)
    b2 = rand(2)
    layer2(x) = W2 * x .+ b2

    model(x) = layer2(σ.(layer1(x)))

    model(rand(5))
end

# 建立分成模型 使用linear函数优化
function build_layers_o1()
    function linear(in, out)
      W = randn(out, in)
      b = randn(out)
      x -> W * x .+ b
    end

    linear1 = linear(5, 3) # we can access linear1.W etc
    linear2 = linear(3, 2)

    model(x) = linear2(σ.(linear1(x)))

    model(rand(5))
end

# 建立分成模型 使用Dense
function build_layers_dense()
    model = Chain(
      Dense(10, 5, σ),
      Dense(5, 2),
      softmax)

    model(rand(10))
end
