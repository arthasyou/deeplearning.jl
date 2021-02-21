# -------------------------------------------------------
# simple ann template
# -------------------------------------------------------

m = Dense(5, 1)
function loss(x, y)
    sum(y .- m(x))^2
end
opt = Descent(0.01)
x, y = rand(5, 5), fill(0.5, 1, 5)
Flux.train!(loss, params(m), [(x, y)], opt)

# view results
params(m)
loss(x, y)
