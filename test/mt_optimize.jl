using Symbolics 

@syms a b

ex = 2a + 2b - (a*(a + b))
res = Symbolics.optimize(ex)
println(res)