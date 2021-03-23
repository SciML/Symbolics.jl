using Symbolics, SparseArrays, Test
@variables a b c1 c2 c3 d e g

# Multiple argument matrix
h = [a + b + c1 + c2,
     c3 + d + e + g,
     0] # uses the same number of arguments as our application
h_julia(a, b, c, d, e, g) = [a[1] + b[1] + c[1] + c[2],
                             c[3] + d[1] + e[1] + g[1],
                             0]
function h_julia!(out, a, b, c, d, e, g)
    out .= [a[1] + b[1] + c[1] + c[2], c[3] + d[1] + e[1] + g[1], 0]
end

h_str = Symbolics.build_function(h, [a], [b], [c1, c2, c3], [d], [e], [g])
h_oop = eval(h_str[1])
h_str_par = Symbolics.build_function(h, [a], [b], [c1, c2, c3], [d], [e], [g], parallel=Symbolics.MultithreadedForm())
h_oop_par = eval(h_str_par[1])
h_par_rgf = Symbolics.build_function(h, [a], [b], [c1, c2, c3], [d], [e], [g], parallel=Symbolics.MultithreadedForm(), expression=false)
h_ip! = eval(h_str[2])
h_ip_skip! = eval(Symbolics.build_function(h, [a], [b], [c1, c2, c3], [d], [e], [g], skipzeros=true, fillzeros=false)[2])
h_ip_skip_par! = eval(Symbolics.build_function(h, [a], [b], [c1, c2, c3], [d], [e], [g], skipzeros=true, parallel=Symbolics.MultithreadedForm(), fillzeros=false)[2])
inputs = ([1], [2], [3, 4, 5], [6], [7], [8])

@test h_oop(inputs...) == h_julia(inputs...)
@test h_oop_par(inputs...) == h_julia(inputs...)
@test h_par_rgf[1](inputs...) == h_julia(inputs...)
out_1 = similar(h, Int)
out_2 = similar(out_1)
h_ip!(out_1, inputs...)
h_julia!(out_2, inputs...)
@test out_1 == out_2
out_1 = similar(h, Int)
h_par_rgf[2](out_1, inputs...)
@test out_1 == out_2
fill!(out_1, 10)
h_ip_skip!(out_1, inputs...)
@test out_1[3] == 10
out_1[3] = 0
@test out_1 == out_2

fill!(out_1, 10)
h_ip_skip_par!(out_1, inputs...)
@test out_1[3] == 10
out_1[3] = 0
@test out_1 == out_2

# Multiple input matrix, some unused arguments
h_skip = [a + b + c1; c2 + c3 + g] # skip d, e
h_julia_skip(a, b, c, d, e, g) = [a[1] + b[1] + c[1]; c[2] + c[3] + g[1]]
function h_julia_skip!(out, a, b, c, d, e, g)
    out .= [a[1] + b[1] + c[1]; c[2] + c[3] + g[1]]
end

h_str_skip = Symbolics.build_function(h_skip, [a], [b], [c1, c2, c3], [], [], [g], checkbounds=true)
h_oop_skip = eval(h_str_skip[1])
h_ip!_skip = eval(h_str_skip[2])
inputs_skip = ([1], [2], [3, 4, 5], [], [], [8])

@test h_oop_skip(inputs_skip...) == h_julia_skip(inputs_skip...)
out_1_skip = Array{Int64}(undef, 2)
out_2_skip = similar(out_1_skip)
h_ip!_skip(out_1_skip, inputs_skip...)
h_julia_skip!(out_2_skip, inputs_skip...)
@test out_1_skip == out_2_skip

# Same as above, except test ability to call with non-matrix arguments (i.e., for `nt`)
inputs_skip_2 = ([1], [2], [3, 4, 5], [], (a = 1, b = 2), [8])
@test h_oop_skip(inputs_skip_2...) == h_julia_skip(inputs_skip_2...)
out_1_skip_2 = Array{Int64}(undef, 2)
out_2_skip_2 = similar(out_1_skip_2)
h_ip!_skip(out_1_skip_2, inputs_skip_2...)
h_julia_skip!(out_2_skip_2, inputs_skip_2...)
@test out_1_skip_2 == out_2_skip_2

# Multiple input scalar
h_scalar = a + b + c1 + c2 + c3 + d + e + g
h_julia_scalar(a, b, c, d, e, g) = a[1] + b[1] + c[1] + c[2] + c[3] + d[1] + e[1] + g[1]
h_str_scalar = Symbolics.build_function(h_scalar, [a], [b], [c1, c2, c3], [d], [e], [g])
h_oop_scalar = eval(h_str_scalar)
@test h_oop_scalar(inputs...) == h_julia_scalar(inputs...)

@variables z[1:100]
@test isequal(simplify(Symbolics.unflatten_long_ops(sum(z))),
              simplify(sum(z)))

@test isequal(simplify(Symbolics.unflatten_long_ops(prod(z))),
              simplify(prod(z)))

@variables t x(t) y(t) k
f = eval(build_function((x+y)/k, [x,y,k]))
@test f([1,1,2]) == 1

f = eval(build_function([(x+y)/k], [x,y,k])[1])
@test f([1,1,2]) == [1]

f = eval(build_function([(x+y)/k], [x,y,k])[2])
z = [0.0]
f(z, [1,1,2])
@test z == [1]

f = eval(build_function(sparse([1],[1], [(x+y)/k], 10,10), [x,y,k])[1])

@test size(f([1.,1.,2])) == (10,10)
@test f([1.,1.,2])[1,1] == 1.0
@test sum(f([1.,1.,2])) == 1.0

let # ModelingToolkit.jl#800
    @variables x
    y = sparse(1:3,1:3,x)

    f1,f2 = build_function(y,x)
    sf1, sf2 = string(f1), string(f2)
    @test !contains(sf1, "CartesianIndex")
    @test !contains(sf2, "CartesianIndex")
    @test contains(sf1, "SparseMatrixCSC(")
    @test contains(sf2, ".nzval")
end

let # Symbolics.jl#123
    ns = 6
    @variables x[1:ns]
    @variables u
    @variables M[1:36]
    @variables qd[1:6]
    output_eq = u*(qd[1]*(M[1]*qd[1] + M[1]*qd[3] + M[1]*qd[4] + M[25]*qd[5] + M[31]*qd[6] + M[7]*qd[2]))

    str = build_function(output_eq, x, target=Symbolics.CTarget())
    @test str == """
    #include <math.h>
    void diffeqf(double* du, const double* RHS1) {
      du[0] = qd₁ * u * (M₁ * qd₁ + M₁ * qd₃ + M₁ * qd₄ + M₂₅ * qd₅ + M₃₁ * qd₆ + M₇ * qd₂);
    }
    """
end

using SymbolicUtils.Code: Func, toexpr
@variables t x(t)
D = Differential(t)
expr = toexpr(Func([D(x)], [], D(x)))
@test expr.args[2].args[end] == expr.args[1].args[1] # check function body and function arg
@test expr.args[2].args[end] == :(xˍt(t))


let # issue#136
    @variables x y
    A = sparse(Tridiagonal([x^i for i in 1:N-1],
                           [x^i * y^(8-i) for i in 1:N],
                           [y^i for i in 1:N-1]))

    val = Dict(x=>1, y=>2)
    B = map(A) do e
        Num(substitute(e, val))
    end

    C = copy(B) - 100*I


    f = build_function(A,[x,y],parallel=Symbolics.MultithreadedForm())[2]
    g = eval(f)

    g(C, [1,2])
    @test contains(repr(f), "schedule")
    @test isequal(C, B)
end
