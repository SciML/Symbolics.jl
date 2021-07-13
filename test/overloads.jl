using Symbolics
using Symbolics: Sym, FnType, Term, value, scalarize
using LinearAlgebra
using SparseArrays: sparse
using Test

a, b, c = :runtime_symbol_value, :value_b, :value_c
vars = @variables t $a $b(t) $c[1:3](t)
@test t isa Num
@test a === :runtime_symbol_value
@test b === :value_b
@test c === :value_c
@test isequal(vars[1], t)
@test isequal(vars[2], Num(Sym{Real}(a)))
@test isequal(vars[3], Num(Sym{FnType{Tuple{Any},Real}}(b)(value(t))))

vars = @variables a,b,c,d,e,f,g,h,i
@test isequal(vars, [a,b,c,d,e,f,g,h,i])
@test isequal(transpose(a), a)
@test isequal(a', a)
@test isequal(sincos(a), (sin(a), cos(a)))

@test substitute(a ~ b, Dict(a=>1, b=>c)) == (1 ~ c)

# test hashing
aa = a; # old a

@variables a

@test isequal(a, aa)
@test hash(a) == hash(aa)

@test isequal(Symbolics.get_variables(a+aa+1), [a])

@test hash(a+b ~ c+d) == hash(a+b ~ c+d)

# test some matrix operations don't throw errors
X = [0 b c;
     d e f;
     g h i]

@test iszero(expand(det(X) - (-b * (d*i-f*g) + c * (d*h - e*g))))


F = lu(X)
@test_nowarn lu(X'), lu(transpose(X))
@test F.p == [2, 1, 3]
R = simplify.(F.L * F.U - X[F.p, :])
@test iszero(R)
@test simplify.(F \ X) == I
@test Symbolics._solve(X, X, true) == I
inv(X)
qr(X)

X2 = [0 b c; 0 0 0; 0 h 0]
@test_throws SingularException lu(X2)
F2 = lu(X2, check=false)
@test F2.info == 1

# test operations with sparse arrays
# note `isequal` instead of `==` because `==` would give another symbolic type

# test that we can create a symbolic sparse array
Oarray = zeros(Num, 2,2)
Oarray[2,2] = a
@test isequal(sparse(Oarray), sparse([2], [2], [a]))

@test isequal(a * sparse([2], [2], [1]), sparse([2], [2], [a * 1]))

A = sparse([2], [2], [a])
B = sparse([2], [2], [b])
@test isequal(A + B, sparse([2], [2], [a+b]))

C = sparse([1, 2], [2, 1], [c, c])
D = sparse([1, 2], [2, 1], [d, d])

@test isequal(C * D, sparse([1,2], [1,2], [c * d, c * d]))

@variables t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]
J = Num[Dx(eqs[1].rhs) Dy(eqs[1].rhs) Dz(eqs[1].rhs)
 Dx(eqs[2].rhs) Dy(eqs[2].rhs) Dz(eqs[2].rhs)
 Dx(eqs[3].rhs) Dy(eqs[3].rhs) Dz(eqs[3].rhs)]

J = expand_derivatives.(J)
using LinearAlgebra
luJ = lu(J,Val(false))

@variables M[1:2,1:2]
inv(M)

@variables b[1:2]
M = [1 0; 0 2]
M \ b
M \ reshape(b,2,1)
M = [1 1; 0 2]
M \ reshape(b,2,1)


M = [1 a; 0 2]
M \ b
M \ [1, 2]

# test det
@variables X[1:4,1:4]
d1 = det(X, laplace=true)
d2 = det(X, laplace=false)
_det1 = eval(build_function(d1,X))
_det2 = eval(build_function(d2,X))
A = [1 1 1 1
     1 0 1 1
     1 1 0 1
     1 1 1 0]
@test _det1(map(Num, A)) == -1
@test _det2(map(Num, A)) == -1

@variables X[1:3,1:3]
d1 = det(X, laplace=true)
d2 = det(X, laplace=false)
_det1 = eval(build_function(d1, X))
_det2 = eval(build_function(d2, X))
A = [1 1 1
     1 0 1
     1 1 1]
@test _det1(map(Num, A)) == 0
@test _det2(map(Num, A)) == 0

@variables a b c d
z1 = a + b * im
z2 = c + d * im
@test z1 * 2 - Complex(2a, 2b) == 0
@test isequal(2z1, Complex(2a, 2b))
@test isequal(z1 / z1, 1)
@test isequal(z1 / z2, Complex((a*c + b*d)/(c^2 + d^2), (b*c - a*d)/(c^2 + d^2)))
@test isequal(1 / z2, Complex(c/(c^2 + d^2), -d/(c^2 + d^2)))
@test isequal(z1 / c, Complex(a/c, b/c))
@test isequal(a / z2, Complex(a*c/(c^2 + d^2), a*d/(c^2 + d^2)))
@test isequal(z1 * z2, Complex(a*c - b*d, a*d + b*c))
@test isequal(z1 - z2, Complex(a - c, b - d))
@test isequal(z1 + z2, Complex(a + c, b + d))
@test isequal(z1 + 2, Complex(a + 2, b))
@test isequal(2 + z1, Complex(2 + a, b))
@test isequal(z1 - 2, Complex(a - 2, b))
@test isequal(2 - z1, Complex(2 - a, -b))
@test isequal(z1 ^ 2, a^2 - b^2 + 2a*b*im)

@test isequal((0 ~ a+0*im), 0 ~ a)
@test isequal((im ~ b+c*im), [0 ~ b; 1 ~ c])
@test isequal((0 ~ z1), [0 ~ a, 0 ~ b])
@test isequal((z1 ~ z2), [a ~ c, b ~ d])

@test a + im === Complex(a, Num(true))
@test real(a) === a
@test conj(a) === a
@test imag(a) === Num(0)

@test isequal(sign(x), Num(SymbolicUtils.Term{Int}(sign, [Symbolics.value(x)])))
@test sign(Num(1)) isa Num
@test isequal(sign(Num(1)), Num(1))
@test isequal(sign(Num(-1)), Num(-1))

using IfElse: ifelse
@test isequal(Symbolics.derivative(abs(x), x), ifelse(signbit(x), -1, 1))
@test isequal(Symbolics.derivative(sign(x), x), 0)
@test isequal(Symbolics.derivative(signbit(x), x), 0)

@test iszero(Num(0.0))
@test isone(Num(1.0))
@test isone(complex(Num(1), Num(0)))
@test iszero(complex(Num(0), Num(0)))

x = Num.(randn(10))
@test norm(x) == norm(Symbolics.value.(x))
@test norm(x, Inf) == norm(Symbolics.value.(x), Inf)
@test norm(x, 1) == norm(Symbolics.value.(x), 1)
@test norm(x, 1.2) == norm(Symbolics.value.(x), 1.2)

@variables x[1:2]
@test isequal(scalarize(norm(x)), sqrt(abs2(x[1]) + abs2(x[2])))
@test isequal(scalarize(norm(x, Inf)), max(abs(x[1]), abs(x[2])))
@test isequal(scalarize(norm(x, 1)), abs(x[1]) + abs(x[2]))
@test isequal(scalarize(norm(x, 1.2)), (abs(x[1])^1.2 + abs(x[2])^1.2)^(1/1.2))

@variables x y
@test isequal(expand((x+y)^2), x^2 + y^2 + 2x*y)

@variables t p x(t) y(t) z(t)
@test isequal(substitute(y ~ x*p, Dict(x => z, y => t)), t ~ z*p)
@test ~(!((1 < x) & (x < 2) | (x >= 100) ⊻ (x <= 1000) & (x != 100))) isa Num

# Maybe move me

@variables x[1:3]
ex = x[1]+x[2]
@test isequal(Symbolics.get_variables(ex), Symbolics.scalarize(x[1:2]))

@variables x
A = [x[1] 2
     2    0.0]
B = [x[1] 1.0
    2.0 0.0]
@test_throws ArgumentError Matrix{Float64}(A)
@test Matrix{Float64}(A-B) isa Matrix{Float64}
@test Matrix{Float64}(A-B) == [0.0 1.0;0.0 0.0]

@test isequal(simplify(cos(x)^2 + sin(x)^2 + im * x), 1 + x*im)
