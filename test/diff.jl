using Symbolics
using Test
using IfElse

# Derivatives
@variables t σ ρ β
@variables x y z
@variables uu(t) uuˍt(t)
D = Differential(t)
D2 = Differential(t)^2
Dx = Differential(x)

@test Symbol(D(D(uu))) === Symbol("uuˍtt(t)")
@test Symbol(D(uuˍt)) === Symbol(D(D(uu)))

test_equal(a, b) = @test isequal(simplify(a), simplify(b))

#@test @macroexpand(@derivatives D'~t D2''~t) == @macroexpand(@derivatives (D'~t), (D2''~t))

@test isequal(expand_derivatives(D(t)), 1)
@test isequal(expand_derivatives(D(D(t))), 0)

dsin = D(sin(t))
@test isequal(expand_derivatives(dsin), cos(t))

dcsch = D(csch(t))
@test isequal(expand_derivatives(dcsch), simplify(-coth(t) * csch(t)))

@test isequal(expand_derivatives(D(-7)), 0)
@test isequal(expand_derivatives(D(sin(2t))), simplify(cos(2t) * 2))
@test isequal(expand_derivatives(D2(sin(t))), simplify(-sin(t)))
@test isequal(expand_derivatives(D2(sin(2t))), simplify(-sin(2t) * 4))
@test isequal(expand_derivatives(D2(t)), 0)
@test isequal(expand_derivatives(D2(5)), 0)

# Chain rule
dsinsin = D(sin(sin(t)))
test_equal(expand_derivatives(dsinsin), cos(sin(t))*cos(t))

d1 = D(sin(t)*t)
d2 = D(sin(t)*cos(t))
@test isequal(expand_derivatives(d1), simplify(t*cos(t)+sin(t)))
@test isequal(expand_derivatives(d2), simplify(cos(t)*cos(t)+(-sin(t))*sin(t)))

eqs = [σ*(y-x),
       x*(ρ-z)-y,
       x*y - β*z]
jac = Symbolics.jacobian(eqs, [x, y, z])
test_equal(jac[1,1], -1σ)
test_equal(jac[1,2], σ)
test_equal(jac[1,3], 0)
test_equal(jac[2,1],  ρ - z)
test_equal(jac[2,2], -1)
test_equal(jac[2,3], -1x)
test_equal(jac[3,1], y)
test_equal(jac[3,2], x)
test_equal(jac[3,3], -1β)

# issue #545
z = t + t^2
#test_equal(expand_derivatives(D(z)), 1 + t * 2)

z = t-2t
#test_equal(expand_derivatives(D(z)), -1)

# Variable dependence checking in differentiation
@variables a(t) b(a)
@test !isequal(D(b), 0)
@test isequal(expand_derivatives(D(t)), 1)
@test isequal(expand_derivatives(Dx(x)), 1)

@variables x(t) y(t) z(t)

@test isequal(expand_derivatives(D(x * y)), simplify(y*D(x) + x*D(y)))
@test isequal(expand_derivatives(D(x * y)), simplify(D(x)*y + x*D(y)))

@test isequal(expand_derivatives(D(2t)), 2)
@test isequal(expand_derivatives(D(2x)), 2D(x))
@test isequal(expand_derivatives(D(x^2)), simplify(2 * x * D(x)))

# n-ary * and +
# isequal(Symbolics.derivative(Term(*, [x, y, z*ρ]), 1), y*(z*ρ))
# isequal(Symbolics.derivative(Term(+, [x*y, y, z]), 1), 1)

@test iszero(expand_derivatives(D(42)))
@test all(iszero, Symbolics.gradient(42, [t, x, y, z]))
@test all(iszero, Symbolics.hessian(42, [t, x, y, z]))
@test isequal(Symbolics.jacobian([t, x, 42], [t, x]),
              Num[1  0
                  Differential(t)(x)           1
                  0  0])

# issue 252
@variables beta, alpha, delta
@variables x1, x2, x3

# expression
tmp = beta * (alpha * exp(x1) * x2 ^ (alpha - 1) + 1 - delta) / x3
# derivative w.r.t. x1 and x2
t1 = Symbolics.gradient(tmp, [x1, x2])
@test_nowarn Symbolics.gradient(t1[1], [beta])

@variables t k
@variables x(t)
D = Differential(k)
@test Symbolics.tosymbol(D(x).val) === Symbol("xˍk(t)")

using Symbolics
@variables t x(t)
∂ₜ = Differential(t)
∂ₓ = Differential(x)
L = .5 * ∂ₜ(x)^2 - .5 * x^2
@test isequal(expand_derivatives(∂ₓ(L)), -1 * x)
test_equal(expand_derivatives(Differential(x)(L) - ∂ₜ(Differential(∂ₜ(x))(L))), -1 * (∂ₜ(∂ₜ(x)) + x))
@test isequal(expand_derivatives(Differential(x)(L) - ∂ₜ(Differential(∂ₜ(x))(L))), (-1 * x) - ∂ₜ(∂ₜ(x)))

@variables x2(t)
@test isequal(expand_derivatives(Differential(x)(2 * x + x2 * x)), 2 + x2)

@variables x y
@variables u(..)
Dy = Differential(y)
Dx = Differential(x)
dxyu = Dx(Dy(u(x,y)))
@test isequal(expand_derivatives(dxyu), dxyu)
dxxu = Dx(Dx(u(x,y)))
@test isequal(expand_derivatives(dxxu), dxxu)

using Symbolics, LinearAlgebra, SparseArrays
using Test

canonequal(a, b) = isequal(simplify(a), simplify(b))

# Calculus
@variables t σ ρ β
@variables x y z
@test isequal(
    (Differential(z) * Differential(y) * Differential(x))(t),
    Differential(z)(Differential(y)(Differential(x)(t)))
)

@test canonequal(
                 Symbolics.derivative(sin(cos(x)), x),
                 -sin(x) * cos(cos(x))
                )
function no_der end
Symbolics.@register no_der(x)
@test canonequal(
                 Symbolics.derivative([sin(cos(x)), hypot(x, no_der(x))], x),
                 [
                  -sin(x) * cos(cos(x)),
                  x/hypot(x, no_der(x)) + no_der(x)*Differential(x)(no_der(x))/hypot(x, no_der(x))
                 ]
                )

function intfun end
Symbolics.@register intfun(x)::Int
@test Symbolics.symtype(intfun(x)) === Int

eqs = [σ*(y-x),
       x*(ρ-z)-y,
       x*y - β*z]

∂ = Symbolics.jacobian(eqs,[x,y,z])
for i in 1:3
    ∇ = Symbolics.gradient(eqs[i],[x,y,z])
    @test canonequal(∂[i,:],∇)
end

@test all(canonequal.(Symbolics.gradient(eqs[1],[x,y,z]),[σ * -1,σ,0]))
@test all(canonequal.(Symbolics.hessian(eqs[1],[x,y,z]),0))

du = [x^2, y^3, x^4, sin(y), x+y, x+z^2, z+x, x+y^2+sin(z)]
reference_jac = sparse(Symbolics.jacobian(du, [x,y,z]))

@test findnz(Symbolics.jacobian_sparsity(du, [x,y,z]))[[1,2]] == findnz(reference_jac)[[1,2]]

let
    @variables t x(t) y(t) z(t)
    @test Symbolics.exprs_occur_in([x,y,z], x^2*y) == [true, true, false]
end

@test isequal(Symbolics.sparsejacobian(du, [x,y,z]), reference_jac)

using Symbolics

rosenbrock(X) = sum(1:length(X)-1) do i
    100 * (X[i+1] - X[i]^2)^2 + (1 - X[i])^2
end

@variables a,b
X = [a,b]

spoly(x) = simplify(x, polynorm=true)
rr = rosenbrock(X)

reference_hes = Symbolics.hessian(rr, X)
@test findnz(sparse(reference_hes))[1:2] == findnz(Symbolics.hessian_sparsity(rr, X))[1:2]

sp_hess = Symbolics.sparsehessian(rr, X)
@test findnz(sparse(reference_hes))[1:2] == findnz(sp_hess)[1:2]
@test isequal(map(spoly, findnz(sparse(reference_hes))[3]), map(spoly, findnz(sp_hess)[3]))

#96
@variables t x[1:4](t) ẋ[1:4](t)
expression = sin(x[1] + x[2] + x[3] + x[4]) |> Differential(t) |> expand_derivatives
expression2 = substitute(expression, Dict(Differential(t).(x) .=> ẋ))
@test isequal(expression2, (ẋ[1] + ẋ[2] + ẋ[3] + ẋ[4])*cos(x[1] + x[2] + x[3] + x[4]))

@test isequal(
    Symbolics.derivative(IfElse.ifelse(signbit(b), b^2, sqrt(b)), b),
    IfElse.ifelse(signbit(b), 2b, (1//2)*(sqrt(b)^-1))
)