module Symbolics

using DocStringExtensions

using LinearAlgebra

using Reexport

using DomainSets

import DomainSets: Domain
@reexport using SymbolicUtils

import SymbolicUtils: Term, Add, Mul, Pow, Sym, symtype,
                      FnType, @rule, Rewriters, substitute, similarterm,
                      promote_symtype, istree, operation, arguments

import SymbolicUtils.Rewriters: Chain, Prewalk, Postwalk, Fixpoint

import SymbolicUtils.Code: toexpr

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# re-export

export simplify, substitute

using SciMLBase, IfElse
export Num
using MacroTools
import MacroTools: splitdef, combinedef, postwalk, striplines
include("wrapper-types.jl")

include("num.jl")
include("complex.jl")

# Hacks to make wrappers "nicer"
const NumberTypes = Union{AbstractFloat,Integer,Complex{<:AbstractFloat},Complex{<:Integer}}
(::Type{T})(x::SymbolicUtils.Symbolic) where {T<:NumberTypes} = throw(ArgumentError("Cannot convert Sym to $T since Sym is symbolic and $T is concrete. Use `substitute` to replace the symbolic unwraps."))
for T in [Num, Complex{Num}]
    @eval begin
        (::Type{S})(x::$T) where {S<:Union{NumberTypes,AbstractArray}} = S(Symbolics.unwrap(x))::S

        SymbolicUtils.simplify(n::$T; kw...) = wrap(SymbolicUtils.simplify(unwrap(n); kw...))
        SymbolicUtils.expand(n::$T) = wrap(SymbolicUtils.expand(unwrap(n)))
        substitute(expr::$T, s::Pair; kw...) = wrap(substituter(s)(unwrap(expr); kw...)) # backward compat
        substitute(expr::$T, s::Vector; kw...) = wrap(substituter(s)(unwrap(expr); kw...))
        substitute(expr::$T, s::Dict; kw...) = wrap(substituter(s)(unwrap(expr); kw...))

        SymbolicUtils.Code.toexpr(x::$T) = SymbolicUtils.Code.toexpr(unwrap(x))

        SymbolicUtils.setmetadata(x::$T, t, v) = wrap(SymbolicUtils.setmetadata(unwrap(x), t, v))
        SymbolicUtils.getmetadata(x::$T, t) = SymbolicUtils.getmetadata(unwrap(x), t)
        SymbolicUtils.hasmetadata(x::$T, t) = SymbolicUtils.hasmetadata(unwrap(x), t)

        (f::Symbolic{<:FnType})(x::$T, y...) = wrap(f(unwrap(x), unwrap.(y)...))
        Broadcast.broadcastable(x::$T) = x
    end
end

"""
    substitute(expr, s)

Performs the substitution on `expr` according to rule(s) `s`.
# Examples
```julia
julia> @variables t x y z(t)
4-element Vector{Num}:
    t
    x
    y
 z(t)
julia> ex = x + y + sin(z)
(x + y) + sin(z(t))
julia> substitute(ex, Dict([x => z, sin(z) => z^2]))
(z(t) + y) + (z(t) ^ 2)
```
"""
substitute

export Equation, ConstrainedEquation
include("equations.jl")

include("utils.jl")
export degree

using ConstructionBase
include("arrays.jl")

export @register
include("register.jl")

using TreeViews
export @variables, Variable
include("variable.jl")

include("linearity.jl")

using DiffRules, SpecialFunctions, NaNMath

using SparseArrays

export Differential, expand_derivatives

include("diff.jl")

export Difference

include("difference.jl")

export Integral
include("integral.jl")

include("array-lib.jl")

include("linear_algebra.jl")

import Libdl
include("build_function.jl")
export build_function

import Distributions
include("extra_functions.jl")

using Latexify
include("latexify_recipes.jl")

using RecipesBase
include("plot_recipes.jl")

include("domains.jl")

using Requires

export symbolics_to_sympy
include("init.jl")

end # module
