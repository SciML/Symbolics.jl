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

export Integral
include("integral.jl")

include("array-lib.jl")

include("linear_algebra.jl")

include("thread_utils.jl")

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
