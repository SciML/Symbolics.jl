@metatheory_init ()

TermInterface.isterm(t::Type{<:Sym}) = false
TermInterface.isterm(t::Type{<:Symbolic}) = true

TermInterface.gethead(t::Symbolic) = :call 
TermInterface.gethead(t::Sym) = t
TermInterface.getargs(t::Symbolic) = [operation(t), arguments(t)...]
TermInterface.arity(t::Symbolic) = length(arguments(t))


function TermInterface.similarterm(x::Type{<:Symbolic{T}}, head, args; metadata=nothing) where T
    @assert head == :call
    Term{T}(args[1], args[2:end])
end

function EGraphs.preprocess(t::Symbolic)
    # TODO change to isterm after PR
    if SymbolicUtils.istree(t)
        f = operation(t)
        if f == (+) || f == (*) || f == (-) # check out for other binary ops TODO
            a = arguments(t)
            if length(a) > 2
                return unflatten_args(f, a, 2)
            end
        end
    end
    return t
end

"""
Equational rewrite rules for optimizing expressions
"""
opt_theory = @methodtheory begin
    a * x == x * a
    a * x + a * y == a*(x+y)
    -1 * a == -a
end

"""
Approximation of costs of operators in number 
of CPU cycles required for the numerical computation
"""
const op_costs = Dict(
    (+) => 1,
    (-) => 1,
    (*) => 3,
    (/) => 24
)

function costfun(n::ENode, g::EGraph, an)
    arity(n) == 0 && return get(op_costs, n.head, 1)

    if !(n.head == :call)
        return 1000000000
    end
    cost = 0

    for id ∈ n.args
        eclass = geteclass(g, id)
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end
    cost
end

function optimize(ex; params=SaturationParams())
    # ex = SymbolicUtils.Code.toexpr(ex)
    g = EGraph()

    settermtype!(g, Term{symtype(ex), Any})

    ec, _ = addexpr!(g, ex)
    g.root = ec.id
    saturate!(g, opt_theory, params)

    extract!(g, costfun) # --> "term" head args
end