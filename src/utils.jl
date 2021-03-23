isblock(x) = length(x) == 1 && x[1] isa Expr && x[1].head == :block
function flatten_expr!(x)
    isblock(x) || return x
    x = MacroTools.striplines(x[1])
    filter!(z->z isa Symbol || z.head != :line, x.args)
    xs = []
    for ex in x.args
        if Meta.isexpr(ex, :tuple)
            append!(xs, ex.args)
        else
            push!(xs, ex)
        end
    end
    xs
end
function build_expr(head::Symbol, args)
    ex = Expr(head)
    append!(ex.args, args)
    ex
end

"""
    get_variables(O) -> Vector{Union{Sym, Term}}

Returns the variables in the expression. Note that the returned variables are
not wrapped in the `Num` type.

# Examples
```julia
julia> @parameters t
(t,)

julia> @variables x y z(t)
(x, y, z(t))

julia> ex = x + y + sin(z)
(x + y) + sin(z(t))

julia> Symbolics.get_variables(ex)
3-element Vector{Any}:
 x
 y
 z(t)
```
"""
get_variables(e::Num, varlist=nothing) = get_variables(value(e), varlist)
get_variables!(vars, e, varlist=nothing) = vars

is_singleton(e::Term) = operation(e) isa Sym
is_singleton(e::Sym) = true
is_singleton(e) = false

get_variables!(vars, e::Number, varlist=nothing) = vars

function get_variables!(vars, e::Symbolic, varlist=nothing)
    if is_singleton(e)
        if isnothing(varlist) || any(isequal(e), varlist)
            push!(vars, e)
        end
    else
        foreach(x -> get_variables!(vars, x, varlist), arguments(e))
    end
    return (vars isa AbstractVector) ? unique!(vars) : vars
end

function get_variables!(vars, e::Equation, varlist=nothing)
  get_variables!(vars, e.rhs, varlist)
end

get_variables(e, varlist=nothing) = get_variables!([], e, varlist)

# Sym / Term --> Symbol
Base.Symbol(x::Union{Num,Symbolic}) = tosymbol(x)
tosymbol(x; kwargs...) = x
tosymbol(x::Sym; kwargs...) = nameof(x)
tosymbol(t::Num; kwargs...) = tosymbol(value(t); kwargs...)

"""
    diff2term(x::Term) -> Term
    diff2term(x) -> x

Convert a differential variable to a `Term`. Note that it only takes a `Term`
not a `Num`.

```julia
julia> @variables t x(t); D = Differential(t);

julia> Symbolics.diff2term(Symbolics.value(D(D(x))))
var"Differential(t)∘Differential(t)"(t)
```
"""
function diff2term(O)
    istree(O) || return O
    ds = []
    while is_derivative(O)
        push!(ds, operation(O).x)
        O = arguments(O)[1]
    end
    op = nothing
    for d in reverse(ds)
        if op === nothing
            op = string("(Differential(", nameof(d), ")")
        else
            op = string(op, "∘Differential(", nameof(d), ")")
        end
    end
    if op === nothing
        return Term{Real}(operation(O), map(diff2term, arguments(O)))
    else
        oldop = operation(O)
        if !(oldop isa Sym)
            throw(ArgumentError("A differentiated state's operation must be a `Sym`, so states like `D(u + u)` are disallowed. Got `$oldop`."))
        end
        op *= ")($(nameof(oldop)))"
        return Term{Real}(rename(oldop, Symbol(op)), arguments(O))
    end
end

"""
    tosymbol(x::Union{Num,Symbolic}; states=nothing, escape=true) -> Symbol

Convert `x` to a symbol. `states` are the states of a system, and `escape`
means if the target has escapes like `val"y(t)"`. If `escape` then it will only
output `y` instead of `y(t)`.

# Examples

```julia
julia> @parameters t; @variables z(t)
(z(t),)

julia> Symbolics.tosymbol(z)
Symbol("z(t)")
```
"""
function tosymbol(t::Term; states=nothing, escape=true)
    if operation(t) isa Sym
        if states !== nothing && !(t in states)
            return nameof(operation(t))
        end
        op = nameof(operation(t))
        args = arguments(t)
    elseif operation(t) isa Differential
        term = diff2term(t)
        op = Symbol(operation(term))
        args = arguments(term)
    else
        @goto err
    end

    return escape ? Symbol(op, "(", join(args, ", "), ")") : op
    @label err
    error("Cannot convert $t to a symbol")
end

function lower_varname(var::Symbolic, idv, order)
    order == 0 && return var
    name = string(nameof(operation(var)))
    underscore = 'ˍ'
    idx = findlast(underscore, name)
    append = string(idv)^order
    if idx === nothing
        newname = Symbol(name, underscore, append)
    else
        nidx = nextind(name, idx)
        newname = Symbol(name[1:idx], name[nidx:end], append)
    end
    return Sym{symtype(operation(var))}(newname)(arguments(var)[1])
end

function lower_varname(t::Symbolic, iv)
    var, order = var_from_nested_derivative(t)
    lower_varname(var, iv, order)
end
lower_varname(t::Sym, iv) = t

"""
    makesym(x::Union{Num,Symbolic}, kwargs...) -> Sym

`makesym` takes the same arguments as [`tosymbol`](@ref), but it converts a
`Term` in the form of `x(t)` to a `Sym` in the form of `x⦗t⦘`.

# Examples
```julia
julia> @parameters t; @variables x(t)
(x(t),)

julia> Symbolics.makesym(x)
x⦗t⦘
```
"""
makesym(t::Symbolic; kwargs...) = Sym{symtype(t)}(tosymbol(t; kwargs...))
makesym(t::Num; kwargs...) = makesym(value(t); kwargs...)

var_from_nested_derivative(x, i=0) = (missing, missing)
var_from_nested_derivative(x::Term,i=0) = operation(x) isa Differential ? var_from_nested_derivative(arguments(x)[1],i+1) : (x,i)
var_from_nested_derivative(x::Sym,i=0) = (x,i)
