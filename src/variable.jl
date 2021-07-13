using SymbolicUtils: FnType, Sym
using Setfield

const IndexMap = Dict{Char,Char}(
            '-' => '₋',
            '0' => '₀',
            '1' => '₁',
            '2' => '₂',
            '3' => '₃',
            '4' => '₄',
            '5' => '₅',
            '6' => '₆',
            '7' => '₇',
            '8' => '₈',
            '9' => '₉')

struct VariableDefaultValue end

const _fail = Dict()
function getdefaultval(x, val=_fail)
    x = unwrap(x)
    if hasmetadata(x, VariableDefaultValue)
        return getmetadata(x, VariableDefaultValue)
    else
        val === _fail && error("$x has no default value")
        return val
    end
end

function recurse_and_apply(f, x)
    if symtype(x) <: AbstractArray
        getindex_posthook(x) do r,x,i...
            recurse_and_apply(f, r)
        end
    else
        f(x)
    end
end

function setdefaultval(x, val)
    if symtype(x) <: AbstractArray
        if val isa AbstractArray
            getindex_posthook(x) do r,x,i...
                setdefaultval(r, val[i...])
            end
        else
            getindex_posthook(x) do r,x,i...
                setdefaultval(r, val)
            end
        end
    else
        setmetadata(x,
                    VariableDefaultValue,
                    val)
    end
end

scalarize_getindex(x) = recurse_and_apply(scalarize, x)

"""
$(TYPEDEF)

A named variable which represents a numerical value. The variable is uniquely
identified by its `name`, and all variables with the same `name` are treated
as equal.

# Fields
$(FIELDS)

For example, the following code defines an independent variable `t`, a parameter
`α`, a function parameter `σ`, a variable `x`, which depends on `t`, a variable
`y` with no dependents, a variable `z`, which depends on `t`, `α`, and `x(t)`
and parameters `β₁` and `β₂`.


```julia
σ = Num(Variable{Symbolics.FnType{Tuple{Any},Real}}(:σ)) # left uncalled, since it is used as a function
w = Num(Variable{Symbolics.FnType{Tuple{Any},Real}}(:w)) # unknown, left uncalled
x = Num(Variable{Symbolics.FnType{Tuple{Any},Real}}(:x))(t)  # unknown, depends on `t`
y = Num(Variable(:y))   # unknown, no dependents
z = Num(Variable{Symbolics.FnType{NTuple{3,Any},Real}}(:z))(t, α, x)  # unknown, multiple arguments
β₁ = Num(Variable(:β, 1)) # with index 1
β₂ = Num(Variable(:β, 2)) # with index 2

expr = β₁ * x + y^α + σ(3) * (z - t) - β₂ * w(t - 1)
```
"""
struct Variable{T} <: Function # backward compat
    """The variable's unique name."""
    name::Symbol
    Variable(name) = Sym{Real}(name)
    Variable{T}(name) where T = Sym{T}(name)
    function Variable{T}(name, indices...) where T
        var_name = Symbol("$(name)$(join(map_subscripts.(indices), "ˏ"))")
        Sym{T}(var_name)
    end
end

function Variable(name, indices...)
    var_name = Symbol("$(name)$(join(map_subscripts.(indices), "ˏ"))")
    Variable(var_name)
end

# TODO: move this to Symutils
function Sym{T}(name, i, indices...) where T
    var_name = Symbol("$(name)$(join(map_subscripts.((i, indices...,)), "ˏ"))")
    Sym{T}(var_name)
end

function map_subscripts(indices)
    str = string(indices)
    join(IndexMap[c] for c in str)
end

rename(x::Sym,name) = @set! x.name = name
function rename(x::Symbolic, name)
    if operation(x) isa Sym
        @assert x isa Term
        @set! x.f = rename(operation(x), name)
        @set! x.hash = Ref{UInt}(0)
        return x
    else
        error("can't rename $x to $name")
    end
end

function unwrap_runtime_var(v)
    isruntime = Meta.isexpr(v, :$) && length(v.args) == 1
    isruntime && (v = v.args[1])
    return isruntime, v
end

# Build variables more easily
function _parse_vars(macroname, type, x, transform=identity)
    ex = Expr(:block)
    var_names = Symbol[]
    # if parsing things in the form of
    # begin
    #     x
    #     y
    #     z
    # end
    x = x isa Tuple && first(x) isa Expr && first(x).head == :tuple ? first(x).args : x # tuple handling
    x = flatten_expr!(x)
    cursor = 0
    isoption(ex) = Meta.isexpr(ex, [:vect, :vcat, :hcat])
    while cursor < length(x)
        cursor += 1
        v = x[cursor]

        # We need lookahead to the next `v` to parse
        # `@variables x [connect=Flow,unit=u]`
        nv = cursor < length(x) ? x[cursor+1] : nothing
        val = unit = connect = options = nothing

        # x = 1, [connect = flow; unit = u"m^3/s"]
        if Meta.isexpr(v, :(=))
            v, val = v.args
            if Meta.isexpr(val, :tuple) && length(val.args) == 2 && isoption(val.args[2])
                options = val.args[2].args
                val = val.args[1]
            end
        end


      # if Meta.isexpr(v, :(::))
      #     v, type′ = v.args
      #     type = type′ === :Complex ? Complex{type} : type′
      # end


        # x [connect = flow; unit = u"m^3/s"]
        if isoption(nv)
            options = nv.args
            cursor += 1
        end

        isruntime, v = unwrap_runtime_var(v)
        iscall = Meta.isexpr(v, :call)
        isarray = Meta.isexpr(v, :ref)
        issym  = v isa Symbol
        @assert iscall || isarray || issym "@$macroname expects a tuple of expressions or an expression of a tuple (`@$macroname x y z(t) v[1:3] w[1:2,1:4]` or `@$macroname x y z(t) v[1:3] w[1:2,1:4] k=1.0`)"

        if iscall
            isruntime, fname = unwrap_runtime_var(v.args[1])
            call_args = map(last∘unwrap_runtime_var, @view v.args[2:end])
            var_name, expr = construct_vars(fname, type, call_args, val, options, transform, isruntime)
        else
            var_name, expr = construct_vars(v, type, nothing, val, options, transform, isruntime)
        end

        push!(var_names, var_name)
        push!(ex.args, expr)
    end
    rhs = build_expr(:vect, var_names)
    push!(ex.args, rhs)
    return ex
end

function construct_vars(v, type, call_args, val, prop, transform, isruntime)
    issym  = v isa Symbol
    isarray = isa(v, Expr) && v.head == :ref
    if isarray
        var_name = v.args[1]
        if Meta.isexpr(var_name, :(::))
            var_name, type′ = var_name.args
            type = type′ === :Complex ? Complex{type} : type′
        end
        isruntime, var_name = unwrap_runtime_var(var_name)
        indices = v.args[2:end]
        expr = _construct_array_vars(isruntime ? var_name : Meta.quot(var_name), type, call_args, val, prop, indices...)
    else
        var_name = v
        if Meta.isexpr(v, :(::))
            var_name, type′ = v.args
            type = type′ === :Complex ? Complex{type} : type′
        end
        expr = construct_var(isruntime ? var_name : Meta.quot(var_name), type, call_args, val, prop)
    end
    lhs = isruntime ? gensym(var_name) : var_name
    rhs = :($transform($expr))
    lhs, :($lhs = $rhs)
end

function option_to_metadata_type(::Val{opt}) where {opt}
    throw(Base.Meta.ParseError("unknown property type $opt"))
end

function setprops_expr(expr, props)
    isnothing(props) && return expr
    for opt in props
        if !Meta.isexpr(opt, :(=))
            throw(Base.Meta.ParseError(
                "Variable properties must be in " *
                "the form of `a = b`. Got $opt."))
        end

        lhs, rhs = opt.args

        @assert lhs isa Symbol "the lhs of an option must be a symbol"
        expr = :($setmetadata($expr,
                              $(option_to_metadata_type(Val{lhs}())),
                       $rhs))
    end
    expr
end


function construct_var(var_name, type, call_args, val, prop)
    expr = if call_args === nothing
        :($Sym{$type}($var_name))
    elseif !isempty(call_args) && call_args[end] == :..
        :($Sym{$FnType{Tuple, $type}}($var_name)) # XXX: using Num as output
    else
        :($Sym{$FnType{NTuple{$(length(call_args)), Any}, $type}}($var_name)($(map(x->:($value($x)), call_args)...)))
    end

    if val !== nothing
        expr = :($setdefaultval($expr, $val))
    end

    :($wrap($(setprops_expr(expr, prop))))
end

function _construct_array_vars(var_name, type, call_args, val, prop, indices...)
    # TODO: just use Sym here
    ndim = length(indices)

    expr = if call_args === nothing
        ex = :($Sym{Array{$type, $ndim}}($var_name))
        :($setmetadata($ex, $ArrayShapeCtx, ($(indices...),)))
    elseif !isempty(call_args) && call_args[end] == :..
        ex = :($Sym{Array{$FnType{Tuple, $type}, $ndim}}($var_name)) # XXX: using Num as output
        :($setmetadata($ex, $ArrayShapeCtx, ($(indices...),)))
    else
        # [(R -> R)(R) ....]
        ex = :($Sym{Array{$FnType{Tuple, $type}, $ndim}}($var_name))
        ex = :($setmetadata($ex, $ArrayShapeCtx, ($(indices...),)))
        :($scalarize_getindex($map(f->$unwrap(f($(call_args...))), $ex)))

    end

    if val !== nothing
        expr = :($setdefaultval($expr, $val))
    end

    expr = setprops_expr(expr, prop)

    return :($wrap($expr))
end


"""

Define one or more unknown variables.

```julia
@variables t α σ(..) β[1:2]
@variables w(..) x(t) y z(t, α, x)

expr = β[1]* x + y^α + σ(3) * (z - t) - β[2] * w(t - 1)
```

`(..)` signifies that the value should be left uncalled.

Symbolics supports creating variables that denote an array of some size.

```julia
julia> @variables x[1:3]
1-element Vector{Symbolics.Arr{Num, 1}}:
 x[1:3]

julia> @variables y[1:3, 1:6] # support for  tensors
1-element Vector{Symbolics.Arr{Num, 2}}:
 y[1:3,1:6]

julia> @variables t z[1:3](t) # also works for dependent variables
2-element Vector{Any}:
 t
  (map(#5, z))[1:3]
```

A symbol or expression that represents an array can be turned into an array of
symbols or expressions using the `scalarize` function.

```julia
julia> Symbolics.scalarize(z)
3-element Vector{Num}:
 z[1](t)
 z[2](t)
 z[3](t)
```

Note that `@variables` returns a vector of all the defined variables.

`@variables` can also take runtime symbol values by the `\$` interpolation
operator, and in this case, `@variables` doesn't automatically assign the value,
instead, it only returns a vector of symbolic variables. All the rest of the
syntax also applies here.

```julia
julia> a, b, c = :runtime_symbol_value, :value_b, :value_c
:runtime_symbol_value

julia> vars = @variables t \$a \$b(t) \$c[1:3](t)
4-element Vector{Any}:
      t
 runtime_symbol_value
   value_b(t)
       (map(#9, value_c))[1:3]

julia> (t, a, b, c)
(t, :runtime_symbol_value, :value_b, :value_c)
```
"""
macro variables(xs...)
    esc(_parse_vars(:variables, Real, xs))
end

TreeViews.hastreeview(x::Sym) = true

function TreeViews.treelabel(io::IO,x::Sym,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Text(x.name))
end
