using SymbolicUtils
using StaticArrays
import Base: eltype, length, ndims, size, axes, eachindex

### Store Shape as a metadata in Term{<:AbstractArray} objects
struct ArrayShapeCtx end


#=
 There are 2 types of array terms:
   `ArrayOp{T<:AbstractArray}` and `Term{<:AbstractArray}`

 - ArrayOp represents a Einstein-notation-inspired array operation.
   it contains a field `term` which is a `Term` that represents the
   operation that resulted in the `ArrayOp`.
   I.e. will be `A*b` for the operation `(i,) => A[i,j] * b[j]` for example.
   It can be `nothing` if not known.
   - calling `shape` on an `ArrayOp` will return the shape of the array or `Unknown()`
   - do not rely on the `symtype` or `shape` information of the `.term` when looking at an `ArrayOp`.
     call `shape`, `symtype` and `ndims` directly on the `ArrayOp`.
 - `Term{<:AbstractArray}`
   - calling `shape` on it will return the shape of the array or `Unknown()`, and uses
     the `ArrayShapeCtx` metadata context to store this.

 The Array type parameter must contain the dimension.
=#

#### ArrayOp ####

"""
    ArrayOp(output_idx, expr, reduce)

A tensor expression where `output_idx` are the output indices
`expr`, is the tensor expression and `reduce` is the function
used to reduce over contracted dimensions.
"""
struct ArrayOp{T<:AbstractArray} <: Symbolic{T}
    output_idx # output indices
    expr       # Used in pattern matching
               # Useful to infer eltype
    reduce
    term
    shape
    boundary_style
    ranges::Dict{Sym, AbstractRange} # index range each index symbol can take,
                                     # optional for each symbol
    metadata
end


struct ZeroBoundary end
struct ClipBoundary end

function ArrayOp(T::Type, output_idx, expr, reduce, term, ranges=Dict(); boundary_style=ZeroBoundary(), metadata=nothing)
    sh = make_shape(output_idx, expr, ranges)
    ArrayOp{T}(output_idx, expr, reduce,
               term, sh, boundary_style,
               ranges, metadata)
end

function ArrayOp(a::AbstractArray)
    i = makesubscripts(ndims(a))
    # TODO: formalize symtype(::Type) then!
    ArrayOp(symtype(a), (i...,), a[i...], +, a)
end

ConstructionBase.constructorof(s::Type{<:ArrayOp{T}}) where {T} = ArrayOp{T}

shape(aop::ArrayOp) = aop.shape

const show_arrayop = Ref{Bool}(false)
function Base.show(io::IO, aop::ArrayOp)
    if istree(aop.term) && !show_arrayop[]
        show(io, aop.term)
    else
        print(io, "@arrayop")
        print(io, "(_[$(join(string.(aop.output_idx), ","))] := $(aop.expr))")
    end
end

symtype(a::ArrayOp{T}) where {T} = T
istree(a::ArrayOp) = true
operation(a::ArrayOp) = typeof(a)
arguments(a::ArrayOp) = [a.output_idx, a.expr, a.reduce, a.term, a.shape, a.ranges, a.metadata]

function Base.isequal(a::ArrayOp, b::ArrayOp)
    a === b && return true
    isequal(operation(a), operation(b)) &&
    isequal(a.output_idx, b.output_idx) &&
    isequal(a.expr, b.expr) &&
    isequal(a.reduce, b.reduce) &&
    isequal(a.shape, b.shape)
end

function Base.hash(a::ArrayOp, u::UInt)
    hash(a.shape, hash(a.expr, hash(a.expr, hash(a.output_idx, hash(operation(a), u)))))
end

macro arrayop(call, output_idx, expr, options...)
    @assert output_idx.head == :tuple
    rs = []
    reduce = +
    for o in options
        if isexpr(o, :call) && o.args[1] == :in
            push!(rs, :($(o.args[2]) => $(o.args[3])))
        elseif isexpr(o, :(=)) && o.args[1] == :reduce
            reduce = o.args[2]
        end
    end

    oidxs = filter(x->x isa Symbol, output_idx.args)
    iidxs = find_indices(expr)
    idxs  = union(oidxs, iidxs)
    fbody = call2term(deepcopy(expr))
    oftype(x,T) = :($x::$T)
    aop = gensym("aop")
    quote
        let
            @syms $(map(x->oftype(x, Int), idxs)...)

            expr = $fbody
            #TODO: proper Atype
            $ArrayOp(Array{$symtype(expr),
                           $(length(output_idx.args))},
                     $output_idx,
                     expr,
                     $reduce,
                     $(call2term(call)),
                     Dict($(rs...)))

        end
    end |> esc
end

const SymArray = Union{ArrayOp, Symbolic{<:AbstractArray}}
const SymMat = Union{ArrayOp{<:AbstractMatrix}, Symbolic{<:AbstractMatrix}}
const SymVec = Union{ArrayOp{<:AbstractVector}, Symbolic{<:AbstractVector}}

### Propagate ###
#
## Shape ##


function axis_in(a, b)
    first(a) >= first(b) && last(a) <= last(b)
end

function make_shape(output_idx, expr, ranges=Dict())
    matches = idx_to_axes(expr)
    for (sym, ms) in matches
        to_check = filter(m->!(shape(m.A) isa Unknown), ms)
        # Only check known dimensions. It may be "known symbolically"
        isempty(to_check) && continue
        restricted = false
        if haskey(ranges, sym)
            ref_axis = ranges[sym]
            restricted = true
        else
            ref_axis = axes(first(to_check).A, first(to_check).dim)
        end
        reference = ref_axis
        for i in (restricted ? 1 : 2):length(ms)
            m = ms[i]
            s=shape(m.A)
            if s !== Unknown()
                if restricted
                    if !axis_in(ref_axis, axes(m.A, m.dim))
                        throw(DimensionMismatch("expected $(ref_axis) to be within axes($(m.A), $(m.dim))"))
                    end
                elseif !isequal(axes(m.A, m.dim), reference)
                    throw(DimensionMismatch("expected axes($(m.A), $(m.dim)) = $(reference)"))
                end
            end
        end
    end

    sz = map(output_idx) do i
        if i isa Sym
            if haskey(ranges, i)
                return axes(ranges[i], 1)
            end
            if !haskey(matches, i)
                error("index $i not found in RHS")
            end
            mi = matches[i]
            @assert !isempty(mi)
            get(first(mi))
        elseif i isa Integer
            return Base.OneTo(1)
        end
    end
    # TODO: maybe we can remove this restriction?
    if any(x->x isa Unknown, sz)
        Unknown()
    else
        sz
    end
end


function ranges(a::ArrayOp)
    rs = Dict{Sym, Any}()
    ax = idx_to_axes(a.expr)
    for i in keys(ax)
        if haskey(a.ranges, i)
            rs[i] = a.ranges[i]
        else
            rs[i] = get(first(ax[i]))
        end
    end
    return rs
end

## Eltype ##

eltype(aop::ArrayOp) = symtype(aop.expr)

## Ndims ##
function ndims(aop::ArrayOp)
    length(aop.output_idx)
end


### Utils ###


# turn `f(x...)` into `term(f, x...)`
#
function call2term(expr, arrs=[])
    !(expr isa Expr) && return expr
    if expr.head == :call
        if expr.args[1] == :(:)
            return expr
        end
        return Expr(:call, term, map(call2term, expr.args)...)
    elseif expr.head == Symbol("'")
        return Expr(:call, term, adjoint, map(call2term, expr.args)...)
    end

    return Expr(expr.head, map(call2term, expr.args)...)
end

# Find all symbolic indices in expr
function find_indices(expr, idxs=[])
    !(expr isa Expr) && return idxs
    if expr.head == :ref
        return append!(idxs, filter(x->x isa Symbol, expr.args[2:end]))
    elseif expr.head == :call && expr.args[1] == :getindex || expr.args[1] == getindex
        return append!(idxs, filter(x->x isa Symbol, expr.args[3:end]))
    else
        foreach(x->find_indices(x, idxs), expr.args)
        return idxs
    end
end

struct AxisOf
    A
    dim
    boundary
end

function Base.get(a::AxisOf)
    @oops shape(a.A)
    axes(a.A, a.dim)
end

function idx_to_axes(expr, dict=Dict{Sym, Vector}(), ranges=Dict())
    if istree(expr)
        if operation(expr) === (getindex)
            args = arguments(expr)
            for (axis, idx_expr) in enumerate(@views args[2:end])
                if idx_expr isa Sym || istree(idx_expr)
                    sym = only(get_variables(idx_expr))
                    axesvec = Base.get!(() -> [], dict, sym)
                    push!(axesvec, AxisOf(first(args), axis, idx_expr - sym))
                end
            end
        else
            idx_to_axes(operation(expr), dict)
            foreach(ex->idx_to_axes(ex, dict), arguments(expr))
        end
    end
    dict
end


#### Term{<:AbstractArray}
#

"""
    arrterm(f, args...; arrayop=nothing)

Create a term of `Term{<: AbstractArray}` which
is the representation of `f(args...)`.

- Calls `propagate_atype(f, args...)` to determine the
  container type, i.e. `Array` or `StaticArray` etc.
- Calls `propagate_eltype(f, args...)` to determine the
  output element type.
- Calls `propagate_ndims(f, args...)` to determine the
  output dimension.
- Calls `propagate_shape(f, args...)` to determine the
  output array shape.

`propagate_shape`, `propagate_atype`, `propagate_eltype` may
return `Unknown()` to say that the output cannot be determined

But `propagate_ndims` must work and return a non-negative integer.
"""
function arrterm(f, args...)
    atype = propagate_atype(f, args...)
    etype = propagate_eltype(f, args...)
    nd    = propagate_ndims(f, args...)

    S = if etype === Unknown() && nd === Unknown()
        atype
    elseif etype === Unknown()
        atype{T, nd} where T
    elseif nd === Unknown()
        atype{etype, N} where N
    else
        atype{etype, nd}
    end

    setmetadata(Term{S}(f, args),
                ArrayShapeCtx,
                propagate_shape(f, args...))
end

"""
    shape(s::Any)

Returns `axes(s)` or throws.
"""
shape(s) = axes(s)

"""
    shape(s::SymArray)

Extract the shape metadata from a SymArray.
If not known, returns `Unknown()`
"""
function shape(s::Symbolic{<:AbstractArray})
    if hasmetadata(s, ArrayShapeCtx)
        getmetadata(s, ArrayShapeCtx)
    else
        Unknown()
    end
end

## `propagate_` interface:
#  used in the `arrterm` construction.

atype(::Type{<:Array}) = Array
atype(::Type{<:SArray}) = SArray
atype(::Type) = AbstractArray

_propagate_atype(::Type{T}, ::Type{T}) where {T} = T
_propagate_atype(::Type{<:Array}, ::Type{<:SArray}) = Array
_propagate_atype(::Type{<:SArray}, ::Type{<:Array}) = Array
_propagate_atype(::Any, ::Any) = AbstractArray
_propagate_atype(T) = T
_propagate_atype() = AbstractArray

function propagate_atype(f, args...)
    As = [atype(symtype(T))
          for T in Iterators.filter(x->x <: Symbolic{<:AbstractArray}, typeof.(args))]
    if length(As) <= 1
        _propagate_atype(As...)
    else
        foldl(_propagate_atype, As)
    end
end

function propagate_eltype(f, args...)
    As = [eltype(symtype(T))
          for T in Iterators.filter(x->symtype(x) <: AbstractArray, args)]
    promote_type(As...)
end

function propagate_ndims(f, args...)
    error("Could not determine the output dimension of $f$args")
end

function propagate_shape(f, args...)
    error("Don't know how to propagate shape for $f$args")
end

### Wrapper type for dispatch

@symbolic_wrap struct Arr{T,N} <: AbstractArray{T, N}
    value
end

Base.hash(x::Arr, u::UInt) = hash(unwrap(x), u)

ArrayOp(x::Arr) = unwrap(x)

function Arr(x)
    A = symtype(x)
    @assert A <: AbstractArray
    Arr{maybewrap(eltype(A)), ndims(A)}(x)
end

unwrap(x::Arr) = x.value

maybewrap(T) = has_symwrapper(T) ? wrapper_type(T) : T
# These methods allow @wrapped methods to be more specific and not overwrite
# each other when defined both for matrix and vector
wrapper_type(::Type{<:AbstractMatrix}) = Arr{<:Any, 2}
wrapper_type(::Type{<:AbstractMatrix{T}}) where {T} = Arr{maybewrap(T), 2}

wrapper_type(::Type{<:AbstractVector}) = Arr{<:Any, 1}
wrapper_type(::Type{<:AbstractVector{T}}) where {T} = Arr{maybewrap(T), 1}

function Base.show(io::IO, ::MIME"text/plain", arr::Arr)
    x = unwrap(arr)
    istree(x) && print(io, "(")
    print(io, unwrap(arr))
    istree(x) && print(io, ")")
    print(io, "[", join(string.(axes(arr)), ","), "]")
end

################# Base array functions
#

# basic
# these methods are not symbolic but work if we know this info.

geteltype(s::SymArray) = geteltype(symtype(s))
geteltype(::Type{<:AbstractArray{T}}) where {T} = T
geteltype(::Type{<:AbstractArray}) = Unknown()

ndims(s::SymArray) = ndims(symtype(s))
ndims(T::Type{<:AbstractArray}) = ndims(T)

function eltype(A::Union{Arr, SymArray})
    T = geteltype(unwrap(A))
    T === Unknown() && error("eltype of $A not known")
    return T
end

function length(A::Union{Arr, SymArray})
    s = shape(unwrap(A))
    s === Unknown() && error("length of $A not known")
    return prod(length, s)
end

function size(A::Union{Arr, SymArray})
    s = shape(unwrap(A))
    s === Unknown() && error("size of $A not known")
    return length.(s)
end

function size(A::SymArray, i::Integer)
    @assert(i > 0)
    i > ndims(A) ? 1 : size(A)[i]
end

function axes(A::Union{Arr, SymArray})
    s = shape(unwrap(A))
    s === Unknown() && error("axes of $A not known")
    return s
end


function axes(A::SymArray, i)
    s = shape(A)
    s === Unknown() && error("axes of $A not known")
    return i <= length(s) ? s[i] : Base.OneTo(1)
end

function eachindex(A::Union{Arr, SymArray})
    s = shape(unwrap(A))
    s === Unknown() && error("eachindex of $A not known")
    return CartesianIndices(s)
end


function SymbolicUtils.Code.toexpr(x::ArrayOp, st)
    haskey(st.symbolify, x) && return st.symbolify[x]

    if istree(x.term)
        toexpr(x.term, st)
    else
        throw(ArgumentError("""Don't know how to turn $x
                               into code yet"""))
    end
end

function SymbolicUtils.Code.toexpr(x::Arr, st)
    toexpr(unwrap(x), st)
end


### Scalarize

scalarize(a::Array) = map(scalarize, a)
scalarize(term::Symbolic{<:AbstractArray}, idx) = term[idx...]
val2num(::Val{n}) where n = n

function replace_by_scalarizing(ex, dict)
    rule = @rule(getindex(~x, ~~i) =>
              scalarize(~x, (map(j->haskey(dict,j) ? dict[j] : j, ~~i)...,)))

    simterm = (x, f, args) -> begin
        if f === Base.literal_pow && length(args) == 3
            #=
            julia> @variables u[1:3]
            1-element Vector{Symbolics.Arr{Num, 1}}:
             u[1:3]

            julia> u.^1
            (broadcast(literal_pow, Base.RefValue{typeof(^)}(^), u, Base.RefValue{Val{1}}(Val{1}())))[1:3]
            =#
            base = args[2]
            exp = val2num(only(args[3]))
            f = only(args[1])
            f(base, exp)
        else
            f(args...)
        end
    end
    function rewrite_operation(x)
        if istree(x) && istree(operation(x))
            f = operation(x)
            replace_by_scalarizing(f, dict)(arguments(x)...)
        else
            nothing
        end
    end

    # This must be a Prewalk to avoid descending into ArrayOp (madness)
    Prewalk(Rewriters.PassThrough(Rewriters.If(x->!(x isa ArrayOp), Chain([rewrite_operation, rule]))), similarterm=simterm)(ex)
end

function scalarize(arr::AbstractArray, idx)
    arr[idx...]
end

function scalarize(arr::Term, idx)
    scalarize_op(operation(arr), arr, idx)
end

scalarize_op(f, arr) = arr

struct ScalarizeCache end

function scalarize_op(f, arr, idx)
    if hasmetadata(arr, ScalarizeCache) && getmetadata(arr, ScalarizeCache)[] !== nothing
        wrap(getmetadata(arr, ScalarizeCache)[][idx...])
    else
        thing = f(scalarize.(map(wrap, arguments(arr)))...)
        getmetadata(arr, ScalarizeCache)[] = thing
        wrap(thing[idx...])
    end
end

@wrapped function Base.:(\)(A::AbstractMatrix, b::AbstractVecOrMat)
    t = arrterm(\, A, b)
    setmetadata(t, ScalarizeCache, Ref{Any}(nothing))
end

@wrapped function Base.inv(A::AbstractMatrix)
    t = arrterm(inv, A)
    setmetadata(t, ScalarizeCache, Ref{Any}(nothing))
end

_det(x, lp) = det(x, laplace=lp)

function scalarize_op(f::typeof(_det), arr)
    det(map(wrap, collect(arguments(arr)[1])), laplace=arguments(arr)[2])
end

@wrapped function LinearAlgebra.det(x::AbstractMatrix; laplace=true)
    Term{eltype(x)}(_det, [x, laplace])
end


# A * x = b
# A ∈ R^(m x n) x ∈ R^(n, k) = b ∈ R^(m, k)
propagate_ndims(::typeof(\), A, b) = ndims(b)
propagate_ndims(::typeof(inv), A) = ndims(A)

# A(m,k) * B(k,n) = C(m,n)
# A(m,k) \ C(m,n)  = B(k,n)
function propagate_shape(::typeof(\), A, b)
    if ndims(b) == 1
        (axes(A,2),)
    else
        (axes(A,2), axes(b, 2))
    end
end

function propagate_shape(::typeof(inv), A)
    @oops shp = shape(A)
    @assert ndims(A) == 2 && reverse(shp) == shp "Inv called on a non-square matrix"
    shp
end

function scalarize(arr::ArrayOp, idx)
    @assert length(arr.output_idx) == length(idx)

    axs = ranges(arr)

    iidx = collect(keys(axs))
    contracted = setdiff(iidx, arr.output_idx)

    dict = Dict(oi => (unwrap(i) isa Symbolic ? unwrap(i) : axs[oi][i])
                for (oi, i) in zip(arr.output_idx, idx))
    partial = replace_by_scalarizing(arr.expr, dict)

    axes = [axs[c] for c in contracted]
    if isempty(contracted)
        partial
    else
        mapreduce(arr.reduce, Iterators.product(axes...)) do idx
            replace_by_scalarizing(partial, Dict(contracted .=> idx))
        end
    end
end

scalarize(arr::Arr, idx) = wrap(scalarize(unwrap(arr),
                                          unwrap.(idx)))

function scalarize(arr)
    if arr isa Arr || arr isa Symbolic{<:AbstractArray}
        map(Iterators.product(axes(arr)...)) do i
            scalarize(arr, i)
        end
    elseif istree(arr) && operation(arr) == getindex
        args = arguments(arr)
        scalarize(args[1], (args[2:end]...,))
    elseif arr isa Num
        wrap(scalarize(unwrap(arr)))
    elseif istree(arr) && symtype(arr) <: Number
        t = similarterm(arr, operation(arr), map(scalarize, arguments(arr)), symtype(arr), metadata=arr.metadata)
        scalarize_op(operation(t), t)
    else
        arr
    end
end

Base.collect(x::Arr) = scalarize(x)
Base.collect(x::SymArray) = scalarize(x)
isarraysymbolic(x) = SymbolicUtils.symtype(unwrap(x)) <: AbstractArray

Base.convert(::Type{<:Array{<:Any, N}}, arr::Arr{<:Any, N}) where {N} = scalarize(arr)
