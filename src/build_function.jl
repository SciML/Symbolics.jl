using SymbolicUtils.Code
using Base.Threads

abstract type BuildTargets end
struct JuliaTarget <: BuildTargets end
struct StanTarget <: BuildTargets end
struct CTarget <: BuildTargets end
struct MATLABTarget <: BuildTargets end

abstract type ParallelForm end
struct SerialForm <: ParallelForm end
struct MultithreadedForm <: ParallelForm
    ntasks::Int
end
MultithreadedForm() = MultithreadedForm(2*nthreads())

"""
`build_function`

Generates a numerically-usable function from a Symbolics `Num`.

```julia
build_function(ex, args...;
               expression = Val{true},
               target = JuliaTarget(),
               kwargs...)
```

Arguments:

- `ex`: The `Num` to compile
- `args`: The arguments of the function
- `expression`: Whether to generate code or whether to generate the compiled form.
  By default, `expression = Val{true}`, which means that the code for the
  function is returned. If `Val{false}`, then the returned value is compiled.

Keyword Arguments:

- `target`: The output target of the compilation process. Possible options are:
    - `JuliaTarget`: Generates a Julia function
    - `CTarget`: Generates a C function
    - `StanTarget`: Generates a function for compiling with the Stan probabilistic
      programming language
    - `MATLABTarget`: Generates an anonymous function for use in MATLAB and Octave
      environments
- `fname`: Used by some targets for the name of the function in the target space.

Note that not all build targets support the full compilation interface. Check the
individual target documentation for details.
"""
function build_function(args...;target = JuliaTarget(),kwargs...)
  _build_function(target,args...;kwargs...)
end

function unflatten_args(f, args, N=4)
    length(args) < N && return Term{Real}(f, args)
    unflatten_args(f, [Term{Real}(f, group)
                                       for group in Iterators.partition(args, N)], N)
end

function unflatten_long_ops(op, N=4)
    op = value(op)
    !istree(op) && return Num(op)
    rule1 = @rule((+)(~~x) => length(~~x) > N ? unflatten_args(+, ~~x, 4) : nothing)
    rule2 = @rule((*)(~~x) => length(~~x) > N ? unflatten_args(*, ~~x, 4) : nothing)

    Num(Rewriters.Postwalk(Rewriters.Chain([rule1, rule2]))(op))
end


# Scalar output

destructure_arg(arg::Union{AbstractArray, Tuple}) = DestructuredArgs(map(value, arg))
destructure_arg(arg) = arg

function _build_function(target::JuliaTarget, op, args...;
                         conv = toexpr,
                         expression = Val{true},
                         expression_module = @__MODULE__(),
                         checkbounds = false,
                         linenumbers = true)

    dargs = map(destructure_arg, [args...])
    expr = toexpr(Func(dargs, [], unflatten_long_ops(op)))

    if expression == Val{true}
        expr
    else
        _build_and_inject_function(expression_module, expr)
    end
end

function _build_and_inject_function(mod::Module, ex)
    if ex.head == :function && ex.args[1].head == :tuple
        ex.args[1] = Expr(:call, :($mod.$(gensym())), ex.args[1].args...)
    elseif ex.head == :(->)
        return _build_and_inject_function(mod, Expr(:function, ex.args...))
    end
    # XXX: Workaround to specify the module as both the cache module AND context module.
    # Currently, the @RuntimeGeneratedFunction macro only sets the context module.
    module_tag = getproperty(mod, RuntimeGeneratedFunctions._tagname)
    RuntimeGeneratedFunctions.RuntimeGeneratedFunction(module_tag, module_tag, ex)
end

toexpr(n::Num, st) = toexpr(value(n), st)

function fill_array_with_zero!(x::AbstractArray)
    if eltype(x) <: AbstractArray
        foreach(fill_array_with_zero!, x)
    else
        fill!(x, false)
    end
    return x
end

"""
Build function target: `JuliaTarget`

```julia
function _build_function(target::JuliaTarget, rhss, args...;
                         conv = toexpr, expression = Val{true},
                         checkbounds = false,
                         linenumbers = false,
                         headerfun = addheader, outputidxs=nothing,
                         convert_oop = true, force_SA = false,
                         skipzeros = outputidxs===nothing,
                         fillzeros = skipzeros && !(typeof(rhss)<:SparseMatrixCSC),
                         parallel=SerialForm(), kwargs...)
```

Generates a Julia function which can then be utilized for further evaluations.
If expression=Val{false}, the return is a Julia function which utilizes
RuntimeGeneratedFunctions.jl in order to be free of world-age issues.

If the `rhss` is a scalar, the generated function is a function
with a scalar output, otherwise if it's an `AbstractArray`, the output
is two functions, one for out-of-place AbstractArray output and a second which
is a mutating function. The outputted functions match the given argument order,
i.e., f(u,p,args...) for the out-of-place and scalar functions and
`f!(du,u,p,args..)` for the in-place version.

Special Keyword Argumnets:

- `parallel`: The kind of parallelism to use in the generated function. Defaults
  to `SerialForm()`, i.e. no parallelism. Note that the parallel forms are not
  exported and thus need to be chosen like `Symbolics.SerialForm()`.
  The choices are:
  - `SerialForm()`: Serial execution.
  - `MultithreadedForm()`: Multithreaded execution with a static split, evenly
    splitting the number of expressions per thread.
- `conv`: The conversion function of symbolic types to Expr. By default this uses
  the `toexpr` function.
- `checkbounds`: For whether to enable bounds checking inside of the generated
  function. Defaults to false, meaning that `@inbounds` is applied.
- `linenumbers`: Determines whether the generated function expression retains
  the line numbers. Defaults to true.
- `convert_oop`: Determines whether the OOP version should try to convert
  the output to match the type of the first input. This is useful for
  cases like LabelledArrays or other array types that carry extra
  information. Defaults to true.
- `force_SA`: Forces the output of the OOP version to be a StaticArray.
  Defaults to `false`, and outputs a static array when the first argument
  is a static array.
- `skipzeros`: Whether to skip filling zeros in the in-place version if the
  filling function is 0.
- `fillzeros`: Whether to perform `fill(out,0)` before the calculations to ensure
  safety with `skipzeros`.
"""
function _build_function(target::JuliaTarget, rhss::AbstractArray, args...;
                       expression = Val{true},
                       expression_module = @__MODULE__(),
                       checkbounds = false,
                       linenumbers = false,
                       outputidxs=nothing,
                       skipzeros = false,
                       wrap_code = (nothing, nothing),
                       fillzeros = skipzeros && !(typeof(rhss)<:SparseMatrixCSC),
                       parallel=SerialForm(), kwargs...)

    dargs = map(destructure_arg, [args...])
    i = findfirst(x->x isa DestructuredArgs, dargs)
    similarto = i === nothing ? Array : dargs[i].name
    oop_expr = Func(dargs, [], make_array(parallel, dargs, rhss, similarto))
    if !isnothing(wrap_code[1])
        oop_expr = wrap_code[1](oop_expr)
    end

    out = Sym{Any}(gensym("out"))
    ip_expr = Func([out, dargs...], [], set_array(parallel, dargs, out, outputidxs, rhss, checkbounds, skipzeros))

    if !isnothing(wrap_code[2])
        ip_expr = wrap_code[2](ip_expr)
    end

    if expression == Val{true}
        return toexpr(oop_expr), toexpr(ip_expr)
    else
        return _build_and_inject_function(expression_module, toexpr(oop_expr)),
        _build_and_inject_function(expression_module, toexpr(ip_expr))
    end
end

function make_array(s, dargs, arr, similarto)
    Base.@warn("Parallel form of $(typeof(s)) not implemented")
    _make_array(arr, similarto)
end

function make_array(s::SerialForm, dargs, arr, similarto)
    _make_array(arr, similarto)
end

function make_array(s::MultithreadedForm, closed_args, arr, similarto)
    per_task = ceil(Int, length(arr) / s.ntasks)
    slices = collect(Iterators.partition(arr, per_task))
    arrays = map(slices) do slice
        Func(closed_args, [], _make_array(slice, similarto)), closed_args
    end
    SpawnFetch{MultithreadedForm}(first.(arrays), last.(arrays), vcat)
end

struct Funcall{F, T}
    f::F
    args::T
end

(f::Funcall)() = f.f(f.args...)

function toexpr(p::SpawnFetch{MultithreadedForm}, st)
    args = isnothing(p.args) ?
              Iterators.repeated((), length(p.exprs)) : p.args
    spawns = map(p.exprs, args) do thunk, a
        ex = :($Funcall($(@RuntimeGeneratedFunction(toexpr(thunk, st))),
                       ($(toexpr.(a, (st,))...),)))
        quote
            let
                task = Base.Threads.Task($ex)
                Base.Threads.schedule(task)
                task
            end
        end
    end
    quote
        $(toexpr(p.combine, st))(map(fetch, ($(spawns...),))...)
    end
end

function _make_array(rhss::AbstractSparseArray, similarto)
    arr = map(x->_make_array(x, similarto), rhss)
    if !(arr isa AbstractSparseArray)
        _make_array(arr, similarto)
    else
        MakeSparseArray(arr)
    end
end

function _make_array(rhss::AbstractArray, similarto)
    arr = map(x->_make_array(x, similarto), rhss)
    # Ugh reshaped array of a sparse array when mapped gives a sparse array
    if arr isa AbstractSparseArray
        _make_array(arr, similarto)
    else
        MakeArray(arr, similarto)
    end
end

_make_array(x, similarto) = unflatten_long_ops(x)

## In-place version

function set_array(p, closed_vars, args...)
    Base.@warn("Parallel form of $(typeof(p)) not implemented")
    _set_array(args...)
end

function set_array(s::SerialForm, closed_vars, args...)
    _set_array(args...)
end

function set_array(s::MultithreadedForm, closed_args, out, outputidxs, rhss, checkbounds, skipzeros)
    if rhss isa AbstractSparseArray
        return set_array(s,
                         closed_args,
                         LiteralExpr(:($out.nzval)),
                         nothing,
                         rhss.nzval,
                         checkbounds,
                         skipzeros)
    end

    outvar = !(out isa Sym) ? gensym("out") : out

    if outputidxs === nothing
        outputidxs = collect(eachindex(rhss))
    end
    per_task = ceil(Int, length(rhss) / s.ntasks)
    # TODO: do better partitioning when skipzeros is present
    slices = collect(Iterators.partition(zip(outputidxs, rhss), per_task))
    arrays = map(slices) do slice
        idxs, vals = first.(slice), last.(slice)
        Func([outvar, closed_args...], [],
             _set_array(outvar, idxs, vals, checkbounds, skipzeros)), [out, closed_args...]
    end
    SpawnFetch{MultithreadedForm}(first.(arrays), last.(arrays), @inline noop(args...) = nothing)
end

function _set_array(out, outputidxs, rhss::AbstractSparseArray, checkbounds, skipzeros)
    _set_array(LiteralExpr(:($out.nzval)), nothing, rhss.nzval, checkbounds, skipzeros)
end

function _set_array(out, outputidxs, rhss::AbstractArray, checkbounds, skipzeros)
    if outputidxs === nothing
        outputidxs = collect(eachindex(rhss))
    end
    # sometimes outputidxs is a Tuple
    ii = findall(i->!(rhss[i] isa AbstractArray) && !(skipzeros && _iszero(rhss[i])), eachindex(outputidxs))
    jj = findall(i->rhss[i] isa AbstractArray, eachindex(outputidxs))
    exprs = []
    rhss_scalar = unflatten_long_ops.(vec(rhss[ii]))
    setterexpr = SetArray(!checkbounds,
                          out,
                          AtIndex.(vec(collect(outputidxs[ii])),
                                   rhss_scalar))
    push!(exprs, setterexpr)
    for j in jj
        push!(exprs, _set_array(LiteralExpr(:($out[$j])), nothing, rhss[j], checkbounds, skipzeros))
    end
    LiteralExpr(quote
                    $(exprs...)
                end)
end

_set_array(out, outputidxs, rhs, checkbounds, skipzeros) = unflatten_long_ops(rhs)


function vars_to_pairs(name,vs::Union{Tuple, AbstractArray}, symsdict=Dict())
    vs_names = tosymbol.(vs)
    for (v,k) in zip(vs_names, vs)
        symsdict[k] = Sym{symtype(k)}(v)
    end
    exs = [:($name[$i]) for (i, u) ∈ enumerate(vs)]
    vs_names,exs
end
function vars_to_pairs(name,vs, symsdict)
    symsdict[vs] = Sym{symtype(vs)}(tosymbol(vs))
    [tosymbol(vs)], [name]
end

get_varnumber(varop, vars::Vector) =  findfirst(x->isequal(x,varop),vars)
get_varnumber(varop, var) =  isequal(var,varop) ? 0 : nothing

function numbered_expr(O::Symbolic,args...;varordering = args[1],offset = 0,
                       lhsname=gensym("du"),rhsnames=[gensym("MTK") for i in 1:length(args)])
    O = value(O)
    if O isa Sym || isa(operation(O), Sym)
        for j in 1:length(args)
            i = get_varnumber(O,args[j])
            if i !== nothing
                return i==0 ? :($(rhsnames[j])) : :($(rhsnames[j])[$(i+offset)])
            end
        end
    end
    if istree(O)
        Expr(:call, Symbol(operation(O)), (numbered_expr(x,args...;offset=offset,lhsname=lhsname,
                                                         rhsnames=rhsnames,varordering=varordering) for x in arguments(O))...)
    elseif O isa Sym
        tosymbol(O, escape=false)
    else
        O
    end
end

function numbered_expr(de::Equation,args...;varordering = args[1],
                       lhsname=gensym("du"),rhsnames=[gensym("MTK") for i in 1:length(args)],offset=0)

    varordering = value.(args[1])
    var = var_from_nested_derivative(de.lhs)[1]
    i = findfirst(x->isequal(tosymbol(x isa Sym ? x : operation(x), escape=false), tosymbol(var, escape=false)),varordering)
    :($lhsname[$(i+offset)] = $(numbered_expr(de.rhs,args...;offset=offset,
                                              varordering = varordering,
                                              lhsname = lhsname,
                                              rhsnames = rhsnames)))
end
numbered_expr(c,args...;kwargs...) = c
numbered_expr(c::Num,args...;kwargs...) = error("Num found")


# Replace certain multiplication and power expressions so they form valid C code
# Extra factors of 1 are hopefully eliminated by the C compiler
function coperators(expr)
    expr isa Expr || return expr
    for e in expr.args
        if e isa Expr
            coperators(e)
        end
    end
    # Introduce another factor 1 to prevent contraction of terms like "5 * t" to "5t" (not valid C code)
    if expr.head==:call && expr.args[1]==:* && length(expr.args)==3 && isa(expr.args[2], Real) && isa(expr.args[3], Symbol)
        push!(expr.args, 1)
    # Power operator does not exist in C, replace by multiplication or "pow"
    elseif expr.head==:call && expr.args[1]==:^
        @assert length(expr.args)==3 "Don't know how to handle ^ operation with <> 2 arguments"
        x = expr.args[2]
        n = expr.args[3]
        empty!(expr.args)
        # Replace by multiplication/division if
        #   x is a symbol and  n is a small integer
        #   x is a more complex expression and n is ±1
        #   n is exactly 0
        if (isa(n,Integer) && ((isa(x, Symbol) && abs(n) <= 3) || abs(n) <= 1)) || n==0
            if n >= 0
                append!(expr.args, [:*, fill(x, n)...])
                # fill up with factor 1 so this expr can still be a multiplication
                while length(expr.args) < 3
                    push!(expr.args, 1)
                end
            else # inverse of the above
                if n==-1
                    term = x
                else
                    term = :( ($(x)) ^ ($(-n)))
                    coperators(term)
                end
                append!(expr.args, [:/, 1., term])
            end
        #... otherwise use "pow" function
        else
            append!(expr.args, [:pow, x, n])
        end
    end
    expr
end


"""
Build function target: `CTarget`

```julia
function _build_function(target::CTarget, eqs::Array{<:Equation}, args...;
                         conv = toexpr, expression = Val{true},
                         fname = :diffeqf,
                         lhsname=:du,rhsnames=[Symbol("RHS\$i") for i in 1:length(args)],
                         libpath=tempname(),compiler=:gcc)
```

This builds an in-place C function. Only works on arrays of equations. If
`expression == Val{false}`, then this builds a function in C, compiles it,
and returns a lambda to that compiled function. These special keyword arguments
control the compilation:

- libpath: the path to store the binary. Defaults to a temporary path.
- compiler: which C compiler to use. Defaults to :gcc, which is currently the
  only available option.
"""
function _build_function(target::CTarget, eqs::Array{<:Equation}, args...;
                         conv = toexpr, expression = Val{true},
                         fname = :diffeqf,
                         lhsname=:du,rhsnames=[Symbol("RHS$i") for i in 1:length(args)],
                         libpath=tempname(),compiler=:gcc)

    @warn "build_function(::Array{<:Equation}...) is deprecated. Use build_function(::AbstractArray...) instead."

    differential_equation = string(join([numbered_expr(eq,args...,lhsname=lhsname,
                                  rhsnames=rhsnames,offset=-1) for
                                  (i, eq) ∈ enumerate(eqs)],";\n  "),";")

    argstrs = join(vcat("double* $(lhsname)",[typeof(args[i])<:Array ? "double* $(rhsnames[i])" : "double $(rhsnames[i])" for i in 1:length(args)]),", ")
    ex = """
    void $fname($(argstrs...)) {
      $differential_equation
    }
    """

    if expression == Val{true}
        return ex
    else
        @assert compiler == :gcc
        ex = build_function(eqs,args...;target=Symbolics.CTarget())
        open(`gcc -fPIC -O3 -msse3 -xc -shared -o $(libpath * "." * Libdl.dlext) -`, "w") do f
            print(f, ex)
        end
        @RuntimeGeneratedFunction(:((du::Array{Float64},u::Array{Float64},p::Array{Float64},t::Float64) -> ccall(("diffeqf", $libpath), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64), du, u, p, t)))
    end
end


"""
Build function target: `CTarget`

```julia
function _build_function(target::CTarget, ex::AbstractArray, args...;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf,
                         lhsname     = :du,
                         rhsnames    = [Symbol("RHS\$i") for i in 1:length(args)],
                         libpath     = tempname(),
                         compiler    = :gcc)
```

This builds an in-place C function. Only works on expressions. If
`expression == Val{false}`, then this builds a function in C, compiles it,
and returns a lambda to that compiled function. These special keyword arguments
control the compilation:

- libpath: the path to store the binary. Defaults to a temporary path.
- compiler: which C compiler to use. Defaults to :gcc, which is currently the
  only available option.
"""
function _build_function(target::CTarget, ex::AbstractArray, args...;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf,
                         lhsname     = :du,
                         rhsnames    = [Symbol("RHS$i") for i in 1:length(args)],
                         libpath     = tempname(),
                         compiler    = :gcc)

    if !columnmajor
        return _build_function(target, hcat([row for row ∈ eachrow(ex)]...), args...;
                               columnmajor = true,
                               conv        = conv,
                               fname       = fname,
                               lhsname     = lhsname,
                               rhsnames    = rhsnames,
                               libpath     = libpath,
                               compiler    = compiler)
    end

    equations = Vector{String}()
    for col ∈ 1:size(ex,2)
        for row ∈ 1:size(ex,1)
            lhs = string(lhsname, "[", (col-1) * size(ex,1) + row-1, "]")
            rhs = numbered_expr(value(ex[row, col]), args...;
                                lhsname  = lhsname,
                                rhsnames = rhsnames,
                                offset   = -1) |> coperators |> string  # Filter through coperators to produce valid C code in more cases
            push!(equations, string(lhs, " = ", rhs, ";"))
        end
    end

    argstrs = join(vcat("double* $(lhsname)",[typeof(args[i])<:Array ? "const double* $(rhsnames[i])" : "const double $(rhsnames[i])" for i in 1:length(args)]),", ")

    ccode = """
    #include <math.h>
    void $fname($(argstrs...)) {$([string("\n  ", eqn) for eqn ∈ equations]...)\n}
    """

    if expression == Val{true}
        return ccode
    else
        @assert compiler == :gcc
        open(`gcc -fPIC -O3 -msse3 -xc -shared -o $(libpath * "." * Libdl.dlext) -`, "w") do f
            print(f, ccode)
        end
        @RuntimeGeneratedFunction(:((du::Array{Float64},u::Array{Float64},p::Array{Float64},t::Float64) -> ccall(("diffeqf", $libpath), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64), du, u, p, t)))
    end

end
_build_function(target::CTarget, ex::Num, args...; kwargs...) = _build_function(target, [ex], args...; kwargs...)


"""
Build function target: `StanTarget`

```julia
function _build_function(target::StanTarget, eqs::Array{<:Equation}, vs, ps, iv;
                         conv = toexpr, expression = Val{true},
                         fname = :diffeqf, lhsname=:internal_var___du,
                         rhsnames=[:internal_var___u,:internal_var___p,:internal_var___t])
```

This builds an in-place Stan function compatible with the Stan differential equation solvers.
Unlike other build targets, this one requestions (vs, ps, iv) as the function arguments.
Only allowed on arrays of equations.
"""
function _build_function(target::StanTarget, eqs::Array{<:Equation}, vs, ps, iv;
                         conv = toexpr, expression = Val{true},
                         fname = :diffeqf, lhsname=:internal_var___du,
                         rhsnames=[:internal_var___u,:internal_var___p,:internal_var___t])

    @warn "build_function(::Array{<:Equation}...) is deprecated. Use build_function(::AbstractArray...) instead."
    @assert expression == Val{true}

    differential_equation = string(join([numbered_expr(eq,vs,ps,lhsname=lhsname,
                                   rhsnames=rhsnames) for
                                   (i, eq) ∈ enumerate(eqs)],";\n  "),";")
    """
    real[] $fname(real $(conv(iv)),real[] $(rhsnames[1]),real[] $(rhsnames[2]),real[] x_r,int[] x_i) {
      real $lhsname[$(length(eqs))];
      $differential_equation
      return $lhsname;
    }
    """
end

"""
Build function target: `StanTarget`

```julia
function _build_function(target::StanTarget, ex::AbstractArray, vs, ps, iv;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf, lhsname=:internal_var___du,
                         rhsnames    =  [:internal_var___u,:internal_var___p,:internal_var___t])
```

This builds an in-place Stan function compatible with the Stan differential equation solvers.
Unlike other build targets, this one requestions (vs, ps, iv) as the function arguments.
Only allowed on expressions, and arrays of expressions.
"""
function _build_function(target::StanTarget, ex::AbstractArray, vs, ps, iv;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf, lhsname=:internal_var___du,
                         rhsnames    =  [:internal_var___u,:internal_var___p,:internal_var___t])

    @assert expression == Val{true}

    if !columnmajor
        return _build_function(target, hcat([row for row ∈ eachrow(ex)]...), vs, ps, iv;
                            columnmajor = true,
                            conv        = conv,
                            expression  = expression,
                            fname       = fname,
                            lhsname     = lhsname,
                            rhsnames    = rhsnames)
    end

    equations = Vector{String}()
    for col ∈ 1:size(ex,2)
        for row ∈ 1:size(ex,1)
            lhs = string(lhsname, "[", (col-1) * size(ex,1) + row, "]")
            rhs = numbered_expr(value(ex[row, col]), vs, ps, iv;
                                lhsname  = lhsname,
                                rhsnames = rhsnames,
                                offset   = 0) |> string
            push!(equations, string(lhs, " = ", rhs, ";"))
        end
    end

    """
    real[] $fname(real $(conv(iv)),real[] $(rhsnames[1]),real[] $(rhsnames[2]),real[] x_r,int[] x_i) {
      real $lhsname[$(length(equations))];
    $([eqn == equations[end] ? string("  ", eqn) : string("  ", eqn, "\n") for eqn ∈ equations]...)
      return $lhsname;
    }
    """
end
_build_function(target::StanTarget, ex::Num, vs, ps, iv; kwargs...) = _build_function(target, [ex], vs, ps, iv; kwargs...)

"""
Build function target: `MATLABTarget`

```julia
function _build_function(target::MATLABTarget, eqs::Array{<:Equation}, args...;
                         conv = toexpr, expression = Val{true},
                         lhsname=:internal_var___du,
                         rhsnames=[:internal_var___u,:internal_var___p,:internal_var___t])
```

This builds an out of place anonymous function @(t,rhsnames[1]) to be used in MATLAB.
Compatible with the MATLAB differential equation solvers. Only allowed on expressions,
and arrays of expressions.
"""
function _build_function(target::MATLABTarget, eqs::Array{<:Equation}, args...;
                         conv = toexpr, expression = Val{true},
                         fname = :diffeqf, lhsname=:internal_var___du,
                         rhsnames=[:internal_var___u,:internal_var___p,:internal_var___t])

    @warn "build_function(::Array{<:Equation}...) is deprecated. Use build_function(::AbstractArray...) instead."
    @assert expression == Val{true}

    matstr = join([numbered_expr(eq.rhs,args...,lhsname=lhsname,
                                  rhsnames=rhsnames) for
                                  (i, eq) ∈ enumerate(eqs)],"; ")

    matstr = replace(matstr,"["=>"(")
    matstr = replace(matstr,"]"=>")")
    matstr = "$fname = @(t,$(rhsnames[1])) ["*matstr*"];"
    matstr
end

"""
Build function target: `MATLABTarget`

```julia
function _build_function(target::MATLABTarget, ex::AbstractArray, args...;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf,
                         lhsname     = :internal_var___du,
                         rhsnames    = [:internal_var___u,:internal_var___p,:internal_var___t])
```

This builds an out of place anonymous function @(t,rhsnames[1]) to be used in MATLAB.
Compatible with the MATLAB differential equation solvers. Only allowed on expressions,
and arrays of expressions.
"""
function _build_function(target::MATLABTarget, ex::AbstractArray, args...;
                         columnmajor = true,
                         conv        = toexpr,
                         expression  = Val{true},
                         fname       = :diffeqf,
                         lhsname     = :internal_var___du,
                         rhsnames    = [:internal_var___u,:internal_var___p,:internal_var___t])

    @assert expression == Val{true}

    if !columnmajor
        return _build_function(target, hcat([row for row ∈ eachrow(ex)]...), args...;
                               columnmajor = true,
                               conv        = conv,
                               expression  = expression,
                               fname       = fname,
                               lhsname     = lhsname,
                               rhsnames    = rhsnames)
    end

    matstr = ""
    for row ∈ 1:size(ex,1)
        row_strings = Vector{String}()
        for col ∈ 1:size(ex,2)
            lhs = string(lhsname, "[", (col-1) * size(ex,1) + row-1, "]")
            rhs = numbered_expr(value(ex[row, col]), args...;
                                lhsname  = lhsname,
                                rhsnames = rhsnames,
                                offset   = 0) |> string
            push!(row_strings, rhs)
        end
        matstr = matstr * "  " * join(row_strings, ", ") * ";\n"
    end

    matstr = replace(matstr,"["=>"(")
    matstr = replace(matstr,"]"=>")")
        matstr = "$fname = @(t,$(rhsnames[1])) [\n"*matstr*"];\n"

    return matstr

end
_build_function(target::MATLABTarget, ex::Num, args...; kwargs...) = _build_function(target, [ex], args...; kwargs...)
