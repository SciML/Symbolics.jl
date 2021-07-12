using SafeTestsets, Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Macro Test" begin include("macro.jl") end
    @safetestset "Arrays" begin include("arrays.jl") end
    @safetestset "Fuzz Arrays" begin include("fuzz-arrays.jl") end
    @safetestset "Differentiation Test" begin include("diff.jl") end
    @safetestset "Difference Test" begin include("difference.jl") end
    @safetestset "Degree Test" begin include("degree.jl") end 
    @safetestset "Is Linear or Affine Test" begin include("islinear_affine.jl") end
    @safetestset "Linear Solver Test" begin include("linear_solver.jl") end
    @safetestset "Overloading Test" begin include("overloads.jl") end
    @safetestset "Build Function Test" begin include("build_function.jl") end
    @safetestset "Build Function Array Test" begin include("build_function_arrayofarray.jl") end
    @safetestset "Build Targets Test" begin include("build_targets.jl") end
    @safetestset "Latexify Test" begin include("latexify.jl") end
    @safetestset "Domain Test" begin include("domains.jl") end
    @safetestset "Metatheory Optimization" begin include("mt_optimize.jl") end
    @safetestset "SymPy Test" begin include("sympy.jl") end

end

if GROUP == "Downstream"
    activate_downstream_env()
    #@time @safetestset "ParameterizedFunctions MATLABDiffEq Regression Test" begin include("downstream/ParameterizedFunctions_MATLAB.jl") end
end
