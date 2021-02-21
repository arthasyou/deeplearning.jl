using Test
using Revise
using Deeplearning

@testset "basic.jl" begin
    Deeplearning.df1()
    Deeplearning.df_arg_matrix()
    Deeplearning.df_m()
    Deeplearning.simple_models()
    Deeplearning.build_layers()
    Deeplearning.build_layers_o1()
    Deeplearning.build_layers_dense()
end
