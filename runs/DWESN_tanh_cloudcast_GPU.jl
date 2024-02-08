include("../ESN.jl")



# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
_data_o = ncread(dir*file, "__xarray_dataarray_variable__")

# u = cc_to_int(_data_o[1,:,:])
# countmap(u)
# Images.Gray.(u./10)


# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => false
    ,:wb_logger_name=> "DWESN_tanh_mnist_GPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta          => 1.0e-8
    ,:initial_transient=>2
    ,:train_length  => size(train_y)[1] -55000
    ,:test_length   => size(test_y)[1]  -9000
    ,:train_f       => __do_train_DWESN_cloudcast!
    ,:test_f        => __do_test_DWESN_cloudcast!
    ,:image_size    => (3,3)
)

if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb]
    using Logging
    using Wandb
end



for _ in 1:repit
    _params[:layers] = [(1,200),(1,1000)]
    sd = rand(1:10000)
    Random.seed!(sd)
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),num_e[1] ) for num_e in _params[:layers] ]
        ,:density  => [rand(Uniform(0.01,0.2),num_e[1] ) for num_e in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.01,0.5),num_e[1] ) for num_e in _params[:layers]]
        ,:rho      => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:_params[:layers][i][1]] for i in 1:length(_params[:layers]) ]
    )

    _params[:train_data]   = train_x
    _params[:test_data]    = test_x
    _params[:train_labels] = train_y
    _params[:test_labels]  = test_y
    par = Dict(
        "Layers" => _params[:layers]
        , "Total nodes"  => sum( map(x -> x[1]*x[2], _params[:layers] ) )
        , "Train length" => _params[:train_length]
        , "Test length"  => _params[:test_length]
        , "Resized"      => _params[:image_size][1]
        , "Initial transient"=> _params[:initial_transient]
        , "seed"         => sd
        , "sgmds"        => _params_esn[:sgmds]
        , "alphas" => _params_esn[:alpha]
        , "densities" => _params_esn[:density]
        , "R_in_densities" => _params_esn[:Rin_dens]
        , "rhos" => _params_esn[:rho]
        , "sigmas" => _params_esn[:sigma]
        , "R_scalings" => _params_esn[:R_scaling]
        )
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end
    display(par)

    r1=[]
    tm = @elapsed begin
        r1 = do_batch_dwesn(_params_esn,_params, sd)
    end
    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", r1.error, "\n", printime  )

end

# EOF