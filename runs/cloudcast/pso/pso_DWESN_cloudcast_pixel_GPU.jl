include("../../../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")



# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "pso_DWESN_cloudcast_pixel_H1to4-100_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 49000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:data              => all
)
_params[:input_size] = ((_params[:radius]*2)+1)^2

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


pso_dict = Dict(
    "N"  => 20
    ,"C1" => 1.5
    ,"C2" => 1.2
    ,"w"  => 0.5
    ,"max_iter" => 30
)

function fitness(_x)

    _params[:layers] = [ [300,300,300],[300,300,300]]
    _params[:connections] = Dict(
         4 => [(1,_x[1]),(2,_x[2]),(3,_x[3])]
        ,5 => [(1,_x[4]),(2,_x[5]),(3,_x[6])]
        ,6 => [(1,_x[7]),(2,_x[8]),(3,_x[9])]
    )
    _params[:active_inputs] = [1,2,3]
    _params[:active_outputs]= [4,5,6]

    sd = 42 #rand(1:10000)
    Random.seed!(sd)
    # _params[:layers] = [(2,300)]; sd=776; Random.seed!(sd) # error 0.2875

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers]]
        ,:density  => [rand(Uniform(0.1,0.3),length(layer) ) for layer in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.1,0.5),length(layer) ) for layer in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),length(layer) ) for layer in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:length(_params[:layers][i])] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Target pixel"        => _params[:target_pixel]
        , "Radius"              => _params[:radius]
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        )
    edges = Dict( "Edge "+string(i) => _x[i] for i in 1:length(_x) )
    
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], merge(par,edges) )
    end
    display(par)

    tm = @elapsed begin
        dwE = do_batch_dwesn(_params_esn,_params)
    end
    _params[:total_time] = tm
    full_log(_params,_params_esn,dwE)

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", dwE.error, "\n", printime  )

    return dwE.error[4]

end



for _ in 1:repit

    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], pso_dict )
    else
        display(par)
        display(pso_dict)
        println(" ")
    end

    pso = PSO(;information=Metaheuristics.Information()
        ,N  = pso_dict["N"]
        ,C1 = pso_dict["C1"]
        ,C2 = pso_dict["C2"]
        ,ω  = pso_dict["w"]
        ,options = Options(iterations=pso_dict["max_iter"])
    )

    lx = zeros(9)'
    ux = ones(9)'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )

    if _params[:wb]
        close(_params[:lg])
    end
 

end


# EOF


