include("../../ESN.jl")
using Metaheuristics

# DATASET
dir       = "data/"
file      = "TrainCloud.nc"

file2     = "newcastle/newcastle_cloud.nc"

_info     = ncinfo(dir*file)

_imgs     = ncread(dir*file, "__xarray_dataarray_variable__")#[:,30-3:30+3,30-3:30+3]
_windD    = ncgetatt(dir*file2, "global", "Wind Direction")
_hum      = ncgetatt(dir*file2, "global", "Humidity")
_windS    = ncgetatt(dir*file2, "global", "Wind Speed")
_press    = ncgetatt(dir*file2, "global", "Pressure")

global maxi, len = 0,length(_hum) 
# for i in 1:len
#     if isnan(_hum[i])
#         global maxi = i-1
#         println(i)
#         break
#     end
# end
maxi = length(_hum)

_imgs, _hum, _windS, _press = _imgs[1:maxi,:,:], _hum[1:maxi]./10.0, _windS[1:maxi]./10.0, _press[1:maxi]./101.325

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "pso_newcastle_attention_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 49000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:target_pixel      => (25,50)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:train_data_extra  => Dict()
    ,:test_data_extra   => Dict()
)

function split_data_extra(train_length, test_length, var)
    tr,te   = train_length, test_length
    train_x = var[1:tr        , : ]
    test_x  = var[tr+1:tr+te  , : ]

    return train_x, test_x
end


_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = _imgs
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )

_params[:train_data_extra][1],  _params[:test_data_extra][1] = split_data_extra( _params[:train_length], _params[:test_length], _press)
_params[:train_data_extra][2],  _params[:test_data_extra][2] = split_data_extra( _params[:train_length], _params[:test_length], _windS)

_params[:input_size] = ((_params[:radius]*2)+1)^2
_params[:extra_data_size] = [size(_params[:train_data_extra][i],2) for i in 1:length(keys(_params[:train_data_extra]))]
    
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
    dwE=[]
    _params[:layers]            = [ [200 for _ in 1:5],[300,300]]
    _params[:active_inputs]     = [1,2,3,6,7]
    _params[:active_outputs]    = [6,7]
    _params[:attention_inputs]  = Dict( 
         4 => [1]
        ,5 => [2]
    )
    _params[:connections]       = Dict(
         6 => [(i,_x[i]) for i in 1:5 ]
        ,7 => [(i,_x[5+i]) for i in 1:5 ]
    )

    sd = 42 #rand(1:10000)
    Random.seed!(sd)

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),num_e[1] ) for num_e in _params[:layers] ]
        ,:density  => [rand(Uniform(0.1,0.3),num_e[1] ) for num_e in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.1,0.5),num_e[1] ) for num_e in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),num_e[1] ) for num_e in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:_params[:layers][i][1]] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Radius"              => _params[:radius]
        , "Initial transient"   => _params[:initial_transient]
        , "Active inputs"       => _params[:active_inputs]
        , "Active outputs"      => _params[:active_outputs]
        , "Layers"              => _params[:layers]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        , "rho"                 => _params_esn[:rho][1][1]
        , "sigma"               => _params_esn[:sigma][1][1]
        , "reservoirs"          => sum([x[1] for x in _params[:layers]])
        , "nodes"               => sum( [ sum(l) for l in _params[:layers] ] )
        , "alpha min"           => minimum( vcat( _params_esn[:alpha]...) )
        , "alpha max"           => maximum( vcat( _params_esn[:alpha]...) )
        , "density min"         => minimum( vcat( _params_esn[:density]...) )
        , "density max"         => maximum( vcat( _params_esn[:density]...) )
    )
    lx    = length(_x)
    edges = Dict( "Edge "*string(i) => _x[i] for i in 1:lx )

    tm = @elapsed begin
        dwE = do_batch_dwesn(_params_esn,_params)
    end

    err_dict = Dict("Error_step_"*string(s) => dwE.error[s] for s in _params[:steps] )

    if _params[:wb]
        Wandb.log(_params[:lg], merge(par,edges, err_dict) )
    end
    # display(par)

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", dwE.error, "\n", printime  )

    return dwE.error[4]
end

for _ in 1:repit

    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], pso_dict )
    else
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

    lx = ones(10)' .* -1
    ux = ones(10)'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )

    if _params[:wb]
        close(_params[:lg])
    end

end


# EOF