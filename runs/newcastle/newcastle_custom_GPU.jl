include("../../ESN.jl")
using Metaheuristics

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"

file2= "newcastle/newcastle_cloud.nc"

_info     = ncinfo(dir*file)

_imgs   = ncread(dir*file, "__xarray_dataarray_variable__")#[:,30-3:30+3,30-3:30+3]
_windD  = ncgetatt(dir*file2, "global", "Wind Direction")
_hum    = ncgetatt(dir*file2, "global", "Humidity")
_windS  = ncgetatt(dir*file2, "global", "Wind Speed")
_press  = ncgetatt(dir*file2, "global", "Pressure")

maxi = length(_hum)

_imgs, _hum, _windS, _press = _imgs[1:maxi,:,:], _hum[1:maxi]./10.0, _windS[1:maxi]./100.0, _press[1:maxi]./1000

_windD = _windD[1:maxi] ./ 360

maximum(_windD)

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "newcastle_GPU_norm"
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 49000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN!
    ,:test_f            => __do_test_DWESN!
    ,:target_pixel      => (25,50)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:data              => _imgs
)
_params[:input_size] = ((_params[:radius]*2)+1)^2

function split_data_newcastle_custom(;data, train_length, test_length, target_pixel, humidity, preassure, wind_speed, wind_dir, radius, steps=[1])

    tp,rd,trl,tel,hum,press,windS, windD = target_pixel, radius, train_length, test_length, humidity, preassure, wind_speed, wind_dir

    d = reshape(cc_to_int(data[:, tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd]), :, (2*rd + 1)^2 )

    train_x = d[1:trl         , : ]
    test_x  = d[trl+1:trl+tel , : ]
    train_y = Dict(s => windD[1+s:trl+s         ] for s in steps)
    test_y  = Dict(s => windD[trl+1+s:trl+tel+s ] for s in steps)
    
    return train_x, train_y, test_x, test_y
end

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_newcastle_custom(
    data              = _imgs
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    , humidity        = _hum
    , preassure       = _press
    , wind_speed      = _windS
    , wind_dir        = _windS
    )


    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


dwE=[]
for _ in 1:repit
    _params[:layers] = [ [200 for _ in 1:5],[300,300]]
    _params[:connections] = Dict(
         6 => [(i,1.0) for i in 1:5 ]
        ,7 => [(i,1.0) for i in 1:5 ]
    )
    _params[:active_inputs] = [1,2,3,4,5,6,7]
    _params[:active_outputs]= [6,7]

    sd = rand(1:10000)
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
        , "Target Pixel"        => _params[:target_pixel]
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

    tm = @elapsed begin
        dwE = do_batch_dwesn(_params_esn,_params)
    end

    # err_dict = Dict("Error_step_"*string(s) => dwE.error[s] for s in _params[:steps] )

    if _params[:wb]
       _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], merge(par, err_dict) )
    end
    display(par)

    _params[:total_time] = tm

    if _params[:wb]
        close(_params[:lg])
    end

end


dwE.Y
dwE.Y_target

for s in _params[:steps]
    err = sum([(dwE.Y[s][i] - dwE.Y_target[s][i])^2 for i in 1:length(dwE.Y)]) / _params[:test_length]
    println("MSE step ", string(s), " -> ", err)
end
# EOF