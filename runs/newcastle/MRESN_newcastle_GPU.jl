include("../../ESN.jl")

# DATASET
dir     = "data/"
file    = "newcastle_cloud.nc"

_info     = ncinfo(dir*file)

_imgs   = ncread(dir*file, "__xarray_dataarray_variable__")
_windD  = ncgetatt(dir*file, "global", "Wind Direction")
_hum    = ncgetatt(dir*file, "global", "Humidity")
_windS  = ncgetatt(dir*file, "global", "Wind Speed")
_press  = ncgetatt(dir*file, "global", "Pressure")

# minimum(_windD)
# maximum(_windD)

# minimum(_hum)
# maximum(_hum)

# minimum(_windS)
# maximum(_windS)

# minimum(_press)
# maximum(_press)

global maxi, len = 0,length(_hum) 
for i in 1:len
    if isnan(_hum[i])
        global maxi = i-1
        println(i)
        break
    end
end

_imgs, _hum, _windS, _press = _imgs[1:maxi,:,:], _hum[1:maxi]./100, _windS[1:maxi]./100, _press[1:maxi]./1013



function split_data_newcastle(imgs, hum, windS, press, steps, tr_len, te_len)

    d = reshape(imgs,:,9)
    tar= d[:,5]
    ext= hcat(hum,windS,press)
    aux = hcat(d,ext)
    
    tr_x = aux[1:tr_len, :]
    te_x = aux[tr_len+1:tr_len+te_len,:]
    
    tr_y = [cc_to_int(tar[1+step:tr_len+step]) for step in steps]
    te_y = [cc_to_int(tar[tr_len+1+step:tr_len+te_len+step]) for step in steps]

    return tr_x, tr_y, te_x, te_y
end

repit = 1000
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => true
    ,:wb_logger_name    => "MRESN_newcastle_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 15000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:steps             => [1]
    ,:data              => all
)

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_newcastle(_imgs, _hum, _windS, _press, _params[:steps][1], _params[:train_length], _params[:test_length])

_params[:input_size] = size(_params[:train_data],2)

# u = cc_to_int(_params[:train_data][1,:,:])
# u2 = cc_to_int(_params[:train_data][2,:,:])
# Images.Gray.(u./10)
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


for _ in 1:repit
    dwE=[]
    _params[:layers] = [(rand([2,3,4,5]),300)]
    sd = rand(1:10000)
    Random.seed!(sd)
    # _params[:layers] = [(2,300)]; sd=776; Random.seed!(sd) # error 0.2875

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
        , "Total nodes"         => sum( map(x -> x[1]*x[2], _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        )
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end
    display(par)

    tm = @elapsed begin
        dwE = do_batch_dwesn(_params_esn,_params)
    end
    dwE.error = dwE.error[1]
    _params[:total_time] = tm
    full_log(_params,_params_esn,dwE)
    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", dwE.error, "\n", printime  )

end

# EOF