
function do_batch_dwesn(_params_esn, _params,sd)
    im_sz    = (_params[:radius]*2 + 1)^2
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    sgmds    = _params_esn[:sgmds]
    densities= _params_esn[:density]
    rin_dens = _params_esn[:Rin_dens]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]
    p_lys    = _params[:layers]

    layers = []
    layer1 = layerESN( esns = [
                ESN( 
                     R      = new_R(p_lys[1][2], density=densities[1][i], rho=rhos[1][i], gpu=_params[:gpu])
                    ,R_in   = new_R_in(p_lys[1][2], im_sz , sigma = sigmas[1][i] ,gpu=_params[:gpu], density=rin_dens[1][i])
                    ,R_scaling = r_scales[1][i], alpha = alphas[1][i], rho = rhos[1][i], sigma = sigmas[1][i], sgmd = sgmds[1][i]
                ) for i in 1:p_lys[1][1]
            ])
    push!(layers,layer1)

    for l in 2:length(_params[:layers])
        layer = layerESN( esns = [
            ESN(
                 R      = new_R(p_lys[l][2], density=densities[l][i], rho=rhos[l][i], gpu=_params[:gpu])
                ,R_in   = new_R_in(p_lys[l][2], layers[l-1].nodes, sigma = sigmas[l][i] ,gpu=_params[:gpu], density=rin_dens[l][i] )
                ,R_scaling = r_scales[l][i], alpha  = alphas[l][i], rho = rhos[l][i], sigma = sigmas[l][i], sgmd = sgmds[l][i]
            ) for i in 1:p_lys[l][1]
        ])
        push!(layers,layer)
    end   

    tms = @elapsed begin
        dwE = DWESN(
            layers = layers
            ,beta=_params[:beta] 
            ,train_function = _params[:train_f]
            ,test_function = _params[:test_f]
            )
        tm_train = @elapsed begin
            dwE.train_function(dwE,_params)
        end
        tm_test = @elapsed begin
            dwE.test_function(dwE,_params)
        end
    end
 
    to_log = Dict(
        "Total time"  => tms
        ,"Train time"=> tm_train
        ,"Test time"=> tm_test
       , "Error"    => dwE.error
    )
    cls_nms = string.(_params[:classes])
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
        Wandb.log(_params[:lg], Dict( "conf_mat"  => Wandb.wandb.plot.confusion_matrix(
                    y_true = _params[:test_labels][1:_params[:test_length]], preds = [x[1] for x in dwE.Y], class_names = cls_nms
                )))
    else
        display(to_log)
        display(confusion_matrix(cls_nms,_params[:test_labels], [x[1] for x in dwE.Y]) )
    end
    return dwE
end
