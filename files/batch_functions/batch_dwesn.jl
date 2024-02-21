
function do_batch_dwesn(_params_esn, _params)

    p,pe  = _params, _params_esn
    im_sz = (p[:radius]*2 + 1)^2

    layers = []
    layer1 = layerESN( esns = [
                ESN( 
                     R      = new_R(p[:layers][1][2], density=pe[:density][1][i], rho=pe[:rho][1][i], gpu=p[:gpu])
                    ,R_in   = new_R_in(p[:layers][1][2], im_sz , sigma = pe[:sigma][1][i] ,gpu=p[:gpu], density=pe[:Rin_dens][1][i])
                    ,R_scaling = pe[:R_scaling][1][i], alpha = pe[:alpha][1][i], rho = pe[:rho][1][i], sigma = pe[:sigma][1][i], sgmd = pe[:sgmds][1][i]
                ) for i in 1:p[:layers][1][1]
            ])
    push!(layers,layer1)

    for l in 2:length(p[:layers])
        layer = layerESN( esns = [
            ESN(
                 R      = new_R(p[:layers][l][2], density=pe[:density][l][i], rho=pe[:rho][l][i], gpu=p[:gpu])
                ,R_in   = new_R_in(p[:layers][l][2], layers[l-1].nodes, sigma = pe[:sigma][l][i] ,gpu=p[:gpu], density=pe[:Rin_dens][l][i] )
                ,R_scaling = pe[:R_scaling][l][i], alpha  = pe[:alpha][l][i], rho = pe[:rho][l][i], sigma = pe[:sigma][l][i], sgmd = pe[:sgmds][l][i]
            ) for i in 1:p[:layers][l][1]
        ])
        push!(layers,layer)
    end   

    tms = @elapsed begin
        dwE = DWESN(
            layers = layers
            ,beta=p[:beta] 
            ,train_function = p[:train_f]
            ,test_function = p[:test_f]
            )
        tm_train = @elapsed begin
            dwE.train_function(dwE,p)
        end
        tm_test = @elapsed begin
            dwE.test_function(dwE,p)
        end
    end
 
    to_log = Dict(
        "Total time"        => tms
        ,"Train time"       => tm_train
        ,"Test time"        => tm_test
        ,"Error"            => dwE.error
        ,"Layers"           => p[:layers]
        , "Sigmoids"        => pe[:sgmds]
        , "Alphas"          => pe[:alpha]
        , "Densities"       => pe[:density]
        , "R_in_densities"  => pe[:Rin_dens]
        , "Rhos"            => pe[:rho]
        , "Sigmas"          => pe[:sigma]
        , "R_scalings"      => pe[:R_scaling]
        , "nodes" => sum( [ l[1]*l[2] for l in p[:layers] ] )
        , "alpha" => pe[:alpha][1][1]
        , "density" => pe[:density][1][1]
        , "rho" => pe[:rho][1][1]
        , "sigma" => pe[:sigma][1][1]
    )
    cls_nms = string.(p[:classes])
    if p[:wb]
        if p[:confusion_matrix]
            to_log["conf_mat"] = Wandb.wandb.plot.confusion_matrix(
                y_true = p[:test_labels][1:p[:test_length]], preds = [x[1] for x in dwE.Y], class_names = cls_nms
            )
        end
        Wandb.log(p[:lg], to_log )
    else
        display(to_log)
        display(confusion_matrix(cls_nms,p[:test_labels], [x[1] for x in dwE.Y]) )
    end
    return dwE
end
