function _step_cloudcast(dwE, data,t,f)
    for _esn in dwE.layers[1].esns
        a = data[t,:,:]
        __update(_esn, a, f )
    end

    for i in 2:length(dwE.layers)
        for _esn in dwE.layers[i].esns
            v = vcat([_e.x for _e in dwE.layers[i-1].esns ]...)
            __update(_esn, v , f )
        end
    end
end

function __fill_X_DWESN_cloudcast!(dwE, args::Dict )

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:args[:initial_transient]
        _step_cloudcast(dwE, args[:train_data], t, f)
    end

    for t in args[:initial_transient]+1:args[:train_length]
        t_in = t - args[:initial_transient]
        _step_cloudcast(dwE, args[:train_data], t, f)

        dwE.X[:,t_in] = vcat(f(args[:train_data][t,:,:]), [ _e.x for l in dwE.layers for _e in l.esns]...  , f([1]) )
    end
end


function __make_Rout_DWESN_cloudcast!(dwE,args)
    X             = dwE.X
    classes       = args[:classes]
    classes_Yt    = Dict( c => zeros(args[:train_length]-args[:initial_transient]) for c in classes )  # New dataset for each class

    for t in 1:args[:train_length]-args[:initial_transient]
        lt = args[:train_labels][t+args[:initial_transient]]
        for c in classes
            y = lt == c ? 1.0 : 0.0
            classes_Yt[c][t] = y
        end
    end
    if args[:gpu]
        classes_Yt = Dict( k => CuArray(classes_Yt[k]) for k in keys(classes_Yt) )
    end

    cudamatrix = args[:gpu] ? CuArray : Matrix
    dwE.classes_Routs = Dict( c => cudamatrix(transpose((X*transpose(X) + dwE.beta*I) \ (X*classes_Yt[c]))) for c in classes )
end


function __do_train_DWESN_cloudcast!(dwE, args)
    num   = args[:train_length]-args[:initial_transient]
    flt = vcat(dwE.layers...)
    dwE.X = zeros( sum([layer.nodes for layer in flt ]) + (args[:radius]*2+1)^2 + 1, num)

    for layer in dwE.layers
        for _esn in layer.esns
            _esn.x = zeros( _esn.R_size, 1)
        end
    end

    if args[:gpu]
        dwE.X = CuArray(dwE.X)
        for layer in dwE.layers
            for _esn in layer.esns
                _esn.x = CuArray( _esn.x)
            end
        end
    end

    __fill_X_DWESN_cloudcast!(dwE,args)
    __make_Rout_DWESN_cloudcast!(dwE,args)
end
