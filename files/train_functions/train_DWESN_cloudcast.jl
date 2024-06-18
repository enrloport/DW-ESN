function __fill_X_DWESN_cloudcast!(dwE, args::Dict )

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:args[:initial_transient]
        _step_cloudcast(dwE, args[:train_data], t, f)
    end

    for t in args[:initial_transient]+1:args[:train_length]
        t_in = t - args[:initial_transient]
        _step_cloudcast(dwE, args[:train_data], t, f)

        dwE.X[:,t_in] = vcat(f(args[:train_data][t,:,:]), [ _e.x for l in dwE.layers for _e in l.esns if _e.output_active]...  , f([1]) )
    end
end


function __make_Rout_DWESN_cloudcast!(dwE,args)
    X             = dwE.X
    classes       = args[:classes]

    for stp in args[:steps]
        # New dataset labels for each class
        classes_Yt    = Dict( c => zeros(args[:train_length]-args[:initial_transient]) for c in classes )
        for t in 1:args[:train_length]-args[:initial_transient]
            lt = args[:train_labels][stp][t+args[:initial_transient]]
            for c in classes
                y = lt == c ? 1.0 : 0.0
                classes_Yt[c][t] = y
            end
        end
        if args[:gpu]
            classes_Yt = Dict( k => CuArray(classes_Yt[k]) for k in keys(classes_Yt) )
        end

        cudamatrix              = args[:gpu] ? CuArray : Matrix
        dwE.classes_Routs[stp]  = Dict( c => cudamatrix(transpose((X*transpose(X) + dwE.beta*I) \ (X*classes_Yt[c]))) for c in classes )
    end

end


function __do_train_DWESN_cloudcast!(dwE, args)
    num               = args[:train_length]-args[:initial_transient]
    dwE.X             = zeros( dwE.output_size + args[:input_size] + 1, num)
    reset_function    = (x) -> zeros(x,1)

    if args[:gpu]
        dwE.X             = CuArray(dwE.X)
        reset_function    = (x) -> CuArray(zeros(x,1))
    end

    # reset states
    map(_e -> _e.x = reset_function(_e.R_size) , values(dwE.esns) )

    __fill_X_DWESN_cloudcast!(dwE,args)
    __make_Rout_DWESN_cloudcast!(dwE,args)
end
