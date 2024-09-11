function __fill_X_DWESN!(dwE, args::Dict )

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


function __make_Rout_DWESN!(dwE,args)
    X             = dwE.X

    for stp in args[:steps]
        y_target    = args[:train_labels][stp][args[:initial_transient]+1:end]
        
        if args[:gpu]
            y_target = CuArray(y_target)
        end

        cudamatrix  = args[:gpu] ? CuArray : Matrix

        t1 = (X*transpose(X) + dwE.beta*I)
        t2 = (X*y_target)

        dwE.R_out[stp]  = cudamatrix(transpose(t1 \ t2))
    end

end


function __do_train_DWESN!(dwE, args)
    num               = args[:train_length]-args[:initial_transient]
    dwE.X             = zeros( dwE.output_size + args[:input_size] + 1, num)
    reset_function    = (x) -> zeros(x,1)

    if args[:gpu]
        dwE.X             = CuArray(dwE.X)
        reset_function    = (x) -> CuArray(zeros(x,1))
    end

    # reset states
    map(_e -> _e.x = reset_function(_e.R_size) , values(dwE.esns) )

    __fill_X_DWESN!(dwE,args)
    __make_Rout_DWESN!(dwE,args)
end
