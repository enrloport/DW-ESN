# Function to test an already trained deepwideESN struct
function __do_test_DWESN!(dwE, args::Dict)
    test_length   = args[:test_length]
    dwE.Y         = Dict( stp => [] for stp in args[:steps])
    f             = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:test_length
        # _step_cloudcast(dwE, args[:test_data], t, f)

        ut = reshape(args[:test_data][t,:,:], :, 1)
        _step_cloudcast(dwE,  ut, f)

        x = vcat(f(args[:test_data][t,:,:]), [ _e.x for l in dwE.layers for _e in l.esns if _e.output_active]...  , f([1]) )

        for stp in args[:steps]
            y = Array(dwE.R_out[stp] * x)[1]
            push!(dwE.Y[stp],y)
        end
    end

    dwE.Y_target   = args[:test_labels]
end
