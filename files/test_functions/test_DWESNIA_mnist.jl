# Function to test an already trained deepESN struct
function __do_test_DWESNIA_mnist!(deepE, args::Dict)
    test_length = args[:test_length]

    classes_Y    = Array{Tuple{Float64,Int,Int}}[]
    wrong_class  = []
    deepE.Y        = []

    if args[:gpu]
        f = (u) -> CuArray(reshape(u, :, 1))
    else
        f = (u) -> reshape(u, :, 1)
    end

    for t in 1:test_length

        for _ in 1:args[:initial_transient]
            _step_dwesnia(deepE, args[:test_data], t, f)
        end
        _step_dwesnia(deepE, args[:test_data], t, f)

        x = vcat(f(args[:test_data][:,:,t]), [ _e.x for l in deepE.layers for _e in l.esns]...  , f([1]) )


        pairs  = []
        for c in args[:classes]
            yc = Array(deepE.classes_Routs[c] * x)[1]
            push!(pairs, (yc, c, args[:test_labels][t]))
        end
        pairs_sorted  = reverse(sort(pairs))
        
        if pairs_sorted[1][2] != pairs_sorted[1][3]
            push!(wrong_class, (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
        end

        push!(deepE.Y,[pairs_sorted[1][2] ;])
        push!(classes_Y, pairs )

        for layer in deepE.layers
            for _esn in layer.esns
                _esn.x[:] = _esn.x .* 0
            end
        end
    end

    deepE.wrong_class= wrong_class
    deepE.classes_Y  = classes_Y
    deepE.Y_target   = args[:test_labels]
    deepE.error      = length(wrong_class) / length(classes_Y)

end
