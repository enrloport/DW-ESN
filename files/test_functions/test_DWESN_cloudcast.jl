# Function to test an already trained deepESN struct
function __do_test_DWESN_cloudcast!(deepE, args::Dict)
    test_length = args[:test_length]

    classes_Y    = Array{Tuple{Float64,Int,Int}}[]
    wrong_class  = []
    deepE.Y      = []

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:test_length

        _step_cloudcast(deepE, args[:test_data], t, f)

        x = vcat(f(args[:test_data][t,:,:]), [ _e.x for l in deepE.layers for _e in l.esns]...  , f([1]) )


        pairs  = []
        for c in args[:classes]
            yc = Array(deepE.classes_Routs[c] * x)[1]
            push!(pairs, (yc, c, args[:test_labels][t]))
        end
        pairs_sorted  = reverse(sort(pairs))
        
        if pairs_sorted[1][2] != pairs_sorted[1][3]
            push!(wrong_class, (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
        end

        push!(deepE.Y,[Int8(pairs_sorted[1][2]) ;])
        push!(classes_Y, pairs )

    end

    deepE.wrong_class= wrong_class
    deepE.classes_Y  = classes_Y
    deepE.Y_target   = args[:test_labels]
    deepE.error      = length(wrong_class) / length(classes_Y)

end
