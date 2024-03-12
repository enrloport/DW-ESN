# Function to test an already trained deepwideESN struct
function __do_test_DWESN_cloudcast_image!(dwE, args::Dict)
    test_length = args[:test_length]

    classes_Y    = Array{Tuple{Float64,Int,Int}}[]
    wrong_class  = []
    dwE.Y      = []

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    d1,d2,d3 = test_length, size(args[:data])[2], size(args[:data])[3]
    res = zeros(d1,d2,d3)

    for i in 1+args[:radius]:d2-args[:radius], j in 1+args[:radius]:d3-args[:radius]
        args[:target_pixel] = (i,j)
        args[:train_data],  args[:train_labels],  args[:test_data],  args[:test_labels] = split_data_cloudcast(
        data              = args[:data]
        , train_length    = args[:train_length]
        , test_length     = args[:test_length]
        , target_pixel    = args[:target_pixel]
        , radius          = args[:radius]
        , step            = args[:step]
        )
        sz_train = size(args[:train_data])[1]
        i_t      = args[:initial_transient]

        for it in sz_train-i_t:sz_train
            _step_cloudcast(dwE, args[:train_data], it, f)
        end

        for t in 1:test_length
            _step_cloudcast(dwE, args[:test_data], t, f)

            x = vcat(f(args[:test_data][t,:,:]), [ _e.x for l in dwE.layers for _e in l.esns]...  , f([1]) )

            pairs  = []
            for c in args[:classes]
                yc = Array(dwE.classes_Routs[c] * x)[1]
                push!(pairs, (yc, c, args[:test_labels][t]))
            end
            pairs_sorted  = reverse(sort(pairs))
            
            if pairs_sorted[1][2] != pairs_sorted[1][3]
                push!(wrong_class, (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
            end

            # push!(dwE.Y,[Int8(pairs_sorted[1][2]) ;])
            push!(classes_Y, pairs )

            res[t,i,j] = Int8(pairs_sorted[1][2])

        end
    end

    dwE.Y = res

    dwE.wrong_class= wrong_class
    dwE.classes_Y  = classes_Y
    dwE.Y_target   = args[:test_labels]
    dwE.error      = length(wrong_class) / length(classes_Y)

end
