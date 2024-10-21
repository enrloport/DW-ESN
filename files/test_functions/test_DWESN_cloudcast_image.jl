# Function to test an already trained deepwideESN struct
function __do_test_DWESN_cloudcast_image!(dwE, args::Dict)
    f           = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)
    test_length = args[:test_length]
    sz1,sz2     = size(args[:data])[2], size(args[:data])[3]

    classes_Y   = Dict( 
        (i,j) => Dict( stp => Array{Tuple{Float64,Int,Int}}[] for stp in args[:steps])
        for i in 1:sz1 for j in 1:sz2 
        )
    wrong_class = Dict(
        (i,j) => Dict( stp => 0 for stp in args[:steps] )
        for i in 1:sz1 for j in 1:sz2 
        )

    dwE.Y       = Dict( 
        (i,j) => Dict( stp => [] for stp in args[:steps] )
        for i in 1:sz1 for j in 1:sz2 
        )

    Threads.@threads for i in 1+args[:radius]:sz1-args[:radius]
        for j in 1+args[:radius]:sz2-args[:radius]
            args[:target_pixel] = (i,j)
            args[:train_data],  args[:train_labels],  args[:test_data],  args[:test_labels] = split_data_cloudcast(
            data              = args[:data]
            , train_length    = args[:train_length]
            , test_length     = args[:test_length]
            , target_pixel    = args[:target_pixel]
            , radius          = args[:radius]
            , steps           = args[:steps]
            )
            sz_train = size(args[:train_data])[1]
            i_t      = args[:initial_transient]


	    for it in sz_train-(2*i_t):sz_train
                ut = reshape(args[:train_data][it,:,:], :, 1)
                _step_cloudcast(dwE, ut, f)
            end


            for t in 1:test_length
                ut      = reshape(args[:test_data][t,:,:], :, 1)
                _step_cloudcast(dwE, ut, f)
                x       = vcat(f(args[:test_data][t,:,:]), [ _e.x for l in dwE.layers for _e in l.esns if _e.output_active]...  , f([1]) )
                pairs   = Dict( stp => [] for stp in args[:steps])

                for stp in args[:steps]
                    for c in args[:classes]
                        yc = Array(dwE.classes_Routs[stp][c] * x)[1]
                        push!(pairs[stp], (yc, c, args[:test_labels][stp][t]))
                    end

                    pairs_sorted  = reverse(sort(pairs[stp]))

                    if pairs_sorted[1][2] != pairs_sorted[1][3]
                        wrong_class[(i,j)][stp] += 1
                    end
                end
            end
	    println( "Pixel (" *string(i)* "," *string(j)* ") -> ", wrong_class[(i,j)][4]/test_length )
        end
    end

    dwE.wrong_class= wrong_class
    dwE.classes_Y  = classes_Y
    dwE.Y_target   = args[:test_labels]
    dwE.error      = Dict( stp => [wrong_class[(i,j)][stp] / test_length for i in 1:sz1 for j in 1:sz2] for stp in args[:steps]   )
end
