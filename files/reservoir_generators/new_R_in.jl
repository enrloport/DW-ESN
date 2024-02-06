

function new_R_in( m,n; sigma::Float64=1.0, density=1.0, distribution=Uniform, gpu=false)
    if density != 1.0
        R_in = sprand(m, n, density, x-> rand(distribution(-sigma, sigma), x) )
        R_in = Array(R_in)
    else
        R_in = rand( distribution( -sigma, sigma ) , m, n )
    end

    if gpu
        R_in=CuArray(R_in)
    end
    return R_in
end