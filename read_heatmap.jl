include("ESN.jl")

using CSV, DataFrames

step = string(1)

# h1 = Matrix(CSV.read("mresn_cloudcast_image_multipred__50000+"*step*".csv", DataFrame, header=false))
h1 = Matrix(CSV.read("mresn_cloudcast_image_multipred__70-105__50000_52500+"*step*".csv", DataFrame, header=false))


# allh = hcat(h1,h2,h3,h4)
hm1 = map( x -> isnan(x) ? 1.0 : x , reshape(h1, 128,128))

Images.Gray.(hm1)

heatmap(hm1, xflip=false, yflip=true, size=(592,512))


# img = Images.Gray.(vcat(allt, allh))
# save(img, "50001+h1,h2,h3,h4.png" )



