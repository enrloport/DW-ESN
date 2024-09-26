include("ESN.jl")

using CSV, DataFrames

step = string(4)

# h1 = Matrix(CSV.read("mresn_cloudcast_image_multipred__50000+"*step*".csv", DataFrame, header=false))
p1 = Matrix(CSV.read("mresn_cloudcast_image_multipred__50000+"*step*".csv", DataFrame, header=false))
p2 = Matrix(CSV.read("mresn_cloudcast_image_multipred__70-105__50000_52500+"*step*".csv", DataFrame, header=false))
# p3 = Matrix(CSV.read("mresn_cloudcast_image_multipred__(103, 93)__50000+"*step*".csv", DataFrame, header=false))
p3 = Matrix(CSV.read("mresn_cloudcast_image_optimized__(103, 93)__500__49500+"*step*".csv", DataFrame, header=false))
p4 = Matrix(CSV.read("mresn_cloudcast_image_optimized__(113, 13)__500__49500+"*step*".csv", DataFrame, header=false))


# allh = hcat(h1,h2,h3,h4)
hm1 = map( x -> isnan(x) ? 1.0 : x , reshape(p1, 128,128))
hm2 = map( x -> isnan(x) ? 1.0 : x , reshape(p2, 128,128))
hm3 = map( x -> isnan(x) ? 1.0 : x , reshape(p3, 128,128))
hm4 = map( x -> isnan(x) ? 1.0 : x , reshape(p4, 128,128))

Images.Gray.(vcat(hm1,hm2,hm3,hm4))

heatmap(hm3, xflip=false, yflip=true, size=(592,512))


m1 = reshape(hm1[4:125,4:125], :, 1)
m2 = reshape(hm2[4:125,4:125], :, 1)
m3 = reshape(hm3[4:125,4:125], :, 1)
m4 = reshape(hm4[4:125,4:125], :, 1)


mins = [ minimum([m1[i],m2[i],m3[i],m4[i]]) for i in 1:length(m1) ]

mn, sd = mean(mins), std(mins)

rmin = reshape(mins, 122,122)

Images.Gray.(rmin)


heatmap(rmin, clim=(0,1), xflip=false, yflip=true, size=(592,512))





mod1 = copy(rmin)

mod1[102,88] = 0
mod1[110,10] = 0 


heatmap(mod1, clim=(0,1), xflip=false, yflip=true, size=(592,512))





# img = Images.Gray.(vcat(allt, allh))
# save(img, "50001+h1,h2,h3,h4.png" )




