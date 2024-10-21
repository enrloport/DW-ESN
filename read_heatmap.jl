include("ESN.jl")

using CSV, DataFrames

step = string(4)
name1= "mresn_cloudcast_4classes_image__(30, 30)__full__70076+"*step*".csv"
name2= "mresn_cloudcast_4classes_image__(113, 13)__full__70076+"*step*".csv"

p1 = Matrix(CSV.read(name1, DataFrame, header=false))
p2 = Matrix(CSV.read(name2, DataFrame, header=false))

hm1 = map( x -> isnan(x) ? 1.0 : x , reshape(p1, 128,128))
hm2 = map( x -> isnan(x) ? 1.0 : x , reshape(p2, 128,128))

# Images.Gray.(vcat(hm1,hm2,hm3,hm4))
Images.Gray.(vcat(hm1,hm2))

heatmap(hm1, clim=(0,1), xflip=false, yflip=true, size=(592,512))
heatmap(hm2, clim=(0,1), xflip=false, yflip=true, size=(592,512))

hm1[4,27:60] = hm1[4,27:60] .* 0
hm1[5,34:53] = hm1[5,34:53] .* 0

hm2[4,27:60] = hm2[4,27:60] .* 0
hm2[5,34:53] = hm2[5,34:53] .* 0

heatmap(hm1, clim=(0,1), xflip=false, yflip=true, size=(592,512))
heatmap(hm2, clim=(0,1), xflip=false, yflip=true, size=(592,512))

mn, sd = 1-mean(hm1), std(hm1)
mn, sd = 1-mean(hm2), std(hm2)

res = zeros(128,128)
for i in 4:125
    for j in 4:125
        res[i,j] = hm1[i,j] < hm2[i,j] ? hm1[i,j] : hm2[i,j]
    end
end

heatmap(res, clim=(0,1), xflip=false, yflip=true, size=(592,512))

mn, sd = 1-mean(res), std(res)


# img = Images.Gray.(vcat(allt, allh))
# save(img, "50001+h1,h2,h3,h4.png" )




