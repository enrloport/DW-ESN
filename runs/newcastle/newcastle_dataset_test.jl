include("../../ESN.jl")

# DATASET
dir     = "data/"
file    = "newcastle_cloud.nc"

_info     = ncinfo(dir*file)

_imgs   = ncread(dir*file, "__xarray_dataarray_variable__")
_windD  = ncgetatt(dir*file, "global", "Wind Direction")
_hum    = ncgetatt(dir*file, "global", "Humidity")
_windS  = ncgetatt(dir*file, "global", "Wind Speed")
_press  = ncgetatt(dir*file, "global", "Pressure")

using CSV

df = CSV.read("data/newcastle/2017-900-Humidity.csv", DataFrame)

_df = df[df."Start Datetime" .== "2017-01-11 00:45:00",[1,4,7,8,9]]
_df2 = df[df."Start Datetime" .== "2017-07-07 08:15:00",[1,4,7,8,9]]

CSV.write("df2.csv", _df2)

sensor = unique(df[!, [1]])."Sensor Name"

candidates = []
for s in sensor
    _d = df[df."Sensor Name" .== s, [1,4,5,7,8]]
    sz=size(_d)

    if sz[1] > 32000
        push!(candidates,s)
    end
end

candidates

s1 = df[df."Sensor Name" .== candidates[8], [1,4,5,7,8,9]]

can = candidates[8]

for Y in ["2017"]
    for M in ["01"]#["01","02","03","04","05","06","07","08","09","10","12","12"]
        for D in 1:31
            _D = D < 10 ? "0"*string(D) : string(D)
            for h in 0:23
                for m in ["00","15","30","45"]
                    _h = string(h)
                    if h < 10 _h = "0"*_h end
                    day = Y*"-"*M*"-"*_D*" "
                    tim = string(_h)*":"*m*":00"
                    # println(day*tim)
                    # display(typeof(s1[s1."Start Datetime" .== day*tim, :]."Median Value"))
                    # println( size(s1[s1."Start Datetime" .== day*tim, :]."Median Value") )
                    if size(s1[s1."Start Datetime" .== day*tim, :]."Median Value",1) == 0
                        println(day*tim)
                    end
                    # println(size(s1[s1."Start Datetime" .== day*tim, :]."Median Value",1))
                end
            end
        end
    end
end
s1[s1."Start Datetime" .== "2017-01-11 00:45:00", :]
s1[940:950,:]
