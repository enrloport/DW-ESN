
#===
Some minor pre-processing is needed to match the cloud types described in CloudCast paper.

1. Remove atipical values (> 253)
2. Give to classes 1, 2, 3 and 4 (which are cloud-free land, cloud-free sea, snow over land and sea ice. I.e., classes that are not actual cloud types) a value of 0.
3. Subtract 4 to match the integer values from cloudcast paper and set NaNs = 0, i.e.,
===#

function cc_to_int(img)
    return map(x-> x > 253 ? Int8(0) : x-4 > 0 ? Int8((x-4)) : Int8(0) , img)
end

function cc_to_float(img)
    return map(x-> x > 253 ? Float16(0.0) : x-4 > 0 ? Float16((x-4)/10) : Float16(0.0) , img)
end