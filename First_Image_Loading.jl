#=
Pkg.add("VegaLite")
Pkg.add("VegaDatasets")
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("Images")
Pkg.add("ImageMagick")
=#

url = "https://i.imgur.com/VGPeJ6s.jpg"
download(url, "phillip.jpg")

using Images
using ImageMagick

begin
    philip = load("phillip.jpg")
end

philip

typeof(philip)

RGB(0.4, 0.6, 0.5)

size(philip)

philip[100,400]

typeof(philip[100,400])

philip[100:400,400:700]

begin
    (h,w) = size(philip)
    dog_head = philip[round(Int,h*.48):round(Int,h*0.9), round(Int,(w*.07)):round(Int,(w*0.6))]
end

dog_head
[dog_head dog_head]
typeof([dog_head dog_head]) 
size([dog_head dog_head])


[dog_head reverse(dog_head, dims=2)
 reverse(dog_head, dims=1) reverse(reverse(dog_head, dims=1), dims=2)
]

new_phil = copy(dog_head)

for i in 1:100
    for j in 1:300
        new_phil[i,j] = RGB(1,0,0)
    end
end

new_phil

new_phil2 = copy(dog_head)

begin
    new_phil2[1:100, 1:300] .= RGB(0,1,0)
    new_phil2
end

dog_head
new_phil
new_phil2

∇
δ
π
# \ plus a character equals symbols! this is rad

arrays_work = [5, 10, 11, 15]

arrays_work[1]

arrays_work[2] == 10

typeof(arrays_work)

data = Dict(:A => [5,10,15], :B => [11, 12, 13, 18])

data[:A]
# Still need colons before "symbol" to define it as a symbol
data2 = Dict(:α => [5,10,15], :B => [11, 12, 13, 18])

arr = [5, 5, 7, 7, 6, 4, 5]
set = Set(arr)
println(set)

# Creating a struct will make a new type that can hold 
# arbitrary and predefined data structures. We can create a 
# struct by using the struct keyword followed by a definition and data
h = 5, 10, 15 # a tuple
v = [6, 11, 16] # an array

struct type_struc
    asdf
    mdfh
end

w = type_struc(5,10)

w.asdf

function addstruc(type_struc)
    return(type_struc.asdf + type_struc.mdfh)
end

addstruc(w)

# Big datatypes on the other hand, are real numbers but go beyond 
# the capabilities of most 64-bit applications 
# Big Int
big(51515235151351335)

# Big Float
big(5.4172473471347374147)


