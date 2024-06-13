# Example using U-net for a noise2noise problem

using UNet, Flux, TestImages, View5D, Noise, NDTools, CUDA

img = 10f0 .* Float32.(testimage("resolution_test_512"))

u = Unet(); 

u = gpu(u);
function loss(u, x, y)
    # return mean(abs2.(u(x) .-y))
    return Flux.mse(u(x), y)
end
opt_state = Flux.setup(Momentum(), u);

# selects a tile at a random (default) or predifined (ctr) position returning tile and center.
function get_tile(img, tile_size=(128,128), ctr = (rand(tile_size[1]÷2:size(img,1)-tile_size[1]÷2),rand(tile_size[2]÷2:size(img,2)-tile_size[2]÷2)) )
    return select_region(img,new_size=tile_size, center=ctr), ctr
end

sz = size(img); 
scale = 0.5f0/maximum(img)
patch = (128, 128)
for n in 1:1000
    println("Iteration: $n")
    myimg, pos = get_tile(img, patch)
    # image to denoise
    nimg1 = gpu(scale.*reshape(Float32.(poisson(Float64.(myimg))), (size(myimg)...,1,1)))
    # goal image (with noise)
    nimg2 = gpu(scale.*reshape(Float32.(poisson(Float64.(myimg))), (size(myimg)...,1,1)))
    rep = [(nimg1, nimg2),] # Iterators.repeated((nimg1, nimg2), 1);
    # Flux.train!(loss, Flux.params(u), rep, opt_state)
    Flux.train!(loss, u, rep, opt_state)
end

# apply the net to the whole image instead:
nimg = gpu(scale.*reshape(Float32.(poisson(Float64.(img))),(size(img)...,1,1)));
# display the images using View5D
@vt img nimg u(nimg)
