# Example using U-net to deconvolve an image

using UNet, Flux, TestImages, View5D, Noise, NDTools, FourierTools, IndexFunArrays

img = 100f0 .* Float32.(testimage("resolution_test_512"))

u = Unet(); 

u = gpu(u);
function loss(u, x, y)
    return Flux.mse(u(x),y)
end

opt_state = Flux.setup(Momentum(), u);

# selects a tile at a random (default) or predifined (ctr) position returning tile and center.
function get_tile(img, tile_size=(128,128), ctr = (rand(tile_size[1]÷2:size(img,1)-tile_size[1]÷2),rand(tile_size[2]÷2:size(img,2)-tile_size[2]÷2)) )
    return select_region(img,new_size=tile_size, center=ctr), ctr
end

R_max = 70;
sz = size(img); psf = abs2.(ift(disc(Float32, sz, R_max))); psf ./= sum(psf); conv_img = conv_psf(img,psf);

scale = 0.5f0/maximum(conv_img)
patch = (128, 128)
for n in 1:2000
    println("Iteration: $n")
    myimg, pos = get_tile(conv_img, patch)
    # image to denoise
    # nimg1 = gpu(reshape(scale .* myimg,(size(myimg)...,1,1))); # gpu(scale.*reshape(poisson(myimg),(size(myimg)...,1,1)))
    nimg1 = gpu(Float32.(scale.*reshape(poisson(Float64.(myimg)), (size(myimg)...,1,1))))
    # goal image (with noise)
    pimg, pos = get_tile(img, patch, pos)
    pimg = gpu(scale.*reshape(pimg,(size(myimg)...,1,1)))
    rep = [(nimg1, pimg)] # Iterators.repeated((nimg1, pimg), 1);
    Flux.train!(loss, u, rep, opt_state)
end

# apply the net to the whole image instead:
nimg = gpu(scale .* reshape(conv_img,(size(conv_img)...,1,1))); # gpu(scale.*reshape(poisson(conv_img),(size(conv_img)...,1,1)))
nimg2 = gpu(scale.*reshape(poisson(Float64.(conv_img)),(size(conv_img)...,1,1)))
# display the images using View5D
@vt img nimg u(nimg) nimg2 u(nimg2)
