using Flux
using Flux: @functor

struct ConvBlock{T}
  op::T
end

@functor ConvBlock

function ConvBlock(in_out_pair::Pair; kwargs...)  
  return ConvBlock([(3, 3), (3, 3)], in_out_pair; kwargs...)
end

"""
  ConvBlock([kernel_sizes,] in_channels, out_channels; activation = NNlib.relu, padding = "valid")

creates a convolution block with the UNet. The optional kernel sizes are given as a vector of tuples,
where each tuple represents the kernel size of a convolution layer. Default is [(3, 3), (3, 3)].
The in_channels and out_channels are the number of input and output channels of the block.
The in_channels and out_channels have to be given as a Pair of integers (in_channels => out_channels).
The activation function is applied after each convolution layer.
"""
function ConvBlock(kernel_sizes, in_out_pair::Pair;
                       activation = NNlib.relu, padding = "valid")
  in_channels, out_channels = in_out_pair
  pad_arg = padding == "same" ? SamePad() : 0
  conv_layers = Any[]
  in_channels_it = in_channels
  for kernel_size in kernel_sizes
    push!(conv_layers,
      Conv(kernel_size, in_channels_it => out_channels, activation; pad=pad_arg)
    )
    in_channels_it = out_channels
  end
  return ConvBlock(Chain(conv_layers...))
end

function (m::ConvBlock)(x)
  return m.op(x)
end
  
  # ConvBlock(in_chs, out_chs, kernel = (3, 3)) =
  #   Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),
	# BatchNormWrap(out_chs),
	# x->leakyrelu.(x,0.2f0))

struct Downsample{T1, T2, T3}
  op::T1
  factor::T2
  pooling_func::T3
end

function Base.show(io::IO, d::Downsample)
  print(io, "Downsample($(d.factor), $(d.pooling_func))")
end

@functor Downsample

function Downsample(downsample_factor; pooling_func = NNlib.maxpool)
    downop = x -> pooling_func(x, downsample_factor, pad=0)
  return Downsample(downop, downsample_factor, pooling_func)
end

function (m::Downsample)(x)
  for (d, x_s, f_s) in zip(1: length(m.factor), size(x), m.factor)
      if (mod(x_s, f_s) !=0)
        throw(DimensionMismatch("Can not downsample $(size(x)) with factor $(m.factor), mismatch in spatial dimension $d"))
      end
  end  
  return m.op(x)
end

struct Upsample{T1, T2}
  op::T1
  factor::T2
end

@functor Upsample

function Upsample(scale_factor, in_out_pair::Pair)
  upop = ConvTranspose(scale_factor, in_out_pair, stride=scale_factor)
  return Upsample(upop, scale_factor)
end

function (m::Upsample)(x)
  return m.op(x)
end

function crop(x, target_size)
  if (size(x) == target_size)
    return x
  else
    offset = Tuple((a-b)÷2+1 for (a,b) in zip(size(x), target_size))
    slice = Tuple(o:o+t-1 for (o,t) in zip(offset,target_size))
    return x[slice...,:,:]
  end
end

function(m::Upsample)(x, y)
  #todo: crop_to_factor
  g_up = m(x)
  k = size(g_up)[1:length(m.factor)]
  f_cropped = crop(y, k)
  new_arr = cat(f_cropped, g_up; dims=length(m.factor)+1)
  return new_arr
end

# holds the information on the unet structure
struct Unet{T1, T2, T3, T4, T5, T6}
  num_levels::T1
  l_conv_chain::T2
  l_down_chain::T3
  r_up_chain::T4
  r_conv_chain::T5
  final_conv::T6
end

@functor Unet

"""
function Unet(;
  in_out_channels_pair = (1 => 1),
  num_fmaps = 64,
  fmap_inc_factor = 2,
  downsample_factors = [(2,2),(2,2),(2,2),(2,2)],
  kernel_sizes_down = [[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]],
  kernel_sizes_up = [[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]],
  activation = NNlib.relu,
  final_activation = NNlib.relu;
  padding="same",
  pooling_func = NNlib.maxpool
  )
    creates a U-net model that can then be used to be trained and to perform predictions. A UNet consists of an initial layer to
    create feature maps, controlled via `num_fmaps`. This is followed by downsampling and umsampling steps,
    which obtain information from the downsampling side of the net via skip-connections, which are automatically inserted.
    The down- and upsampling steps contain on each level a number of consequtive convolutions controlled via the arguments `kernel_sizes_down`
    and `kernel_sizes_up` respectively.

# Paramers
+ `in_out_channels_pair`: channels of the input to the U-net and channels of the output of the U-net as a Pair of integers

+ `num_fmaps`: number of feature maps that the input gets expanded to in the first step

+ `fmap_inc_factor`: the factor that the feature maps get expanded by in every level of the U-net

+ `downsample_factors`: vector of downsampling factors of individual U-net levels

+ `kernel_sizes_down`: vector of vectors of tuples of individual kernel_sizes used in the convolutions on the way down
  e.g. 5 lists of convolutions. 4 before downsampling and one final after the downsample, each with 2 consecutive 3x3 convolutions.

+ `kernel_sizes_up`: vector of vectors of tuples of individual kernel_sizes used in the convolutions on the way up (backwards)
  similar but but after each upsampling step, starting from the top, but not initial one before upsampling.

+ `activation`: activation function after each convolution layer

+ `final_activation`: activation function for the final step

+ `padding="valid"`: method of padding during convolution and upsampling

+ `pooling_func` = NNlib.maxpool

# Example
```jldoctest
```
"""
function Unet(;  # all arguments are named and ahve defaults
  in_out_channels_pair = (1 => 1),
  num_fmaps = 64,
  fmap_inc_factor = 2,
  downsample_factors = [(2,2),(2,2),(2,2),(2,2)],
  kernel_sizes_down = [[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]],
  kernel_sizes_up = [[(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)], [(3,3), (3,3)]],
  activation = NNlib.relu,
  final_activation = NNlib.relu,
  padding ="same",
  pooling_func = NNlib.maxpool
  )
  in_channels, out_channels = in_out_channels_pair
  num_levels = length(downsample_factors) + 1
  dims = length(downsample_factors[1])
  l_convs = Any[]
  for level in 1:num_levels
    in_ch = (level == 1) ? in_channels : num_fmaps * fmap_inc_factor ^ (level - 2)

    cb = ConvBlock(
      kernel_sizes_down[level],
      in_ch => 
      num_fmaps * fmap_inc_factor ^ (level - 1),
      activation=activation,
      padding=padding
      )
    push!(l_convs, cb)
  end

  l_downs = Any[]
  for level in 1:num_levels - 1
    push!(l_downs,
      Downsample(
        downsample_factors[level];
        pooling_func = pooling_func
      )
    )
  end
  
  r_ups = Any[]
  for level in 1:num_levels - 1
    push!(r_ups,
      Upsample(
        downsample_factors[level],
        num_fmaps * fmap_inc_factor ^ level => num_fmaps * fmap_inc_factor ^ level
      )
    )
  end

  r_convs = Any[]
  for level in 1:num_levels - 1
    push!(r_convs,
      ConvBlock(
        kernel_sizes_up[level],
        num_fmaps * fmap_inc_factor ^ (level - 1) +
        num_fmaps * fmap_inc_factor ^ level => num_fmaps * fmap_inc_factor ^ (level -  1),
        activation=activation,
        padding=padding
      )
    )
  end

  final_conv = ConvBlock(
    [ntuple((i) -> 1, dims)],
    num_fmaps => out_channels,
    activation=final_activation,
    padding=padding
  )
  return Unet(num_levels, l_convs, l_downs, r_ups, r_convs, final_conv)
end


function (m::Unet)(x::AbstractArray; level=1)
  f_left = m.l_conv_chain[level](x)
  fs_out = let
      if (level == m.num_levels)
        f_left
      else
        g_in = m.l_down_chain[level](f_left)
        gs_out = m(g_in; level=level+1)
        fs_right = m.r_up_chain[level](gs_out, f_left)
        m.r_conv_chain[level](fs_right)
      end
    end

  if (level == 1)
    return m.final_conv(fs_out)
  else
    return fs_out
  end
end

function Base.show(io::IO, u::Unet)
  ws = size(u.l_conv_chain[1].op[1].weight)
  println(io, "UNet, Input Channels: $(ws[end-1])")
  lvl = ""
  for (c, d) in zip(u.l_conv_chain, u.l_down_chain)
      println(io, "$(lvl)Conv: $c")
      println(io, "$(lvl)| \\")
      println(io, "$(lvl)|  \\DownSample: $d")
      println(io, "$(lvl)|   \\")
      lvl *= "|    "
  end
  println(io, "$(lvl)Conv: $(u.l_conv_chain[end])")
  for (c, d) in zip(u.r_conv_chain[end:-1:1], u.r_up_chain[end:-1:1])
      lvl = lvl[1:end-5]
      println(io, "$(lvl)|   /")
      println(io, "$(lvl)|  /UpSample: $d ")
      println(io, "$(lvl)| /")
      println(io, "$(lvl)Concat")
      println(io, "$(lvl)Conv: $(c)")
  end
  println(io, "FinalConv: $(u.final_conv)")
end
