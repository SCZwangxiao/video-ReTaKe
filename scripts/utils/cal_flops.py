# QWen2-VL

C3 =  3584 # attention dim	dm	3584
C4 =  28 # attention head number	head_num	28
C5 =  18944 # FF hidden layer dim	df	18944
C6 =  28 # Total Layer	L	28
C7 =  152064 # vocabulary size	V	152064
# C8 =  2304 # maximum token number	s	2304
C9 =  1 # Data format	FP16	1
C10 =  128 # attention dim for each head	dk	128
C11 =  7 # attention head number each chip	dhead_num	7
C12 =  896 # attention dim for each chip	dt	896
C13 =  4736 # FF hidden layer dim for each chip	dft	4736

C16 = 1 # batch size	b	1
C17 = 1 # current token number	n	1
# C18 = 3456 # KV Cache Size		3456

C32 = 1000000000

def calculate_flops_prefill(input_length, kvcache_length):
    C8 = input_length
    C18 = kvcache_length
    layernorm = 5*C16*C8*C12/1000/1000/1000
    QKV计算 = 2*3*C16*C8*C12*C3/C32
    RoPE = 6*C16*C8*C12/C32
    # attention计算 = (2*C16*C8*(C8+C18)*C12 + 3*C16*(C8+C18)*C12*C12 + C16*C8*(C8+C18))/C32
    attention计算 = (4*C16*C8*(C8+C18)*C12 - 2*C16*C8*C8*C12 + 3*C16*C8*C12*C12 + 2*C16*C8*(C8+C18) - C16*C8*C8)/C32
    linear计算 = 2*C16*C8*C12*C3/C32
    layernorm = 5*C16*C8*C12/1000/1000/1000
    gate门控 = (2*C16*C8*C3*C13 + 4*C16*C8*C13)/1000/1000/1000
    FF计算1 = 2*C16*C8*C3*C13/1000/1000/1000
    FF计算2 = 2*C16*C8*C3*C13/1000/1000/1000
    return sum([
        layernorm,
        QKV计算,
        RoPE,
        attention计算,
        linear计算,
        layernorm,
        gate门控,
        FF计算1,
        FF计算2
    ])

def cal_pivotkv_flops(num_frames, chunk_size, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0):
    tokens_per_chunk = (448/14/2)*(448/14/2)*chunk_size/2/(1280/720) * visual_compression_ratio
    flops = 0
    kvcache_size = 0
    for idx in range(num_frames // chunk_size):
        flops += calculate_flops_prefill(tokens_per_chunk, kvcache_size)
        kvcache_size += tokens_per_chunk * kvcache_compression_ratio
    return flops

# ax*x + bx
# a2c*2c + b2c = ac*c + bc + ac*2c + b2c
# 4ac^2 + 2bc = ac^2 + bc + 2ac^2 + 2bc

# 9ac^2 + 3bc = ac*c + bc + ac*2c + b2c + ac*3c + b3c
#             = (1+2+3)ac^2 + (1+2+3)bc

# k^2ac^2 + kbc = (k^2 + k)ac^2 + kbc - kac^2

# f(n) = 2ac^2 + bc - ac^2


print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.25))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.8660254037844386, kvcache_compression_ratio=0.28867513459481287))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.7071067811865476, kvcache_compression_ratio=0.3535533905932738))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.5, kvcache_compression_ratio=0.5))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.3535533905932738, kvcache_compression_ratio=0.7071067811865476))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.28867513459481287, kvcache_compression_ratio=0.8660254037844386))
print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=0.25, kvcache_compression_ratio=1.0))

# print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.1))
# print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.3))
# print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.5))
# print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.7))
# print(cal_pivotkv_flops(num_frames=1024, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0))


print(cal_pivotkv_flops(num_frames=256, chunk_size=256, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0))
print(cal_pivotkv_flops(num_frames=256, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0))
print(cal_pivotkv_flops(num_frames=256, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.5))



# LLaVA-Video

C3 =  3584 # attention dim	dm	3584
C4 =  28 # attention head number	head_num	28
C5 =  18944 # FF hidden layer dim	df	18944
C6 =  28 # Total Layer	L	28
C7 =  152064 # vocabulary size	V	152064
# C8 =  2304 # maximum token number	s	2304
C9 =  1 # Data format	FP16	1
C10 =  128 # attention dim for each head	dk	128
C12 =  896 # attention dim for each chip	dt	896
C13 =  4736 # FF hidden layer dim for each chip	dft	4736

C16 = 1 # batch size	b	1
C17 = 1 # current token number	n	1
# C18 = 3456 # KV Cache Size		3456

C32 = 1000000000

def calculate_flops_prefill(input_length, kvcache_length):
    C8 = input_length
    C18 = kvcache_length
    layernorm = 5*C16*C8*C12/1000/1000/1000
    QKV计算 = 2*3*C16*C8*C12*C3/C32
    RoPE = 6*C16*C8*C12/C32
    # attention计算 = (2*C16*C8*(C8+C18)*C12 + 3*C16*(C8+C18)*C12*C12 + C16*C8*(C8+C18))/C32
    attention计算 = (4*C16*C8*(C8+C18)*C12 - 2*C16*C8*C8*C12 + 3*C16*C8*C12*C12 + 2*C16*C8*(C8+C18) - C16*C8*C8)/C32
    linear计算 = 2*C16*C8*C12*C3/C32
    layernorm = 5*C16*C8*C12/1000/1000/1000
    gate门控 = (2*C16*C8*C3*C13 + 4*C16*C8*C13)/1000/1000/1000
    FF计算1 = 2*C16*C8*C3*C13/1000/1000/1000
    FF计算2 = 2*C16*C8*C3*C13/1000/1000/1000
    return sum([
        layernorm,
        QKV计算,
        RoPE,
        attention计算,
        linear计算,
        layernorm,
        gate门控,
        FF计算1,
        FF计算2
    ])

def cal_pivotkv_flops(num_frames, chunk_size, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0):
    tokens_per_chunk = (384/14/2)**2*chunk_size * visual_compression_ratio
    flops = 0
    kvcache_size = 0
    for idx in range(num_frames // chunk_size):
        flops += calculate_flops_prefill(tokens_per_chunk, kvcache_size)
        kvcache_size += tokens_per_chunk * kvcache_compression_ratio
    return flops

# ax*x + bx
# a2c*2c + b2c = ac*c + bc + ac*2c + b2c
# 4ac^2 + 2bc = ac^2 + bc + 2ac^2 + 2bc

# 9ac^2 + 3bc = ac*c + bc + ac*2c + b2c + ac*3c + b3c
#             = (1+2+3)ac^2 + (1+2+3)bc

# k^2ac^2 + kbc = (k^2 + k)ac^2 + kbc - kac^2

# f(n) = 2ac^2 + bc - ac^2

print(cal_pivotkv_flops(num_frames=256, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=1.0))
print(cal_pivotkv_flops(num_frames=256, chunk_size=32, visual_compression_ratio=1.0, kvcache_compression_ratio=0.5))
