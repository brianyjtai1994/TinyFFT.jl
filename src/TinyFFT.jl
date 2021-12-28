module TinyFFT

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

export fft, fft!

function clp2(x::Int)
    x == 0 && error("_log2(x): x cannot be zero!")
    r = 0
    x > 4294967295 && (x >>= 32; r += 32)
    x >      65535 && (x >>= 16; r += 16)
    x >        255 && (x >>=  8; r +=  8)
    x >         15 && (x >>=  4; r +=  4)
    x >          3 && (x >>=  2; r +=  2)
    x >          1 && (x >>=  1; r +=  1)
    return r
end

function swap!(m::MatI, i1::Int, j1::Int, i2::Int, j2::Int) # @code_warntype ✓
    @inbounds temp     = m[i1,j1]
    @inbounds m[i1,j1] = m[i2,j2]
    @inbounds m[i2,j2] = temp
    return nothing
end

function apy2(x::Real, y::Real)
    isnan(x) && return x
    isnan(y) && return y
    # general case
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    iszero(z) && return w
    return w * sqrt(1.0 + abs2(z / w))
end

function twiddle(sz::Int)
    ws = Matrix{Float64}(undef, sz, 2)
    hz = sz >> 1
    ix = 1
    rc = 1.0
    rs = 0.0
    cf = cospi(inv(sz))
    sf = sinpi(inv(sz))
    #### initialize: θ = 0
    @inbounds ws[ix,1] = rc
    @inbounds ws[ix,2] = rs
    while ix < hz
        ix += 1
        tc  = cf * rc - sf * rs
        ts  = sf * rc + cf * rs
        @inbounds ws[ix,1] = rc = tc
        @inbounds ws[ix,2] = rs = ts
    end
    #### specific case: θ = π / 2
    ix += 1
    @inbounds ws[ix,1] = 0.0
    @inbounds ws[ix,2] = 1.0
    #### remaining cases: θ > π / 2
    jx = ix
    while ix < sz
        ix += 1
        jx -= 1
        @inbounds ws[ix,1] = -ws[jx,1]
        @inbounds ws[ix,2] =  ws[jx,2]
    end
    return ws
end

function butterfly!(y::MatO, x::MatI, w::MatI, jx::Int, kx::Int, H::Int, h::Int, S::Int, s::Int)
    wx = 1
    for _ in 1:h
        jy = jx + s

        @inbounds begin
            wr = w[wx,1]
            wi = w[wx,2]
            xr = x[kx,1]
            xi = x[kx,2]
            Xr = x[kx+H,1]
            Xi = x[kx+H,2]
    
            y[jx,1] = xr + Xr
            y[jx,2] = xi + Xi
            y[jy,1] = wr * (xr - Xr) + wi * (xi - Xi)
            y[jy,2] = wr * (xi - Xi) - wi * (xr - Xr)
        end

        jx += S
        kx += s
        wx += s
    end
end

# a := signal, b:= buffer
function difnn!(a::MatIO, b::MatIO, w::MatI, H::Int)
    h = H
    s = 1
    S = 2
    r = false

    while h > 0
        for ix in 1:s
            if r
                butterfly!(a, b, w, ix, ix, H, h, S, s)
            else
                butterfly!(b, a, w, ix, ix, H, h, S, s)
            end
        end

        h >>= 1
        s <<= 1
        S <<= 1
        r = !r
    end
    return nothing
end

function fftshift!(x::MatI)
    N = size(x, 1) >> 1
    for j in axes(x, 2), i in eachindex(1:N)
        swap!(x, i, j, i+N, j)
    end
end

function fftfreq!(f::VecI, Δf::Real)
    nhalfp1 = length(f) >> 1 + 1
    @simd for i in eachindex(f)
        @inbounds f[i] = Δf * (i - nhalfp1)
    end
    return nothing
end

struct FFT
    cxy::Matrix{Float64}
    crθ::Matrix{Float64}
    fac::Matrix{Float64}
    frq::Vector{Float64}

    function FFT(N::Int)
        cxy = Matrix{Float64}(undef, N, 2)
        crθ = Matrix{Float64}(undef, N, 2)
        frq = Vector{Float64}(undef, N)
        return new(cxy, crθ, twiddle(N >> 1), frq)
    end
end

function fft!(o::FFT, sig::VecI, timestep::Real; unwrap::Bool=false)
    cxy = o.cxy
    crθ = o.crθ
    toN = eachindex(sig)

    N = length(toN)
    H = N >> 1

    if isone(clp2(N) & 1)
        @inbounds for i in toN
            crθ[i,1] = sig[i]
            crθ[i,2] = 0.0
        end
        difnn!(crθ, cxy, o.fac, H)
    else
        @inbounds for i in toN
            cxy[i,1] = sig[i]
            cxy[i,2] = 0.0
        end
        difnn!(cxy, crθ, o.fac, H)
    end

    # circular shift for FFT
    fftshift!(cxy)

    if unwrap
        #### Phases unwrapping
        halfπ = 0.5 * π
        @inbounds crθ[1,2] = prevθ = atan(cxy[1,2], cxy[1,1])
        @inbounds crθ[1,1] = 0.0
        for i in 2:N
            @inbounds crθ[i,2] = thisθ = atan(cxy[i,2], cxy[i,1])
            diff = thisθ - prevθ
            θmod = rem2pi(diff + halfπ, RoundDown) - halfπ
            diff > 0.0 && θmod == -halfπ && (θmod = halfπ)
            @inbounds crθ[i,1] = abs(diff) < halfπ ? crθ[i-1,1] : θmod - diff + crθ[i-1,1]
            prevθ = thisθ
        end
        @simd for i in toN
            @inbounds crθ[i,2] += crθ[i,1]
        end
        @inbounds refθ = crθ[H+1,2]
        @simd for i in toN
            @inbounds crθ[i,2] -= refθ
        end
    end

    #### Compute amplitudes
    @inbounds for i in toN
        crθ[i,1] = apy2(cxy[i,1], cxy[i,2]) / H
    end

    #### Generate FFT frequency coordinate, frq[H+1] = 0
    fftfreq!(o.frq, inv(timestep * N))
    return o
end

fft(sig::VecI, timestep::Real; unwrap::Bool=false) = fft!(FFT(length(sig)), sig, timestep; unwrap)

end # module
