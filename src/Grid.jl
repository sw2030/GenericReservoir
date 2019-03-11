import Base.@propagate_inbounds

struct Grid{T,N,P,S<:AbstractArray}<:AbstractArray{T,N}
    A::S
end

Base.size(g::Grid{T,3,7}) where {T} = (size(g.A, 1)-2, size(g.A, 2)-2, size(g.A, 3)-2)
@propagate_inbounds Base.getindex(g::Grid{T,3,7}, i, j, k) where {T}       = g.A[i+1,j+1,k+1]

function makegrid(x::Array{T,3},P) where {T}
    r = Grid{T,3,P,Array{T,3}}(zeros(eltype(x),size(x,1)+2,size(x,2)+2,size(x,3)+2))
    r.A[2:end-1, 2:end-1, 2:end-1] = x
    r
end