using LinearAlgebra
import Base.@propagate_inbounds

struct PhaseVector{T,S} <: AbstractArray{T,1}
    vecs::S ## 2 by n Array
end
PhaseVector(A::AbstractArray{T}) where {T} = PhaseVector{T,typeof(A)}(A)

Base.zero(A::PhaseVector{T}) where {T} = PhaseVector{T}(zero(A.vecs))
Base.copy(A::PhaseVector{T}) where {T} = PhaseVector{T}(copy(A.vecs))
Base.copyto!(A::PhaseVector{T}, B::PhaseVector{T}) where {T} = copyto!(A.vecs, B.vecs)
Base.size(A::PhaseVector) = (size(A.vecs, 2), )
Base.length(A::PhaseVector) = size(A.vecs, 2)
Base.similar(A::PhaseVector{T,S}) where {T,S} = PhaseVector{T,S}(similar(A.vecs)) 
@propagate_inbounds Base.getindex(A::PhaseVector, i) = getindex(A.vecs, :, i)
@propagate_inbounds Base.setindex!(A::PhaseVector, a, i) = setindex!(A.vecs, a, :, i)

LinearAlgebra.rmul!(A::PhaseVector, β::Number) = LinearAlgebra.rmul!(A.vecs, β)
LinearAlgebra.norm(A::PhaseVector) = LinearAlgebra.norm(A.vecs)
LinearAlgebra.dot(A::PhaseVector, B::PhaseVector) = LinearAlgebra.dot(A.vecs, B.vecs)
LinearAlgebra.axpy!(α, A::PhaseVector{T}, B::PhaseVector{T}) where {T} = LinearAlgebra.axpy!(α, A.vecs, B.vecs)
