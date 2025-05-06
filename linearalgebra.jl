using LinearAlgebra
using Random
using Graphs
using SparseArrays

G_ER = erdos_renyi(100000,400)
adj = adjacency_matrix(G_ER)
e = ones(size(adj,1))
D = Diagonal(adj*e)
P_ref = D \ adj
C = copy(adj)
T =1 
W = P_ref.* exp.(-C./T)
Z = (I - W) \ I
cholesky(Q)
Zh = Z * (1 ./Diagonal(Z))


n = 100_000
k = 4
m = Int(n * k / 2)

G = erdos_renyi(n, m)
A = adjacency_matrix(G)             # sparse n×n matrix
e = ones(n)  
d = A * e                           # out-degree vector
D = spdiagm(0 => d)                 # sparse diagonal matrix

invD = spdiagm(0 => 1.0 ./ d)       # D⁻¹
P_ref = invD * A                    # reference transition matrix

θ = 1.0
C = A .!= 0.0                       # for now, just use unit cost on edges
C = θ * sparse(C)                  # multiply by θ
W = P_ref .* exp.(-C)              # elementwise exponential and multiply

