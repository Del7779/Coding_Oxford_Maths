# Union-Find data structure for percolation.
mutable struct UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
end

# Constructor: initialize each node as its own parent; ranks start at zero.
function UnionFind(n::Int)
    parent = collect(1:n)    # Julia uses 1-indexing.
    rank = zeros(Int, n)
    return UnionFind(parent, rank)
end

# Find with path compression.
function uf_find(uf::UnionFind, x::Int)
    if uf.parent[x] != x
        uf.parent[x] = uf_find(uf, uf.parent[x])
    end
    return uf.parent[x]
end

# Union by rank.
function uf_union(uf::UnionFind, x::Int, y::Int)
    root_x = uf_find(uf, x)
    root_y = uf_find(uf, y)
    if root_x != root_y
        if uf.rank[root_x] > uf.rank[root_y]
            uf.parent[root_y] = root_x
        elseif uf.rank[root_x] < uf.rank[root_y]
            uf.parent[root_x] = root_y
        else
            uf.parent[root_y] = root_x
            uf.rank[root_x] += 1
        end
    end
end