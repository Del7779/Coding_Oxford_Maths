# --- Optimized Union-Find Data Structure with Component Sizes ---
mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}  # Component sizes: size[i] is meaningful only if parent[i] == i.
end

function UnionFind(n::Int)
    parent = collect(1:n)
    size = ones(Int, n)
    return UnionFind(parent, size)
end

# Find with path compression.
function uf_find(uf::UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]  # Path halving.
        x = uf.parent[x]
    end
    return x
end

# Union by size.
function uf_union!(uf::UnionFind, x::Int, y::Int)
    root_x = uf_find(uf, x)
    root_y = uf_find(uf, y)
    if root_x == root_y
        return
    end
    # Attach the smaller tree to the larger tree.
    if uf.size[root_x] < uf.size[root_y]
        uf.parent[root_x] = root_y
        uf.size[root_y] += uf.size[root_x]
    else
        uf.parent[root_y] = root_x
        uf.size[root_x] += uf.size[root_y]
    end
end