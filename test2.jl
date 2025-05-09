

using Graphs, StatsBase, Distributions, BenchmarkTools
include("Newman_Ziff.jl")
using .Newman_Ziff

# Prepare the od set

function find_node_pairs_within_distance(g::Graph, C::Int)
    pairs = Set{Tuple{Int, Int}}()
    for u in 1:nv(g)
        visited = Dict{Int, Int}()  # node => distance
        queue = [(u, 0)]

        while !isempty(queue)
            (current, dist) = popfirst!(queue)
            if dist >= C
                continue
            end
            for v in neighbors(g, current)
                if !haskey(visited, v)
                    visited[v] = dist + 1
                    push!(queue, (v, dist + 1))
                    if u < v  # avoid duplicates like (2,1) if (1,2) is already added
                        push!(pairs, (u, v))
                    end
                elseif dist + 1 <= visited[v]
                    visited[v] = dist + 1
                    pos = findfirst(x->x[1]==v,queue)
                    queue[pos] = (v, dist+1)
                    if u < v  # avoid duplicates like (2,1) if (1,2) is already added
                        push!(pairs, (u, v))
                    end    
                end
            end
        end
    end

    return collect(pairs)
end



# Geometric distribution
function sample_tau(L::Int, N::Int)
    p = 2L / (N * (N - 1))
    d = Geometric(p)
    return rand(d)  # returns τ ∈ {1, 2, 3, ...}
end


"""
    build_indexed_adjacency(g)

Returns `adj[u] == Vector{(v,eid)}` for each undirected edge (u<->v).
`eid` runs from 1 to `ne(g)` in the same order as `edges(g)`.
"""
function build_indexed_adjacency(g::SimpleGraph)
    n = nv(g)
    adj = [Vector{Tuple{Int,Int}}() for _ in 1:n]
    for (eid, e) in enumerate(edges(g))
        u, v = src(e), dst(e)
        push!(adj[u], (v, eid))
        push!(adj[v], (u, eid))
    end
    return adj
end

"""
    init_sppstate!(st, g)

Resize & clear `st` to be ready for BFS on graph `g`.
"""
struct SPPState
    dist    :: Vector{Int}
    σ       :: Vector{Int}
    parents :: Vector{Vector{Int}}
    queue   :: Vector{Int}
    adj     :: Vector{Vector{Tuple{Int,Int}}}
    removed :: BitVector
    seen    :: Vector{Int}        # NEW
  end
  
  function init_sppstate!(st, g::SimpleGraph)
    n = nv(g)    
    resize!(st.dist, n);    # no fill! here
    resize!(st.σ,    n);
    resize!(st.queue, n+1);
    st.removed .= false
    empty!(st.seen)
    for p in st.parents
      empty!(p)
    end
    for i in 1:n 
        st.dist[i] = typemax(Int)
        st.σ[i] = 0
    end
  end
  
  function sample_shortest_path!(o,d,g,C,st)::Union{Vector{Int},Nothing}
    ### 1) partial reset ###
    for u in st.seen
      st.dist[u]    = typemax(Int)
      st.σ[u]       = 0
      empty!(st.parents[u])
    end
    empty!(st.seen)
  
    ### 2) seed BFS ###
    head = 1; tail = 1
    st.queue[1] = o
    st.dist[o]  = 0
    st.σ[o]     = 1
    push!(st.seen, o)
    target = typemax(Int)
  
    ### 3) BFS up to C ###
    while head ≤ tail
      u = st.queue[head]; head += 1
      if u==d || st.dist[u] ≥ C
        break
      end
      @inbounds for (v,eid) in st.adj[u]
        if st.removed[eid]
          continue
        end
        if st.dist[v] == typemax(Int)
          st.dist[v] = st.dist[u] + 1
          st.σ[v]    = st.σ[u]
          push!(st.parents[v], u)
          push!(st.seen, v)             # track newly visited
          tail += 1
          st.queue[tail] = v
          if v == d
            target = st.dist[v]
          end
        elseif st.dist[v] == st.dist[u] + 1
          st.σ[v] += st.σ[u]
          push!(st.parents[v], u)
        end
      end
    end
  
    ### 4) early-out no path ###
    if st.dist[d] == typemax(Int) || st.dist[d] > C
      return nothing
    end
  
    ### 5) backtrack (unchanged) ###
    L = st.dist[d]
    path = Vector{Int}(undef, L+1)
    idx  = L+1
    cur  = d
    while true
      path[idx] = cur
      idx -= 1
      if cur == o
        break
      end
      ps = st.parents[cur]
      ws = st.σ[ps]
      cur = sample(ps, StatsBase.weights(ws))
    end
  
    return path
  end
# ─────────────────────────────────────────────────────────────────────────────
# 3) High-level run_single_spp_fast
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_single_spp_fast(g, C, OD_pairs)

Optimized SSP run on graph `g` with max cost `C` and initial `OD_pairs`.
Returns `(τs, edge_list_tau)`, where
• `τs[i]` is the i-th sampled demand,
• `edge_list_tau[i]` is the list of edges removed at that demand.
"""
function run_single_spp(
    g::SimpleGraph,
    C::Int,
    OD_pairs::Vector{Tuple{Int,Int}}
)
    N = nv(g)
    # build adj+eid once
    adj = build_indexed_adjacency(g)

    # one‐time state allocation
    st = SPPState(
      Vector{Int}(undef, N),
      Vector{Int}(undef, N),
      [Int[] for _ in 1:N],
      Vector{Int}(undef, N+1),
      adj,
      BitVector(undef,ne(g)),
      Vector{Int64}(),
    )
    init_sppstate!(st, g)

    # local copy of OD set
    od = copy(OD_pairs)
    τs = Int[]                                # sampled taus
    edge_logs = Vector{Vector{Tuple{Int,Int}}}()

    while !isempty(od)
        L = length(od)
        push!(τs, sample_tau(L, N))

        # pick random OD
        i = rand(1:L)
        o, d = od[i]

        # sample one shortest path
        path = sample_shortest_path!(o, d, g, C, st)
        if path === nothing
            # no path ≤C → drop this OD
            od[i] = od[end]; pop!(od)
            push!(edge_logs, Tuple{Int,Int}[])
            continue
        end

        # record & mask each edge
        elist = Vector{Tuple{Int,Int}}(undef, length(path)-1)
        for j in 1:length(path)-1
            u, v = minmax(path[j], path[j+1])
            elist[j] = (u, v)
            # find its eid in adj[u]
            # (we know it must exist, so use `findfirst` once)
            pos = findfirst(x -> x[1] == v, st.adj[u])
            eid = st.adj[u][pos][2] 
            st.removed[eid] = true
        end
        push!(edge_logs, elist)

        # check connectivity: another quick BFS on the masked graph
        if sample_shortest_path!(o, d, g, C, st) === nothing
            od[i] = od[end]; pop!(od)
        end
    end

    return τs, edge_logs
end

function run_single_spp_newman(G::SimpleGraph, C, OD_pairs::Vector{Tuple{Int,Int}};points::Int64=50)
    t, edge_list_tau = run_single_spp(G,C,OD_pairs)
    edge_list = [(x...,) for ee in edge_list_tau for x in ee]
    edge_list = reverse(edge_list)
    windows = round.(Int,collect(range(1, ne(G)-1;length=points)))
    p_values, s_max_trial, chi_trial = run_single_trial(nv(G), edge_list, windows;shuffle=false)
    return p_values, s_max_trial, chi_trial
end






N = 10000
C = 3
G = erdos_renyi(N, Int(4 * N / 2))
adj = build_indexed_adjacency(G)
# prepare state
st = SPPState(
    Vector{Int}(undef, N),
    Vector{Int}(undef, N),
    [Int[] for _ in 1:N],
    Vector{Int}(undef, N+1),
    adj,
    BitVector(undef,(ne(G))),
    Vector{Int}()
  )
init_sppstate!(st, G)

N =10000
G = erdos_renyi(N, Int(4*N/2))
@btime find_node_pairs_within_distance(G,3)

Ns = 10 .^[2,3,4,5,7]
C = 3
times = Float64[]
for n in Ns
    m = Int(n*4/2)
    G = erdos_renyi(n, m)
    OD_pairs = find_node_pairs_within_distance(G, C)
    b = @benchmark run_single_spp_newman($G, $C, $OD_pairs)
    @info "Generating graph with N=$n, minimum time: $(minimum(b).time / 1e9)"
    push!(times, minimum(b).time / 1e9)
end


# Calculate slope of log-log plot
log_N = log.(Ns)
log_t = log.(times)
slope = (length(Ns) * sum(log_N .* log_t) - sum(log_N) * sum(log_t)) / 
        (length(Ns) * sum(log_N.^2) - sum(log_N)^2)

using Plots
# Plot the results
p = plot(Ns, times, 
    label="C=3, Slope = $(round(slope, digits=2))",
    marker=:circle,
    xlabel="Network size (N)",
    ylabel="Time (seconds)",
    title="Time Complexity Analysis",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10)




N =100
G = erdos_renyi(N, Int(4*N/2))
C = [1,2,3,10, 20, 40, 80]
times = Float64[]
for c in C
    od_pairs = find_node_pairs_within_distance(G, c)
    
    b = @benchmark run_single_spp_newman($G, $c, $od_pairs)

    t = minimum(b).time / 1e9  # Convert to seconds
    
    push!(times, t)
end

# Calculate average shortest path length in the graph
path_lengths = []
for u in 1:nv(G)
  for v in u+1:nv(G)
    path = dijkstra_shortest_paths(G, u)
    if path.dists[v] != typemax(Int)
      push!(path_lengths, path.dists[v])
    end
  end
end

avg_path_length = mean(path_lengths)
@info "Average shortest path length: $avg_path_length"

histogram(path_lengths)

log(N)/log(4)





using Plots
p = plot(C, times, 
    label="N=$N",
    marker=:circle,
    xlabel="Cost threshold (C)",
    ylabel="Time (seconds)",
    title="Time Complexity Analysis",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,dpi=600)
# Fit and plot trend line
log_C = log.(C)
log_t = log.(times)
slope = (length(C) * sum(log_C .* log_t) - sum(log_C) * sum(log_t)) / 
        (length(C) * sum(log_C.^2) - sum(log_C)^2)

@info "Empirical complexity: O(N^$(round(slope, digits=2)))"