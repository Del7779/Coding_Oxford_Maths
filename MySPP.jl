include("Newman_Ziff.jl")

module MySPP
using Graphs, StatsBase, Distributions
using ..Newman_Ziff
export find_node_pairs_within_distance, sample_tau, build_indexed_adjacency, SPPState, init_sppstate!, sample_shortest_path!, run_single_spp, run_single_trial_spp_newman


# Prepare the od set
function find_node_pairs_within_distance(g::SimpleGraph, C::Int)
    n = nv(g)
    pairs = Vector{Tuple{Int,Int}}()

    seen  = falses(n)                  # BitVector
    dist  = Vector{Int}(undef, n)
    queue = Vector{Int}(undef, n+1)

    for u in 1:n
        # reset for this source
        fill!(seen, false)
        # note: dist need only be assigned when seen[v] goes true

        head, tail = 1, 1
        queue[1]  = u
        seen[u]   = true
        dist[u]   = 0

        while head ≤ tail
            v = queue[head]; head += 1
            d = dist[v]
            if d == C
                continue
            end
            @inbounds for w in neighbors(g, v)
                if !seen[w]
                    seen[w]      = true
                    dist[w]      = d + 1
                    tail        += 1
                    queue[tail] = w
                    if u < w
                        push!(pairs, (u, w))
                    end
                end
            end
        end
    end

    return pairs
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
    # resize!(st.dist, n);    # no fill! here
    # resize!(st.σ,    n);
    # resize!(st.queue, n+1);
    # resize!(st.removed,ne(g))

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
  
function sample_shortest_path!(o,d,C,st)::Union{Vector{Int},Nothing}
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
    OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState)


    N = nv(g)
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
        path = sample_shortest_path!(o, d, C, st)
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
        if sample_shortest_path!(o, d, C, st) === nothing
            od[i] = od[end]; pop!(od)
        end
    end

    return τs, edge_logs
end

function run_single_trial_spp_newman(G::SimpleGraph, 
    C::Int64, 
    OD_pairs::Vector{Tuple{Int,Int}},
    st::SPPState,
    ;points::Int64=50)
    init_sppstate!(st,G)
    t, edge_list_tau = run_single_spp(G,C,OD_pairs,st)
    edge_list = [(x...,) for ee in edge_list_tau for x in ee]
    edge_list = reverse(edge_list)
    windows = round.(Int,collect(range(1, ne(G)-1;length=points)))
    p_values, s_max_trial, chi_trial = run_single_trial(nv(G), edge_list, windows;shuffle=false)
    return p_values, s_max_trial, chi_trial
end

end



