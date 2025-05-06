using Graphs, Random, Statistics, GraphPlot

# Function to simulate influence spread using the Independent Cascade (IC) model.
function simulate_IC(g::SimpleGraph, seeds::Vector{Int}; p=0.1)
    activated = Set(seeds)
    newly_activated = Set(seeds)
    while !isempty(newly_activated)
        next_activated = Set{Int}()
        for u in newly_activated
            for v in neighbors(g, u)
                # Only try to activate nodes that are not already activated.
                if !(v in activated)
                    # With probability p, v becomes activated.
                    if rand() < p
                        push!(next_activated, v)
                    end
                end
            end
        end
        activated = union(activated, next_activated)
        newly_activated = next_activated
    end
    return length(activated)
end

# Function to estimate the average spread of influence given a set of seed nodes.
function estimate_spread(g::SimpleGraph, seeds::Vector{Int}; p=0.1, num_simulations=100)
    spreads = [simulate_IC(g, seeds; p=p) for _ in 1:num_simulations]
    return mean(spreads)
end

# Greedy algorithm for influence maximization: select k seeds.
function greedy_influence(g::SimpleGraph, k::Int; p=0.1, num_simulations=100)
    n = nv(g)
    seeds = Int[]
    candidate_nodes = collect(1:n)
    for i in 1:k
        best_node = nothing
        best_spread = -Inf
        # Evaluate the marginal gain for each candidate node.
        for v in candidate_nodes
            candidate_seeds = union(seeds, [v])
            spread = estimate_spread(g, candidate_seeds; p=p, num_simulations=num_simulations)
            if spread > best_spread
                best_spread = spread
                best_node = v
            end
        end
        push!(seeds, best_node)
        deleteat!(candidate_nodes, findfirst(x -> x == best_node, candidate_nodes))
        println("Selected seed: ", best_node, " with estimated spread: ", best_spread)
    end
    return seeds
end

# Example usage:
n = 100             # Number of nodes in the graph
p_edge = 0.05        # Probability for edge creation (Erdős–Rényi graph)
g = erdos_renyi(n, p_edge)
k = 5                # Number of influencers to select

# Run the greedy influencer selection with an activation probability of 0.1.
selected_seeds = greedy_influence(g, k; p=0.1, num_simulations=100)
println("Greedy selected seeds: ", selected_seeds)

# Create a vector of colors where selected nodes are red and others are lightblue
nodefillc = [v in selected_seeds ? "red" : "lightblue" for v in 1:nv(g)]

# Plot the graph with highlighted seed nodes
gplot(g, nodefillc=nodefillc)