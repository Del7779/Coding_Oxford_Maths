using Graphs, Random
using CategoricalArrays
using StatsBase,LsqFit
using Plots
using Graphs, GraphMakie, CairoMakie



# Function to generate an initial scale-free network using preferential attachment
function generate_scale_free_network_optimized(n::Int, m::Int)
    g = SimpleGraph(n)
    add_edge!(g, 1, 2)  # Start with an initial edge
    degrees = [1, 1]  # Track degrees
    node_list = [1, 2]  # List for preferential selection
    
    for new_node in 3:n
        targets = Set{Int}()
        
        while length(targets) < m
            selected_node = rand(node_list)  # Fast sampling from weighted list
            push!(targets, selected_node)
        end
        
        for target in targets
            add_edge!(g, new_node, target)
            degrees[target] += 1
            push!(node_list, target)  # Update selection list
        end
        
        push!(degrees, m)
        append!(node_list, fill(new_node, m))  # Maintain weighted list for future selections
    end
    
    return g
end


g = generate_scale_free_network_optimized(100, 2)
random_link_swapping!(g, 100)
d = degree(g)
plot1 = Figure();
ax, plt = graphplot(plot1[1,1], g, layout=Spring())
display(plot1)

dict_count = countmap(d)
size = collect(keys(dict_count))
prob = collect(values(dict_count))/sum(collect(values(dict_count)))

# histogram(d, bins = 10 .^(range(log10(minimum(d)),log10(maximum(d));length=10)), label="Degree Distribution",xlabel="Degree",ylabel="Frequency",legend=:topleft)
bin_edges = 10 .^(range(log10(minimum(d)),log10(maximum(d));length=20))

hist = fit(Histogram,d, bin_edges)
edge_ = hist.edges[1]
size = 0.5*(edge_[1:end-1]+edge_[2:end])
binwidth = diff(edge_)
prob = hist.weights./sum(hist.weights)./binwidth
index = findall(x -> x != 0.0, prob)


f(x,p) = p[1] .+ x.* p[2]
curve_fitted = curve_fit(f, log10.(size[index]), log10.(prob[index]), [1.0, -3.0])
h2 = scatter(size[index],prob[index],xscale=:log10,yscale=:log10,label="Degree Distribution",xlabel="Degree",ylabel="Frequency",legend=:outerright)
curve_fitted.param[2]
f2(x) = 10^(curve_fitted.param[1]) .*x.^ curve_fitted.param[2]
Plots.plot!(h2,1:maximum(size),f2,linestyle=:dash,linewidth=2,label="Power Law Fit Slope=$(round.(curve_fitted.param[2],digits=3)) ")


graphplot(g, nodelabel=1:100, nodeshape=:circle, nodefillc=:blue, nodesize=0.1, linecolor=:gray, linewidth=0.1, linealpha=0.5, curves=false, arrow=false, names=false, fontsize=5)



