using Random, Statistics, Plots
using BenchmarkTools
using Plots
using LsqFit
using SpecialFunctions
using Latexify, LaTeXStrings
include("Union_Find.jl")
println(L"$(\omega = 1)$")

# Check percolation: here we check if there is a spanning cluster connecting
# the top row (nodes 1:L) to the bottom row (nodes (L^2-L+1):L^2).
function check_percolation(uf::UnionFind, L::Int)
    top_nodes = 1:L                   # First row nodes
    bottom_nodes = (L^2 - L + 1):L^2     # Last row nodes
    top_clusters = Set(uf_find(uf, node) for node in top_nodes)
    for node in bottom_nodes
        if uf_find(uf, node) in top_clusters
            return true
        end
    end
    return false
end

"""
    newman_ziff_bond_percolation(L, realizations; windows)

Perform a Newman–Ziff bond percolation simulation on an L×L lattice.
For each realization, bonds are added in a random order and at specified 
windows (i.e. numbers of bonds added), the following are computed:
  - GCC: the fraction of sites in the largest cluster.
  - Susceptibility: sum(s^2) of cluster sizes excluding the largest cluster, normalized by N.
  - Percolation probability: whether the lattice percolates (top to bottom).

The results are averaged over all realizations.
`windows` should be a vector of bond counts at which measurements are taken.
If not provided, measurements will be taken at every bond.
Returns (avg_gcc, avg_susceptibility, avg_perc, p_values) where p_values = windows/num_bonds.
"""
function newman_ziff_bond_percolation(L::Int, realizations::Int; windows::Union{Vector{Int}, Nothing}=nothing)
    N = L * L  # Number of lattice sites
    edges = Tuple{Int, Int}[]
    # Generate all possible bonds in the lattice.
    for x in 1:L, y in 1:L
        site = (x - 1) * L + (y - 1)
        if x < L
            push!(edges, (site + 1, site + L + 1))
        end
        if y < L
            push!(edges, (site + 1, site + 2))
        end
    end
    num_bonds = length(edges)
    if isnothing(windows)
        windows = collect(1:num_bonds)
    end

    # Initialize accumulators for GCC, susceptibility, and percolation probability.
    avg_gcc = zeros(Float64, length(windows))
    avg_susceptibility = zeros(Float64, length(windows))
    avg_perc = zeros(Float64, length(windows))
    
    for _ in 1:realizations
        uf = UnionFind(N)
        shuffle!(edges)  # Randomize bond order
        gcc = zeros(Float64, length(windows))
        susceptibility = zeros(Float64, length(windows))
        perc_curve = zeros(Float64, length(windows))
        index_window = 1

        for (i, (a, b)) in enumerate(edges)
            uf_union(uf, a, b)
            if i in windows
                # Compute cluster sizes
                all_cluster_sizes = Dict{Int, Int}()
                for node in 1:N
                    root = uf_find(uf, node)
                    all_cluster_sizes[root] = get(all_cluster_sizes, root, 0) + 1
                end
                max_cluster_size = maximum(values(all_cluster_sizes))
                gcc[index_window] = max_cluster_size / N
                # Exclude the largest cluster for susceptibility calculation.
                reduced_cluster_sizes = filter(s -> s != max_cluster_size, collect(values(all_cluster_sizes)))
                susceptibility[index_window] = sum(map(s -> s^2, reduced_cluster_sizes)) / N
                # Record percolation (1 if percolated, 0 otherwise)
                perc_curve[index_window] = check_percolation(uf, L) ? 1.0 : 0.0
                index_window += 1
            end 
            if i == windows[end]
                break
            end
        end
        avg_gcc .+= gcc / realizations
        avg_susceptibility .+= susceptibility / realizations
        avg_perc .+= perc_curve / realizations
    end
    p_values = windows / num_bonds
    return avg_gcc, avg_susceptibility, avg_perc, p_values
end

## Single simulation
# L = 200
# realizations = 500
# n_data = 300
# # Fit the probability curve with an error function
# function erf_model(x, p)
#     # Model: A * [1 + erf((x-μ)/(σ√2))] + B
#     return p[1] * (1 .+ erf.((x .- p[2]) ./ (p[3]*sqrt(2)))) .+ p[4]
# end
# max_bonds = 2*L*(L-1)
# windows_data_pc = round.(Int,range(0.47*max_bonds, stop=0.53*max_bonds, length=n_data))
# windows_data_away_pc = round.(Int,range(0.4*max_bonds, stop=0.6*max_bonds, length=100))
# windows_data = sort(unique([windows_data_pc; windows_data_away_pc]))
# avg_gcc, avg_susceptibility, perc_prob, p_values = newman_ziff_bond_percolation(L, realizations; windows=windows_data);

# # Fit the percolation probability curve with an error function
# curve_fitted = curve_fit(erf_model, p_values, perc_prob, [0.5, 0.5, 0.1, 0.0]);
# fit_perc_prob = erf_model(p_values, curve_fitted.param);


# # Plot first simulation
# p1 = plot(p_values, avg_gcc, 
#     label="L= $L (sparse sampling)", series=:scatter,
#     xlabel="p", 
#     ylabel="Fraction of largest cluster");

# # Plot susceptibility
# p2 = scatter(p_values, perc_prob, series= :scatter,color=:white,strokewidth=1,
#     label="L = $L (sparse sampling)", markersize=2,
#     xlabel="p",
#     ylabel="Susceptibility");
# plot!(p2, p_values, fit_perc_prob, label="Fit")
# vline!(p2, [curve_fitted.param[2]], label="Estimated pc", linestyle=:dash)
# plot(p1, p2, layout=(2,1),size=(1200,800),legend=:outertopright)

# Example usage
L = [256, 512, 1024, 2048]  # Lattice size
realizations = 1000  # Number of realizations
simulation_data = Dict{Int,Tuple}()

hf_order = plot(legend=:outerright,title="Fraction of largest cluster vs p",xlabel="p",ylabel="Fraction of largest cluster",figsize=(1200,800));
hf_prob = plot(legend=:outerright, title="Percolation probability vs p", xlabel="p", ylabel="Percolation probability", figsize=(1200,800));
n_data = 300;
function erf_model(x, p)
    # Model: A * [1 + erf((x-μ)/(σ√2))] + B
    return p[1] * (1 .+ erf.((x .- p[2]) ./ (p[3]*sqrt(2)))) .+ p[4]
end

for l in L
    t_start = time()
    max_bonds = 2*l*(l-1)
    windows_data_pc = round.(Int,range(0.47*max_bonds, stop=0.53*max_bonds, length=n_data))
    windows_data_away_pc = round.(Int,range(0.4*max_bonds, stop=0.6*max_bonds, length=100))
    windows_data = sort(unique([windows_data_pc; windows_data_away_pc]))

    # simulate and store the data
    avg_gcc, avg_susceptibility, perc_prob, p_values = newman_ziff_bond_percolation(l, realizations; windows=windows_data)
    t_end = time()
    elasp_time = t_end - t_start
    simulation_data[l] = (avg_gcc, avg_susceptibility, perc_prob, p_values, elasp_time)

    # Fit with error function
    curve_fitted = curve_fit(erf_model, p_values, perc_prob, [0.5, 0.5, 0.1, 0.0])
    est_pc = curve_fitted.param[2]
    
    # Plots
    plot!(hf_order , p_values, avg_gcc, label="$l")
    vline!(hf_order, [est_pc], label="Estimated pc = $est_pc for L = $l", linestyle=:dash)
    scatter!(hf_prob, p_values, perc_prob, label="$l", markersize=2,color=:white,strokewidth=1)
    fit_perc_prob = erf_model(p_values, curve_fitted.param)
    plot!(hf_prob, p_values, fit_perc_prob, label="Fit for L = $l")
    vline!(hf_prob, [est_pc], label="Estimated pc = $est_pc for L = $l", linestyle=:dash)
    println("L = $l: Estimated pc = $est_pc, Time = $elasp_time")
    println("L = $l: Average GCC = $(mean(avg_gcc)), Average Susceptibility = $(mean(avg_susceptibility))")
end

display(plot(hf_order,hf_prob, layout=(2,1), legend=:topleft,size=(1200,800)))


# Plot
hf_order = plot(legend=:outerright,title=L"Fraction of largest cluster vs $p$",xlabel=L"p",ylabel=L"S_L",dpi=300);
hf_prob = plot(legend=:outerright, title=L"Percolation probability vs $p$", xlabel=L"p", ylabel=L"R_L",dpi=300);
hf_finite_size = plot(legend=:outerright, title=L"Finite size scaling of $p_c$", xlabel=L"L^{-4/3}", ylabel=L"\p_{c}^{L}",dpi=300); 
# Detailed Plot
colors = [:blue, :red, :green, :purple]  # Define colors for each L
for (i, l) in enumerate(L)
    avg_gcc, avg_susceptibility, perc_prob, p_values = simulation_data[l]
    curve_fitted = curve_fit(erf_model, p_values, perc_prob, [0.5, 0.5, 0.1, 0.0])
    est_pc = curve_fitted.param[2]
    scatter!(hf_finite_size,[l].^(-4/3),[curve_fitted.param[2]], label="L = $l", markersize=5, color=colors[i],
            xscale=:log10, yscale=:log10)
    hline!(hf_finite_size,[0.5],linestyle=:dash, label="", color=:black)
    plot!(hf_order, p_values, avg_gcc, label=L"L = %$l", color=colors[i])
    vline!(hf_order, [est_pc], label="", color=colors[i], linestyle=:dash)
    
    scatter!(hf_prob, p_values, perc_prob, label=L"L = %$l", markersize=2, 
            color=colors[i], markerstrokecolor=colors[i])
    fit_perc_prob = erf_model(p_values, curve_fitted.param)
    plot!(hf_prob, p_values, fit_perc_prob, label=L"Error Function Fit", color=colors[i])
    vline!(hf_prob, [est_pc], label="", color=colors[i], linestyle=:dash)
end

ylims!(hf_finite_size, (10^(-0.3012), 10^(-0.3008)))
display(hf_finite_size)
display(hf_order)  
dipslay(hf_prob)

h_all=plot(hf_order,hf_prob,hf_finite_size, layout=(3,1), legend=:topleft,size=(1400,1200),dpi=300)

# Or you can save as PNG
savefig(h_all, "Figure/bond_percolation_results.png")  # For PNG format

# Save simulation data to a file
using JLD2

# Create data dictionary to save
save_data = Dict(
    "L" => L,
    "realizations" => realizations,
    "simulation_data" => simulation_data
)


@save "Data/bond_percolation_data.jld2" save_data

# Load data
@load "Data/bond_percolation_data.jld2" save_data
# Plot
hf_order = plot(title=L"Fraction of largest cluster vs $p$",xlabel=L"p",ylabel=L"S_L",dpi=600,size=(1000,800),margin=10Plots.mm);
hf_prob = plot(title=L"Percolation probability vs $p$", xlabel=L"p", ylabel=L"R_L",dpi=600,size=(1000,800),margin=10Plots.mm);
hf_finite_size = plot(title=L"Finite size scaling of $p_c$", xlabel=L"L^{-4/3}", ylabel=L"\p_{c}^{L}",dpi=600,size=(1000,800),margin=10Plots.mm); 
function erf_model(x, p)
    # Model: A * [1 + erf((x-μ)/(σ√2))] + B
    return p[1] * (1 .+ erf.((x .- p[2]) ./ (p[3]*sqrt(2)))) .+ p[4]
end
# Detailed Plot
colors = [:blue, :red, :green, :purple]
L = save_data["L"]
simulation_data = save_data["simulation_data"]
# Define colors for each L
for (i, l) in enumerate(L)
    avg_gcc, avg_susceptibility, perc_prob, p_values = simulation_data[l]
    curve_fitted = curve_fit(erf_model, p_values, perc_prob, [0.5, 0.5, 0.1, 0.0])
    est_pc = curve_fitted.param[2]
    scatter!(hf_finite_size,[l].^(-4/3),[curve_fitted.param[2]], label="L = $l", markersize=5, color=colors[i],
            xscale=:log10, yscale=:log10)
    hline!(hf_finite_size,[0.5],linestyle=:dash, label="", color=:black)
    plot!(hf_order, p_values, avg_gcc, label=L"L = %$l", color=colors[i])
    vline!(hf_order, [est_pc], label="", color=colors[i], linestyle=:dash)
    
    scatter!(hf_prob, p_values, perc_prob, label=L"L = %$l", markersize=2, 
            color=colors[i], markerstrokecolor=colors[i])
    fit_perc_prob = erf_model(p_values, curve_fitted.param)
    plot!(hf_prob, p_values, fit_perc_prob, label=L"Error Function Fit", color=colors[i])
    vline!(hf_prob, [est_pc], label="", color=colors[i], linestyle=:dash)
end

ylims!(hf_finite_size, (10^(-0.3012), 10^(-0.3008)))
display(hf_finite_size)
display(hf_order)  
display(hf_prob)
mkdir("Figure/bond_percolation")
savefig(hf_order, "Figure/bond_percolation/bond_percolation_order.png")  # For PNG format with high DPI
savefig(hf_prob, "Figure/bond_percolation/bond_percolation_prob.png")  # For PNG format
savefig(hf_finite_size, "Figure/bond_percolation/bond_percolation_finitesize.png")  # For PNG format

y_data = Vector{Float64}()
x_data = Float64.(copy(L))
for (i,j) in enumerate(L)
    avg_gcc, avg_susceptibility, perc_prob, p_values = simulation_data[j]
    push!(y_data,avg_gcc[argmin(abs.(p_values.-1/2))])
end

f(x,β) = β[1].*x.^(β[2])
curve_fitted = curve_fit(f,x_data,y_data,[1,-4/3])
est_exp = curve_fitted.param[2]
scatter(x_data,y_data,xscale=:log10,yscale=:log10,markersize = 5)
plot!(x_data,f(x_data,curve_fitted.param),linestyle=:dash,xlabel = "L", ylabel = L"S_c",label = "Exponent $(round(curve_fitted.param[2],digits = 3))")
savefig("Figure/bond_percolation/Sc(L).png")



# S ~ (p-p_c)^A
p_vals = simulation_data[L[end]][4]
index_c = findall(x -> x > 1/2 && x < 0.51, p_vals)
x_data = p_vals[index_c].-1/2
y_data = simulation_data[L[end]][1][index_c]

curve_fitted2 = curve_fit(f,x_data,y_data,[0.1,-5/36])
curve_fitted2.param

scatter(x_data,y_data,xscale=:log10,yscale=:log10)
plot!(x_data,f(x_data,curve_fitted2.param))