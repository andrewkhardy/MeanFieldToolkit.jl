module MFTDecompose
    export InterQuarticToHopping, InterQuarticToPairing, IntraQuarticToHopping, IntraQuarticToPairing, InterBondQuarticToHopping, InterBondQuarticToPairing
    
    using TightBindingToolkit, LinearAlgebra, Tullio


@doc """
```julia
InterQuarticToHopping(Uij::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose a general density-type, inter-site quartic interaction into MFT hopping channels (both inter-site and on-site).
The interaction is provided as a rank-4 array living on the bond.
The expectation values as a dictionary of matrices containing the expectation values on the two sites, i and j, and the bond i ↔ j with keys "ii", "jj", and "ij".

"""
    function InterQuarticToHopping(Uij::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}
        
        Chi_ij  =   Chis["ij"]
        Chi_ii  =   Chis["ii"] 
        Chi_jj  =   Chis["jj"]

        @tullio t_ij[ a , b]    :=   - conj(Chi_ij[c, d]) * Uij[a, c, d, b]
        @tullio t_ii[ a , b]    :=   Chi_jj[c, d] * Uij[a, b, c, d]
        @tullio t_jj[ a , b]    :=   Chi_ii[c, d] * Uij[c, d, a, b]

        return Dict("ij" => t_ij, 
                    "ii" => t_ii,
                    "jj" => t_jj)
    end


@doc """
```julia
InterQuarticToPairing(Uij::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose a general density-type, inter-site quartic interaction into MFT pairing channels.
The interaction is provided as a rank-4 array living on the bond.
The expectation values as a dictionary of matrices containing the expectation values on the two sites, i and j, and the bond i ↔ j with keys "ii", "jj", and "ij".

"""
    function InterQuarticToPairing(Uij::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}
        
        Delta_ij    =  Deltas["ij"]

        @tullio p_ij[ a , b]    :=    conj(Delta_ij[c, d]) * Uij[a, c, b, d]

        return Dict("ij" => p_ij)
    end


@doc """
```julia
IntraQuarticToHopping(Uii::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose a general density-type, intra-site quartic interaction into MFT hopping channels.
The interaction is provided as a rank-4 array living on the site.
The expectation values as a dictionary of matrices containing the expectation values on the site, i with key "ii".

"""
    function IntraQuarticToHopping(Uii::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Chi_ii =    Chis["ii"] 

        @tullio t_ii[ a , b]    :=   Chi_ii[c, d] * (Uii[a, b, c, d] + Uii[c, d, a, b] - Uii[c, b, a, d] - Uii[a, d, c, b])

        return Dict("ii" => t_ii)
    end


@doc """
```julia
IntraQuarticToHopping(Uii::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose a general density-type, intra-site quartic interaction into MFT pairing channels.
The interaction is provided as a rank-4 array living on the site.
The expectation values as a dictionary of matrices containing the expectation values on the site, i with key "ii".

"""
    function IntraQuarticToPairing(Uii::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Delta_ii =    Deltas["ii"] 

        @tullio p_ii[ a , b]    :=   conj(Delta_ii[c, d]) * (Uii[a, c, b, d])

        return Dict("ii" => p_ii)
    end


@doc """
```julia
InterBondQuarticToHopping(Uij::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose an inter-site quartic interaction whose two bilinears both hop across the bond,
Σ U[a, b, c, d] (f†_{ia} f_{jb})(f†_{ic} f_{jd}) + h.c., into MFT hopping channels. This is the form of pair hopping
between two orbitals held as different sites: J f†_{i↑} f†_{i↓} f_{j↓} f_{j↑} = J (f†_{i↑} f_{j↑})(f†_{i↓} f_{j↓}).
Both Hartree and Fock terms live on the bond, so only "ij" is returned; it is the on-site formula with the bond correlation.

"""
    function InterBondQuarticToHopping(Uij::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Chi_ij  =   Chis["ij"]

        @tullio t_ij[ a , b]    :=   Chi_ij[c, d] * (Uij[a, b, c, d] + Uij[c, d, a, b] - Uij[c, b, a, d] - Uij[a, d, c, b])

        return Dict("ij" => t_ij)
    end


@doc """
```julia
InterBondQuarticToPairing(Uij::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) --> Dict{String, Matrix{ComplexF64}}
```
Function to decompose the interaction of [`InterBondQuarticToHopping`](@ref), Σ U[a, b, c, d] (f†_{ia} f_{jb})(f†_{ic} f_{jd}) + h.c.,
into MFT pairing channels. It moves a pair from site j to site i, so it couples the on-site pair amplitudes of the two sites:
the pairing on i comes from the pair amplitude on j, and the pairing on j (from the Hermitian conjugate) from that on i.

"""
    function InterBondQuarticToPairing(Uij::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Delta_ii    =   Deltas["ii"]
        Delta_jj    =   Deltas["jj"]

        @tullio p_ii[ a , c]    :=   conj(Delta_jj[b, d]) * Uij[a, b, c, d]
        @tullio p_jj[ d , b]    :=   conj(Delta_ii[c, a]) * conj(Uij[a, b, c, d])

        return Dict("ii" => p_ii, "jj" => p_jj)
    end


    ##### The built-in decompositions, and the one to use instead on the other kind of bond
    const IntraCounterpart  =   Dict{Function, Function}(InterQuarticToHopping => IntraQuarticToHopping, InterQuarticToPairing => IntraQuarticToPairing,
                                                         InterBondQuarticToHopping => IntraQuarticToHopping, InterBondQuarticToPairing => IntraQuarticToPairing)
    const InterCounterpart  =   Dict{Function, Function}(IntraQuarticToHopping => InterQuarticToHopping, IntraQuarticToPairing => InterQuarticToPairing)

@doc """
```julia
CheckDecompositions(Interactions::Vector{<:Param}, Decompositions::AbstractVector)
CheckDecompositions(Interactions::Vector{<:Param}, Decomposition::Function)
```
Throws an `ArgumentError` if a built-in decomposition is used on the wrong kind of bond: the `Intra` decompositions only handle on-site bonds `(i, i, 0)`, and the `Inter` decompositions assume two different sites.
User-defined decomposition functions are not checked.

"""
    function CheckDecompositions(Interactions::Vector{<:Param}, Decompositions::AbstractVector)

        for (Interaction, Decomposition) in zip(Interactions, Decompositions)
            for bond in Interaction.unitBonds
                onsite  =   bond.base == bond.target && iszero(bond.offset)

                if onsite && haskey(IntraCounterpart, Decomposition)
                    throw(ArgumentError("Interaction \"$(Interaction.label)\" has an on-site bond on site $(bond.base), but uses $(Decomposition), which assumes two different sites. Use $(IntraCounterpart[Decomposition]) for on-site interactions, and put on-site and inter-site bonds in separate interaction Params."))
                elseif !onsite && haskey(InterCounterpart, Decomposition)
                    throw(ArgumentError("Interaction \"$(Interaction.label)\" has a bond between different sites ($(bond.base) → $(bond.target), offset $(bond.offset)), but uses $(Decomposition), which only handles on-site interactions. Use $(InterCounterpart[Decomposition]) for inter-site interactions, and put on-site and inter-site bonds in separate interaction Params."))
                end
            end
        end
    end

    CheckDecompositions(Interactions::Vector{<:Param}, Decomposition::Function) = CheckDecompositions(Interactions, fill(Decomposition, length(Interactions)))




end