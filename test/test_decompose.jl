##### Analytic checks of the mean-field decompositions against textbook Hartree-Fock(-Bogoliubov)

@testset "Decompositions" begin
    U   =   2.0
    Chi =   ComplexF64[0.3 0.1+0.2im; 0.1-0.2im 0.6]     ##### generic Hermitian on-site <f^dag_a f_b>

    @testset "IntraQuarticToHopping: Hubbard Hartree + Fock" begin
        t   =   IntraQuarticToHopping(U .* Hubbard, Dict("ii" => Chi))["ii"]
        ##### U n_up n_down -> U <n_dn> n_up + U <n_up> n_dn - U <f^dag_dn f_up> f^dag_up f_dn - U <f^dag_up f_dn> f^dag_dn f_up
        @test t ≈ [U * Chi[2, 2]  -U * Chi[2, 1];
                  -U * Chi[1, 2]   U * Chi[1, 1]]
        @test t ≈ adjoint(t)
    end

    @testset "IntraQuarticToPairing: Hubbard" begin
        Delta   =   ComplexF64[0.0 0.4+0.1im; -0.4-0.1im 0.0]
        p       =   IntraQuarticToPairing(U .* Hubbard, Dict("ii" => Delta))["ii"]
        @test p ≈ [0.0  U * conj(Delta[1, 2]); 0.0 0.0]
    end

    @testset "InterQuarticToHopping: density-density V n_i n_j" begin
        V       =   0.7
        DD      =   DensityToPartonCoupling(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2))
        Chi_jj  =   ComplexF64[0.2 0.0; 0.0 0.5]
        Chi_ij  =   ComplexF64[0.1+0.3im 0.05; -0.2im 0.4]
        t       =   InterQuarticToHopping(V .* DD, Dict("ii" => Chi, "jj" => Chi_jj, "ij" => Chi_ij))
        @test t["ii"] ≈ V * tr(Chi_jj) * I          ##### Hartree on i from the density on j
        @test t["jj"] ≈ V * tr(Chi) * I             ##### Hartree on j from the density on i
        @test t["ij"] ≈ -V .* conj.(Chi_ij)         ##### Fock on the bond
    end
end

@testset "InterBond decompositions (pair hopping)" begin
    J       =   0.6
    PH      =   zeros(ComplexF64, 2, 2, 2, 2)
    PH[1, 1, 2, 2]  =   J                                   ##### J (f†_{i↑} f_{j↑})(f†_{i↓} f_{j↓}) = J f†_{i↑} f†_{i↓} f_{j↓} f_{j↑}
    Chi_ij  =   ComplexF64[0.1+0.3im 0.05-0.1im; -0.2im 0.4+0.1im]

    @testset "hopping channel" begin
        t   =   InterBondQuarticToHopping(PH, Dict("ij" => Chi_ij))
        @test keys(t) == Set(["ij"])
        @test t["ij"] ≈ J .* [Chi_ij[2, 2] -Chi_ij[2, 1]; -Chi_ij[1, 2] Chi_ij[1, 1]]
    end

    ##### for any tensor, t is the derivative of <O> = Σ U (χ_ab χ_cd - χ_ad χ_cb) and Re Σ t χ = <O + O†>
    @testset "derivative and energy" begin
        Random.seed!(5)
        U       =   randn(ComplexF64, 2, 2, 2, 2)
        O(χ)    =   sum(U[a, b, c, d] * (χ[a, b] * χ[c, d] - χ[a, d] * χ[c, b]) for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
        t       =   InterBondQuarticToHopping(U, Dict("ij" => Chi_ij))["ij"]
        ε       =   1e-6
        for a in 1:2, b in 1:2
            E   =   zeros(ComplexF64, 2, 2); E[a, b] = ε
            @test t[a, b] ≈ (O(Chi_ij + E) - O(Chi_ij - E)) / (2ε) atol = 1e-8
        end
        @test real(sum(t .* Chi_ij)) ≈ 2 * real(O(Chi_ij))
    end

    @testset "pairing channel" begin
        δi, δj  =   0.3 + 0.1im, -0.2 + 0.25im
        Δi, Δj  =   δi .* ComplexF64[0 1; -1 0], δj .* ComplexF64[0 1; -1 0]
        p       =   InterBondQuarticToPairing(PH, Dict("ii" => Δi, "jj" => Δj))
        @test p["ii"] ≈ [0 J * conj(δj); 0 0]               ##### the pair on j induces pairing on i
        @test p["jj"] ≈ [0 0; -J * conj(δi) 0]              ##### and the pair on i pairing on j (f†_2 f†_1 = -f†_1 f†_2)
        ##### double-counting energy Re Σ (p Δ) = <O + O†> in the pairing channel, for any tensor
        Random.seed!(6)
        U   =   randn(ComplexF64, 2, 2, 2, 2)
        q   =   InterBondQuarticToPairing(U, Dict("ii" => Δi, "jj" => Δj))
        X   =   -sum(U[a, b, c, d] * Δi[a, c] * conj(Δj[d, b]) for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
        @test real(sum(q["ii"] .* Δi) + sum(q["jj"] .* Δj)) ≈ 2 * real(X)
    end

    @testset "on-site use is rejected" begin
        uc  =   UnitCell([[1.0, 0.0], [0.0, 1.0]], 2)
        AddBasisSite!(uc, [0.0, 0.0])
        Up  =   Param(1.0, 4)
        AddAnisotropicBond!(Up, uc, 1, 1, [0, 0], PH, 0.0, "pair hopping")
        @test_throws ArgumentError MeanFieldToolkit.MFTDecompose.CheckDecompositions([Up], InterBondQuarticToHopping)
    end
end
