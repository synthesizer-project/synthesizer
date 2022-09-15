





from synthesizer.abundances_sw import Abundances



abundances = Abundances()

print(abundances.sol)

Z = abundances.Z_sol
alpha = 0.0
CO = 0.0
d2m = None
scaling = None


a = abundances.generate_abundances(Z, alpha, CO, d2m, scaling = scaling) # generate AbundancePattern object

print(a.solar_relative_abundance('O')) # [O/H], should be 0.0, comes out as -0.027


Z = 0.0001
alpha = 0.4
CO = 0.0
d2m = None
scaling = None

a = abundances.generate_abundances(Z, alpha, CO, d2m, scaling = scaling) # generate AbundancePattern object


print(a.solar_relative_abundance('O', ref_element = 'Fe')) # [O/Fe], should be 0.4, comes out as 0.4
print(a.Z)

a.plot_abundances()
