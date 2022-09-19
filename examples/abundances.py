

from synthesizer.abundances_sw import Abundances, plot_abundance_patterns

""" Demonstrate the use of the abundances module """



abundances = Abundances()


# --- generate the default abundance pattern

Z = 0.01
CO = 0.0
d2m = None
scaling = None
alpha = 0.0

a = abundances.generate_abundances(Z, alpha, CO, d2m, scaling = scaling) # generate AbundancePattern object

# a.plot_abundances()

# --- generate an alpha enhanced abundance pattern while keeping the same total metallicity default abundance pattern

alpha = 0.6

a_ae = abundances.generate_abundances(Z, alpha, CO, d2m, scaling = scaling) # generate AbundancePattern object

print(a_ae.solar_relative_abundance('O', ref_element = 'Fe')) # [O/Fe], should be 0.4, comes out as 0.4


fig, ax = plot_abundance_patterns([a, a_ae], ['default', r'\alpha = 0.6'], show = True, ylim = [-7., -3.])


fig.savefig('abundance_comparison.pdf')
