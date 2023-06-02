"""
Demonstrate the use of the abundances module
"""

from synthesizer.abundances import Abundances, plot_abundance_pattern


# by default Abundances creates a solar abundance pattern with no depletion
a1 = Abundances()
print(a1)
print(f'max d2m: {a1.get_max_d2m():.2f}')

# you can access the logarithmic abundances (log10(X/H)) of an element like this:
print(f"log10(O/H): {a1.a['O']:.2f}")
# or this
print(f"log10(O/H): {a1['O']:.2f}")

# we can change the metallicity
a2 = Abundances(Z=0.01)
print(a2)


# or alpha-enhancement
a3 = Abundances(Z=0.01, alpha=0.6, CO=0.0)
print(a3)
# print(a3.metallicity())

print(f"[O/Fe] = {a3.solar_relative_abundance('O', ref_element='Fe'):.2f}")  # [O/Fe]

plot_abundance_pattern([a2, a3], [r'Z=0.01', r'Z=0.01; \alpha = 0.6'],
                       show=True, ylim=[-7., -3.])


# or the dust-to-metal ratio
a4 = Abundances(Z=0.01, d2m=0.3)
print(a4)

print(f"[O/Fe] = {a4.solar_relative_abundance('O', ref_element='Fe'):.2f}")  # [O/Fe]
plot_abundance_pattern([a2, a4], [r'Z=0.01', rf'Z=0.01; d2m = {a4.d2m}'],
                       show=True, ylim=[-7., -3.])


# asking for a dust-to-metal ratio above the maximum set by the default depletion pattern should throw an exception
# a5 = Abundances(Z=0.01, d2m=0.7)
# print(a5)
