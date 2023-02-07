"""
Demonstrate the use of the abundances module
"""

from synthesizer.abundances import Abundances, plot_abundance_pattern


# by default Abundances creates a solar abundance pattern
a1 = Abundances()
print(a1)

# can access the logarithmic abundances of an element like this:
print(a1['O'])
print(a1.a['O'])

# we can change the metallicity
a2 = Abundances(Z=0.01)
print(a2)

# or other parameters
a3 = Abundances(Z=0.01, alpha=0.6, CO=0.0)
print(a3)
# print(a3.metallicity())

print(f"[O/Fe] = {a3.solar_relative_abundance('O', ref_element='Fe'):.2f}")  # [O/Fe]


plot_abundance_pattern([a2, a3], [r'Z=0.01', r'Z=0.01; \alpha = 0.6'],
                       show=True, ylim=[-7., -3.])
