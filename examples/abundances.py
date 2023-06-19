"""
Demonstrate the use of the abundances module
"""

from synthesizer.abundances import Abundances, plot_abundance_pattern, plot_multiple_abundance_patterns


# by default Abundances creates a solar abundance pattern with no depletion
a1 = Abundances()
print(a1)

# you can access the logarithmic abundances (log10(X/H)) of an element like this:
print(f"log10(O/H): {a1.total['O']:.2f}")
# or this
print(f"log10(O/H): {a1['O']:.2f}")

# we can change the metallicity
a2 = Abundances(Z=0.01) 
print(a2)


# or alpha-enhancement
a3 = Abundances(Z=0.01, alpha=0.6)
print(a3)

# we can print a relative solar abundance like this:
print(f"[O/Fe] = {a3.solar_relative_abundance('O', ref_element='Fe'):.2f}")  
# or like this:
print(f"[O/Fe] = {a3['[O/Fe]']:.2f}")  

# there are also a helper functions for plotting one or more abundance patterns, here we plot two abundance patterns with different alpha abundances
plot_multiple_abundance_patterns([a2, a3], labels=[r'Z=0.01', r'Z=0.01; \alpha = 0.6'], show=True, ylim=[-7., -3.])


# or the dust-to-metal ratio
a4 = Abundances(Z=0.01, d2m=0.3)
print(a4)

# when d2m > 0.0 total, gas, and dust abundance patterns are provided e.g.

print(f'log10(C/H) total: {a4.total["C"]:.2f}')
print(f'log10(C/H) gas: {a4.gas["C"]:.2f}')
print(f'log10(C/H) dust: {a4.dust["C"]:.2f}')

# we can plot the abundance pattern of each component

plot_abundance_pattern(a4, show=True, ylim=[-7., -3.], lines = ['total', 'gas', 'dust'])



