Spectra
*****

Synthesizer enables the generation of multiple different spectra which are associated with galaxies (Galaxy objects) of their components.

**incident** spectra are the spectra that serve as an input to the photoionisation modelling. In the context of stellar population synthesis these are the spectra that are produced by these codes and equivalent to the "pure stellar" spectra.

**transmitted** spectra is the incident spectra that is transmitted through the gas in the photoionisation modelling. Functionally the main difference between **transmitted** and **incident** is that the **transmitted** has little flux below the Lyman-limit, since this has been absorbed by the gas.

**nebular** is the nebular continuum and line emission predicted by the photoionisation model.

**intrinsic** is, for fesc=0.0, is the sum of the **nebular** and **transmitted** emission. For fesc!=0.0 it also includes a contribution from **incident**.

**attenuated** is the the dust attenuated **intrinsic** spectrum. This does not include thermal dust emission so is only valid from UV to near-IR.

**dust** is the thermal dust emission calculated using an energy balance approach and assuming a dist emission model.

**total** is the sum of **attenuated** and **dust**.