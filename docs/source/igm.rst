InterGalactic Medium
********************

`Synthesizer` provides some common parametrisations of the transmission through the InterGalactic Medium (IGM)::

    from synthesizer.igm import Madau96, Inoue14

We can plot Madau+96 and Inoue+94 at a range of redshifts::

    import matplotlib.pyplot as plt
    import numpy as np
    
    lam = np.arange(0, 50000)
    z = 7.
    
    for z in [3., 5., 7.]:
        for IGM, ls in zip([Inoue14, Madau96], ['-', ':']):
            igm = IGM()
            plt.plot(lam, igm.T(z, lam), ls=ls)
    
    
    plt.ylim([0, 1.1])
    plt.show()

.. image:: images/igm.png
