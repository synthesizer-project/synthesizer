FAQ
===

A collection of frequently asked questions about Synthesizer.

Why do I get an SVO inaccessible warning on a HPC but it works locally?
-----------------------------------------------------------------------

Your compute node doesn't have internet access. Use the ``write`` method to
save your instruments or filters locally, then use the ``load`` method to read
them back in on the HPC.
