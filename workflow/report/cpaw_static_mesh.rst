CPAW Static Mesh Structure
==========================

MeshBlock boundaries for the static-refinement layout used by the three-dimensional
circularly polarized traveling Alfvén wave convergence study. This plot has its own
output namespace even when CPAW and the linear-wave study use the same layout.

The base mesh has ``nx1 = 2*N``, ``nx2 = N``, and ``nx3 = N``. The configured
MeshBlock dimensions preserve the same block topology as ``N`` changes.
