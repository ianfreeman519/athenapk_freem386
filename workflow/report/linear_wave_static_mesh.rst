Three-dimensional Mesh Structure
================================

MeshBlock boundaries for the static-refinement layout used by the three-dimensional
linear MHD wave convergence study.

Using:
    mesh:
      - nx1 = 2*N
      - nx2 = N
      - nx3 = N
      - meshblock_nx1 = nx1 // 4
      - meshblock_nx2 = nx2 // 2
      - meshblock_nx3 = nx3 // 2

    refinement_regions:
      - x1min: 0.5
        x1max: 2.5
        x2min: 0.25
        x2max: 1.25
        x3min: 0.25
        x3max: 1.25
        level: 1

      - x1min: 1.0
        x1max: 2.0
        x2min: 0.5
        x2max: 1.0
        x3min: 0.5
        x3max: 1.0
        level: 2

gives 240 meshblocks
