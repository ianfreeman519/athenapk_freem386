# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2025, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# Any external modules that are required by python can be added to REQUIRED_PYTHON_MODULES
# list variable, before including TestSetup.cmake.
list(APPEND REQUIRED_PYTHON_MODULES numpy unyt matplotlib h5py scipy)
message(STATUS
  "Checking for Python modules (${REQUIRED_PYTHON_MODULES}) required for regression tests.\n"
  "   Check can be disabled by setting PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF but then\n"
  "   tests are not guaranteed to work anymore.")

# import Parthenon setup_test_serial and setup_test_parallel
include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)

function(setup_test_both dir arg test_labels)
  setup_test_serial(${dir} "${arg}" "${test_labels}")
  # just using 2 ranks for MPI tests
  setup_test_parallel(2 ${dir} "${arg}" "${test_labels}")
endfunction()

setup_test_both("riemann_hydro" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sod.in --num_steps 39" "other")

setup_test_both("tabulated_eos" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/tabulated_sod.in --num_steps 1" "other")

setup_test_both("turbulence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 1" "other")

setup_test_both("convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 40" "convergence")

setup_test_both("mhd_convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 48" "convergence")

setup_test_serial("performance" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 21" "performance")

setup_test_both("cluster_hse" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hse.in --num_steps 2" "convergence")

setup_test_serial("cluster_tabular_cooling" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/cooling.in --num_steps 11" "convergence")

setup_test_both("aniso_therm_cond_ring_conv" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")
  
setup_test_both("aniso_therm_cond_ring_multid" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")
 
setup_test_both("aniso_therm_cond_gauss_conv" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 24" "convergence")

setup_test_both("diffusion" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 12" "convergence")

setup_test_both("resistive_ct_analytic" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/ct_glm_compatible/resistive_diffusion.in --num_steps 22" "convergence")

  setup_test_both("diffusion_linwave3d" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 2" "convergence")

setup_test_both("field_loop" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/field_loop.in --num_steps 15" "convergence")

setup_test_both("cluster_magnetic_tower" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/magnetic_tower.in --num_steps 4" "convergence")

setup_test_both("cluster_hydro_agn_feedback" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hydro_agn_feedback.in --num_steps 5" "convergence")

setup_test_both("cluster_agn_triggering" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/agn_triggering.in --num_steps 3" "convergence")

  setup_test_both("particle_advection" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 2" "regression")

  setup_test_serial("lw_implode_symmetry" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/lw_implode.in --num_steps 1" "regression")

add_test(NAME free_free_cooling
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/free_free_cooling/check_free_free.py
    --driver $<TARGET_FILE:athenaPK>)
set_tests_properties(free_free_cooling PROPERTIES LABELS "regression" TIMEOUT 600)

add_test(NAME reservoir_mask
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/reservoir_mask/check_reservoir_mask.py
    --driver $<TARGET_FILE:athenaPK>)
set_tests_properties(reservoir_mask PROPERTIES LABELS "regression" TIMEOUT 600)

add_test(NAME reservoir_mask_cooling
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/reservoir_mask/check_reservoir_cooling.py
    --driver $<TARGET_FILE:athenaPK>)
set_tests_properties(reservoir_mask_cooling PROPERTIES LABELS "regression" TIMEOUT 600)

add_test(NAME compatible_ct_energy
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
    --driver $<TARGET_FILE:athenaPK>)
set_tests_properties(compatible_ct_energy PROPERTIES LABELS "regression" TIMEOUT 1800)

# Independent exact magnetic-decay and thermal-gain benchmarks.
add_test(NAME compatible_ct_energy_analytic
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
    --driver $<TARGET_FILE:athenaPK> --group analytic)
set_tests_properties(compatible_ct_energy_analytic PROPERTIES
  LABELS "regression;convergence" TIMEOUT 1800)
if(MPI_FOUND AND MPIEXEC_EXECUTABLE)
  add_test(NAME compatible_ct_energy_analytic_mpi
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
      --driver $<TARGET_FILE:athenaPK> --group analytic
      --launcher "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}")
  set_tests_properties(compatible_ct_energy_analytic_mpi PROPERTIES
    LABELS "regression;convergence" PROCESSORS 2 TIMEOUT 1800)
endif()
add_test(NAME compatible_ct_energy_analytic_reference
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/analytic_diffusion.py)
set_tests_properties(compatible_ct_energy_analytic_reference PROPERTIES LABELS "regression" TIMEOUT 60)

# RKL2 physical validation is separate from the fast driver-free analyzer checks.
foreach(section fixed spitzer temporal controls tabulated)
  if(section STREQUAL "tabulated" AND
     NOT EXISTS "${PROJECT_SOURCE_DIR}/aluminum_saha_eos_three_chart.h5")
    message(STATUS "RKL2 tabulated validation unavailable: supply the aluminum EOS table")
    continue()
  endif()
  add_test(NAME compatible_ct_rkl2_${section}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
      --driver $<TARGET_FILE:athenaPK> --group rkl2 --rkl2-section ${section}
      --batch-seconds 0)
  set_tests_properties(compatible_ct_rkl2_${section} PROPERTIES
    LABELS "regression;convergence;rkl2" TIMEOUT 1800)
  if(MPI_FOUND AND MPIEXEC_EXECUTABLE)
    string(REPLACE ";" " " rkl_mpi_preflags "${MPIEXEC_PREFLAGS}")
    add_test(NAME compatible_ct_rkl2_${section}_mpi
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
        --driver $<TARGET_FILE:athenaPK> --group rkl2 --rkl2-section ${section}
        --batch-seconds 0
        --launcher "${MPIEXEC_EXECUTABLE} ${rkl_mpi_preflags} ${MPIEXEC_NUMPROC_FLAG} 2")
    set_tests_properties(compatible_ct_rkl2_${section}_mpi PROPERTIES
      LABELS "regression;convergence;rkl2" PROCESSORS 2 TIMEOUT 1800)
  endif()
endforeach()
add_test(NAME compatible_ct_rkl2_reference
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/rkl2_diffusion.py)
set_tests_properties(compatible_ct_rkl2_reference PROPERTIES LABELS "regression;rkl2" TIMEOUT 60)

add_test(NAME compatible_ct_refinement_flow
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
    --driver $<TARGET_FILE:athenaPK> --group refinement-flow --batch-seconds 0)
set_tests_properties(compatible_ct_refinement_flow PROPERTIES
  LABELS "regression;refinement" TIMEOUT 600)
add_test(NAME compatible_ct_refinement_flow_reference
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/refinement_flow.py)
set_tests_properties(compatible_ct_refinement_flow_reference PROPERTIES
  LABELS "regression;refinement" TIMEOUT 60)

foreach(section base resolution temporal)
  add_test(NAME compatible_ct_interface_thermal_${section}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
      --driver $<TARGET_FILE:athenaPK> --group interface-thermal
      --interface-section ${section} --batch-seconds 0)
  set_tests_properties(compatible_ct_interface_thermal_${section} PROPERTIES
    LABELS "regression;refinement;diagnostic" TIMEOUT 1800)
endforeach()
add_test(NAME compatible_ct_interface_thermal_reference
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/interface_thermal.py)
set_tests_properties(compatible_ct_interface_thermal_reference PROPERTIES
  LABELS "regression;refinement;diagnostic" TIMEOUT 60)

# Compiled checks use the production stencil, state validation, retry policy,
# and collective reduction helpers. Build these along with athenaPK.
add_executable(compatible_ct_rkl_bounds
  test_suites/compatible_ct_energy/ct_rkl_bounds.cpp)
target_link_libraries(compatible_ct_rkl_bounds PRIVATE parthenon)
add_test(NAME compatible_ct_rkl_bounds_serial COMMAND compatible_ct_rkl_bounds)
set_tests_properties(compatible_ct_rkl_bounds_serial PROPERTIES
  LABELS "regression;refinement;rkl2" TIMEOUT 60)
if(MPI_FOUND AND MPIEXEC_EXECUTABLE)
  add_test(NAME compatible_ct_rkl_bounds_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:compatible_ct_rkl_bounds> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(compatible_ct_rkl_bounds_mpi PROPERTIES
    LABELS "regression;refinement;rkl2;mpi" PROCESSORS 2 TIMEOUT 60)
endif()

add_test(NAME compatible_ct_spitzer_smr_guard
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
    --driver $<TARGET_FILE:athenaPK> --group refinement-flow
    --refinement-section guard --batch-seconds 0)
set_tests_properties(compatible_ct_spitzer_smr_guard PROPERTIES
  LABELS "regression;refinement;rkl2;spitzer" TIMEOUT 1800)
if(MPI_FOUND AND MPIEXEC_EXECUTABLE)
  string(REPLACE ";" " " guard_mpi_preflags "${MPIEXEC_PREFLAGS}")
  add_test(NAME compatible_ct_spitzer_smr_guard_mpi
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/compatible_ct_energy/check_compatible_energy.py
      --driver $<TARGET_FILE:athenaPK> --group refinement-flow
      --refinement-section guard --batch-seconds 0
      --launcher "${MPIEXEC_EXECUTABLE} ${guard_mpi_preflags} ${MPIEXEC_NUMPROC_FLAG} 2")
  set_tests_properties(compatible_ct_spitzer_smr_guard_mpi PROPERTIES
    LABELS "regression;refinement;rkl2;spitzer;mpi" PROCESSORS 2 TIMEOUT 1800)
endif()
