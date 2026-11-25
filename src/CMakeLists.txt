# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_glmmhd.cpp
        units.hpp
        bvals/boundary_conditions_apk.hpp
        eos/adiabatic_hydro.cpp
        hydro/diffusion/conduction.cpp
        hydro/diffusion/diffusion.cpp
        hydro/diffusion/diffusion.hpp
        hydro/diffusion/resistivity.cpp
        hydro/diffusion/viscosity.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/glmmhd/dedner_source.cpp
        hydro/prolongation/custom_ops.hpp
        hydro/srcterms/gravitational_field.hpp
        hydro/srcterms/tabular_cooling.hpp
        hydro/srcterms/tabular_cooling.cpp
        refinement/gradient.cpp
        refinement/other.cpp
        tracers/tracers.cpp
        tracers/tracers.hpp
        utils/few_modes_ft.cpp
        utils/few_modes_ft.hpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon singularity-eos)

if(AthenaPK_ENABLE_EOS_TESTING)
  add_executable(eos_tests ../tst/unit/eos.cpp eos/adiabatic_hydro.cpp eos/adiabatic_glmmhd.cpp)
  target_include_directories(eos_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(eos_tests PRIVATE parthenon singularity-eos)
  add_test(NAME eos_device COMMAND eos_tests)
endif()

if(AthenaPK_ENABLE_EOS_TESTING)
  add_executable(eos_benchmark ../tst/unit/eos_benchmark.cpp)
  target_include_directories(eos_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(eos_benchmark PRIVATE parthenon singularity-eos)
endif()
