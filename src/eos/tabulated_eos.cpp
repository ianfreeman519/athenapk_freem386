#include "tabulated_eos.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <hdf5.h>

#include "config.hpp"
#include "utils/error_checking.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
using Real = parthenon::Real;

struct RawTable {
  std::vector<Real> rho_e, eps, pressure, temperature, sound_speed, zbar;
  std::vector<Real> rho_p, pres, energy_p;
  std::vector<Real> rho_t, temp, energy_t;
  int nrho_e, neps, nrho_p, npres, nrho_t, ntemp;
};

void H5Check(const herr_t status, const std::string &what) {
  PARTHENON_REQUIRE_THROWS(status >= 0, "Tabulated EOS HDF5 failure: " + what);
}

std::vector<Real> ReadDataset(hid_t file, const std::string &name,
                              std::vector<hsize_t> &dims) {
  const hid_t ds = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  PARTHENON_REQUIRE_THROWS(ds >= 0, "Missing EOS dataset " + name);
  const hid_t space = H5Dget_space(ds);
  const int rank = H5Sget_simple_extent_ndims(space);
  PARTHENON_REQUIRE_THROWS(rank == 1 || rank == 2,
                           "EOS dataset must have rank one or two: " + name);
  dims.resize(rank);
  H5Check(H5Sget_simple_extent_dims(space, dims.data(), nullptr), name + " dimensions");
  size_t n = 1;
  for (const auto d : dims) n *= d;
  std::vector<Real> data(n);
  const hid_t memory_type = sizeof(Real) == sizeof(double) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
  H5Check(H5Dread(ds, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
          "read " + name);
  H5Sclose(space);
  H5Dclose(ds);
  return data;
}

long long ReadIntegerAttribute(hid_t obj, const std::string &name) {
  PARTHENON_REQUIRE_THROWS(H5Aexists(obj, name.c_str()) > 0,
                           "Missing EOS attribute " + name);
  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  long long value;
  H5Check(H5Aread(attr, H5T_NATIVE_LLONG, &value), "read attribute " + name);
  H5Aclose(attr);
  return value;
}

double ReadRealAttribute(hid_t obj, const std::string &name) {
  PARTHENON_REQUIRE_THROWS(H5Aexists(obj, name.c_str()) > 0,
                           "Missing EOS attribute " + name);
  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  double value;
  H5Check(H5Aread(attr, H5T_NATIVE_DOUBLE, &value), "read attribute " + name);
  H5Aclose(attr);
  return value;
}

std::string ReadStringAttribute(hid_t obj, const std::string &name) {
  PARTHENON_REQUIRE_THROWS(H5Aexists(obj, name.c_str()) > 0,
                           "Missing EOS attribute " + name);
  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  const hid_t type = H5Aget_type(attr);
  PARTHENON_REQUIRE_THROWS(H5Tget_class(type) == H5T_STRING,
                           "EOS attribute is not a string: " + name);
  std::string value;
  if (H5Tis_variable_str(type) > 0) {
    char *raw = nullptr;
    H5Check(H5Aread(attr, type, &raw), "read attribute " + name);
    if (raw != nullptr) {
      value = raw;
      H5Check(H5free_memory(raw), "free attribute " + name);
    }
  } else {
    const size_t n = H5Tget_size(type);
    std::vector<char> buf(n + 1, '\0');
    H5Check(H5Aread(attr, type, buf.data()), "read attribute " + name);
    value.assign(buf.data(), n);
    while (!value.empty() && (value.back() == '\0' || value.back() == ' '))
      value.pop_back();
  }
  H5Tclose(type);
  H5Aclose(attr);
  return value;
}

void RequireUnits(hid_t file, const std::string &dataset, const std::string &units) {
  const hid_t ds = H5Dopen2(file, dataset.c_str(), H5P_DEFAULT);
  PARTHENON_REQUIRE_THROWS(ds >= 0, "Missing EOS dataset " + dataset);
  const auto actual = ReadStringAttribute(ds, "linear_units");
  H5Dclose(ds);
  PARTHENON_REQUIRE_THROWS(actual == units,
                           dataset + " has units '" + actual + "', expected '" + units + "'");
}

void ValidateAxis(const std::vector<Real> &x, const std::string &name) {
  PARTHENON_REQUIRE_THROWS(x.size() >= 2, name + " must contain at least two points");
  const Real dx = x[1] - x[0];
  PARTHENON_REQUIRE_THROWS(std::isfinite(dx) && dx > 0.0,
                           name + " must be finite and strictly increasing");
  for (size_t i = 0; i < x.size(); ++i) {
    PARTHENON_REQUIRE_THROWS(std::isfinite(x[i]), name + " contains non-finite data");
    if (i > 0) {
      PARTHENON_REQUIRE_THROWS(x[i] > x[i - 1], name + " is not increasing");
      const auto err = std::abs((x[i] - x[i - 1]) - dx);
      PARTHENON_REQUIRE_THROWS(err <= 1.e-10 * std::max(std::abs(dx), Real(1.0)),
                               name + " is not uniformly spaced");
    }
  }
}

void ValidateField(const std::vector<Real> &q, const size_t expected,
                   const std::string &name) {
  PARTHENON_REQUIRE_THROWS(q.size() == expected, "Wrong shape for " + name);
  PARTHENON_REQUIRE_THROWS(
      std::all_of(q.begin(), q.end(), [](Real x) { return std::isfinite(x); }),
      name + " contains non-finite data");
}

void ValidateShape(const std::vector<hsize_t> &dims, const int n0, const int n1,
                   const std::string &name) {
  PARTHENON_REQUIRE_THROWS(
      dims.size() == 2 && dims[0] == static_cast<hsize_t>(n0) &&
          dims[1] == static_cast<hsize_t>(n1),
      "Wrong [density, thermodynamic-coordinate] shape for " + name);
}

void ValidateMonotoneRows(const std::vector<Real> &q, int nr, int nx,
                          const std::string &name) {
  for (int i = 0; i < nr; ++i)
    for (int j = 1; j < nx; ++j)
      PARTHENON_REQUIRE_THROWS(q[i * nx + j] > q[i * nx + j - 1],
                               name + " is not strictly increasing in its second axis");
}

Real InterpolateHost(const std::vector<Real> &q, const std::vector<Real> &xaxis,
                     const std::vector<Real> &yaxis, Real x, Real y) {
  const Real fx = (x - xaxis.front()) / (xaxis[1] - xaxis[0]);
  const Real fy = (y - yaxis.front()) / (yaxis[1] - yaxis[0]);
  int i = std::clamp(static_cast<int>(std::floor(fx)), 0,
                     static_cast<int>(xaxis.size()) - 2);
  int j = std::clamp(static_cast<int>(std::floor(fy)), 0,
                     static_cast<int>(yaxis.size()) - 2);
  const Real tx = fx - i, ty = fy - j;
  const int ny = yaxis.size();
  return (1 - tx) * ((1 - ty) * q[i * ny + j] + ty * q[i * ny + j + 1]) +
         tx * ((1 - ty) * q[(i + 1) * ny + j] + ty * q[(i + 1) * ny + j + 1]);
}

void ValidateCrossCharts(const RawTable &r) {
  Real worst_p = 0.0, worst_t = 0.0;
  constexpr int nsample = 33;
  for (int a = 0; a < nsample; ++a) {
    const int i = a * (r.nrho_e - 1) / (nsample - 1);
    for (int b = 0; b < nsample; ++b) {
      const int j = b * (r.neps - 1) / (nsample - 1);
      const Real lr = r.rho_e[i], le = r.eps[j];
      const Real lp = r.pressure[i * r.neps + j];
      const Real lt = r.temperature[i * r.neps + j];
      worst_p = std::max(worst_p,
                         std::abs(InterpolateHost(r.energy_p, r.rho_p, r.pres, lr, lp) - le));
      worst_t = std::max(worst_t,
                         std::abs(InterpolateHost(r.energy_t, r.rho_t, r.temp, lr, lt) - le));
    }
  }
  PARTHENON_REQUIRE_THROWS(worst_p < 3.e-3 && worst_t < 3.e-3,
                           "EOS forward/inverse chart inconsistency exceeds 0.003 dex");
}

RawTable ReadRaw(const std::string &filename) {
  const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  PARTHENON_REQUIRE_THROWS(file >= 0, "Unable to open tabulated EOS file " + filename);
  PARTHENON_REQUIRE_THROWS(ReadIntegerAttribute(file, "format_version") == 1,
                           "Unsupported tabulated EOS format version");
  PARTHENON_REQUIRE_THROWS(ReadRealAttribute(file, "log_base") == 10.0,
                           "The v1 EOS format requires base-10 logarithms");
  PARTHENON_REQUIRE_THROWS(ReadRealAttribute(file, "inverse_margin_decades") >= 1.0,
                           "Inverse EOS charts must include at least one decade of margin");
  PARTHENON_REQUIRE_THROWS(
      ReadStringAttribute(file, "sound_speed_definition") ==
          "adiabatic sound speed sqrt((dP/drho)_s)",
      "EOS sound speed must be the adiabatic thermodynamic sound speed");
  PARTHENON_REQUIRE_THROWS(
      ReadStringAttribute(file, "array_index_order") ==
          "quantity[density_index, thermodynamic_coordinate_index]",
      "Unsupported tabulated EOS array ordering");

  RawTable r;
  std::vector<hsize_t> d;
  r.rho_e = ReadDataset(file, "/eos_rho_e/axes/log10_density", d);
  r.nrho_e = d[0];
  r.eps = ReadDataset(file, "/eos_rho_e/axes/log10_specific_internal_energy", d);
  r.neps = d[0];
  r.pressure = ReadDataset(file, "/eos_rho_e/log10_pressure", d);
  ValidateShape(d, r.nrho_e, r.neps, "/eos_rho_e/log10_pressure");
  r.temperature = ReadDataset(file, "/eos_rho_e/log10_temperature", d);
  ValidateShape(d, r.nrho_e, r.neps, "/eos_rho_e/log10_temperature");
  r.sound_speed = ReadDataset(file, "/eos_rho_e/log10_sound_speed", d);
  ValidateShape(d, r.nrho_e, r.neps, "/eos_rho_e/log10_sound_speed");
  r.zbar = ReadDataset(file, "/eos_rho_e/Zbar", d);
  ValidateShape(d, r.nrho_e, r.neps, "/eos_rho_e/Zbar");

  r.rho_p = ReadDataset(file, "/eos_rho_p/axes/log10_density", d);
  r.nrho_p = d[0];
  r.pres = ReadDataset(file, "/eos_rho_p/axes/log10_pressure", d);
  r.npres = d[0];
  r.energy_p = ReadDataset(file, "/eos_rho_p/log10_specific_internal_energy", d);
  ValidateShape(d, r.nrho_p, r.npres,
                "/eos_rho_p/log10_specific_internal_energy");

  r.rho_t = ReadDataset(file, "/eos_rho_T/axes/log10_density", d);
  r.nrho_t = d[0];
  r.temp = ReadDataset(file, "/eos_rho_T/axes/log10_temperature", d);
  r.ntemp = d[0];
  r.energy_t = ReadDataset(file, "/eos_rho_T/log10_specific_internal_energy", d);
  ValidateShape(d, r.nrho_t, r.ntemp,
                "/eos_rho_T/log10_specific_internal_energy");

  // The v1 fixture has an explicit and deliberately mixed SI/eV unit contract.
  RequireUnits(file, "/eos_rho_e/axes/log10_density", "kg/m^3");
  RequireUnits(file, "/eos_rho_e/axes/log10_specific_internal_energy", "J/kg");
  RequireUnits(file, "/eos_rho_e/log10_pressure", "Pa");
  RequireUnits(file, "/eos_rho_e/log10_temperature", "eV");
  RequireUnits(file, "/eos_rho_e/log10_sound_speed", "m/s");
  RequireUnits(file, "/eos_rho_p/axes/log10_density", "kg/m^3");
  RequireUnits(file, "/eos_rho_p/axes/log10_pressure", "Pa");
  RequireUnits(file, "/eos_rho_p/log10_specific_internal_energy", "J/kg");
  RequireUnits(file, "/eos_rho_T/axes/log10_density", "kg/m^3");
  RequireUnits(file, "/eos_rho_T/axes/log10_temperature", "eV");
  RequireUnits(file, "/eos_rho_T/log10_specific_internal_energy", "J/kg");
  H5Fclose(file);

  ValidateAxis(r.rho_e, "rho-e density axis");
  ValidateAxis(r.eps, "rho-e energy axis");
  ValidateAxis(r.rho_p, "rho-P density axis");
  ValidateAxis(r.pres, "rho-P pressure axis");
  ValidateAxis(r.rho_t, "rho-T density axis");
  ValidateAxis(r.temp, "rho-T temperature axis");
  const auto ne = static_cast<size_t>(r.nrho_e) * r.neps;
  ValidateField(r.pressure, ne, "rho-e pressure");
  ValidateField(r.temperature, ne, "rho-e temperature");
  ValidateField(r.sound_speed, ne, "rho-e sound speed");
  ValidateField(r.zbar, ne, "rho-e Zbar");
  ValidateField(r.energy_p, static_cast<size_t>(r.nrho_p) * r.npres,
                "rho-P energy");
  ValidateField(r.energy_t, static_cast<size_t>(r.nrho_t) * r.ntemp,
                "rho-T energy");
  ValidateMonotoneRows(r.pressure, r.nrho_e, r.neps, "rho-e pressure");
  ValidateMonotoneRows(r.temperature, r.nrho_e, r.neps, "rho-e temperature");
  ValidateMonotoneRows(r.energy_p, r.nrho_p, r.npres, "rho-P energy");
  ValidateMonotoneRows(r.energy_t, r.nrho_t, r.ntemp, "rho-T energy");
  PARTHENON_REQUIRE_THROWS(
      std::all_of(r.zbar.begin(), r.zbar.end(), [](Real z) { return z >= 0.0; }),
      "Zbar must be nonnegative");
  ValidateCrossCharts(r);
  return r;
}

template <typename View>
void Copy2D(View &dst, const std::vector<Real> &src, int n0, int n1,
            const std::string &label, Real offset = 0.0) {
  dst = View(label, n0, n1);
  auto h = Kokkos::create_mirror_view(dst);
  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j) h(i, j) = src[i * n1 + j] + offset;
  Kokkos::deep_copy(dst, h);
}

uint64_t Hash(const RawTable &r, const std::array<Real, 6> &scales) {
  uint64_t h = 1469598103934665603ULL;
  auto add = [&h](const void *p, size_t n) {
    const auto *c = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ULL;
  };
  for (const auto *v : {&r.rho_e, &r.eps, &r.pressure, &r.temperature,
                        &r.sound_speed, &r.zbar, &r.rho_p, &r.pres,
                        &r.energy_p, &r.rho_t, &r.temp, &r.energy_t})
    add(v->data(), v->size() * sizeof(Real));
  add(scales.data(), scales.size() * sizeof(Real));
  return h;
}
} // namespace

TabulatedEOS TabulatedEOS::Load(const std::string &filename, Real density_factor,
                                Real energy_factor, Real pressure_factor,
                                Real sound_speed_factor, Real temperature_factor,
                                Real mean_ion_mass) {
  for (const auto x : {density_factor, energy_factor, pressure_factor,
                       sound_speed_factor, temperature_factor, mean_ion_mass})
    PARTHENON_REQUIRE_THROWS(std::isfinite(x) && x > 0.0,
                             "Tabulated EOS conversion factors and A must be positive");

  // Initialization-only replicated reads avoid coupling the table format to parallel-HDF5
  // hyperslab semantics. Every rank validates identical immutable input.
  auto r = ReadRaw(filename);
  TabulatedEOS t;
  t.nrho_e_ = r.nrho_e;
  t.neps_ = r.neps;
  t.nrho_p_ = r.nrho_p;
  t.npres_ = r.npres;
  t.nrho_t_ = r.nrho_t;
  t.ntemp_ = r.ntemp;
  const Real lrd = std::log10(density_factor), lee = std::log10(energy_factor);
  const Real lpp = std::log10(pressure_factor), lcc = std::log10(sound_speed_factor);
  const Real ltt = std::log10(temperature_factor);
  t.rho_e_lr0_ = r.rho_e.front() + lrd;
  t.rho_e_dlr_ = r.rho_e[1] - r.rho_e[0];
  t.rho_e_lx0_ = r.eps.front() + lee;
  t.rho_e_dlx_ = r.eps[1] - r.eps[0];
  t.rho_p_lr0_ = r.rho_p.front() + lrd;
  t.rho_p_dlr_ = r.rho_p[1] - r.rho_p[0];
  t.rho_p_lx0_ = r.pres.front() + lpp;
  t.rho_p_dlx_ = r.pres[1] - r.pres[0];
  t.rho_t_lr0_ = r.rho_t.front() + lrd;
  t.rho_t_dlr_ = r.rho_t[1] - r.rho_t[0];
  t.rho_t_lx0_ = r.temp.front() + ltt;
  t.rho_t_dlx_ = r.temp[1] - r.temp[0];
  t.mean_ion_mass_ = mean_ion_mass;
  Copy2D(t.log_pressure_, r.pressure, r.nrho_e, r.neps, "eos pressure", lpp);
  Copy2D(t.log_temperature_, r.temperature, r.nrho_e, r.neps, "eos temperature", ltt);
  Copy2D(t.log_sound_speed_, r.sound_speed, r.nrho_e, r.neps, "eos sound speed", lcc);
  Copy2D(t.zbar_, r.zbar, r.nrho_e, r.neps, "eos zbar");
  Copy2D(t.log_energy_rho_p_, r.energy_p, r.nrho_p, r.npres, "eos energy(rho,p)", lee);
  Copy2D(t.log_energy_rho_t_, r.energy_t, r.nrho_t, r.ntemp, "eos energy(rho,T)", lee);
  t.diagnostics_ = parthenon::ParArray1D<int>("tabulated eos diagnostics",
                                               static_cast<int>(ndiagnostics));
  Kokkos::deep_copy(t.diagnostics_, 0);
  t.fingerprint_ = Hash(r, {density_factor, energy_factor, pressure_factor,
                            sound_speed_factor, temperature_factor, mean_ion_mass});
  return t;
}

std::string TabulatedEOS::Fingerprint() const {
  std::ostringstream fp;
  fp << std::hex << fingerprint_;
  return fp.str();
}

void TabulatedEOS::ResetDiagnostics() const { Kokkos::deep_copy(diagnostics_, 0); }

void TabulatedEOS::CheckDiagnostics(const std::string &where) const {
  auto h = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), diagnostics_);
  std::array<long long, ndiagnostics> counts{};
  for (int i = 0; i < ndiagnostics; ++i) counts[i] = h(i);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, counts.data(), ndiagnostics, MPI_LONG_LONG,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
  if (counts[invalid_input] > 0 || counts[far_out_of_bounds] > 0) {
    std::ostringstream msg;
    msg << "Tabulated EOS fatal diagnostic after " << where << ": invalid="
        << counts[invalid_input] << ", more-than-one-decade="
        << counts[far_out_of_bounds];
    PARTHENON_FAIL(msg.str());
  }
  long long outside = 0;
  for (int i = rho_low; i <= temperature_high; ++i) outside += counts[i];
  if (outside > 0) {
    std::ostringstream msg;
    msg << "Tabulated EOS extrapolated " << outside << " query coordinates after " << where;
    PARTHENON_WARN(msg.str());
  }
  ResetDiagnostics();
}
