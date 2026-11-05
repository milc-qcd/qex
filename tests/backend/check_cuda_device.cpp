// Host-side preflight: CPU execution must not count as CUDA GPU validation.
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2 || (std::string(argv[1]) != "H100" && std::string(argv[1]) != "A100")) {
    std::cerr << "Usage: check_cuda_device H100|A100\n";
    return 2;
  }
  const std::string expected = argv[1];
  try {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
      std::cerr << "No GPU is visible through the selected SYCL runtime.\n";
      return 1;
    }
    for (const auto &device : devices) {
      const auto name = device.get_info<sycl::info::device::name>();
      if (device.get_backend() != sycl::backend::ext_oneapi_cuda ||
          device.get_info<sycl::info::device::vendor_id>() != 0x10de ||
          name.find(expected) == std::string::npos ||
          !device.has(sycl::aspect::fp64) ||
          !device.has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "Expected an NVIDIA " << expected << " with CUDA, FP64 and device USM; got "
                  << name << '\n';
        return 1;
      }
      std::cout << expected << " device: " << name << '\n';
    }
    std::cout << "PASS: " << expected << " device preflight (" << devices.size() << " visible)\n";
    return 0;
  } catch (const sycl::exception &error) {
    std::cerr << "SYCL " << expected << " preflight failed: " << error.what() << '\n';
    return 1;
  }
}
