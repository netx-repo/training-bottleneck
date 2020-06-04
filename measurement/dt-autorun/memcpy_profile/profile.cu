#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>

long avgTime(std::vector<long> times) {
  long long total = 0;
  for (double t : times) {
    total += t;
  }
  return total / times.size();
}

std::vector<int> readLayerSize(std::string logpath) {
  std::ifstream infile(logpath);
  if(!infile.good()) {
    std::cout << "open file " << logpath << "error\n";
    std::vector<int> empty;
    return empty;
  }
  std::string line;
  std::vector<int> sizes;
  while (std::getline(infile, line)) {
    sizes.push_back(std::stoi(line));
  }
  return sizes;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage" << argv[0] << "<size> <repeat-times>\n";
    return 1;
  }
  std::string arg1 = argv[1];
  std::string arg2 = argv[2];
  const unsigned int N = std::stoi(arg1);
  const unsigned int bytes = N * sizeof(float);
  std::cout << "transfer data size: " << bytes << " bytes" << std::endl;
  int *h_a = (int *)malloc(bytes);
  int *d_a;
  cudaMalloc((int **)&d_a, bytes);

  memset(h_a, 0, bytes);
  std::vector<long> h2d_times;
  std::vector<long> d2h_times;
  for (int i = 0; i < stoi(arg2); i++) {
    auto s = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    auto e = std::chrono::high_resolution_clock::now();
    h2d_times.push_back((e - s).count());

    s = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    e = std::chrono::high_resolution_clock::now();
    d2h_times.push_back((e - s).count());
  }

  std::cout << "Host to Device memcopy " << avgTime(h2d_times) << " ns\n";
  std::cout << "Device to Host " << avgTime(d2h_times) << " ns\n";

  return 0;
}