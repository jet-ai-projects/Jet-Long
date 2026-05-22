// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

torch::Tensor jetlong_merge2_bshd_cuda(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b);

torch::Tensor jetlong_merge2_bshd_out_cuda(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b,
    torch::Tensor out);

torch::Tensor jetlong_merge2_bshd(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b) {
  TORCH_CHECK(out_a.is_cuda(), "out_a must be a CUDA tensor");
  return jetlong_merge2_bshd_cuda(out_a, lse_a, out_b, lse_b);
}

torch::Tensor jetlong_merge2_bshd_out(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b,
    torch::Tensor out) {
  TORCH_CHECK(out_a.is_cuda() && out.is_cuda(), "out_a/out must be CUDA tensors");
  return jetlong_merge2_bshd_out_cuda(out_a, lse_a, out_b, lse_b, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge2_bshd", &jetlong_merge2_bshd, "Two-branch LSE merge for BSHD tensors");
  m.def("merge2_bshd_out", &jetlong_merge2_bshd_out, "Two-branch LSE merge for BSHD tensors with preallocated output");
}
