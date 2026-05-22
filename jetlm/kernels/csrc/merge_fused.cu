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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float read_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float read_as_float<at::Half>(const at::Half* ptr, int64_t offset) {
  const auto* half_ptr = reinterpret_cast<const __half*>(ptr);
  return __half2float(half_ptr[offset]);
}

template <>
__device__ __forceinline__ float read_as_float<at::BFloat16>(const at::BFloat16* ptr, int64_t offset) {
  const auto* bf16_ptr = reinterpret_cast<const __nv_bfloat16*>(ptr);
  return __bfloat162float(bf16_ptr[offset]);
}

template <typename scalar_t>
__device__ __forceinline__ void write_from_float(scalar_t* ptr, int64_t offset, float value) {
  ptr[offset] = static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ void write_from_float<at::Half>(at::Half* ptr, int64_t offset, float value) {
  auto* half_ptr = reinterpret_cast<__half*>(ptr);
  half_ptr[offset] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void write_from_float<at::BFloat16>(at::BFloat16* ptr, int64_t offset, float value) {
  auto* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(ptr);
  bf16_ptr[offset] = __float2bfloat16(value);
}

template <typename scalar_t>
__device__ __forceinline__ float round_like_scalar(float value) {
  return value;
}

template <>
__device__ __forceinline__ float round_like_scalar<at::Half>(float value) {
  return __half2float(__float2half_rn(value));
}

template <>
__device__ __forceinline__ float round_like_scalar<at::BFloat16>(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

template <typename scalar_t>
__global__ void merge2_bshd_kernel(
    const scalar_t* __restrict__ out_a,
    const float* __restrict__ lse_a,
    const scalar_t* __restrict__ out_b,
    const float* __restrict__ lse_b,
    scalar_t* __restrict__ out,
    int64_t total,
    int64_t seqlen,
    int64_t nheads,
    int64_t head_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t tmp0 = idx / head_dim;
  const int64_t h = tmp0 % nheads;
  const int64_t tmp1 = tmp0 / nheads;
  const int64_t s = tmp1 % seqlen;
  const int64_t b = tmp1 / seqlen;
  const int64_t lse_idx = (b * nheads + h) * seqlen + s;

  const float la = lse_a[lse_idx];
  const float lb = lse_b[lse_idx];
  const float m = fmaxf(la, lb);
  const float wa = expf(la - m);
  const float wb = expf(lb - m);
  const float denom = fmaxf(wa + wb, 1.0e-7f);
  const float av = read_as_float(out_a, idx);
  const float bv = read_as_float(out_b, idx);
  const float wa_norm = round_like_scalar<scalar_t>(wa / denom);
  const float wb_norm = round_like_scalar<scalar_t>(wb / denom);
  const float term_a = round_like_scalar<scalar_t>(wa_norm * av);
  const float term_b = round_like_scalar<scalar_t>(wb_norm * bv);
  write_from_float(out, idx, round_like_scalar<scalar_t>(term_a + term_b));
}

void check_bshd_inputs(const torch::Tensor& out, const torch::Tensor& lse) {
  TORCH_CHECK(out.dim() == 4, "output tensors must be BSHD: (batch, seqlen, nheads, head_dim)");
  TORCH_CHECK(lse.dim() == 3, "lse tensors must be BHS: (batch, nheads, seqlen)");
  TORCH_CHECK(out.is_cuda() && lse.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(out.is_contiguous(), "output tensors must be contiguous BSHD tensors");
  TORCH_CHECK(lse.is_contiguous(), "lse tensors must be contiguous BHS tensors");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse tensors must be float32");
  TORCH_CHECK(lse.size(0) == out.size(0), "lse batch size mismatch");
  TORCH_CHECK(lse.size(1) == out.size(2), "lse head count mismatch");
  TORCH_CHECK(lse.size(2) == out.size(1), "lse seqlen mismatch");
}

void check_like_contiguous(const torch::Tensor& out, const torch::Tensor& ref, const char* name) {
  TORCH_CHECK(out.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(out.sizes() == ref.sizes(), name, " shape mismatch");
  TORCH_CHECK(out.scalar_type() == ref.scalar_type(), name, " dtype mismatch");
}

}  // namespace

torch::Tensor jetlong_merge2_bshd_out_cuda(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b,
    torch::Tensor out) {
  check_bshd_inputs(out_a, lse_a);
  check_bshd_inputs(out_b, lse_b);
  TORCH_CHECK(out_b.sizes() == out_a.sizes(), "output shape mismatch");
  TORCH_CHECK(lse_b.sizes() == lse_a.sizes(), "lse shape mismatch");
  TORCH_CHECK(out_b.scalar_type() == out_a.scalar_type(), "dtype mismatch");
  check_like_contiguous(out, out_a, "out");

  const c10::cuda::CUDAGuard device_guard(out_a.device());
  const int64_t total = out.numel();
  constexpr int threads = 512;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, out_a.scalar_type(), "jetlong_merge2_bshd_cuda", [&] {
    merge2_bshd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        out_a.data_ptr<scalar_t>(),
        lse_a.data_ptr<float>(),
        out_b.data_ptr<scalar_t>(),
        lse_b.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        total,
        out_a.size(1),
        out_a.size(2),
        out_a.size(3));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor jetlong_merge2_bshd_cuda(
    torch::Tensor out_a,
    torch::Tensor lse_a,
    torch::Tensor out_b,
    torch::Tensor lse_b) {
  auto out = torch::empty_like(out_a);
  return jetlong_merge2_bshd_out_cuda(out_a, lse_a, out_b, lse_b, out);
}
