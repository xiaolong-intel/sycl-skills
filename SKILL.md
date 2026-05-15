---
name: hardware-info
description: "Measure Intel GPU hardware specifications using SYCL. USE FOR: querying GPU device info, memory hierarchy, bandwidth estimates, compute throughput, roofline analysis ridge points, architecture constants for Xe2-HPG. Compiles and runs device_info.cpp to produce a full hardware report."
---

# Hardware Info Measurement

This skill compiles and runs a SYCL program to measure Intel GPU hardware specifications, including compute units, memory hierarchy, theoretical bandwidth, peak compute, and roofline ridge points.

## Prerequisites

- Intel oneAPI DPC++ compiler (`icpx`) available in PATH
- Intel GPU with Level-Zero runtime installed
- Environment sourced: `source /opt/intel/oneapi/setvars.sh` (or equivalent)

## Steps

### 1. Compile the device_info program

```bash
cd /xiaolong/sycl-skills/hardware_info
icpx -fsycl -o device_info device_info.cpp
```

### 2. Run the program

```bash
./device_info
```

### 3. Interpret the output

The program reports the following sections:

| Section | Key Metrics |
|---------|-------------|
| **Device Info** | GPU name, vendor, driver version, XVE count, Xe-cores, max clock, sub-group sizes |
| **Memory Hierarchy** | Global memory (VRAM), L2 cache size, SLM per Xe-core, cache line size |
| **Bandwidth (Theoretical Peak)** | VRAM BW, L2 BW, SLM BW, GRF BW |
| **Compute (Theoretical Peak)** | FP32 TFLOPS, FP16 TFLOPS, INT8 TOPS |
| **Roofline Key Points** | Ridge point for VRAM / L2 / SLM (FLOP/Byte threshold for compute-bound vs memory-bound) |
| **Architecture Constants** | GRF per thread, register width, threads per XVE, L1 cache estimate |

### 4. Using the results for kernel optimization

- **Arithmetic Intensity > Ridge Point** → kernel is compute-bound → optimize ALU utilization
- **Arithmetic Intensity < Ridge Point** → kernel is memory-bound → optimize data locality / reduce memory traffic
- Use the bandwidth hierarchy (GRF >> SLM >> L2 >> VRAM) to decide which memory level to target for data reuse

## Output Example (Intel Arc B60, 8-card system)

```
Found 8 GPU(s) (Level-Zero backend)

============================================
  Device Info (Card 0, all cards identical)
============================================
Name:                 Intel(R) Graphics [0xe211]
Vendor:               Intel(R) Corporation
Driver Version:       1.6.33578+15
Max Compute Units:    160 (= XVE count)
Xe-cores:             20 (= XVE / 8)
Max Work Group Size:  1024
Max Clock Freq(MHz):  2400
Max Work Item Sizes:  1024 x 1024 x 1024
Sub-Group Sizes:      16 32 (= supported SIMD widths)
Max Sub-Groups/WG:    64

============================================
  Memory Hierarchy
============================================
Global Memory:        23256 MB (VRAM)
L2 Cache (shared):    18432 KB
SLM (per Xe-core):    128 KB
Cache Line Size:      1 bytes
Mem Address Bits:     64

============================================
  Bandwidth (Theoretical Peak)
============================================
VRAM Bus Width:       192 bit (from hardware)
VRAM Clock Rate:      4750 MHz
VRAM Data Rate:       19.0 Gbps
VRAM Bandwidth:       456.0 GB/s (per card)
L2 BW (per Xe-core):  153.6 GB/s
L2 BW (total):        3072.0 GB/s
SLM Read BW/Xe-core:  153.6 GB/s
SLM Write BW/Xe-core: 76.8 GB/s
SLM Total BW (all):   3072.0 GB/s (read)
GRF BW (per XVE):     307.2 GB/s
GRF BW (total):       49.2 TB/s

============================================
  Compute (Theoretical Peak)
============================================
FP32:                 6.1 TFLOPS
FP16:                 12.3 TFLOPS
INT8:                 24.6 TOPS
FP32 (8-card total):  49.2 TFLOPS

============================================
  Roofline Key Points (FP32)
============================================
Ridge Point (VRAM):   13.5 FLOP/Byte
Ridge Point (L2):     2.0 FLOP/Byte
Ridge Point (SLM):    2.0 FLOP/Byte

=> Kernel arithmetic intensity > Ridge Point => compute-bound
=> Kernel arithmetic intensity < Ridge Point => memory-bound

============================================
  Memory Bandwidth Hierarchy Summary
============================================
  GRF (Register) :  49.2 TB/s
  SLM (Local Mem):  3.1 TB/s
  L2 Cache       :  3072.0 GB/s
  VRAM (Global)  :  456.0 GB/s
  --------------------------------
  GRF >> SLM >> L2 >> VRAM (fast to slow)

============================================
  Architecture Constants (Xe2-HPG)
============================================
GRF per thread:       128 (default) / 256 (large GRF)
GRF register width:   32 bytes (256-bit)
GRF total per thread: 128 x 32B = 4 KB (default)
Threads per XVE:      8 (default) / 4 (large GRF)
L1 Cache/Xe-core:     ~192 KB (estimated)
Total GPUs:           8
```

## Notes

- The program auto-detects hardware via Level-Zero backend and filters out duplicate OpenCL devices.
- If the driver returns invalid VRAM bus width or clock rate, it falls back to known B60 specs (192-bit GDDR6, 19 Gbps).
- All bandwidth/compute numbers are **theoretical peaks**; real-world performance will be lower due to occupancy, bank conflicts, etc.
