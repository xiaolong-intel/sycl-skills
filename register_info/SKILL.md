---
name: register-info
description: "Diagnose register pressure, spill, and GRF allocation for Intel GPU SYCL kernels. USE FOR: checking register spill count, comparing large vs small GRF modes, dumping ISA assembly, analyzing compiler register allocation decisions, fixing register spill performance issues on Xe2-HPG."
---

# Register & Compiler Diagnostics (Xe2-HPG)

This skill provides tools and techniques to diagnose register pressure issues in SYCL kernels targeting Intel Xe2-HPG GPUs.

## Prerequisites

- Intel oneAPI DPC++ compiler (`icpx`) with `-fsycl`
- Intel GPU with Level-Zero runtime
- Environment: `source /opt/intel/oneapi/setvars.sh`

## 1. GRF Architecture Overview

| Property | Xe2-HPG (B60) |
|----------|---------------|
| GRF per thread (small mode) | 128 registers × 32B = 4 KB |
| GRF per thread (large mode) | 256 registers × 32B = 8 KB |
| Threads per XVE (small GRF) | 8 |
| Threads per XVE (large GRF) | 4 |
| Total GRF per XVE | 32 KB |

**Trade-off**: Large GRF = more registers per thread (less spill) but halved occupancy (fewer threads to hide latency).

## 2. Controlling GRF Mode

### 2.1 Compiler Flag

```bash
# Auto (compiler decides per kernel)
icpx -fsycl -ftarget-register-alloc-mode=pvc:auto -o kernel kernel.cpp

# Force small GRF (128 regs, high occupancy)
icpx -fsycl -ftarget-register-alloc-mode=pvc:small -o kernel kernel.cpp

# Force large GRF (256 regs, less spill)
icpx -fsycl -ftarget-register-alloc-mode=pvc:large -o kernel kernel.cpp
```

Note: `pvc` target is also used for Xe2-HPG in current icpx versions.

### 2.2 Per-Kernel Attribute (preferred)

```cpp
// Request large GRF for a register-heavy kernel
[[intel::reqd_work_group_size(256)]]
[[intel::register_alloc_mode(large)]]  // if supported
void my_kernel(sycl::nd_item<1> it) { ... }
```

### 2.3 Decision Guide

| Kernel Type | Recommended GRF | Reason |
|-------------|----------------|--------|
| Memory-bound, simple | Small (128) | Need occupancy to hide latency |
| Compute-bound, many accumulators | Large (256) | Avoid spill which adds memory traffic |
| Mixed (e.g., fused GEMM + post-ops) | Auto or Large | Let compiler decide, verify with ISA dump |

## 3. Dumping ISA Assembly

### 3.1 Enable IGC Shader Dump

```bash
# Set environment variables before running
export IGC_ShaderDumpEnable=1
export IGC_DumpToCurrentDir=1

# Compile and run — ISA will be dumped to current directory
icpx -fsycl -O2 -o kernel kernel.cpp
./kernel

# Look for .asm files
ls *.asm 2>/dev/null || ls /tmp/IntelIGC/*.asm 2>/dev/null
```

### 3.2 Alternative: Use `-save-temps`

```bash
icpx -fsycl -O2 -save-temps -o kernel kernel.cpp
# Generates .spv (SPIRV) and other intermediate files
```

### 3.3 Using `ocloc` to disassemble

```bash
# If you have the .spv file
ocloc disasm -file kernel.spv -device bmg
```

## 4. Detecting Register Spill

### 4.1 What is Spill

When a kernel uses more live registers than available GRF, the compiler "spills" values to scratch memory (off-chip VRAM). Each spill = extra load/store to global memory.

**Performance impact**: A spill in a hot loop can cost 100+ cycles per iteration (VRAM latency).

### 4.2 Checking Spill in ISA

```bash
# After dumping ISA, search for scratch/spill operations
grep -c "scratch" *.asm        # count scratch (spill) operations
grep -c "spill\|fill" *.asm    # spill = write to scratch, fill = read back
```

### 4.3 Using IGC Stats

```bash
export IGC_ShaderDumpEnable=1
export IGC_EnableVISAOutput=1
export IGC_DumpToCurrentDir=1

./kernel

# Look for stats in the dump
grep -i "spill\|scratch\|GRF" *.dump 2>/dev/null
grep -i "RegPressure" *.dump 2>/dev/null
```

### 4.4 Automated Spill Check Script

```bash
#!/bin/bash
# Usage: ./spill_check.sh <source.cpp> [kernel_name]
SRC=${1:?Usage: spill_check.sh source.cpp}
NAME=${2:-"kernel"}

export IGC_ShaderDumpEnable=1
export IGC_DumpToCurrentDir=1

rm -f *.asm 2>/dev/null
icpx -fsycl -O2 -o /tmp/spill_test "$SRC" && /tmp/spill_test

echo "=== Spill Report ==="
for f in *.asm; do
    spills=$(grep -c "scratch" "$f" 2>/dev/null || echo 0)
    echo "  $f: $spills scratch ops"
done
```

## 5. Reducing Register Pressure

### 5.1 Techniques (ordered by effectiveness)

| Technique | Example | Saves |
|-----------|---------|-------|
| **Reduce live range** | Move variable declaration closer to use | Frees GRF earlier |
| **Stream instead of buffer** | Read `b[i]` and immediately use, don't store all `b[]` first | N×VEC GRF |
| **Reduce VEC width** | VEC=8→4 if register limited | Each halving saves 50% per array |
| **Reduce unroll factor** | `#pragma unroll 2` instead of full unroll | Trades ILP for GRF |
| **Use SLM for intermediates** | Spill explicitly to SLM (faster than scratch) | GRF at cost of SLM BW |
| **Switch to large GRF** | `-ftarget-register-alloc-mode=pvc:large` | 2x registers |

### 5.2 Register Budget Estimation

Quick formula for a SYCL kernel:

```
GRF_used ≈ (float arrays in registers × elements × 4B) / 32B_per_GRF
         + overhead (pointers, loop vars, etc.) ~10-15 GRF
```

**Example (mhc_post, HC=4, VEC=8)**:
- `a_reg[4][4]` = 16 floats = 2 GRF
- `c_reg[4]` = 4 floats = 0.5 GRF
- `acc[4][8]` = 32 floats = 4 GRF
- `bv` (8 bf16 → 8 float temp) = 1 GRF
- `dv` (8 bf16) = 0.5 GRF
- Overhead = ~12 GRF
- **Total ≈ 20 GRF** → 128 GRF mode safe, no spill expected

**Example (mhc_pre, HC=4, FN_BATCH=4, TOKENS_PER_WG=2)**:
- `local_mix[2][24]` = 48 floats = 6 GRF
- `local_sq[2]` = 2 floats = 0.25 GRF
- `fnf[4][8]` = 32 floats = 4 GRF
- `xf[8]` = 8 floats = 1 GRF
- `partials[25]` = 25 floats = ~4 GRF (reuses local_mix memory)
- Overhead = ~15 GRF
- **Total ≈ 30 GRF** → safe in 128 mode

If FN_BATCH increases to 12: `fnf[12][8]` = 96 floats = 12 GRF → total ~42 GRF → still safe.

## 6. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Full unroll of large loop | Huge spill count in ISA | Use `#pragma unroll N` with smaller N |
| Storing all data before computing | Many arrays live simultaneously | Stream: load→compute→discard |
| VEC too large | Spill in inner loop | Reduce VEC or switch to large GRF |
| Compiler inlines too aggressively | Code bloat → register pressure | Mark helper functions `__attribute__((noinline))` |
| Auto-vectorization conflict | Compiler widens SIMD unexpectedly | Use `[[intel::reqd_sub_group_size(16)]]` |

## 7. Quick Commands Reference

```bash
# Compile with optimization report
icpx -fsycl -O2 -Rpass=loop-unroll -Rpass-missed=loop-vectorize -o kernel kernel.cpp

# Compile with large GRF
icpx -fsycl -O2 -ftarget-register-alloc-mode=pvc:large -o kernel kernel.cpp

# Dump ISA and check spill
IGC_ShaderDumpEnable=1 IGC_DumpToCurrentDir=1 ./kernel
grep -c "scratch" *.asm

# Compare small vs large GRF performance
icpx -fsycl -O2 -ftarget-register-alloc-mode=pvc:small -o k_small kernel.cpp
icpx -fsycl -O2 -ftarget-register-alloc-mode=pvc:large -o k_large kernel.cpp
echo "Small GRF:" && ./k_small
echo "Large GRF:" && ./k_large
```
