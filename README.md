# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# [zero-keras](https://github.com/SamuelMarks/zero-keras)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-keras/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-keras/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-96.3%25-green.svg)](https://github.com/SamuelMarks/zero-keras/actions/workflows/ci.yml)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](https://github.com/SamuelMarks/zero-keras/tree/master/docs)

## System Architecture & Purpose

**zero-keras** is a 1:1 API-compatible shell for the Keras framework. It exists to provide the exact same user-facing frontend API, abstractions, and syntactic sugar as native Keras, but it routes all underlying mathematical operations, automatic differentiation, and graph execution down to the unified [`ml-switcheroo-compiler`](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend.

This architectural split allows developers to write standard Keras code while taking advantage of our compiler's multiple internal execution backends (`numpy`, `jax`, `mlx`, `cupy`, `dusk`, `torch`). 

To ensure absolute reliability, the [`zero-zoo`](https://github.com/SamuelMarks/zero-zoo) verification tier rigorously tests **zero-keras** against the native Keras implementation. It injects identical seeds and inputs into both frameworks and asserts float-for-float parity across complex model architectures during both forward and backward passes.

```mermaid
graph TD
    subgraph "Verification Tier"
        ZZ[zero-zoo / The Model Zoo]
    end

    subgraph "Native Frameworks"
        KERAS[Native Keras]
    end

    subgraph "API-Compatible Shells"
        ZJ[zero-jax]
        ZF[zero-flax]
        ZP[zero-pytorch]
        ZK[zero-keras]
        ZT[zero-tensorflow]
        ZM[zero-mlx]
        ZPX[zero-pax]
    end

    subgraph "Compilation Core"
        COMP[ml-switcheroo-compiler]
    end

    subgraph "Internal Backends"
        NUMPY[numpy]
        JAX_B[jax]
        MLX_B[mlx]
        CUPY[cupy]
        DUSK[dusk]
        TORCH_B[torch]
    end

    ZZ -.->|Validates Float Equivalence| KERAS
    ZZ -.->|Validates Float Equivalence| ZJ
    ZZ -.->|Validates Float Equivalence| ZF
    ZZ -.->|Validates Float Equivalence| ZP
    ZZ -.->|Validates Float Equivalence| ZK
    ZZ -.->|Validates Float Equivalence| ZT
    ZZ -.->|Validates Float Equivalence| ZM
    ZZ -.->|Validates Float Equivalence| ZPX

    ZJ --> COMP
    ZF --> ZJ
    ZP --> COMP
    ZK --> COMP
    ZT --> ZK
    ZM --> COMP
    ZPX --> ZJ
    
    COMP --> NUMPY
    COMP --> JAX_B
    COMP --> MLX_B
    COMP --> CUPY
    COMP --> DUSK
    COMP --> TORCH_B

    style ZK fill:#D00000,stroke:#333,stroke-width:2px,color:#fff
    style KERAS fill:#D00000,stroke:#333,stroke-width:2px,color:#fff
```

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0), shall be
dual licensed as above, without any additional terms or conditions.
