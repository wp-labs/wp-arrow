# wp-arrow

[![Crates.io](https://img.shields.io/crates/v/wp-arrow.svg)](https://crates.io/crates/wp-arrow)
[![CI](https://img.shields.io/github/actions/workflow/status/wp-labs/wp-arrow/ci.yml?branch=main)](https://github.com/wp-labs/wp-arrow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wp-labs/wp-arrow/graph/badge.svg?token=6SVCXBHB6B)](https://codecov.io/gh/wp-labs/wp-arrow)
[![Crates.io downloads](https://img.shields.io/crates/d/wp-arrow)](https://crates.io/crates/wp-arrow)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust Edition](https://img.shields.io/badge/edition-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)

Apache Arrow utilities for wp-model, providing schema mapping, data conversion, and IPC support.

## Modules

- **schema** - Arrow schema definitions and mapping from wp-model types
- **convert** - Data conversion between wp-model and Arrow arrays
- **ipc** - Arrow IPC serialization and deserialization
- **error** - Error types

## Dependencies

- [Apache Arrow](https://docs.rs/arrow) (IPC feature)
- [wp-model-core](https://github.com/wp-labs/wp-model-core)
- [chrono](https://docs.rs/chrono)

## License

Apache-2.0
