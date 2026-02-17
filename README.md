![CI](https://github.com/wp-labs/wp-arrow/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/wp-labs/wp-arrow/graph/badge.svg?token=6SVCXBHB6B)](https://codecov.io/gh/wp-labs/wp-arrow)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

# wp-arrow

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
