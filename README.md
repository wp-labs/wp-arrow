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
