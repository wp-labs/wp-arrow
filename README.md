# wp-arrow

[![Crates.io](https://img.shields.io/crates/v/wp-arrow.svg)](https://crates.io/crates/wp-arrow)
[![CI](https://img.shields.io/github/actions/workflow/status/wp-labs/wp-arrow/ci.yml?branch=main)](https://github.com/wp-labs/wp-arrow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wp-labs/wp-arrow/graph/badge.svg?token=6SVCXBHB6B)](https://codecov.io/gh/wp-labs/wp-arrow)
[![Crates.io downloads](https://img.shields.io/crates/d/wp-arrow)](https://crates.io/crates/wp-arrow)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust Edition](https://img.shields.io/badge/edition-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)

Apache Arrow utilities for wp-model, providing schema mapping, data conversion, and IPC support.

## What this crate is (read before touching type mapping)

`wp-arrow` is the **home of the `wparse(sink) ↔ wfusion(receiver)` Arrow column-type contract**.
The contract's single source of truth is the spec table in
[`wp-reactor/docs/design/arrow-type-mapping.md`](https://github.com/wp-labs/wp-reactor/blob/main/docs/design/arrow-type-mapping.md);
this crate is where its implementation is meant to live.

**Current state (2026-09):** the contract's schema table now **also** lives here as
[`contract::wp_type_to_arrow`](https://docs.rs/wp-arrow/latest/wp_arrow/contract/fn.wp_type_to_arrow.html)
(exhaustive over the 37 variants of `wp_model_core::model::DataType`, with its own pinning test) —
that is migration step A-2/2a, a **pure addition**.

**Not switched over yet:** `wp-connector-utils` still holds its own copy of that table, and the
receiver side (`wf-runtime`) derives its expectations from it. Step A-2/2b is to make
`wp-connector-utils` delegate here and drop its copy; that step is **gated on publishing a new
version of this crate** (a published crate may only depend on crates.io versions). Until then,
the *effective* implementation stays in `wp-connector-utils`.

> ⚠️ **Do not use `schema` / `convert` (the 9-variant typed front-end) to judge wire-contract
> consistency** — the contract is `contract`. `WpDataType`'s `Array` → `List(inner)` and
> `BigInt` → `Decimal256(39,0)` are its own choices, deliberately different from the wire contract
> (which is conservative: structured fields and big integers are `Utf8`).
> `wp-labs/warp-fusion#102` was exactly the opposite misjudgement: "find the authoritative
> implementation from the self-description → land on the typed front-end → report an inconsistency".

The full migration plan is the A-2 section of the spec table.

## Modules

- **contract** - Wire-contract mapping: `wp_model_core::model::DataType` (37 variants, exhaustive) → Arrow column type. **This is `wparse` ↔ `wfusion`'s Arrow contract.**
- **schema** - Arrow schema definitions and mapping from wp-model types (9-variant `WpDataType` **typed front-end**, not the contract)
- **convert** - Data conversion between wp-model and Arrow arrays (currently unused; to be merged with `wp-connector-utils`' value layer in A-2/2c)
- **ipc** - Arrow IPC serialization and deserialization (`[4B tag_len BE][tag][Arrow IPC stream]` frame = the de-facto wire format used by the family's live producer/consumer, which currently implement it independently)
- **error** - Error types

## Dependencies

- [Apache Arrow](https://docs.rs/arrow) (IPC feature)
- [wp-model-core](https://github.com/wp-labs/wp-model-core)
- [chrono](https://docs.rs/chrono)

## License

Apache-2.0
