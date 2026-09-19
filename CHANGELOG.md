# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-09-19

### Added

- **契约的值层迁入 `contract::value`（A-2 第 3 步 / 2c）**：`DataRecord` → 列 的编码
  （`encode_record` / `encode_records`）从 `wp-connector-utils/src/arrow/record.rs`
  **逐字移植**（只把错误类型从 `SinkResult`/`SinkReason` 换成 `WpArrowError`；派发仍
  **按 Arrow 列类型**，所以口径仍由 `contract::wp_type_to_arrow` 单点决定）。
  至此线协议契约的两层 —— 列类型表 + 值编码 —— 都在本 crate。
- 新增依赖 `serde_json`：结构化字段（`Obj`/`Array`）按 JSON 文本写入 Utf8 列，
  与移植前口径一致。
- 等价性凭据：同一份「金标准」夹具与期望在**迁移前**（`wp-connector-utils` 的实现）
  与**迁移后**（`contract::value`）两处各有一份且同时通过
  （`wire_value_encoding_is_pinned_by_golden_values`）→ 搬迁是等价改动，
  而不只是「看起来一样」。

### Changed

- `contract` 拆为 `contract/mod.rs`（列类型表）+ `contract/value.rs`（值层），两者均公开；
  `contract::{encode_record, encode_records}` 是值层的稳定入口。

### Tests

- 73 → 76 项测试。

## [0.4.1] - 2026-09-19

### Added

#### 线协议契约表：`contract::wp_type_to_arrow`（A-2 第 1 步）

- 新增 `contract` 模块：`wp_model_core::model::DataType`（穷尽 **37 变体**，无 `_` 兜底）
  → Arrow 列类型。它是 **wparse（sink）↔ wfusion（接收）Arrow 列类型契约**的实现，
  口径表见 `wp-reactor/docs/design/arrow-type-mapping.md` §3。
- 该入口不依赖任何 connector crate，也不依赖本 crate 的 `WpDataType`——入参就是
  `wp_model_core::model::DataType`。
- 测试：全 37 变体钉桩（`wire_contract_full_mapping_is_pinned`）、
  `hex` 必须为 `Utf8` 的 DIV-1 回归（`hex_must_be_utf8`）、
  以及「契约 ≠ 9 变体类型化前端」的防呆两条
  （`wire_contract_differs_from_the_typed_frontend_on_two_rows`、`wire_contract_and_frontend_agree_on_the_overlap`）。

### Changed

- 文档定位澄清（不涉及行为）：本 crate 是 wp-model ↔ Arrow **契约**的归属地；
  `schema` / `convert`（9 变体的 `WpDataType`）是**类型化前端**，其 `Array` → `List(inner)`、
  `BigInt` → `Decimal256(39,0)` 与线协议无关——契约口径是保守的（结构化与大整数一律 `Utf8`）。
  这是对 `wp-labs/warp-fusion#102` 那类「按自我声明找权威实现」误判的防呆。
- 版本 0.4.0 → 0.4.1（纯新增，不改现有行为；`^0.4` 的下游无需改版本要求）。

### Tests

- 69 → 73 项测试全部通过（新增 4 项 `contract` 测试）。

## [0.4.0] - 2026-09-19

### ⚠️ BREAKING CHANGES

- 依赖 `arrow` 59 → 60（arrow 类型出现在公开 API，大版本升级即破坏性变更）
- 依赖 `wp-model-core` 0.9 → 0.10（整型正名：`Value::Digit` → `Value::Int`、`DataType::Digit` → `DataType::Int`）

### Changed

- 跟进 wp-model-core 0.10 正名：`Value::Int` / `DataType::Int` / `Field::from_int`（编解码两侧与测试）
- `wp_type_to_model_meta`：`WpDataType::Digit` → `DataType::Int`；`DataType::Array` 改用 `ArraySubtype`
- README 徽章补齐为 6 枚并移至标题下方
- LICENSE 补齐 Apache-2.0 版权行
- 版本 0.3.1 → 0.4.0

### Dependencies

- `arrow`：`59` → `60`
- `wp-model-core`：`0.9` → `0.10`（`num-bigint` 保持 `0.4`，与上游 `BigUint` 对齐）

### Tests

- 既有 69 项测试在 arrow 60 / wp-model-core 0.10 下全部通过

## [0.3.0] - 2026-08-04

### ⚠️ BREAKING CHANGES

- 依赖 `wp-model-core` 0.8 → 0.9（上游新增 `Value::BigUint` / `DataType::BigInt` 变体）

### Added

#### 任意精度整数（`Value::BigUint`）Arrow 传输

- `WpDataType::BigInt` 变体：IPv4/IPv6 统一数值键等超出 `i64` 范围的整数可经 Arrow 通道无损传输
- `to_arrow_type(BigInt)` → `Decimal256(39, 0)`：保留数值语义，精度 39 位足以表示 `2^129-1`（IPv6 统一键上限）
- `build_bigint_column`：`BigUint → i256`（借道十进制字符串，无损）
- `build_list_bigint`：`array<bigint>` → `List<Decimal256(39, 0)>`
- `extract_value(BigInt)`：`i256 → 十进制 → BigUint`，meta 类型保留为 `DataType::BigInt`
- `parse_wp_type("bigint")` 支持
- 常量 `BIGINT_DECIMAL_PRECISION = 39`（未来编码位数变化时需同步调整）

### Dependencies

- `wp-model-core`：`0.8` → `0.9`
- 新增 `num-bigint = "0.4"`（与 `Value::BigUint` 互操作）

### Tests

- `arrow_type_bigint`：`Decimal256(39, 0)` 映射断言
- `parse_bigint`：`"bigint"` / `"BIGINT"` 解析
- `roundtrip_bigint_ipv6_key`：IPv4/IPv6 统一键无损往返（含 IPv6 键 `382824323044708348099391746388336347272`）+ meta 保留
- `roundtrip_bigint_list`：`array<bigint>` 往返

### Changed

- 版本 0.2.0 → 0.3.0

[Unreleased]: https://github.com/wp-labs/wp-arrow/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/wp-labs/wp-arrow/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/wp-labs/wp-arrow/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/wp-labs/wp-arrow/compare/v0.3.1...v0.4.0
[0.3.0]: https://github.com/wp-labs/wp-arrow/compare/v0.2.0...v0.3.0
