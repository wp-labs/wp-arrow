# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/wp-labs/wp-arrow/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/wp-labs/wp-arrow/compare/v0.2.0...v0.3.0
