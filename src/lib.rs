//! wp-model ↔ Apache Arrow 工具库：schema 映射、值/列转换、IPC 帧。
//!
//! # 定位（改映射前先读这段）
//!
//! **本 crate 是「wparse（sink 侧）↔ wfusion（接收侧）Arrow 列类型契约」的归属地。**
//! 契约的单一事实来源（规格表 + 已知差异登记）在
//! [`wp-reactor/docs/design/arrow-type-mapping.md`](https://github.com/wp-labs/wp-reactor/blob/main/docs/design/arrow-type-mapping.md)，
//! 本 crate 是它的**实现落点**。
//!
//! # 当前状态（2026-09，A-2 已完成 2a/2b/2c）
//!
//! - **契约的两层都在本 crate 的 [`contract`]**：列类型表（[`contract::wp_type_to_arrow`]，穷尽
//!   `wp_model_core::model::DataType` 的 37 个变体）与值层（[`contract::value`]，`DataRecord` → 列）。
//! - **生产路径**：`wp-connector-utils` 的 `arrow::wp_type_to_arrow` 与 `arrow::record`
//!   已改为**转发**到这里（公开路径与签名不变），经 `wf-runtime` 到达生产。
//! - [`schema`] / [`convert`]（9 变体的**类型化前端**，`Array` → `List`、`BigInt` → `Decimal256`）
//!   **不是契约**，仍不在生产路径上（全家族生产调用 0 处）。
//!
//! ⚠️ **不要用 [`schema`] / [`convert`] 的口径去判断线协议是否一致** —— 契约看 [`contract`]。
//! 历史上 `wp-labs/warp-fusion#102` 正是「按自我声明找权威实现 → 找到类型化前端 →
//! 报口径不一致」这一误判。
//!
//! 完整迁移记录见规格表 A-2 一节。

pub mod contract;
pub mod convert;
pub mod error;
pub mod ipc;
pub mod schema;

pub use contract::wp_type_to_arrow;
