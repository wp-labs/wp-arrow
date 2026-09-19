//! wp-model ↔ Apache Arrow 工具库：schema 映射、值/列转换、IPC 帧。
//!
//! # 定位（改映射前先读这段）
//!
//! **本 crate 是「wparse（sink 侧）↔ wfusion（接收侧）Arrow 列类型契约」的归属地。**
//! 契约的单一事实来源（规格表 + 已知差异登记）在
//! [`wp-reactor/docs/design/arrow-type-mapping.md`](https://github.com/wp-labs/wp-reactor/blob/main/docs/design/arrow-type-mapping.md)，
//! 本 crate 是它的**实现落点**。
//!
//! # 当前状态（2026-09）
//!
//! - **契约的 schema 表已在本 crate**： [`contract::wp_type_to_arrow`]（穷尽
//!   `wp_model_core::model::DataType` 的 37 个变体），A-2 第 1 步已落地。
//! - **尚未切换**：`wp-connector-utils` 仍持有自己的一份，接收侧（`wf-runtime`）也经它推导；
//!   第 2 步是让前者改为转发到本模块、并把契约的**值层**（`DataRecord` → 列）一并搬来
//!   —— 需要先发布本 crate（跨仓发布顺序见规格表 §5）。
//! - [`schema`] / [`convert`]（9 变体的**类型化前端**）仍不在生产路径上（全家族生产调用 0 处）。
//!
//! ⚠️ **在 A-2 第 2 步完成前，不要用 [`schema`] / [`convert`] 的口径去判断线协议是否一致**
//! —— 契约看 [`contract`]。历史上 `wp-labs/warp-fusion#102` 正是
//! 「按自我声明找权威实现 → 找到类型化前端 → 报口径不一致」这一误判。
//!
//! 完整迁移计划见规格表 A-2 一节。

pub mod contract;
pub mod convert;
pub mod error;
pub mod ipc;
pub mod schema;

pub use contract::wp_type_to_arrow;
