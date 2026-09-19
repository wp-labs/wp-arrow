//! 线协议契约：`wp_model_core::model::DataType` → Arrow 列类型，以及值层（`DataRecord` → 列）。
//!
//! 这是 **wparse（sink 侧）↔ wfusion（接收侧）Arrow 列类型契约**的实现，
//! 规格表见 `wp-reactor/docs/design/arrow-type-mapping.md`（§3 是逐行口径表，§4 是已知差异登记）。
//!
//! 两层：
//! - [`wp_type_to_arrow`]（本模块）—— 表：某类型该是什么 Arrow 列；
//! - [`value`] —— 值：值怎么写进那一列（[`encode_record`] / [`encode_records`]）。
//! - 两层都按 **Arrow 列类型**说事，所以口径只有一处：先查表得到列类型，值层只认列类型。
//!
//! # 归属（A-2）
//!
//! 本模块是契约实现的**目标落点**。迁移前它住在 `wp-connector-utils`
//! （`arrow::wp_type_to_arrow`）——那是「面向 sink 的 connector 工具」crate，
//! 把线协议契约放在那里是**定位倒置**：按自我声明去找权威实现的人会找到
//! `wp-arrow`（`schema.rs`/`convert.rs` 的 9 变体类型化前端），拿到的是另一套口径，
//! 于2026-09-19 报出「`wp-arrow` 与 `wp-connector-utils` 不一致」（wp-labs/warp-fusion#102）。
//!
//! **迁移状态**：本模块已落地（A-2 2a），值层已随 2c 迁入（[`value`]）；
//! `wp-connector-utils` 的 `arrow::wp_type_to_arrow` / `arrow::record` 已改为转发到这里，
//! 并经 `wp-arrow` 发布版进入生产（跨仓发布顺序见规格表 §5）。
//!
//! # 不要把本表与 [`crate::schema`] 混为一谈
//!
//! [`crate::schema::WpDataType`] 是 9 变体的**类型化前端**，它的 `Array` → `List(inner)`、
//! `BigInt` → `Decimal256(39,0)` 是它自己的口径（保留结构化/数值语义），**与线协议无关**
//! （规格表 §1 A-0、§4 DIV-2/DIV-3）。契约口径是**保守的**：结构化字段一律 `Utf8`（JSON 文本），
//! 任意精度整数一律十进制 `Utf8`。

use arrow::datatypes::{DataType, TimeUnit};

pub mod value;

pub use value::{encode_record, encode_records};

/// `wp_model_core::model::DataType` → Arrow 列类型（**线协议契约口径**）。
///
/// match 是**穷尽的**（无 `_` 兜底）：`wp-model-core` 新增变体会直接**编译失败** ——
/// 这是刻意的，逼对新类型表态，而不是静默兜到 `Utf8`。
///
/// 每个变体的理由与已知差异（DIV-1/2/3）见 `wp-reactor/docs/design/arrow-type-mapping.md`。
pub fn wp_type_to_arrow(dt: &wp_model_core::model::DataType) -> DataType {
    use wp_model_core::model::DataType as WpDt;
    match dt {
        WpDt::Bool => DataType::Boolean,
        WpDt::Int => DataType::Int64,
        // 任意精度整数（BigUint）：以十进制字符串输出（与 format_utf8_value 的 to_string 一致）
        WpDt::BigInt => DataType::Utf8,
        WpDt::Float => DataType::Float64,
        WpDt::Port => DataType::Int32,
        WpDt::Time
        | WpDt::TimeISO
        | WpDt::TimeRFC3339
        | WpDt::TimeRFC2822
        | WpDt::TimeTIMESTAMP
        | WpDt::TimeCLF => DataType::Timestamp(TimeUnit::Nanosecond, None),
        // DIV-1（已修复）：`hex` 走 Utf8（十六进制字符串）。期望侧（`wf-runtime`）与
        // `wp-arrow` 的类型化前端都是 Utf8，且 `Value::Hex` 的 `Display` 就是 `{:#X}`
        // （`wp-model-core` primitive.rs），与 sink 值层的 Utf8 输出同形 ——
        // 所以只需这一行对齐，值层不用改。规格表：DIV-1。
        WpDt::Hex => DataType::Utf8,
        WpDt::Base64 => DataType::Binary,
        WpDt::Chars
        | WpDt::Symbol
        | WpDt::PeekSymbol
        | WpDt::IP
        | WpDt::IpNet
        | WpDt::Domain
        | WpDt::Email
        | WpDt::Url
        | WpDt::SN
        | WpDt::IdCard
        | WpDt::MobilePhone
        | WpDt::KV
        | WpDt::KvArr
        | WpDt::Json
        | WpDt::ExactJson
        | WpDt::HttpRequest
        | WpDt::HttpStatus
        | WpDt::HttpAgent
        | WpDt::HttpMethod
        | WpDt::Auto
        | WpDt::ProtoText
        | WpDt::Obj
        | WpDt::Ignore => DataType::Utf8,
        // DIV-3：结构化数组走 `Utf8`（JSON 文本），**不带** `wfl_field_type` 元数据；
        // 接收侧对该形态返回「兼容」，语义在 coerce 阶段兑现（有意设计，规格表 §4）。
        WpDt::Array(_) => DataType::Utf8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 规格表钉桩：`wp_type_to_arrow` 的**全 37 个** `DataType` 变体映射。
    ///
    /// 规格表：`wp-reactor/docs/design/arrow-type-mapping.md` §3。该表是 sink 侧与
    /// 接收侧（`wf-runtime`）之间 Arrow 列类型契约的单一事实来源；本测试把它钉死，
    /// 使任何口径漂移都以测试失败暴露，而不是线上静默出错。
    ///
    /// match 是穷尽的（无 `_` 兜底），所以 wp-model-core 新增变体会直接编译失败；
    /// `cases.len()` 断言则保证本表与文档同步更新。
    ///
    /// 迁移期（A-2 第 2 步完成前）`wp-connector-utils` 也有一份同样的钉桩 —— 两份同时
    /// 存在是刻意的：任一侧漂移都会在**两个**仓里各报一次，直到第 2 步把那份删掉。
    /// ⚠️ 该编译期守卫的前提：`wp_model_core::model::DataType` **不是** `#[non_exhaustive]`
    /// （2026-09 现状如此）。若上游改成 non_exhaustive，下游会被迫补 `_` 兜底，守卫就**静默失效**
    /// （而 `cases.len()` 仍为 37）——那时必须把 `_` 分支改成显式拒绝，或在 `_` 上挂
    /// `deny(non_exhaustive_omitted_patterns)`。
    #[test]
    fn wire_contract_full_mapping_is_pinned() {
        use wp_model_core::model::{ArraySubtype, DataType as WpDt};

        let ts = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let cases: Vec<(WpDt, DataType)> = vec![
            (WpDt::Bool, DataType::Boolean),
            (WpDt::Chars, DataType::Utf8),
            (WpDt::Symbol, DataType::Utf8),
            (WpDt::PeekSymbol, DataType::Utf8),
            (WpDt::Int, DataType::Int64),
            // DIV-2：wp-arrow 的**类型化前端**为 Decimal256(39,0)，契约是十进制字符串
            (WpDt::BigInt, DataType::Utf8),
            (WpDt::Float, DataType::Float64),
            // `Ignore` 本身映 Utf8；"剔除 Ignore 字段" 的语义在 sink 侧的 schema 推断里，
            // 本模块只做类型映射，不做字段取舍
            (WpDt::Ignore, DataType::Utf8),
            (WpDt::Time, ts.clone()),
            (WpDt::TimeISO, ts.clone()),
            (WpDt::TimeRFC3339, ts.clone()),
            (WpDt::TimeRFC2822, ts.clone()),
            (WpDt::TimeTIMESTAMP, ts.clone()),
            (WpDt::TimeCLF, ts.clone()),
            (WpDt::IP, DataType::Utf8),
            (WpDt::IpNet, DataType::Utf8),
            (WpDt::Domain, DataType::Utf8),
            (WpDt::Email, DataType::Utf8),
            (WpDt::Port, DataType::Int32),
            (WpDt::SN, DataType::Utf8),
            // DIV-1（已修复）：Hex 走 Utf8（十六进制字符串）
            (WpDt::Hex, DataType::Utf8),
            (WpDt::Base64, DataType::Binary),
            (WpDt::KV, DataType::Utf8),
            (WpDt::KvArr, DataType::Utf8),
            (WpDt::Json, DataType::Utf8),
            (WpDt::ExactJson, DataType::Utf8),
            (WpDt::HttpRequest, DataType::Utf8),
            (WpDt::HttpStatus, DataType::Utf8),
            (WpDt::HttpAgent, DataType::Utf8),
            (WpDt::HttpMethod, DataType::Utf8),
            (WpDt::Url, DataType::Utf8),
            (WpDt::Auto, DataType::Utf8),
            (WpDt::ProtoText, DataType::Utf8),
            // DIV-3：结构化字段只给 Utf8 且不带 `wfl_field_type` 元数据（有意设计）
            (WpDt::Obj, DataType::Utf8),
            (WpDt::Array(ArraySubtype::new("int")), DataType::Utf8),
            (WpDt::IdCard, DataType::Utf8),
            (WpDt::MobilePhone, DataType::Utf8),
        ];

        assert_eq!(
            cases.len(),
            37,
            "wp-model-core DataType 变体数变化 → 同步更新规格表 §3"
        );
        for (dt, expected) in cases {
            assert_eq!(wp_type_to_arrow(&dt), expected, "DataType::{dt:?}");
        }
    }

    /// DIV-1 回归：`hex` 必须是 `Utf8`（十六进制字符串），不是 `Binary`。
    ///
    /// 断链症状：sink 侧若退回 `Binary`，接收侧期望 `Utf8` 且 `hex` 不带结构化元数据
    /// → 走严格相等 → `arrow source schema mismatch`；即使绕过校验，`coerce_column`
    /// 的 `_ => NullArray` 也会把整列变 null。
    #[test]
    fn hex_must_be_utf8() {
        assert_eq!(
            wp_type_to_arrow(&wp_model_core::model::DataType::Hex),
            DataType::Utf8
        );
    }

    /// 与本 crate 的类型化前端**刻意不同**的两行：契约是保守口径，前端保留结构与数值语义。
    ///
    /// 这个 `assert_ne!`/差异断言是防呆：若有人「顺手」把两边对齐，这里会失败 ——
    /// 对齐本身不是目标，目标是不让人把 [`crate::schema`] 当成契约
    /// （规格表 §1 A-0 / §4 DIV-2·DIV-3）。
    ///
    /// 差异**只有两行**（`BigInt` / `Array`）；其余重叠行反而应当一致 —— 一并钉住，
    /// 免得读者以为前端「到处都是另一套口径」。
    #[test]
    fn wire_contract_differs_from_the_typed_frontend_on_two_rows() {
        use crate::schema::{BIGINT_DECIMAL_PRECISION, WpDataType, to_arrow_type};
        use wp_model_core::model::{ArraySubtype, DataType as WpDt};

        // ── 仅有的两行差异 ──
        // DIV-2 BigInt：契约十进制字符串 vs 前端 Decimal256(39,0)（数值语义）
        assert_eq!(wp_type_to_arrow(&WpDt::BigInt), DataType::Utf8);
        assert_eq!(
            to_arrow_type(&WpDataType::BigInt),
            DataType::Decimal256(BIGINT_DECIMAL_PRECISION, 0)
        );

        // DIV-3 结构化数组：契约 Utf8（JSON 文本） vs 前端 List(inner)（逐元素带类型）
        assert_eq!(
            wp_type_to_arrow(&WpDt::Array(ArraySubtype::new("int"))),
            DataType::Utf8
        );
        assert!(
            matches!(
                to_arrow_type(&WpDataType::Array(Box::new(WpDataType::Digit))),
                DataType::List(_)
            ),
            "前端对结构化数组给 List(inner)（它自己的口径）"
        );
    }

    /// 两侧**都能表达**的重叠行必须一致（差异只限于上面那两行）。
    ///
    /// 这栏是给下一个读者看的：前端的价值在于“保留结构/数值语义的强类型 API”，
    /// 不在于另一套列类型口径 —— 除 BigInt/Array 外，它与契约同口径。
    #[test]
    fn wire_contract_and_frontend_agree_on_the_overlap() {
        use crate::schema::{WpDataType, to_arrow_type};
        use wp_model_core::model::DataType as WpDt;

        let rows = [
            (WpDt::Bool, WpDataType::Bool),
            (WpDt::Int, WpDataType::Digit),
            (WpDt::Float, WpDataType::Float),
            (WpDt::Time, WpDataType::Time),
            (WpDt::Chars, WpDataType::Chars),
            (WpDt::IP, WpDataType::Ip),
            // DIV-1：Hex 两边都是 Utf8（十六进制字符串）
            (WpDt::Hex, WpDataType::Hex),
        ];
        for (model, frontend) in rows {
            assert_eq!(
                wp_type_to_arrow(&model),
                to_arrow_type(&frontend),
                "重叠行应一致：{model:?} vs {frontend:?}"
            );
        }
    }
}
