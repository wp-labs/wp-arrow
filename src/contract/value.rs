//! 线协议契约的**值层**：`DataRecord` → 列（按 Arrow 列类型派发）。
//!
//! 与 [`crate::contract`] 的 schema 表配套：表回答「这一列该是什么 Arrow 类型」，
//! 本模块回答「值怎么写进那一列」。两者合起来才是完整的线协议契约。
//!
//! # 归属（A-2 第 3 步 / 2c）
//!
//! 这些函数**逐字移植自** `wp-connector-utils/src/arrow/record.rs`（那个 crate 是
//! 「面向 sink 的 connector 工具」，契约的语义归属不在这里）。移植时只改了错误类型：
//! `SinkResult`/`SinkReason`（来自 `wp-connector-api`，本 crate 不能依赖）→
//! [`WpArrowError::ArrowBuildError`]。行为、错误文案形状与派发顺序保持不变。
//!
//! 派发**按 Arrow 列类型**（不是按 wp-model 类型）：所以「某类型编成什么列」由
//! [`crate::contract::wp_type_to_arrow`] 决定，本模块只认列类型 —— 两处不会各有一套口径。

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, Float64Builder, Int32Builder, Int64Builder,
    StringBuilder, TimestampNanosecondBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use wp_model_core::model::{DataRecord, Value};

use crate::error::WpArrowError;

/// 单条 `DataRecord` → 一行 `RecordBatch`。
///
/// 每个列按名字在记录里查找；缺字段 → null。
pub fn encode_record(
    record: &DataRecord,
    schema: &Arc<Schema>,
) -> Result<RecordBatch, WpArrowError> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let records = [Arc::new(record.clone())];
        columns.push(build_column_from_field(field, &records)?);
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|e| WpArrowError::ArrowBuildError(e.to_string()))
}

/// 多条 `DataRecord` → 一个 `RecordBatch`。
///
/// 每个列按名字在**每条**记录里查找；缺字段 → null。`records` 为空时按 schema
/// 产零行（列类型齐全，便于下游直接 append）。
pub fn encode_records(
    records: &[Arc<DataRecord>],
    schema: &Arc<Schema>,
) -> Result<RecordBatch, WpArrowError> {
    if records.is_empty() {
        let empty_columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|f| empty_column_for_type(f.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        return RecordBatch::try_new(Arc::clone(schema), empty_columns)
            .map_err(|e| WpArrowError::ArrowBuildError(e.to_string()));
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        columns.push(build_column_from_field(field, records)?);
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|e| WpArrowError::ArrowBuildError(e.to_string()))
}

// ---------------------------------------------------------------------------
// 列构造：按 Arrow 列类型派发
// ---------------------------------------------------------------------------

fn build_column_from_field(
    field: &Field,
    records: &[Arc<DataRecord>],
) -> Result<ArrayRef, WpArrowError> {
    let field_name = field.name();
    match field.data_type() {
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(records.len());
            for record in records {
                match record.field(field_name).map(|f| f.get_value()) {
                    Some(Value::Bool(v)) => builder.append_value(*v),
                    Some(Value::Chars(s)) => builder.append_value(s.eq_ignore_ascii_case("true")),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(records.len());
            for record in records {
                match record
                    .field(field_name)
                    .and_then(|f| parse_digit(f.get_value()))
                {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int32 => {
            let mut builder = Int32Builder::with_capacity(records.len());
            for record in records {
                match record
                    .field(field_name)
                    .and_then(|f| parse_digit(f.get_value()))
                {
                    Some(v) => builder.append_value(v as i32),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Binary => {
            let mut builder = BinaryBuilder::with_capacity(records.len(), records.len() * 64);
            for record in records {
                match record.field(field_name).map(|f| f.get_value()) {
                    Some(v) => {
                        let bytes = to_raw_bytes(v);
                        builder.append_value(&bytes[..]);
                    }
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(records.len());
            for record in records {
                match record
                    .field(field_name)
                    .and_then(|f| parse_float(f.get_value()))
                {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            let mut builder = TimestampNanosecondBuilder::with_capacity(records.len());
            for record in records {
                match record
                    .field(field_name)
                    .and_then(|f| parse_timestamp_ns(f.get_value()))
                {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        // Utf8 与其余列类型（结构化字段走的就是这里：JSON 文本）
        _ => {
            let mut builder = StringBuilder::with_capacity(records.len(), records.len() * 32);
            for record in records {
                match record.field(field_name) {
                    Some(f) => builder.append_value(format_utf8_value(f.get_value())),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
    }
}

// ---------------------------------------------------------------------------
// 值格式化
// ---------------------------------------------------------------------------

/// [`Value`] → Utf8 列文本。
///
/// 结构化（`Obj` / `Array`）序列化为 **JSON**；其余走 `Display`。
fn format_utf8_value(v: &Value) -> String {
    match v {
        Value::Obj(_) | Value::Array(_) => {
            serde_json::to_string(v).unwrap_or_else(|_| format!("{v:?}"))
        }
        _ => v.to_string(),
    }
}

/// [`Value`] → Binary 列的原始字节。
///
/// `Value::Hex` 取其 `u128` 的**最小大端**字节；其余退化为 Utf8 文本的字节。
///
/// 注意：`hex` 字段**不再**走 Binary 列（DIV-1 已对齐为 Utf8），所以 `Value::Hex`
/// 分支只在**显式声明 Binary** 的列上才可达。
fn to_raw_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::Hex(h) => {
            if h.0 == 0 {
                return vec![0];
            }
            let be = h.0.to_be_bytes();
            let start = be.iter().position(|&b| b != 0).unwrap();
            be[start..].to_vec()
        }
        _ => format_utf8_value(v).into_bytes(),
    }
}

// ---------------------------------------------------------------------------
// 取值助手（带 Chars 回退）
// ---------------------------------------------------------------------------

fn parse_digit(v: &Value) -> Option<i64> {
    match v {
        Value::Int(d) => Some(*d),
        Value::Float(f) => Some(*f as i64),
        Value::Chars(s) => s.parse().ok(),
        _ => None,
    }
}

fn parse_float(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(d) => Some(*d as f64),
        Value::Chars(s) => s.parse().ok(),
        _ => None,
    }
}

fn parse_timestamp_ns(v: &Value) -> Option<i64> {
    match v {
        Value::Time(t) => Some(t.and_utc().timestamp_nanos_opt()?),
        // 整数时间戳按**毫秒**解读（与移植前的 sink 口径一致）
        Value::Int(d) => d.checked_mul(1_000_000),
        Value::Chars(s) => chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .or_else(|| {
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .map(|dt| dt.and_utc().fixed_offset())
            })
            .and_then(|dt| dt.timestamp_nanos_opt()),
        _ => None,
    }
}

fn empty_column_for_type(data_type: &DataType) -> Result<ArrayRef, WpArrowError> {
    let arr: ArrayRef = match data_type {
        DataType::Boolean => Arc::new(arrow::array::BooleanArray::from(Vec::<bool>::new())),
        DataType::Int32 => Arc::new(arrow::array::Int32Array::from(Vec::<i32>::new())),
        DataType::Int64 => Arc::new(arrow::array::Int64Array::from(Vec::<i64>::new())),
        DataType::Float64 => Arc::new(arrow::array::Float64Array::from(Vec::<f64>::new())),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => Arc::new(
            arrow::array::TimestampNanosecondArray::from(Vec::<i64>::new()),
        ),
        DataType::Binary => Arc::new(arrow::array::BinaryArray::from(Vec::<Option<&[u8]>>::new())),
        _ => Arc::new(arrow::array::StringArray::from(Vec::<Option<&str>>::new())),
    };
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `is_null` / `data_type` 来自 `Array` trait（B 侧的测试模块顶部已导入，这里同样需要）
    use arrow::array::Array as _;
    use arrow::array::{
        BinaryArray, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
        TimestampNanosecondArray,
    };
    use wp_model_core::model::types::value::{HexT, ObjectValue};
    use wp_model_core::model::{DataField, Field as ModelField, FieldStorage};

    /// 线协议**值层**的金标准：覆盖全部列类型 + 缺字段 + 类型回退（Chars/Int/Float/Time 互转）。
    ///
    /// 本测与 `wp-connector-utils/src/arrow/record.rs` 的同名测试**逐字同构**
    /// （同一份夹具、同一组期望）。两份同时通过即证明 A-2 2c 的值层搬迁是**等价**改动，
    /// 而不只是「看起来一样」；搬迁后本测是**实现侧**钉桩，那边那份变成消费侧拼线。
    #[test]
    fn wire_value_encoding_is_pinned_by_golden_values() {
        let epoch =
            chrono::NaiveDateTime::parse_from_str("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap();
        let ts_val = epoch + chrono::Duration::seconds(5);

        // row0：全字段齐、尽量走「正路」
        let mut obj = ObjectValue::new();
        obj.insert("k", DataField::from_chars("k", "v"));
        let row0 = DataRecord::from(vec![
            FieldStorage::from(DataField::from_bool("b", true)),
            FieldStorage::from(DataField::from_int("i64", 42)),
            FieldStorage::from(DataField::from_int("i32", 70_000)),
            FieldStorage::from(DataField::from_float("f", 1.5)),
            FieldStorage::from(DataField::from_time("ts", ts_val)),
            FieldStorage::from(DataField::from_chars("bin", "hi")),
            FieldStorage::from(DataField::from_obj("s", obj)),
        ]);

        // row1：类型回退（Chars 解析 / Float→Int / Int(ms)→时间戳 / Hex→Binary / Array→JSON）
        let arr = DataField::from_arr(
            "s",
            vec![
                DataField::from_chars("c", "x"),
                DataField::from_int("i", 22),
            ],
        );
        let row1 = DataRecord::from(vec![
            FieldStorage::from(DataField::from_chars("b", "TRUE")),
            FieldStorage::from(DataField::from_chars("i64", "42")),
            FieldStorage::from(DataField::from_float("i32", 3.9)),
            FieldStorage::from(DataField::from_chars("f", "2.71")),
            FieldStorage::from(DataField::from_int("ts", 1_700_000_000_000)),
            FieldStorage::from(DataField::from_hex("bin", HexT(0x1A2B))),
            FieldStorage::from(arr),
        ]);

        // row2：除 `s` 外全缺 → 其余列 null；`s` 走 Hex 的 Utf8 形态
        let row2 = DataRecord::from(vec![FieldStorage::from(DataField::from_hex(
            "s",
            HexT(0x1A2B),
        ))]);

        let rows = vec![Arc::new(row0), Arc::new(row1), Arc::new(row2)];
        let schema = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Boolean, true),
            Field::new("i64", DataType::Int64, true),
            Field::new("i32", DataType::Int32, true),
            Field::new("f", DataType::Float64, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
            Field::new("bin", DataType::Binary, true),
            Field::new("s", DataType::Utf8, true),
        ]));

        let batch = encode_records(&rows, &schema).unwrap();
        assert_eq!(batch.num_rows(), 3);

        let b = batch
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!((b.value(0), b.value(1)), (true, true));
        assert!(b.is_null(2));

        let i64c = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!((i64c.value(0), i64c.value(1)), (42, 42));
        assert!(i64c.is_null(2));

        let i32c = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!((i32c.value(0), i32c.value(1)), (70_000, 3));
        assert!(i32c.is_null(2));

        let fc = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(fc.value(0), 1.5);
        assert_eq!(fc.value(1), 2.71);
        assert!(fc.is_null(2));

        let tsc = batch
            .column(4)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        assert_eq!(
            tsc.value(0),
            ts_val.and_utc().timestamp_nanos_opt().unwrap()
        );
        assert_eq!(tsc.value(1), 1_700_000_000_000 * 1_000_000);
        assert!(tsc.is_null(2));

        let binc = batch
            .column(5)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(binc.value(0), b"hi");
        assert_eq!(binc.value(1), &[0x1A, 0x2B]);
        assert!(binc.is_null(2));

        let sc = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(!sc.is_null(2), "row2 的 s 是有的（Hex），不应 null");
        // 结构化字段走 JSON（`serde_json::to_string` 同一套规则）——不硬编 JSON 形状，
        // 但把「必须等于源 Value 的 serde_json 渲染」钉住。
        for (row, field) in [(0usize, "s"), (1usize, "s")] {
            let src = rows[row].field(field).unwrap().get_value();
            let rendered = sc.value(row);
            match serde_json::to_string(src) {
                Ok(json) => {
                    assert_eq!(
                        rendered, json,
                        "row{row} 结构化字段应等于源 Value 的 JSON 渲染"
                    )
                }
                Err(e) => panic!(
                    "row{row}: serde_json 渲染失败（{e}）→ 列里实际是 {rendered:?}，src={src:?}"
                ),
            }
            assert!(
                serde_json::from_str::<serde_json::Value>(rendered).is_ok(),
                "结构化字段在 Utf8 列里必须是合法 JSON"
            );
        }
        // row2：Hex 走 Utf8 时是 `{:#X}` 形态（与 `Value::Hex` 的 Display 同形）
        assert_eq!(sc.value(2), "0x1A2B");
        assert_eq!(sc.value(2), format!("{:#X}", 0x1A2Bu128));
    }

    /// 单条入口（`encode_record`）与整批入口行为一致（都按名字查字段）。
    #[test]
    fn single_record_entry_matches_batch_entry() {
        let rec = DataRecord::from(vec![FieldStorage::from(ModelField::from_chars("x", "v"))]);
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
        let one = encode_record(&rec, &schema).unwrap();
        let many = encode_records(&[Arc::new(rec)], &schema).unwrap();
        assert_eq!(one.num_rows(), 1);
        assert_eq!(many.num_rows(), 1);
        let a = one
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let b = many
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(a.value(0), b.value(0));
    }

    /// 空批次：按 schema 产零行，列类型齐全（便于下游 append）。
    #[test]
    fn empty_records_produce_typed_zero_row_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = encode_records(&[], &schema).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.column(0).data_type(), &DataType::Int64);
        assert_eq!(batch.column(1).data_type(), &DataType::Utf8);
    }
}
