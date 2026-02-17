use std::net::IpAddr;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float64Array, Int64Array, ListArray, StringArray, TimestampNanosecondArray,
};
use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, RecordBatch,
    StringBuilder, TimestampNanosecondBuilder,
};
use chrono::DateTime;

use wp_model_core::model::{
    DataRecord, DataType, FValueStr, Field, FieldStorage, HexT, IpNetValue, Value,
};

use crate::error::WpArrowError;
use crate::schema::{FieldDef, WpDataType, to_arrow_schema};

/// Convert row-oriented DataRecords to a columnar Arrow RecordBatch.
///
/// Schema is driven by `field_defs`. For each FieldDef, the corresponding value
/// is looked up by name in every record. Missing nullable fields become null;
/// missing non-nullable fields produce an error.
pub fn records_to_batch(
    records: &[DataRecord],
    field_defs: &[FieldDef],
) -> Result<RecordBatch, WpArrowError> {
    let schema = to_arrow_schema(field_defs)?;
    let columns: Vec<ArrayRef> = field_defs
        .iter()
        .map(|fd| build_column(fd, records))
        .collect::<Result<_, _>>()?;
    RecordBatch::try_new(Arc::new(schema), columns)
        .map_err(|e| WpArrowError::ArrowBuildError(e.to_string()))
}

/// Convert a columnar Arrow RecordBatch back to row-oriented DataRecords.
///
/// `field_defs` provides WpDataType metadata for distinguishing Arrow Utf8 columns
/// (which may represent Chars, Ip, or Hex). Record IDs are set to sequential row indices.
pub fn batch_to_records(
    batch: &RecordBatch,
    field_defs: &[FieldDef],
) -> Result<Vec<DataRecord>, WpArrowError> {
    if field_defs.len() != batch.num_columns() {
        return Err(WpArrowError::SchemaMismatch {
            expected: field_defs.len(),
            actual: batch.num_columns(),
        });
    }

    let num_rows = batch.num_rows();
    let mut records = Vec::with_capacity(num_rows);

    for row_idx in 0..num_rows {
        let mut items = Vec::with_capacity(field_defs.len());
        for (col_idx, fd) in field_defs.iter().enumerate() {
            let col = batch.column(col_idx);
            if col.is_null(row_idx) {
                continue;
            }
            let value = extract_value(col, row_idx, &fd.data_type, &fd.name)?;
            let meta = wp_type_to_model_meta(&fd.data_type);
            let field = Field::new(meta, fd.name.as_str(), value);
            items.push(FieldStorage::from_owned(field));
        }
        let mut record = DataRecord::from(items);
        record.id = row_idx as u64;
        records.push(record);
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// Internal helpers for records_to_batch
// ---------------------------------------------------------------------------

fn build_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    match &fd.data_type {
        WpDataType::Chars | WpDataType::Ip | WpDataType::Hex => build_string_column(fd, records),
        WpDataType::Digit => build_digit_column(fd, records),
        WpDataType::Float => build_float_column(fd, records),
        WpDataType::Bool => build_bool_column(fd, records),
        WpDataType::Time => build_time_column(fd, records),
        WpDataType::Array(inner) => build_list_column(fd, records, inner),
    }
}

fn build_string_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = StringBuilder::with_capacity(records.len(), records.len() * 32);
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(val) => {
                let s = value_to_string(val, &fd.data_type, &fd.name)?;
                builder.append_value(&s);
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_digit_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = Int64Builder::with_capacity(records.len());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Digit(v)) => builder.append_value(*v),
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Digit".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_float_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = Float64Builder::with_capacity(records.len());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Float(v)) => builder.append_value(*v),
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Float".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_bool_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = BooleanBuilder::with_capacity(records.len());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Bool(v)) => builder.append_value(*v),
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Bool".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_time_column(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = TimestampNanosecondBuilder::with_capacity(records.len());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Time(ndt)) => {
                let nanos = ndt.and_utc().timestamp_nanos_opt().ok_or_else(|| {
                    WpArrowError::TimestampOverflow {
                        field_name: fd.name.clone(),
                    }
                })?;
                builder.append_value(nanos);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Time".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_list_column(
    fd: &FieldDef,
    records: &[DataRecord],
    inner_type: &WpDataType,
) -> Result<ArrayRef, WpArrowError> {
    match inner_type {
        WpDataType::Chars | WpDataType::Ip | WpDataType::Hex => {
            build_list_string(fd, records, inner_type)
        }
        WpDataType::Digit => build_list_digit(fd, records),
        WpDataType::Float => build_list_float(fd, records),
        WpDataType::Bool => build_list_bool(fd, records),
        WpDataType::Time => build_list_time(fd, records),
        WpDataType::Array(_) => Err(WpArrowError::UnsupportedDataType(
            "nested array<array<...>> not supported".to_string(),
        )),
    }
}

fn build_list_string(
    fd: &FieldDef,
    records: &[DataRecord],
    inner_type: &WpDataType,
) -> Result<ArrayRef, WpArrowError> {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Array(items)) => {
                for item in items {
                    let val = item.get_value();
                    if matches!(val, Value::Null) {
                        builder.values().append_null();
                    } else {
                        let s = value_to_string(val, inner_type, &fd.name)?;
                        builder.values().append_value(&s);
                    }
                }
                builder.append(true);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Array".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_list_digit(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Array(items)) => {
                for item in items {
                    match item.get_value() {
                        Value::Digit(v) => builder.values().append_value(*v),
                        Value::Null => builder.values().append_null(),
                        other => {
                            return Err(WpArrowError::ValueConversionError {
                                field_name: fd.name.clone(),
                                expected: "Digit".to_string(),
                                actual: other.tag().to_string(),
                            });
                        }
                    }
                }
                builder.append(true);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Array".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_list_float(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = ListBuilder::new(Float64Builder::new());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Array(items)) => {
                for item in items {
                    match item.get_value() {
                        Value::Float(v) => builder.values().append_value(*v),
                        Value::Null => builder.values().append_null(),
                        other => {
                            return Err(WpArrowError::ValueConversionError {
                                field_name: fd.name.clone(),
                                expected: "Float".to_string(),
                                actual: other.tag().to_string(),
                            });
                        }
                    }
                }
                builder.append(true);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Array".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_list_bool(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = ListBuilder::new(BooleanBuilder::new());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Array(items)) => {
                for item in items {
                    match item.get_value() {
                        Value::Bool(v) => builder.values().append_value(*v),
                        Value::Null => builder.values().append_null(),
                        other => {
                            return Err(WpArrowError::ValueConversionError {
                                field_name: fd.name.clone(),
                                expected: "Bool".to_string(),
                                actual: other.tag().to_string(),
                            });
                        }
                    }
                }
                builder.append(true);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Array".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_list_time(fd: &FieldDef, records: &[DataRecord]) -> Result<ArrayRef, WpArrowError> {
    let mut builder = ListBuilder::new(TimestampNanosecondBuilder::new());
    for rec in records {
        match rec.get_value(&fd.name) {
            Some(Value::Null) | None => {
                handle_null(&mut builder, fd, |b| b.append_null())?;
            }
            Some(Value::Array(items)) => {
                for item in items {
                    match item.get_value() {
                        Value::Time(ndt) => {
                            let nanos = ndt.and_utc().timestamp_nanos_opt().ok_or_else(|| {
                                WpArrowError::TimestampOverflow {
                                    field_name: fd.name.clone(),
                                }
                            })?;
                            builder.values().append_value(nanos);
                        }
                        Value::Null => builder.values().append_null(),
                        other => {
                            return Err(WpArrowError::ValueConversionError {
                                field_name: fd.name.clone(),
                                expected: "Time".to_string(),
                                actual: other.tag().to_string(),
                            });
                        }
                    }
                }
                builder.append(true);
            }
            Some(other) => {
                return Err(WpArrowError::ValueConversionError {
                    field_name: fd.name.clone(),
                    expected: "Array".to_string(),
                    actual: other.tag().to_string(),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Convert a Value to its string representation for Arrow Utf8 columns.
fn value_to_string(
    val: &Value,
    wp_type: &WpDataType,
    field_name: &str,
) -> Result<String, WpArrowError> {
    match (wp_type, val) {
        // Chars accepts any text-like Value
        (WpDataType::Chars, Value::Chars(s)) => Ok(s.to_string()),
        (WpDataType::Chars, Value::Domain(d)) => Ok(d.to_string()),
        (WpDataType::Chars, Value::Url(u)) => Ok(u.to_string()),
        (WpDataType::Chars, Value::Email(e)) => Ok(e.to_string()),
        // Ip accepts IpAddr, IpNet, and Chars fallback
        (WpDataType::Ip, Value::IpAddr(ip)) => Ok(ip.to_string()),
        (WpDataType::Ip, Value::IpNet(net)) => Ok(net.to_string()),
        (WpDataType::Ip, Value::Chars(s)) => Ok(s.to_string()),
        // Hex
        (WpDataType::Hex, Value::Hex(h)) => Ok(format!("{:#X}", h.0)),
        _ => Err(WpArrowError::ValueConversionError {
            field_name: field_name.to_string(),
            expected: format!("{:?}", wp_type),
            actual: val.tag().to_string(),
        }),
    }
}

/// Handle null/missing values: append null if nullable, error if required.
fn handle_null<B, F>(builder: &mut B, fd: &FieldDef, append_null: F) -> Result<(), WpArrowError>
where
    F: FnOnce(&mut B),
{
    if fd.nullable {
        append_null(builder);
        Ok(())
    } else {
        Err(WpArrowError::MissingRequiredField {
            field_name: fd.name.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers for batch_to_records
// ---------------------------------------------------------------------------

/// Extract a Value from an Arrow array column at the given row index.
fn extract_value(
    col: &ArrayRef,
    row_idx: usize,
    wp_type: &WpDataType,
    field_name: &str,
) -> Result<Value, WpArrowError> {
    match wp_type {
        WpDataType::Chars => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| WpArrowError::ArrowBuildError("expected StringArray".to_string()))?;
            Ok(Value::Chars(FValueStr::from(arr.value(row_idx))))
        }
        WpDataType::Digit => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| WpArrowError::ArrowBuildError("expected Int64Array".to_string()))?;
            Ok(Value::Digit(arr.value(row_idx)))
        }
        WpDataType::Float => {
            let arr = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                WpArrowError::ArrowBuildError("expected Float64Array".to_string())
            })?;
            Ok(Value::Float(arr.value(row_idx)))
        }
        WpDataType::Bool => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                WpArrowError::ArrowBuildError("expected BooleanArray".to_string())
            })?;
            Ok(Value::Bool(arr.value(row_idx)))
        }
        WpDataType::Time => {
            let arr = col
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .ok_or_else(|| {
                    WpArrowError::ArrowBuildError("expected TimestampNanosecondArray".to_string())
                })?;
            let nanos = arr.value(row_idx);
            let ndt = DateTime::from_timestamp_nanos(nanos).naive_utc();
            Ok(Value::Time(ndt))
        }
        WpDataType::Ip => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| WpArrowError::ArrowBuildError("expected StringArray".to_string()))?;
            let s = arr.value(row_idx);
            Ok(parse_ip_value(s, field_name)?)
        }
        WpDataType::Hex => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| WpArrowError::ArrowBuildError("expected StringArray".to_string()))?;
            let s = arr.value(row_idx);
            Ok(parse_hex_value(s, field_name)?)
        }
        WpDataType::Array(inner) => {
            let arr = col
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| WpArrowError::ArrowBuildError("expected ListArray".to_string()))?;
            let inner_arr = arr.value(row_idx);
            let inner_meta = wp_type_to_model_meta(inner);
            let mut items = Vec::new();
            for i in 0..inner_arr.len() {
                if inner_arr.is_null(i) {
                    items.push(FieldStorage::from_owned(Field::new(
                        inner_meta.clone(),
                        "item",
                        Value::Null,
                    )));
                } else {
                    let val = extract_value(&inner_arr, i, inner, field_name)?;
                    items.push(FieldStorage::from_owned(Field::new(
                        inner_meta.clone(),
                        "item",
                        val,
                    )));
                }
            }
            Ok(Value::Array(items))
        }
    }
}

/// Parse a string as an IP address or CIDR network.
fn parse_ip_value(s: &str, field_name: &str) -> Result<Value, WpArrowError> {
    if s.contains('/') {
        // Try parsing as IpNet (CIDR notation)
        let parts: Vec<&str> = s.splitn(2, '/').collect();
        let addr: IpAddr = parts[0].parse().map_err(|e| WpArrowError::ParseError {
            field_name: field_name.to_string(),
            detail: format!("invalid IP address: {e}"),
        })?;
        let prefix: u8 = parts[1].parse().map_err(|e| WpArrowError::ParseError {
            field_name: field_name.to_string(),
            detail: format!("invalid prefix length: {e}"),
        })?;
        let net = IpNetValue::new(addr, prefix).ok_or_else(|| WpArrowError::ParseError {
            field_name: field_name.to_string(),
            detail: format!("invalid prefix length {prefix} for {addr}"),
        })?;
        Ok(Value::IpNet(net))
    } else {
        // Try parsing as plain IpAddr
        let addr: IpAddr = s.parse().map_err(|e| WpArrowError::ParseError {
            field_name: field_name.to_string(),
            detail: format!("invalid IP address: {e}"),
        })?;
        Ok(Value::IpAddr(addr))
    }
}

/// Parse a hex string (with optional 0x/0X prefix) into a HexT value.
fn parse_hex_value(s: &str, field_name: &str) -> Result<Value, WpArrowError> {
    let hex_str = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let v = u128::from_str_radix(hex_str, 16).map_err(|e| WpArrowError::ParseError {
        field_name: field_name.to_string(),
        detail: format!("invalid hex: {e}"),
    })?;
    Ok(Value::Hex(HexT(v)))
}

/// Map a WpDataType to the corresponding wp-model-core DataType for Field.meta.
fn wp_type_to_model_meta(wp_type: &WpDataType) -> DataType {
    match wp_type {
        WpDataType::Chars => DataType::Chars,
        WpDataType::Digit => DataType::Digit,
        WpDataType::Float => DataType::Float,
        WpDataType::Bool => DataType::Bool,
        WpDataType::Time => DataType::Time,
        WpDataType::Ip => DataType::IP,
        WpDataType::Hex => DataType::Hex,
        WpDataType::Array(inner) => {
            let inner_name = match inner.as_ref() {
                WpDataType::Chars => "chars",
                WpDataType::Digit => "digit",
                WpDataType::Float => "float",
                WpDataType::Bool => "bool",
                WpDataType::Time => "time",
                WpDataType::Ip => "ip",
                WpDataType::Hex => "hex",
                WpDataType::Array(_) => "array",
            };
            DataType::Array(inner_name.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldDef, WpDataType};
    use arrow::array::AsArray;
    use chrono::NaiveDateTime;
    use std::net::{IpAddr, Ipv4Addr};
    use wp_model_core::model::{DataField, DataRecord, Field, Value};

    // Helper to build a DataRecord from a list of Fields
    fn make_record(fields: Vec<DataField>) -> DataRecord {
        DataRecord::from(fields)
    }

    // =======================================================================
    // records_to_batch tests
    // =======================================================================

    #[test]
    fn r2b_basic_types() {
        let fds = vec![
            FieldDef::new("name", WpDataType::Chars),
            FieldDef::new("count", WpDataType::Digit),
            FieldDef::new("ratio", WpDataType::Float),
            FieldDef::new("active", WpDataType::Bool),
        ];
        let records = vec![
            make_record(vec![
                Field::from_chars("name", "Alice"),
                Field::from_digit("count", 10),
                Field::from_float("ratio", 1.5),
                Field::from_bool("active", true),
            ]),
            make_record(vec![
                Field::from_chars("name", "Bob"),
                Field::from_digit("count", 20),
                Field::from_float("ratio", 2.5),
                Field::from_bool("active", false),
            ]),
        ];

        let batch = records_to_batch(&records, &fds).unwrap();
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.num_rows(), 2);

        let names = batch.column(0).as_string::<i32>();
        assert_eq!(names.value(0), "Alice");
        assert_eq!(names.value(1), "Bob");

        let counts = batch
            .column(1)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(counts.value(0), 10);
        assert_eq!(counts.value(1), 20);

        let ratios = batch
            .column(2)
            .as_primitive::<arrow::datatypes::Float64Type>();
        assert!((ratios.value(0) - 1.5).abs() < f64::EPSILON);

        let actives = batch.column(3).as_boolean();
        assert!(actives.value(0));
        assert!(!actives.value(1));
    }

    #[test]
    fn r2b_time_field() {
        let fds = vec![FieldDef::new("ts", WpDataType::Time)];
        let ndt =
            NaiveDateTime::parse_from_str("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let records = vec![make_record(vec![Field::from_time("ts", ndt)])];

        let batch = records_to_batch(&records, &fds).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let expected_nanos = ndt.and_utc().timestamp_nanos_opt().unwrap();
        assert_eq!(arr.value(0), expected_nanos);
    }

    #[test]
    fn r2b_ip_field() {
        let fds = vec![FieldDef::new("addr", WpDataType::Ip)];
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let net = IpNetValue::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)), 8).unwrap();
        let records = vec![
            make_record(vec![Field::from_ip("addr", ip)]),
            make_record(vec![Field::new(DataType::IP, "addr", Value::IpNet(net))]),
        ];

        let batch = records_to_batch(&records, &fds).unwrap();
        let arr = batch.column(0).as_string::<i32>();
        assert_eq!(arr.value(0), "192.168.1.1");
        assert_eq!(arr.value(1), "10.0.0.0/8");
    }

    #[test]
    fn r2b_hex_field() {
        let fds = vec![FieldDef::new("color", WpDataType::Hex)];
        let records = vec![make_record(vec![Field::from_hex("color", HexT(255))])];

        let batch = records_to_batch(&records, &fds).unwrap();
        let arr = batch.column(0).as_string::<i32>();
        assert_eq!(arr.value(0), "0xFF");
    }

    #[test]
    fn r2b_nullable_missing() {
        let fds = vec![
            FieldDef::new("name", WpDataType::Chars),
            FieldDef::new("opt", WpDataType::Digit), // nullable by default
        ];
        let records = vec![
            make_record(vec![Field::from_chars("name", "Alice")]),
            // "opt" missing => should be null
        ];

        let batch = records_to_batch(&records, &fds).unwrap();
        assert!(batch.column(1).is_null(0));
    }

    #[test]
    fn r2b_required_missing() {
        let fds = vec![FieldDef::new("required", WpDataType::Digit).with_nullable(false)];
        let records = vec![make_record(vec![Field::from_chars("other", "x")])];

        let err = records_to_batch(&records, &fds).unwrap_err();
        assert!(matches!(err, WpArrowError::MissingRequiredField { .. }));
    }

    #[test]
    fn r2b_null_value_nullable() {
        let fds = vec![FieldDef::new("val", WpDataType::Chars)];
        let records = vec![make_record(vec![Field::new(
            DataType::Chars,
            "val",
            Value::Null,
        )])];

        let batch = records_to_batch(&records, &fds).unwrap();
        assert!(batch.column(0).is_null(0));
    }

    #[test]
    fn r2b_empty_records() {
        let fds = vec![FieldDef::new("x", WpDataType::Digit)];
        let records: Vec<DataRecord> = vec![];

        let batch = records_to_batch(&records, &fds).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 1);
    }

    #[test]
    fn r2b_extra_fields_ignored() {
        let fds = vec![FieldDef::new("a", WpDataType::Digit)];
        let records = vec![make_record(vec![
            Field::from_digit("a", 1),
            Field::from_chars("extra", "ignored"),
        ])];

        let batch = records_to_batch(&records, &fds).unwrap();
        assert_eq!(batch.num_columns(), 1);
        let arr = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(arr.value(0), 1);
    }

    #[test]
    fn r2b_array_field() {
        let fds = vec![FieldDef::new(
            "tags",
            WpDataType::Array(Box::new(WpDataType::Digit)),
        )];
        let items: Vec<DataField> =
            vec![Field::from_digit("item", 10), Field::from_digit("item", 20)];
        let records = vec![make_record(vec![Field::from_arr("tags", items)])];

        let batch = records_to_batch(&records, &fds).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(arr.len(), 1);
        let inner = arr.value(0);
        let inner_vals = inner.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(inner_vals.value(0), 10);
        assert_eq!(inner_vals.value(1), 20);
    }

    #[test]
    fn r2b_type_mismatch() {
        let fds = vec![FieldDef::new("num", WpDataType::Digit)];
        let records = vec![make_record(vec![Field::from_chars("num", "not_a_number")])];

        let err = records_to_batch(&records, &fds).unwrap_err();
        assert!(matches!(err, WpArrowError::ValueConversionError { .. }));
    }

    #[test]
    fn r2b_large_batch() {
        let fds = vec![
            FieldDef::new("id", WpDataType::Digit),
            FieldDef::new("name", WpDataType::Chars),
        ];
        let records: Vec<DataRecord> = (0..10000)
            .map(|i| {
                make_record(vec![
                    Field::from_digit("id", i),
                    Field::from_chars("name", format!("row_{i}")),
                ])
            })
            .collect();

        let batch = records_to_batch(&records, &fds).unwrap();
        assert_eq!(batch.num_rows(), 10000);

        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.value(0), 0);
        assert_eq!(ids.value(9999), 9999);
    }

    // =======================================================================
    // batch_to_records tests
    // =======================================================================

    #[test]
    fn b2r_basic_types() {
        let fds = vec![
            FieldDef::new("name", WpDataType::Chars),
            FieldDef::new("count", WpDataType::Digit),
            FieldDef::new("ratio", WpDataType::Float),
            FieldDef::new("active", WpDataType::Bool),
        ];
        // Build batch from records first
        let records_in = vec![make_record(vec![
            Field::from_chars("name", "Alice"),
            Field::from_digit("count", 42),
            Field::from_float("ratio", 1.23),
            Field::from_bool("active", true),
        ])];
        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out.len(), 1);
        let rec = &records_out[0];
        assert_eq!(
            rec.get_value("name"),
            Some(&Value::Chars(FValueStr::from("Alice")))
        );
        assert_eq!(rec.get_value("count"), Some(&Value::Digit(42)));
        assert_eq!(rec.get_value("ratio"), Some(&Value::Float(1.23)));
        assert_eq!(rec.get_value("active"), Some(&Value::Bool(true)));
    }

    #[test]
    fn b2r_timestamp() {
        let fds = vec![FieldDef::new("ts", WpDataType::Time)];
        let ndt =
            NaiveDateTime::parse_from_str("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let records_in = vec![make_record(vec![Field::from_time("ts", ndt)])];
        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out[0].get_value("ts"), Some(&Value::Time(ndt)));
    }

    #[test]
    fn b2r_ip_parsing() {
        let fds = vec![FieldDef::new("addr", WpDataType::Ip)];
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let net = IpNetValue::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)), 8).unwrap();
        let records_in = vec![
            make_record(vec![Field::from_ip("addr", ip)]),
            make_record(vec![Field::new(
                DataType::IP,
                "addr",
                Value::IpNet(net.clone()),
            )]),
        ];
        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out[0].get_value("addr"), Some(&Value::IpAddr(ip)));
        assert_eq!(records_out[1].get_value("addr"), Some(&Value::IpNet(net)));
    }

    #[test]
    fn b2r_hex_parsing() {
        let fds = vec![FieldDef::new("color", WpDataType::Hex)];
        let records_in = vec![make_record(vec![Field::from_hex("color", HexT(255))])];
        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(
            records_out[0].get_value("color"),
            Some(&Value::Hex(HexT(255)))
        );
    }

    #[test]
    fn b2r_sequential_ids() {
        let fds = vec![FieldDef::new("x", WpDataType::Digit)];
        let records_in = vec![
            make_record(vec![Field::from_digit("x", 1)]),
            make_record(vec![Field::from_digit("x", 2)]),
            make_record(vec![Field::from_digit("x", 3)]),
        ];
        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out[0].id, 0);
        assert_eq!(records_out[1].id, 1);
        assert_eq!(records_out[2].id, 2);
    }

    #[test]
    fn b2r_schema_mismatch() {
        let fds_2 = vec![
            FieldDef::new("a", WpDataType::Digit),
            FieldDef::new("b", WpDataType::Digit),
        ];
        let fds_1 = vec![FieldDef::new("a", WpDataType::Digit)];
        let records = vec![make_record(vec![Field::from_digit("a", 1)])];
        let batch = records_to_batch(&records, &fds_1).unwrap();

        let err = batch_to_records(&batch, &fds_2).unwrap_err();
        assert!(matches!(
            err,
            WpArrowError::SchemaMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    // =======================================================================
    // Roundtrip tests
    // =======================================================================

    #[test]
    fn roundtrip_all_types() {
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let net = IpNetValue::new(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 0)), 12).unwrap();
        let ndt =
            NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();

        let fds = vec![
            FieldDef::new("chars", WpDataType::Chars),
            FieldDef::new("digit", WpDataType::Digit),
            FieldDef::new("float", WpDataType::Float),
            FieldDef::new("bool", WpDataType::Bool),
            FieldDef::new("time", WpDataType::Time),
            FieldDef::new("ip", WpDataType::Ip),
            FieldDef::new("hex", WpDataType::Hex),
            FieldDef::new("nums", WpDataType::Array(Box::new(WpDataType::Digit))),
        ];

        let arr_items: Vec<DataField> = vec![
            Field::from_digit("item", 100),
            Field::from_digit("item", 200),
        ];

        let records_in = vec![
            make_record(vec![
                Field::from_chars("chars", "hello"),
                Field::from_digit("digit", 42),
                Field::from_float("float", 9.876),
                Field::from_bool("bool", true),
                Field::from_time("time", ndt),
                Field::from_ip("ip", ip),
                Field::from_hex("hex", HexT(0xDEAD)),
                Field::from_arr("nums", arr_items),
            ]),
            make_record(vec![
                Field::from_chars("chars", "world"),
                Field::from_digit("digit", -1),
                Field::from_float("float", 0.0),
                Field::from_bool("bool", false),
                Field::from_time("time", ndt),
                Field::new(DataType::IP, "ip", Value::IpNet(net.clone())),
                Field::from_hex("hex", HexT(0)),
                Field::from_arr("nums", vec![Field::from_digit("item", 300)]),
            ]),
        ];

        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out.len(), 2);

        // Row 0
        assert_eq!(
            records_out[0].get_value("chars"),
            Some(&Value::Chars(FValueStr::from("hello")))
        );
        assert_eq!(records_out[0].get_value("digit"), Some(&Value::Digit(42)));
        assert_eq!(
            records_out[0].get_value("float"),
            Some(&Value::Float(9.876))
        );
        assert_eq!(records_out[0].get_value("bool"), Some(&Value::Bool(true)));
        assert_eq!(records_out[0].get_value("time"), Some(&Value::Time(ndt)));
        assert_eq!(records_out[0].get_value("ip"), Some(&Value::IpAddr(ip)));
        assert_eq!(
            records_out[0].get_value("hex"),
            Some(&Value::Hex(HexT(0xDEAD)))
        );

        // Verify array field
        if let Some(Value::Array(items)) = records_out[0].get_value("nums") {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].get_value(), &Value::Digit(100));
            assert_eq!(items[1].get_value(), &Value::Digit(200));
        } else {
            panic!("expected Array value for 'nums'");
        }

        // Row 1
        assert_eq!(records_out[1].get_value("ip"), Some(&Value::IpNet(net)));
        assert_eq!(records_out[1].get_value("hex"), Some(&Value::Hex(HexT(0))));
    }

    #[test]
    fn roundtrip_with_nulls() {
        let fds = vec![
            FieldDef::new("name", WpDataType::Chars),
            FieldDef::new("opt_digit", WpDataType::Digit),
        ];

        let records_in = vec![
            make_record(vec![
                Field::from_chars("name", "row1"),
                Field::from_digit("opt_digit", 100),
            ]),
            make_record(vec![
                Field::from_chars("name", "row2"),
                // opt_digit missing => null
            ]),
        ];

        let batch = records_to_batch(&records_in, &fds).unwrap();
        let records_out = batch_to_records(&batch, &fds).unwrap();

        assert_eq!(records_out.len(), 2);
        assert_eq!(
            records_out[0].get_value("opt_digit"),
            Some(&Value::Digit(100))
        );
        // null field should be absent from the record (we skip nulls in batch_to_records)
        assert_eq!(records_out[1].get_value("opt_digit"), None);
    }
}
