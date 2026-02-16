use std::sync::Arc;

use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema, TimeUnit};

use crate::error::WpArrowError;

/// WPL data types that can be mapped to Apache Arrow types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WpDataType {
    Chars,
    Digit,
    Float,
    Bool,
    Time,
    Ip,
    Hex,
    Array(Box<WpDataType>),
}

/// A named, typed field definition for building Arrow schemas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
    pub name: String,
    pub data_type: WpDataType,
    pub nullable: bool,
}

impl FieldDef {
    pub fn new(name: impl Into<String>, data_type: WpDataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: true,
        }
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// Maps a [`WpDataType`] to the corresponding [`ArrowDataType`].
pub fn to_arrow_type(wp_type: &WpDataType) -> ArrowDataType {
    match wp_type {
        WpDataType::Chars => ArrowDataType::Utf8,
        WpDataType::Digit => ArrowDataType::Int64,
        WpDataType::Float => ArrowDataType::Float64,
        WpDataType::Bool => ArrowDataType::Boolean,
        WpDataType::Time => ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
        WpDataType::Ip => ArrowDataType::Utf8,
        WpDataType::Hex => ArrowDataType::Utf8,
        WpDataType::Array(inner) => {
            let inner_arrow = to_arrow_type(inner);
            ArrowDataType::List(Arc::new(ArrowField::new("item", inner_arrow, true)))
        }
    }
}

/// Converts a [`FieldDef`] into an Arrow [`ArrowField`].
///
/// Returns an error if the field name is empty.
pub fn to_arrow_field(field: &FieldDef) -> Result<ArrowField, WpArrowError> {
    if field.name.is_empty() {
        return Err(WpArrowError::EmptyFieldName);
    }
    let arrow_type = to_arrow_type(&field.data_type);
    Ok(ArrowField::new(&field.name, arrow_type, field.nullable))
}

/// Converts a slice of [`FieldDef`] into an Arrow [`Schema`].
pub fn to_arrow_schema(fields: &[FieldDef]) -> Result<Schema, WpArrowError> {
    let arrow_fields: Vec<ArrowField> = fields
        .iter()
        .map(to_arrow_field)
        .collect::<Result<_, _>>()?;
    Ok(Schema::new(arrow_fields))
}

/// Parses a WPL type string into a [`WpDataType`].
///
/// Supported formats:
/// - Basic types: `"chars"`, `"digit"`, `"float"`, `"bool"`, `"time"`, `"ip"`, `"hex"`
/// - Array types: `"array<chars>"`, `"array<array<digit>>"`
///
/// Type names are case-insensitive.
pub fn parse_wp_type(s: &str) -> Result<WpDataType, WpArrowError> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();

    match lower.as_str() {
        "chars" => Ok(WpDataType::Chars),
        "digit" => Ok(WpDataType::Digit),
        "float" => Ok(WpDataType::Float),
        "bool" => Ok(WpDataType::Bool),
        "time" => Ok(WpDataType::Time),
        "ip" => Ok(WpDataType::Ip),
        "hex" => Ok(WpDataType::Hex),
        _ if lower.starts_with("array<") && lower.ends_with('>') => {
            let inner_str = &s[6..s.len() - 1];
            let inner_trimmed = inner_str.trim();
            if inner_trimmed.is_empty() {
                return Err(WpArrowError::InvalidArrayInnerType(String::new()));
            }
            let inner = parse_wp_type(inner_trimmed)?;
            Ok(WpDataType::Array(Box::new(inner)))
        }
        _ => Err(WpArrowError::UnsupportedDataType(s.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // to_arrow_type: basic types
    // ---------------------------------------------------------------

    #[test]
    fn arrow_type_chars() {
        assert_eq!(to_arrow_type(&WpDataType::Chars), ArrowDataType::Utf8);
    }

    #[test]
    fn arrow_type_digit() {
        assert_eq!(to_arrow_type(&WpDataType::Digit), ArrowDataType::Int64);
    }

    #[test]
    fn arrow_type_float() {
        assert_eq!(to_arrow_type(&WpDataType::Float), ArrowDataType::Float64);
    }

    #[test]
    fn arrow_type_bool() {
        assert_eq!(to_arrow_type(&WpDataType::Bool), ArrowDataType::Boolean);
    }

    #[test]
    fn arrow_type_time() {
        assert_eq!(
            to_arrow_type(&WpDataType::Time),
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
        );
    }

    #[test]
    fn arrow_type_ip() {
        assert_eq!(to_arrow_type(&WpDataType::Ip), ArrowDataType::Utf8);
    }

    #[test]
    fn arrow_type_hex() {
        assert_eq!(to_arrow_type(&WpDataType::Hex), ArrowDataType::Utf8);
    }

    // ---------------------------------------------------------------
    // to_arrow_type: array types
    // ---------------------------------------------------------------

    #[test]
    fn arrow_type_array_digit() {
        let wp = WpDataType::Array(Box::new(WpDataType::Digit));
        let arrow = to_arrow_type(&wp);
        assert_eq!(
            arrow,
            ArrowDataType::List(Arc::new(ArrowField::new("item", ArrowDataType::Int64, true)))
        );
    }

    #[test]
    fn arrow_type_array_chars() {
        let wp = WpDataType::Array(Box::new(WpDataType::Chars));
        let arrow = to_arrow_type(&wp);
        assert_eq!(
            arrow,
            ArrowDataType::List(Arc::new(ArrowField::new("item", ArrowDataType::Utf8, true)))
        );
    }

    #[test]
    fn arrow_type_nested_array() {
        let wp = WpDataType::Array(Box::new(WpDataType::Array(Box::new(WpDataType::Float))));
        let inner_list =
            ArrowDataType::List(Arc::new(ArrowField::new("item", ArrowDataType::Float64, true)));
        let expected = ArrowDataType::List(Arc::new(ArrowField::new("item", inner_list, true)));
        assert_eq!(to_arrow_type(&wp), expected);
    }

    // ---------------------------------------------------------------
    // to_arrow_field
    // ---------------------------------------------------------------

    #[test]
    fn arrow_field_basic() {
        let fd = FieldDef::new("src_ip", WpDataType::Ip);
        let field = to_arrow_field(&fd).unwrap();
        assert_eq!(field.name(), "src_ip");
        assert_eq!(field.data_type(), &ArrowDataType::Utf8);
        assert!(field.is_nullable());
    }

    #[test]
    fn arrow_field_non_nullable() {
        let fd = FieldDef::new("count", WpDataType::Digit).with_nullable(false);
        let field = to_arrow_field(&fd).unwrap();
        assert!(!field.is_nullable());
    }

    #[test]
    fn arrow_field_empty_name_errors() {
        let fd = FieldDef::new("", WpDataType::Chars);
        assert_eq!(to_arrow_field(&fd), Err(WpArrowError::EmptyFieldName));
    }

    // ---------------------------------------------------------------
    // to_arrow_schema
    // ---------------------------------------------------------------

    #[test]
    fn arrow_schema_firewall_log() {
        let fields = vec![
            FieldDef::new("src_ip", WpDataType::Ip),
            FieldDef::new("dst_ip", WpDataType::Ip),
            FieldDef::new("port", WpDataType::Digit),
            FieldDef::new("protocol", WpDataType::Chars),
            FieldDef::new("timestamp", WpDataType::Time),
            FieldDef::new("allowed", WpDataType::Bool),
        ];
        let schema = to_arrow_schema(&fields).unwrap();
        assert_eq!(schema.fields().len(), 6);
        assert_eq!(schema.field(0).name(), "src_ip");
        assert_eq!(schema.field(2).data_type(), &ArrowDataType::Int64);
        assert_eq!(
            schema.field(4).data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
        );
    }

    #[test]
    fn arrow_schema_with_array_field() {
        let fields = vec![
            FieldDef::new("name", WpDataType::Chars),
            FieldDef::new("tags", WpDataType::Array(Box::new(WpDataType::Chars))),
        ];
        let schema = to_arrow_schema(&fields).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert!(matches!(schema.field(1).data_type(), ArrowDataType::List(_)));
    }

    #[test]
    fn arrow_schema_empty_fields() {
        let schema = to_arrow_schema(&[]).unwrap();
        assert_eq!(schema.fields().len(), 0);
    }

    #[test]
    fn arrow_schema_error_propagation() {
        let fields = vec![
            FieldDef::new("ok", WpDataType::Chars),
            FieldDef::new("", WpDataType::Digit),
        ];
        assert_eq!(to_arrow_schema(&fields), Err(WpArrowError::EmptyFieldName));
    }

    // ---------------------------------------------------------------
    // parse_wp_type: basic types
    // ---------------------------------------------------------------

    #[test]
    fn parse_chars() {
        assert_eq!(parse_wp_type("chars"), Ok(WpDataType::Chars));
    }

    #[test]
    fn parse_digit() {
        assert_eq!(parse_wp_type("digit"), Ok(WpDataType::Digit));
    }

    #[test]
    fn parse_float() {
        assert_eq!(parse_wp_type("float"), Ok(WpDataType::Float));
    }

    #[test]
    fn parse_bool() {
        assert_eq!(parse_wp_type("bool"), Ok(WpDataType::Bool));
    }

    #[test]
    fn parse_time() {
        assert_eq!(parse_wp_type("time"), Ok(WpDataType::Time));
    }

    #[test]
    fn parse_ip() {
        assert_eq!(parse_wp_type("ip"), Ok(WpDataType::Ip));
    }

    #[test]
    fn parse_hex() {
        assert_eq!(parse_wp_type("hex"), Ok(WpDataType::Hex));
    }

    // ---------------------------------------------------------------
    // parse_wp_type: case insensitivity
    // ---------------------------------------------------------------

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(parse_wp_type("CHARS"), Ok(WpDataType::Chars));
        assert_eq!(parse_wp_type("Digit"), Ok(WpDataType::Digit));
        assert_eq!(parse_wp_type("BOOL"), Ok(WpDataType::Bool));
    }

    // ---------------------------------------------------------------
    // parse_wp_type: array types
    // ---------------------------------------------------------------

    #[test]
    fn parse_array_chars() {
        assert_eq!(
            parse_wp_type("array<chars>"),
            Ok(WpDataType::Array(Box::new(WpDataType::Chars)))
        );
    }

    #[test]
    fn parse_array_digit() {
        assert_eq!(
            parse_wp_type("array<digit>"),
            Ok(WpDataType::Array(Box::new(WpDataType::Digit)))
        );
    }

    #[test]
    fn parse_nested_array() {
        assert_eq!(
            parse_wp_type("array<array<float>>"),
            Ok(WpDataType::Array(Box::new(WpDataType::Array(Box::new(
                WpDataType::Float
            )))))
        );
    }

    #[test]
    fn parse_array_with_whitespace() {
        assert_eq!(
            parse_wp_type("  array< chars >  "),
            Ok(WpDataType::Array(Box::new(WpDataType::Chars)))
        );
    }

    // ---------------------------------------------------------------
    // parse_wp_type: error cases
    // ---------------------------------------------------------------

    #[test]
    fn parse_unsupported_type() {
        let err = parse_wp_type("unknown").unwrap_err();
        assert_eq!(err, WpArrowError::UnsupportedDataType("unknown".to_string()));
    }

    #[test]
    fn parse_array_empty_inner() {
        let err = parse_wp_type("array<>").unwrap_err();
        assert_eq!(err, WpArrowError::InvalidArrayInnerType(String::new()));
    }

    #[test]
    fn parse_array_invalid_inner() {
        let err = parse_wp_type("array<invalid>").unwrap_err();
        assert_eq!(
            err,
            WpArrowError::UnsupportedDataType("invalid".to_string())
        );
    }

    // ---------------------------------------------------------------
    // Property tests: Clone, Eq, Hash, FieldDef defaults
    // ---------------------------------------------------------------

    #[test]
    fn wf_data_type_clone_eq() {
        let a = WpDataType::Array(Box::new(WpDataType::Chars));
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn wf_data_type_hash_consistent() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WpDataType::Digit);
        set.insert(WpDataType::Digit);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn field_def_default_nullable() {
        let fd = FieldDef::new("test", WpDataType::Bool);
        assert!(fd.nullable);
    }
}
