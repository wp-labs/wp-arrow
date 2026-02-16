use std::fmt;

/// Errors that can occur during WPL-to-Arrow type conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WpArrowError {
    /// The given type name is not a recognized WPL data type.
    UnsupportedDataType(String),
    /// An `array<...>` type has an invalid or missing inner type.
    InvalidArrayInnerType(String),
    /// A field definition has an empty name.
    EmptyFieldName,
    /// A required (non-nullable) field is missing from the record.
    MissingRequiredField { field_name: String },
    /// Wrapper for `arrow::error::ArrowError` (stored as String because ArrowError is not Clone/Eq).
    ArrowBuildError(String),
    /// A field value does not match the expected WpDataType.
    ValueConversionError {
        field_name: String,
        expected: String,
        actual: String,
    },
    /// The number of FieldDefs does not match the number of columns in the RecordBatch.
    SchemaMismatch { expected: usize, actual: usize },
    /// A NaiveDateTime value overflows the i64 nanosecond representation.
    TimestampOverflow { field_name: String },
    /// Failed to parse a string value back into the expected type.
    ParseError { field_name: String, detail: String },
    /// IPC encoding failed (wraps Arrow IPC writer errors).
    IpcEncodeError(String),
    /// IPC decoding failed (malformed frame, incomplete data, etc.).
    IpcDecodeError(String),
}

impl fmt::Display for WpArrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WpArrowError::UnsupportedDataType(t) => {
                write!(f, "unsupported WPL data type: {t}")
            }
            WpArrowError::InvalidArrayInnerType(t) => {
                write!(f, "invalid array inner type: {t}")
            }
            WpArrowError::EmptyFieldName => {
                write!(f, "field name must not be empty")
            }
            WpArrowError::MissingRequiredField { field_name } => {
                write!(f, "missing required field: {field_name}")
            }
            WpArrowError::ArrowBuildError(msg) => {
                write!(f, "arrow build error: {msg}")
            }
            WpArrowError::ValueConversionError {
                field_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "value conversion error for field '{field_name}': expected {expected}, got {actual}"
                )
            }
            WpArrowError::SchemaMismatch { expected, actual } => {
                write!(
                    f,
                    "schema mismatch: expected {expected} columns, got {actual}"
                )
            }
            WpArrowError::TimestampOverflow { field_name } => {
                write!(
                    f,
                    "timestamp overflow for field '{field_name}': value out of i64 nanosecond range"
                )
            }
            WpArrowError::ParseError { field_name, detail } => {
                write!(f, "parse error for field '{field_name}': {detail}")
            }
            WpArrowError::IpcEncodeError(msg) => {
                write!(f, "IPC encode error: {msg}")
            }
            WpArrowError::IpcDecodeError(msg) => {
                write!(f, "IPC decode error: {msg}")
            }
        }
    }
}

impl std::error::Error for WpArrowError {}
