use arrow::array::RecordBatch;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;

use crate::error::WpArrowError;

/// Decoded IPC data frame.
pub struct IpcFrame {
    pub tag: String,
    pub batch: RecordBatch,
}

/// Encode a RecordBatch into an IPC frame: `[4B tag_len BE][tag bytes][Arrow IPC stream]`.
pub fn encode_ipc(tag: &str, batch: &RecordBatch) -> Result<Vec<u8>, WpArrowError> {
    let tag_bytes = tag.as_bytes();
    let tag_len = tag_bytes.len() as u32;

    let mut buf = Vec::new();
    buf.extend_from_slice(&tag_len.to_be_bytes());
    buf.extend_from_slice(tag_bytes);

    let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
        .map_err(|e| WpArrowError::IpcEncodeError(e.to_string()))?;
    writer
        .write(batch)
        .map_err(|e| WpArrowError::IpcEncodeError(e.to_string()))?;
    writer
        .finish()
        .map_err(|e| WpArrowError::IpcEncodeError(e.to_string()))?;

    Ok(buf)
}

/// Decode a complete IPC frame from bytes.
pub fn decode_ipc(data: &[u8]) -> Result<IpcFrame, WpArrowError> {
    if data.len() < 4 {
        return Err(WpArrowError::IpcDecodeError(format!(
            "frame too short: {} bytes, minimum 4",
            data.len()
        )));
    }

    let tag_len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    let tag_end = 4 + tag_len;
    if data.len() < tag_end {
        return Err(WpArrowError::IpcDecodeError(format!(
            "frame truncated: tag_len={tag_len} but only {} bytes remain after header",
            data.len() - 4
        )));
    }

    let tag = String::from_utf8(data[4..tag_end].to_vec())
        .map_err(|e| WpArrowError::IpcDecodeError(format!("invalid UTF-8 in tag: {e}")))?;

    let ipc_payload = &data[tag_end..];
    let mut reader = StreamReader::try_new(ipc_payload, None)
        .map_err(|e| WpArrowError::IpcDecodeError(e.to_string()))?;

    let batch = reader
        .next()
        .ok_or_else(|| WpArrowError::IpcDecodeError("no RecordBatch in IPC payload".to_string()))?
        .map_err(|e| WpArrowError::IpcDecodeError(e.to_string()))?;

    Ok(IpcFrame { tag, batch })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(num_rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let ids: Vec<i32> = (0..num_rows as i32).collect();
        let names: Vec<Option<&str>> = (0..num_rows)
            .map(|i| if i % 2 == 0 { Some("even") } else { None })
            .collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn ipc_roundtrip_basic() {
        let batch = make_batch(5);
        let encoded = encode_ipc("test-tag", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, "test-tag");
        assert_eq!(frame.batch.num_rows(), batch.num_rows());
        assert_eq!(frame.batch.num_columns(), batch.num_columns());
        assert_eq!(frame.batch.schema(), batch.schema());
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_roundtrip_empty_batch() {
        let batch = make_batch(0);
        let encoded = encode_ipc("empty", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.batch.num_rows(), 0);
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_roundtrip_large_batch() {
        let batch = make_batch(1000);
        let encoded = encode_ipc("large", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.batch.num_rows(), 1000);
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_tag_preserved() {
        let batch = make_batch(1);
        let encoded = encode_ipc("my-tag", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, "my-tag");
    }

    #[test]
    fn ipc_utf8_tag() {
        let batch = make_batch(1);
        let tag = "数据标签-🚀";
        let encoded = encode_ipc(tag, &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, tag);
    }

    #[test]
    fn ipc_empty_tag() {
        let batch = make_batch(1);
        let encoded = encode_ipc("", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, "");
    }

    #[test]
    fn decode_ipc_too_short() {
        let result = decode_ipc(&[0x00; 2]);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }

    #[test]
    fn decode_ipc_truncated_tag() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_be_bytes()); // tag_len = 100 but no data
        let result = decode_ipc(&data);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }
}
