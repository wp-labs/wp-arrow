use arrow::array::RecordBatch;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;

use crate::error::WpArrowError;

const FRAME_TYPE_IPC: u8 = 0x01;
const FRAME_TYPE_ACK: u8 = 0x02;
const IPC_HEADER_MIN_SIZE: usize = 21; // 1 + 8 + 8 + 4
const ACK_FRAME_SIZE: usize = 17; // 1 + 8 + 8

/// Decoded IPC data frame.
pub struct IpcFrame {
    pub source_id: u64,
    pub batch_seq: u64,
    pub tag: String,
    pub batch: RecordBatch,
}

/// Decoded ACK confirmation frame.
pub struct AckFrame {
    pub source_id: u64,
    pub ack_seq: u64,
}

/// Encode a RecordBatch into an IPC data frame with header fields.
pub fn encode_ipc(
    source_id: u64,
    batch_seq: u64,
    tag: &str,
    batch: &RecordBatch,
) -> Result<Vec<u8>, WpArrowError> {
    let tag_bytes = tag.as_bytes();
    let tag_len = tag_bytes.len() as u32;

    let mut buf = Vec::new();
    buf.push(FRAME_TYPE_IPC);
    buf.extend_from_slice(&source_id.to_le_bytes());
    buf.extend_from_slice(&batch_seq.to_le_bytes());
    buf.extend_from_slice(&tag_len.to_le_bytes());
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

/// Decode a complete IPC data frame from bytes.
pub fn decode_ipc(data: &[u8]) -> Result<IpcFrame, WpArrowError> {
    if data.len() < IPC_HEADER_MIN_SIZE {
        return Err(WpArrowError::IpcDecodeError(format!(
            "frame too short: {} bytes, minimum {}",
            data.len(),
            IPC_HEADER_MIN_SIZE
        )));
    }
    if data[0] != FRAME_TYPE_IPC {
        return Err(WpArrowError::IpcDecodeError(format!(
            "unexpected frame type: 0x{:02X}, expected 0x{:02X}",
            data[0], FRAME_TYPE_IPC
        )));
    }

    let source_id = u64::from_le_bytes(data[1..9].try_into().unwrap());
    let batch_seq = u64::from_le_bytes(data[9..17].try_into().unwrap());
    let tag_len = u32::from_le_bytes(data[17..21].try_into().unwrap()) as usize;

    let tag_end = 21 + tag_len;
    if data.len() < tag_end {
        return Err(WpArrowError::IpcDecodeError(format!(
            "frame truncated: tag_len={tag_len} but only {} bytes remain after header",
            data.len() - 21
        )));
    }

    let tag = String::from_utf8(data[21..tag_end].to_vec())
        .map_err(|e| WpArrowError::IpcDecodeError(format!("invalid UTF-8 in tag: {e}")))?;

    let ipc_payload = &data[tag_end..];
    let mut reader = StreamReader::try_new(ipc_payload, None)
        .map_err(|e| WpArrowError::IpcDecodeError(e.to_string()))?;

    let batch = reader
        .next()
        .ok_or_else(|| WpArrowError::IpcDecodeError("no RecordBatch in IPC payload".to_string()))?
        .map_err(|e| WpArrowError::IpcDecodeError(e.to_string()))?;

    Ok(IpcFrame {
        source_id,
        batch_seq,
        tag,
        batch,
    })
}

/// Encode an ACK frame (fixed 17 bytes).
pub fn encode_ack(source_id: u64, ack_seq: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ACK_FRAME_SIZE);
    buf.push(FRAME_TYPE_ACK);
    buf.extend_from_slice(&source_id.to_le_bytes());
    buf.extend_from_slice(&ack_seq.to_le_bytes());
    buf
}

/// Decode an ACK frame from bytes.
pub fn decode_ack(data: &[u8]) -> Result<AckFrame, WpArrowError> {
    if data.len() != ACK_FRAME_SIZE {
        return Err(WpArrowError::IpcDecodeError(format!(
            "ACK frame size mismatch: expected {} bytes, got {}",
            ACK_FRAME_SIZE,
            data.len()
        )));
    }
    if data[0] != FRAME_TYPE_ACK {
        return Err(WpArrowError::IpcDecodeError(format!(
            "unexpected frame type: 0x{:02X}, expected 0x{:02X}",
            data[0], FRAME_TYPE_ACK
        )));
    }

    let source_id = u64::from_le_bytes(data[1..9].try_into().unwrap());
    let ack_seq = u64::from_le_bytes(data[9..17].try_into().unwrap());

    Ok(AckFrame { source_id, ack_seq })
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
        let encoded = encode_ipc(1, 10, "test-tag", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.batch.num_rows(), batch.num_rows());
        assert_eq!(frame.batch.num_columns(), batch.num_columns());
        assert_eq!(frame.batch.schema(), batch.schema());
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_roundtrip_empty_batch() {
        let batch = make_batch(0);
        let encoded = encode_ipc(0, 0, "empty", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.batch.num_rows(), 0);
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_roundtrip_large_batch() {
        let batch = make_batch(1000);
        let encoded = encode_ipc(42, 999, "large", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.batch.num_rows(), 1000);
        assert_eq!(frame.batch, batch);
    }

    #[test]
    fn ipc_header_fields() {
        let batch = make_batch(1);
        let encoded = encode_ipc(123, 456, "my-tag", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.source_id, 123);
        assert_eq!(frame.batch_seq, 456);
        assert_eq!(frame.tag, "my-tag");
    }

    #[test]
    fn ipc_utf8_tag() {
        let batch = make_batch(1);
        let tag = "数据标签-🚀";
        let encoded = encode_ipc(1, 1, tag, &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, tag);
    }

    #[test]
    fn ipc_empty_tag() {
        let batch = make_batch(1);
        let encoded = encode_ipc(1, 1, "", &batch).unwrap();
        let frame = decode_ipc(&encoded).unwrap();
        assert_eq!(frame.tag, "");
    }

    #[test]
    fn ack_roundtrip() {
        let encoded = encode_ack(7, 42);
        assert_eq!(encoded.len(), ACK_FRAME_SIZE);
        let frame = decode_ack(&encoded).unwrap();
        assert_eq!(frame.source_id, 7);
        assert_eq!(frame.ack_seq, 42);
    }

    #[test]
    fn ack_max_values() {
        let encoded = encode_ack(u64::MAX, u64::MAX);
        let frame = decode_ack(&encoded).unwrap();
        assert_eq!(frame.source_id, u64::MAX);
        assert_eq!(frame.ack_seq, u64::MAX);
    }

    #[test]
    fn decode_ipc_too_short() {
        let result = decode_ipc(&[0x01; 10]);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }

    #[test]
    fn decode_ipc_wrong_type() {
        let mut data = vec![0x02]; // ACK type byte
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        let result = decode_ipc(&data);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }

    #[test]
    fn decode_ipc_truncated_tag() {
        let mut data = vec![FRAME_TYPE_IPC];
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&100u32.to_le_bytes()); // tag_len = 100 but no data
        let result = decode_ipc(&data);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }

    #[test]
    fn decode_ack_wrong_size() {
        let result = decode_ack(&[0x02; 10]);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }

    #[test]
    fn decode_ack_wrong_type() {
        let mut data = vec![FRAME_TYPE_IPC]; // IPC type byte instead of ACK
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let result = decode_ack(&data);
        assert!(matches!(result, Err(WpArrowError::IpcDecodeError(_))));
    }
}
