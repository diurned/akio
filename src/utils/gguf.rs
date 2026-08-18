#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

// Arrays larger than this are counted but not stored (e.g. tokenizer vocab ~150k strings).
const MAX_ARRAY_STORE: u64 = 4096;

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    LargeArray { elem_type: u32, count: u64 },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Int64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint8(v) => Some(*v as u64),
            GgufValue::Uint16(v) => Some(*v as u64),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Number of elements in an array value (small or large).
    pub fn array_len(&self) -> Option<u64> {
        match self {
            GgufValue::Array(arr) => Some(arr.len() as u64),
            GgufValue::LargeArray { count, .. } => Some(*count),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> u64 {
        let elements = self.element_count();
        let (type_size, block_size) = tensor_type_layout(self.tensor_type);
        elements.saturating_mul(type_size) / block_size
    }

    pub fn dim1(&self) -> u64 {
        self.shape.get(1).copied().unwrap_or(0)
    }
}

// (type_size, block_size) per ggml_type enum in ggml.h.
// bytes = elements * type_size / block_size
fn tensor_type_layout(t: u32) -> (u64, u64) {
    match t {
        0 => (4, 1),                                   // F32
        1 => (2, 1),                                   // F16
        2 => (2 + 32 / 2, 32),                         // Q4_0   (18, 32)
        3 => (2 + 2 + 32 / 2, 32),                     // Q4_1   (20, 32)
        6 => (2 + 4 + 32 / 2, 32),                     // Q5_0   (22, 32)
        7 => (2 + 2 + 4 + 32 / 2, 32),                 // Q5_1   (24, 32)
        8 => (2 + 32, 32),                             // Q8_0   (34, 32)
        9 => (2 + 2 + 32, 32),                         // Q8_1   (36, 32)
        10 => (256 / 16 + 256 / 4 + 4, 256),           // Q2_K   (84, 256)
        11 => (256 / 8 + 256 / 4 + 14, 256),           // Q3_K  (110, 256)
        12 => (2 + 2 + 12 + 256 / 2, 256),             // Q4_K  (144, 256)
        13 => (2 + 2 + 12 + 256 / 8 + 256 / 2, 256),   // Q5_K (176, 256)
        14 => (256 / 2 + 256 / 4 + 256 / 16 + 2, 256), // Q6_K (210, 256)
        15 => (4 + 256 + 2 * 256 / 16, 256),           // Q8_K  (292, 256)
        // IQ types – approximate as slightly under 2 bytes per element
        16..=23 | 29 => (2, 1),
        24 => (1, 1), // I8
        25 => (2, 1), // I16
        26 => (4, 1), // I32
        27 => (8, 1), // I64
        28 => (8, 1), // F64
        30 => (2, 1), // BF16
        _ => (2, 1),  // unknown – assume F16
    }
}

#[derive(Debug)]
pub struct GgufMeta {
    pub version: u32,
    pub kv: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufMeta {
    fn arch_str(&self) -> &str {
        self.kv
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    }

    pub fn get_kv(&self, key: &str) -> Option<&GgufValue> {
        if !key.starts_with("general.") && !key.starts_with("tokenizer.") {
            let prefixed = format!("{}.{}", self.arch_str(), key);
            if let Some(v) = self.kv.get(&prefixed) {
                return Some(v);
            }
        }
        self.kv.get(key)
    }

    pub fn architecture(&self) -> String {
        self.arch_str().to_owned()
    }

    pub fn block_count(&self) -> u64 {
        self.get_kv("block_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    pub fn embedding_length(&self) -> u64 {
        self.get_kv("embedding_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    pub fn head_count(&self) -> u64 {
        self.get_kv("attention.head_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
    }

    pub fn head_count_kv(&self) -> u64 {
        self.get_kv("attention.head_count_kv")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| self.head_count())
    }

    pub fn context_length(&self) -> u64 {
        self.get_kv("context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048)
    }

    fn embedding_head_size(&self) -> u64 {
        let heads = self.head_count();
        if heads > 0 {
            self.embedding_length() / heads
        } else {
            0
        }
    }

    pub fn embedding_head_k(&self) -> u64 {
        self.get_kv("attention.key_length")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| self.embedding_head_size())
    }

    pub fn embedding_head_v(&self) -> u64 {
        self.get_kv("attention.value_length")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| self.embedding_head_size())
    }

    pub fn vocab_size(&self) -> u64 {
        // Primary source: tokenizer token array length.
        if let Some(v) = self.kv.get("tokenizer.ggml.tokens") {
            if let Some(n) = v.array_len() {
                return n;
            }
        }
        // fallback: explicit vocab_size key
        self.get_kv("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32_000)
    }

    pub fn feed_forward_length(&self) -> u64 {
        self.get_kv("feed_forward_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[inline]
fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(b[0])
}

#[inline]
fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(u16::from_le_bytes(b))
}

#[inline]
fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(f32::from_le_bytes(b))
}

#[inline]
fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)
        .map_err(|e| format!("read error: {e}"))?;
    Ok(f64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read error: {e}"))?;
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8 in GGUF string: {e}"))
}

/// Read and discard a single GGUF value of the given type.
fn skip_gguf_value<R: Read>(r: &mut R, type_id: u32) -> Result<(), String> {
    match type_id {
        0 | 1 | 7 => {
            read_u8(r)?;
        }
        2 | 3 => {
            read_u16_le(r)?;
        }
        4 | 5 | 6 => {
            read_u32_le(r)?;
        }
        8 => {
            let len = read_u64_le(r)? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)
                .map_err(|e| format!("read error: {e}"))?;
        }
        9 => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            for _ in 0..count {
                skip_gguf_value(r, elem_type)?;
            }
        }
        10 | 11 | 12 => {
            read_u64_le(r)?;
        }
        _ => return Err(format!("unknown GGUF type {type_id} while skipping")),
    }
    Ok(())
}

fn read_gguf_value<R: Read>(r: &mut R, type_id: u32) -> Result<GgufValue, String> {
    match type_id {
        0 => Ok(GgufValue::Uint8(read_u8(r)?)),
        1 => Ok(GgufValue::Int8(read_u8(r)? as i8)),
        2 => Ok(GgufValue::Uint16(read_u16_le(r)?)),
        3 => Ok(GgufValue::Int16(read_u16_le(r)? as i16)),
        4 => Ok(GgufValue::Uint32(read_u32_le(r)?)),
        5 => Ok(GgufValue::Int32(read_u32_le(r)? as i32)),
        6 => Ok(GgufValue::Float32(read_f32_le(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::Str(read_gguf_string(r)?)),
        9 => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            if count > MAX_ARRAY_STORE {
                for _ in 0..count {
                    skip_gguf_value(r, elem_type)?;
                }
                Ok(GgufValue::LargeArray { elem_type, count })
            } else {
                let mut arr = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    arr.push(read_gguf_value(r, elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(r)?)),
        11 => Ok(GgufValue::Int64(read_u64_le(r)? as i64)),
        12 => Ok(GgufValue::Float64(read_f64_le(r)?)),
        _ => Err(format!("unknown GGUF value type: {type_id}")),
    }
}

/// Parse GGUF metadata (KV pairs + tensor info) without loading tensor data.
pub fn parse_gguf(path: &str) -> Result<GgufMeta, String> {
    let file = File::open(path).map_err(|e| format!("cannot open '{}': {e}", path))?;
    let mut r = BufReader::with_capacity(64 * 1024, file);

    let magic = read_u32_le(&mut r)?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "'{}' is not a GGUF file (magic 0x{:08X})",
            path, magic
        ));
    }

    let version = read_u32_le(&mut r)?;
    if version < 2 {
        return Err(format!("unsupported GGUF version {version} (need ≥ 2)"));
    }

    let tensor_count = read_u64_le(&mut r)? as usize;
    let kv_count = read_u64_le(&mut r)? as usize;

    let mut kv = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)?;
        let type_id = read_u32_le(&mut r)?;
        let value = read_gguf_value(&mut r, type_id)?;
        kv.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut r)?;
        let dims = read_u32_le(&mut r)? as usize;
        let mut shape = Vec::with_capacity(dims);
        for _ in 0..dims {
            shape.push(read_u64_le(&mut r)?);
        }
        let tensor_type = read_u32_le(&mut r)?;
        let offset = read_u64_le(&mut r)?;
        tensors.push(TensorInfo {
            name,
            shape,
            tensor_type,
            offset,
        });
    }

    Ok(GgufMeta {
        version,
        kv,
        tensors,
    })
}
