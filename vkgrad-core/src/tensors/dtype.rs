use std::num::NonZeroUsize;

use crate::bits::BITS_PER_BYTE;

/// based on the dlpack API
/// don't worry, i won't use anything further than F16
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDataType {
    Int(usize),
    UInt(usize),
    Float(usize),
    BF16,
    F8e3m4,
    F8e4m3,
    F8e4m3b11fnuz,
    F8e4m3fn,
    F8e4m3fnuz,
    F8e5m2fn,
    F8e5m2fnuz,
    F6e2m3fn,
    F6e3m2fn,
    F4e2m1fn,
    Bool,
}

impl TensorDataType {
    pub fn size_in_bits(self) -> usize {
        match self {
            TensorDataType::Int(bits) => bits,
            TensorDataType::UInt(bits) => bits,
            TensorDataType::Float(bits) => bits,
            TensorDataType::BF16 => 16,
            TensorDataType::F8e3m4 => 8,
            TensorDataType::F8e4m3 => 8,
            TensorDataType::F8e4m3b11fnuz => 8,
            TensorDataType::F8e4m3fn => 8,
            TensorDataType::F8e4m3fnuz => 8,
            TensorDataType::F8e5m2fn => 8,
            TensorDataType::F8e5m2fnuz => 8,
            TensorDataType::F6e2m3fn => 6,
            TensorDataType::F6e3m2fn => 6,
            TensorDataType::F4e2m1fn => 4,
            TensorDataType::Bool => 1,
        }
    }

    pub fn size_in_bytes(self) -> usize {
        self.size_in_bits().div_ceil(BITS_PER_BYTE)
    }

    pub fn is_bit_format(self) -> bool {
        self.size_in_bits() % BITS_PER_BYTE != 0
    }

    pub fn alignment(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.size_in_bits()).expect("size in bits should never be 0")
    }
}

pub trait TensorDataTypeTrait {
    fn data_type() -> TensorDataType;
}

impl TensorDataTypeTrait for f32 {
    fn data_type() -> TensorDataType {
        TensorDataType::Float(32)
    }
}

impl TensorDataTypeTrait for f64 {
    fn data_type() -> TensorDataType {
        TensorDataType::Float(64)
    }
}
