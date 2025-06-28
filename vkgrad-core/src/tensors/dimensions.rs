use smallvec::SmallVec;

use crate::tensors::{dtype::TensorDataType, views::TENSOR_DIM_SMALL_VEC_CAP};

#[derive(Clone, Copy, Debug)]
pub struct TensorDimension {
    pub shape: usize,
    pub stride: isize, // in bits, not elements!, unlike dlpack
}

impl TensorDimension {
    pub fn new(shape: usize, stride: isize) -> Self {
        Self { shape, stride }
    }

    /// output bits...
    pub fn calc_offsets_and_size<'a>(
        dtype: TensorDataType,
        dims: impl Iterator<Item = &'a TensorDimension>,
    ) -> (isize, isize, usize) {
        let mut min_offset = 0isize;
        let mut max_offset = 0isize;

        for TensorDimension { shape, stride } in dims {
            let extent = (*shape as isize - 1) * stride;
            min_offset += extent.min(0);
            max_offset += extent.max(0);
        }

        assert!(min_offset <= max_offset);
        (
            min_offset,
            max_offset,
            (max_offset - min_offset) as usize + dtype.size_in_bits(),
        )
    }

    // Packed: elements are tightly packed
    // Homogenous: no two elements occupies the same memory (not byte)
    // Linear: strides are positive
    pub fn is_packed_homogenous_linear<'a>(
        dtype: TensorDataType,
        dims: impl Iterator<Item = &'a TensorDimension>,
    ) -> bool {
        let mut dims: SmallVec<[_; TENSOR_DIM_SMALL_VEC_CAP]> = dims.collect();
        dims.sort_by_key(|d| d.stride);
        let mut expected_stride = dtype.size_in_bits();
        for TensorDimension { shape, stride } in dims {
            if Ok(expected_stride) != (*stride).try_into() {
                return false;
            }

            expected_stride *= shape;
        }

        true
    }
}
