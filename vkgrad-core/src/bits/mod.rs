pub const BITS_PER_BYTE: usize = 8;

pub fn bit_offset_to_byte_offset(offset: isize) -> isize {
    offset.div_euclid(BITS_PER_BYTE as _)
}

pub fn bit_offset_to_byte_offset_unsigned(offset: usize) -> usize {
    offset.div_euclid(BITS_PER_BYTE)
}

pub fn memcpy_bit(
    src: &[u8],
    dst: &mut [u8],
    src_offset: usize,
    dst_offset: usize,
    num_bits: usize,
) {
    if src_offset >= BITS_PER_BYTE || dst_offset >= BITS_PER_BYTE {
        memcpy_bit(
            &src[src_offset / BITS_PER_BYTE..],
            &mut dst[dst_offset / BITS_PER_BYTE..],
            src_offset % BITS_PER_BYTE,
            dst_offset % BITS_PER_BYTE,
            num_bits,
        );
        return;
    }

    assert!(src.len() >= src_offset + num_bits / BITS_PER_BYTE);
    assert!(dst.len() >= dst_offset + num_bits / BITS_PER_BYTE);

    if src_offset == 0 && dst_offset == 0 {
        dst.copy_from_slice(src);
        return;
    }

    log::trace!("Performing bitwise copying: very slow!");
    for i in 0..num_bits {
        let src_bit_idx = src_offset + i;
        let dst_bit_idx = dst_offset + i;

        let src_byte = src_bit_idx / BITS_PER_BYTE;
        let src_bit = src_bit_idx % BITS_PER_BYTE;
        let bit = (src[src_byte] >> src_bit) & 1;

        let dst_byte = dst_bit_idx / BITS_PER_BYTE;
        let dst_bit = dst_bit_idx % BITS_PER_BYTE;

        dst[dst_byte] &= !(1 << dst_bit);
        dst[dst_byte] |= bit << dst_bit;
    }
}
