pub(crate) unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}

pub fn gcd_unsigned(mut a: u64, mut b: u64) -> u64 {
    let mut c;
    while a != 0 {
        c = a;
        a = b % a;
        b = c;
    }
    b
}

pub fn gcd_signed(mut a: i64, mut b: i64) -> u64 {
    let mut c;
    while a != 0 {
        c = a;
        // only wraps when i64::MIN % -1 and that still yields 0
        a = b.wrapping_rem(a);
        b = c;
    }
    b.unsigned_abs()
}

pub fn gcd_signed_i128(mut a: i128, mut b: i128) -> u128 {
    let mut c;
    while a != 0 {
        c = a;
        // only wraps when i128::MIN % -1 and that still yields 0
        a = b.wrapping_rem(a);
        b = c;
    }
    b.unsigned_abs()
}
