use smallvec::SmallVec;

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
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

pub fn gcd_signed(mut a: i64, mut b: i64) -> i64 {
    let mut c;
    while a != 0 {
        c = a;
        a = b % a;
        b = c;
    }
    b.abs()
}

pub struct CombinationIterator {
    indices: SmallVec<[u32; 10]>,
    k: u32,
    init: bool,
}

impl CombinationIterator {
    pub fn new(n: usize, k: u32) -> CombinationIterator {
        CombinationIterator {
            indices: (0..n).map(|_| 0).collect(),
            k,
            init: false,
        }
    }

    pub fn next<'a>(&'a mut self) -> Option<&'a [u32]> {
        if self.indices.len() == 0 {
            return None;
        }

        if !self.init {
            self.init = true;
            self.indices[0] = self.k;
            return Some(&self.indices);
        }

        if self.k == 0 {
            return None;
        }

        // find the last non-zero index that is not at the end
        let mut i = self.indices.len() - 1;
        while self.indices[i] == 0 {
            i -= 1;
        }

        // cannot move to the right more
        // find the next index
        let mut last_val = 0;
        if i == self.indices.len() - 1 {
            last_val = self.indices[i];
            self.indices[i] = 0;

            if self.indices.len() == 1 {
                return None;
            }

            i = self.indices.len() - 2;
            while self.indices[i] == 0 {
                if i == 0 {
                    return None;
                }

                i -= 1;
            }
        }

        self.indices[i] -= 1;
        self.indices[i + 1] = last_val + 1;

        Some(&self.indices)
    }
}
