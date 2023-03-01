const HENSEL_LIFTING_MASK: [u8; 128] = [
    255, 85, 51, 73, 199, 93, 59, 17, 15, 229, 195, 89, 215, 237, 203, 33, 31, 117, 83, 105, 231,
    125, 91, 49, 47, 5, 227, 121, 247, 13, 235, 65, 63, 149, 115, 137, 7, 157, 123, 81, 79, 37, 3,
    153, 23, 45, 11, 97, 95, 181, 147, 169, 39, 189, 155, 113, 111, 69, 35, 185, 55, 77, 43, 129,
    127, 213, 179, 201, 71, 221, 187, 145, 143, 101, 67, 217, 87, 109, 75, 161, 159, 245, 211, 233,
    103, 253, 219, 177, 175, 133, 99, 249, 119, 141, 107, 193, 191, 21, 243, 9, 135, 29, 251, 209,
    207, 165, 131, 25, 151, 173, 139, 225, 223, 53, 19, 41, 167, 61, 27, 241, 239, 197, 163, 57,
    183, 205, 171, 1,
];

/// A 64-bit number representing a number in Montgomory form.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MontgomeryNumber(pub(crate) u64);

/// A finite field over a prime that uses Montgomery modular arithmetic
/// to increase the performance of the multiplication operator.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FiniteFieldU64 {
    p: u64,
    m: u64,
    one: MontgomeryNumber,
}

impl FiniteFieldU64 {
    /// Create a new finite field. `n` must be a prime larger than 2.
    pub fn new(p: u64) -> FiniteFieldU64 {
        assert!(p % 2 != 0);

        //println!("{} {} {}", p, Self::mod_2_64(p), 1 + u64::MAX as u128 % p as u128);
        assert!(Self::get_one(p) as u128 == 1 + u64::MAX as u128 % p as u128);

        FiniteFieldU64 {
            p,
            m: Self::inv_2_64(p),
            one: MontgomeryNumber(Self::get_one(p)),
        }
    }

    pub fn get_prime(&self) -> u64 {
        self.p
    }

    pub fn get_magic(&self) -> u64 {
        self.m
    }

    /// Returns the unit element in Montgomory form, ie.e 1 + 2^64 mod a.
    fn get_one(a: u64) -> u64 {
        if a as u128 <= 1u128 << 63 {
            let res = (((1u128 << 63) % a as u128) << 1) as u64;

            if res < a {
                res
            } else {
                res - a
            }
        } else {
            a.wrapping_neg()
        }
    }

    /// Returns -a^-1 mod 2^64.
    fn inv_2_64(a: u64) -> u64 {
        let mut ret: u64 = HENSEL_LIFTING_MASK[((a >> 1) & 127) as usize] as u64;
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret
    }

    /// Convert a number in a prime field a % n to Montgomory form.
    #[inline(always)]
    pub fn to_montgomery(&self, a: u64) -> MontgomeryNumber {
        // TODO: slow, faster alternatives may need assembly
        MontgomeryNumber((((a as u128) << 64) % self.p as u128) as u64)
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    pub fn mul(&self, a: MontgomeryNumber, b: MontgomeryNumber) -> MontgomeryNumber {
        let t = a.0 as u128 * b.0 as u128;
        let m = (t as u64).wrapping_mul(self.m);
        let u = ((t.wrapping_add(m as u128 * self.p as u128)) >> 64) as u64;

        // correct for overflow
        if u < (t >> 64) as u64 {
            return MontgomeryNumber(u.wrapping_sub(self.p));
        }

        if u >= self.p {
            MontgomeryNumber(u - self.p)
        } else {
            MontgomeryNumber(u)
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    pub fn add(&self, a: MontgomeryNumber, b: MontgomeryNumber) -> MontgomeryNumber {
        let mut t = a.0 as u128 + b.0 as u128;

        if t > self.p as u128 {
            t -= self.p as u128;
        }

        MontgomeryNumber(t as u64)
    }

    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    pub fn sub(&self, a: MontgomeryNumber, b: MontgomeryNumber) -> MontgomeryNumber {
        if a.0 >= b.0 {
            MontgomeryNumber(a.0 - b.0)
        } else {
            MontgomeryNumber(a.0 + (self.p - b.0))
        }
    }

    /// Return the unit element in Montgomory form.
    pub fn one(&self) -> MontgomeryNumber {
        self.one
    }

    /// Computes -x mod n.
    #[inline]
    pub fn neg(&self, a: MontgomeryNumber) -> MontgomeryNumber {
        MontgomeryNumber(self.p - a.0)
    }

    /// Computes x^-1 mod n.
    #[inline]
    pub fn inv(&self, x: MontgomeryNumber) -> MontgomeryNumber {
        assert!(x.0 != 0, "0 is not invertible");

        // apply multiplication with 1 twice to get the correct scaling of R=2^64
        // see the paper [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf).
        let x_mont = self
            .mul(self.mul(x, MontgomeryNumber(1)), MontgomeryNumber(1))
            .0;

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u64 = 1;
        let mut u3 = x_mont;
        let mut v1: u64 = 0;
        let mut v3 = self.p;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        debug_assert!(u3 == 1);
        if even_iter {
            MontgomeryNumber(u1)
        } else {
            MontgomeryNumber(self.p - u1)
        }
    }

    /// Compute b^e % n.
    #[inline(always)]
    pub fn pow(&self, mut b: MontgomeryNumber, mut e: u64) -> MontgomeryNumber {
        let mut x = self.one();
        while e != 0 {
            if e & 1 != 0 {
                x = self.mul(x, b);
            }
            b = self.mul(b, b);
            e /= 2;
        }

        x
    }

    /// Convert a number from Montgomory form to standard form.
    #[inline(always)]
    pub fn to_u64(&self, a: MontgomeryNumber) -> u64 {
        self.mul(a, MontgomeryNumber(1)).0
    }
}

/// Do a deterministic Miller test to check if `n` is a prime.
/// Since `n` is a `u64`, a basis of only 7 witnesses has to be tested.
///
/// Based on [Wojciech Izykowski's implementation](https://github.com/wizykowski/miller-rabin).
pub fn is_prime_u64(n: u64) -> bool {
    // shortest SPRP basis from Jim Sinclair for testing primality of u64
    let witnesses: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];

    if n < 2 {
        return false;
    }

    if n % 2 == 0 {
        return n == 2;
    }

    let mut s = 0;
    let mut d = n - 1;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }

    let f = FiniteFieldU64::new(n);
    let neg_one = MontgomeryNumber(n.wrapping_sub(f.one().0));

    'test: for a in witnesses {
        let a = f.to_montgomery(a);

        if a.0 == 0 {
            continue;
        }

        let mut x = f.pow(a, d);

        if x == f.one() || x == neg_one {
            continue;
        }

        for _ in 0..s {
            x = f.mul(x, x);

            if x == f.one() {
                return false;
            }
            if x == neg_one {
                continue 'test;
            }
        }

        return false;
    }

    true
}

/// An interator over consecutive 64-bit primes.
pub struct PrimeIteratorU64 {
    current_number: u64,
}

impl PrimeIteratorU64 {
    /// Create a new prime iterator that is larger than `start`.
    pub fn new(start: u64) -> PrimeIteratorU64 {
        PrimeIteratorU64 {
            current_number: start.max(1),
        }
    }
}

impl Iterator for PrimeIteratorU64 {
    type Item = u64;

    /// Yield the next prime or `None` if `u64::MAX` has been reached.
    fn next(&mut self) -> Option<u64> {
        while self.current_number < u64::MAX {
            self.current_number += 1;

            if is_prime_u64(self.current_number) {
                return Some(self.current_number);
            }
        }

        None
    }
}
