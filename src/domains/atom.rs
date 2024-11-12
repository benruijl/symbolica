use crate::{
    atom::{Atom, AtomView},
    poly::Variable,
};

use super::{integer::Integer, Derivable, EuclideanDomain, Field, InternalOrdering, Ring};

use rand::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomField {}

impl Default for AtomField {
    fn default() -> Self {
        AtomField::new()
    }
}

impl AtomField {
    pub fn new() -> AtomField {
        AtomField {}
    }
}

impl std::fmt::Display for AtomField {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::fmt::Debug for AtomField {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl InternalOrdering for Atom {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
}

impl Ring for AtomField {
    type Element = Atom;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a + b;
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a - b;
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = &*a + self.mul(b, c);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = &*a - self.mul(b, c);
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    fn zero(&self) -> Self::Element {
        Atom::new_num(0)
    }

    fn one(&self) -> Self::Element {
        Atom::new_num(1)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.npow(Integer::from(e))
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        if let AtomView::Num(n) = a.as_view() {
            n.is_one()
        } else {
            false
        }
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Atom::new_num(r)
    }

    fn nth(&self, n: u64) -> Self::Element {
        Atom::new_num(Integer::from(n))
    }

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn size(&self) -> Integer {
        0.into()
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<(), std::fmt::Error> {
        element.as_view().format(f, opts, state)
    }
}

impl EuclideanDomain for AtomField {
    fn rem(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (a / b, self.zero())
    }

    fn gcd(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        // FIXME: return something else?
        self.one()
    }
}

impl Field for AtomField {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a / b
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        let one = Atom::new_num(1);
        self.div(&one, a)
    }
}

impl Derivable for AtomField {
    fn derivative(&self, e: &Atom, x: &Variable) -> Atom {
        match x {
            Variable::Symbol(s) => e.derivative(*s),
            _ => panic!("Cannot take derivative of non-symbol"),
        }
    }
}
