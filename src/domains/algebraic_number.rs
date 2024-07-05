use std::{rc::Rc, sync::Arc};

use rand::Rng;

use crate::{
    combinatorics::CombinationIterator,
    poly::{factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Variable},
    printer::PolynomialPrinter,
};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::Integer,
    EuclideanDomain, Field, Ring,
};

/// An algebraic number ring, with a monic, irreducible defining polynomial.
// TODO: make special case for degree two and three and hardcode the multiplication table
#[derive(Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct AlgebraicExtension<R: Ring> {
    poly: Rc<MultivariatePolynomial<R, u8>>, // TODO: convert to univariate polynomial
}

impl<UField: FiniteFieldWorkspace> AlgebraicExtension<FiniteField<UField>>
where
    FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u8>,
{
    /// Construct the Galois field GF(prime^exp).
    /// The irreducible polynomial is determined automatically.
    pub fn galois_field(prime: UField, exp: usize) -> Self {
        assert!(exp > 0);

        let field = FiniteField::<UField>::new(prime.clone());

        if exp == 1 {
            let mut poly =
                MultivariatePolynomial::new(&field, None, Arc::new(vec![Variable::Temporary(0)]));

            poly.append_monomial(field.one(), &[1]);
            return AlgebraicExtension::new(poly);
        }

        fn is_irreducible<UField: FiniteFieldWorkspace>(
            coeffs: &[u64],
            poly: &mut MultivariatePolynomial<FiniteField<UField>, u8>,
        ) -> bool
        where
            FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u8>,
        {
            poly.clear();
            for (i, c) in coeffs.iter().enumerate() {
                poly.append_monomial(poly.field.nth(*c), &[i as u8]);
            }

            poly.is_irreducible()
        }

        let mut coeffs = vec![0; exp as usize + 1];
        coeffs[exp as usize] = 1;
        let mut poly = MultivariatePolynomial::new(
            &field,
            Some(coeffs.len()),
            Arc::new(vec![Variable::Temporary(0)]),
        );

        // find the minimal polynomial
        if prime.to_u64() == 2 {
            coeffs[0] = 1;

            // try all odd number of non-zero coefficients
            for g in 0..exp / 2 {
                let g = 2 * g + 1;

                let mut c = CombinationIterator::new(exp as usize - 1, g as usize);
                while let Some(comb) = c.next() {
                    for i in 0..g as usize {
                        coeffs[comb[i]] = 1;
                    }

                    if is_irreducible(&coeffs, &mut poly) {
                        return AlgebraicExtension::new(poly);
                    }

                    for i in 0..g as usize {
                        coeffs[comb[i]] = 0;
                    }
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        if exp == 2 {
            for k in 1..prime.to_u64() {
                coeffs[0] = k;

                if is_irreducible(&coeffs, &mut poly) {
                    return AlgebraicExtension::new(poly);
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        // try shape x^n+a*x+b for fast division
        for k in 1..prime.to_u64() {
            for k2 in 1..prime.to_u64() {
                coeffs[0] = k;
                coeffs[1] = k2;

                if is_irreducible(&coeffs, &mut poly) {
                    return AlgebraicExtension::new(poly);
                }
            }
        }

        // try random polynomials
        let mut r = rand::thread_rng();
        loop {
            for c in coeffs.iter_mut() {
                *c = r.gen_range(0..prime.to_u64());
            }
            coeffs[exp as usize] = 1;

            if is_irreducible(&coeffs, &mut poly) {
                return AlgebraicExtension::new(poly);
            }
        }
    }
}

impl<R: Ring> AlgebraicExtension<R> {
    pub fn new(poly: MultivariatePolynomial<R, u8>) -> AlgebraicExtension<R> {
        AlgebraicExtension {
            poly: Rc::new(poly),
        }
    }

    pub fn constant(&self, c: R::Element) -> AlgebraicNumber<R> {
        AlgebraicNumber {
            poly: self.poly.constant(c),
        }
    }

    /// Get the minimal polynomial.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u8> {
        &self.poly
    }

    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> AlgebraicExtension<FiniteField<UField>>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        AlgebraicExtension {
            poly: Rc::new(
                self.poly
                    .map_coeff(|c| c.to_finite_field(field), field.clone()),
            ),
        }
    }

    pub fn to_element(&self, poly: MultivariatePolynomial<R, u8>) -> AlgebraicNumber<R> {
        assert!(poly.nvars() == 1);

        if poly.degree(0) >= self.poly.degree(0) {
            AlgebraicNumber {
                poly: poly.quot_rem_univariate_monic(&self.poly).1,
            }
        } else {
            AlgebraicNumber { poly }
        }
    }
}

impl<R: Ring> std::fmt::Debug for AlgebraicExtension<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " % {:?}", self.poly)
    }
}

impl<R: Ring> std::fmt::Display for AlgebraicExtension<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " % {}", self.poly)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicNumber<R: Ring> {
    pub(crate) poly: MultivariatePolynomial<R, u8>,
}

impl<R: Ring> PartialOrd for AlgebraicNumber<R> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.poly.partial_cmp(&other.poly)
    }
}

impl<R: Ring> std::fmt::Debug for AlgebraicNumber<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.poly)
    }
}

impl<R: Ring> std::fmt::Display for AlgebraicNumber<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.poly)
    }
}

impl<R: Ring> AlgebraicNumber<R> {
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> AlgebraicNumber<FiniteField<UField>>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        AlgebraicNumber {
            poly: self
                .poly
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
        }
    }

    pub fn into_poly(self) -> MultivariatePolynomial<R, u8> {
        self.poly
    }
}

impl<R: Ring> Ring for AlgebraicExtension<R> {
    type Element = AlgebraicNumber<R>;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: &a.poly + &b.poly,
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: &a.poly - &b.poly,
        }
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: (&a.poly * &b.poly).quot_rem_univariate_monic(&self.poly).1,
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = self.add(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = self.sub(a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: -a.poly.clone(),
        }
    }

    fn zero(&self) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.zero(),
        }
    }

    fn one(&self) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.one(),
        }
    }

    fn nth(&self, n: u64) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.constant(self.poly.field.nth(n)),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        let mut result = self.one();
        for _ in 0..e {
            result = self.mul(&result, b);
        }
        result
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.poly.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.poly.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.poly.field.characteristic()
    }

    fn size(&self) -> Integer {
        &self.poly.field.characteristic() * self.poly.degree(0) as i64
    }

    /// Sample a monic polynomial.
    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let coeffs: Vec<_> = (0..self.poly.degree(0))
            .map(|_| self.poly.field.sample(rng, range))
            .collect();

        let mut poly = self.poly.zero_with_capacity(coeffs.len());
        let mut exp = vec![0];
        for (i, c) in coeffs.into_iter().enumerate() {
            exp[0] = i as u8;
            poly.append_monomial(c, &exp);
        }

        AlgebraicNumber { poly }
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        in_product: bool, // can be used to add parentheses
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        if f.sign_plus() {
            f.write_str("+")?;
        }

        if in_product {
            f.write_str("(")?;
        }

        write!(
            f,
            "{}",
            PolynomialPrinter {
                poly: &element.poly,
                opts: *opts,
            }
        )?;

        if in_product {
            f.write_str(")")?;
        }

        Ok(())
    }
}

impl<R: Field + PolynomialGCD<u8>> EuclideanDomain for AlgebraicExtension<R> {
    fn rem(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        // TODO: due to the remainder requiring an inverse, we need to have R be a field
        // instead of a Euclidean domain. Relax this condition by doing a pseudo-division
        // to get the case where rem = 0 without requiring a field?
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let c1 = a.poly.content();
        let c2 = b.poly.content();
        AlgebraicNumber {
            poly: a.poly.constant(a.poly.field.gcd(&c1, &c2)),
        }
    }
}

impl<R: Field + PolynomialGCD<u8>> Field for AlgebraicExtension<R> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        if a.poly.is_zero() {
            panic!("Division by zero");
        }

        AlgebraicNumber {
            poly: a.poly.eea_univariate(&self.poly).1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::Atom;
    use crate::domains::algebraic_number::AlgebraicExtension;
    use crate::domains::finite_field::{PrimeIteratorU64, Zp64};
    use crate::domains::rational::Q;

    #[test]
    fn gcd_number_field() -> Result<(), String> {
        let ring = Atom::parse("a^3 + 3a^2 - 46*a + 1")?.to_polynomial(&Q, None);
        let ring = AlgebraicExtension::new(ring);

        let a = Atom::parse("x^3-2x^2+(-2a^2+8a+2)x-a^2+11a-1")?
            .to_polynomial::<_, u8>(&Q, None)
            .to_number_field(&ring);

        let b = Atom::parse("x^3-2x^2-x+1")?
            .to_polynomial(&Q, a.variables.clone().into())
            .to_number_field(&ring);

        let r = a.gcd(&b).from_number_field();

        let expected =
            Atom::parse("-50/91+x-23/91*a-1/91*a^2")?.to_polynomial(&Q, a.variables.clone().into());
        assert_eq!(r, expected);
        Ok(())
    }

    #[test]
    fn galois() {
        for i in PrimeIteratorU64::new(2).take(20) {
            for j in 1..10 {
                let _ = AlgebraicExtension::<Zp64>::galois_field(i, j);
            }
        }
    }
}
