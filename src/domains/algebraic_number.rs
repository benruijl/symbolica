//! Algebraic number fields, e.g. fields supporting sqrt(2).

use std::sync::Arc;

use rand::Rng;

use crate::{
    coefficient::ConvertToRing,
    combinatorics::CombinationIterator,
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial,
        PositiveExponent, Variable,
    },
    tensors::matrix::Matrix,
};

use super::{
    finite_field::{
        FiniteField, FiniteFieldCore, FiniteFieldWorkspace, GaloisField, ToFiniteField,
    },
    integer::Integer,
    rational::Rational,
    EuclideanDomain, Field, InternalOrdering, Ring, SelfRing,
};

/// An algebraic number ring, with a monic, irreducible defining polynomial.
///
/// # Examples
///
/// ```
/// use symbolica::{
///     atom::{Atom, AtomCore},
///     domains::{algebraic_number::AlgebraicExtension, rational::Q, Ring},
/// };
///
/// let extension = AlgebraicExtension::new(Atom::parse("x^2-2").unwrap().to_polynomial(&Q, None));
/// let sqrt_2 = extension.to_element(Atom::parse("x").unwrap().to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.to_element(Atom::parse("2").unwrap().to_polynomial(&Q, None))
/// );
///```
///
/// Galois field:
///
/// ```
/// use symbolica::{
///     atom::{Atom, AtomCore, Symbol},
///     domains::{algebraic_number::AlgebraicExtension, finite_field::Zp, rational::Q, Ring},
/// };
///
/// let field = AlgebraicExtension::galois_field(Zp::new(17), 4, Symbol::new("x0").into());
/// ```
///
// TODO: make special case for degree two and three and hardcode the multiplication table
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicExtension<R: Ring> {
    poly: Arc<MultivariatePolynomial<R, u16>>, // TODO: convert to univariate polynomial
}

impl<T: FiniteFieldWorkspace> GaloisField for AlgebraicExtension<FiniteField<T>>
where
    FiniteField<T>: FiniteFieldCore<T> + PolynomialGCD<u16>,
{
    type Base = FiniteField<T>;

    fn get_extension_degree(&self) -> u64 {
        self.poly.degree(0) as u64
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        let mut p = Integer::zero();
        for x in a.poly.into_iter() {
            p += &(self.poly.ring.to_integer(x.coefficient)
                * &self.characteristic().pow(x.exponents[0] as u64));
        }
        p
    }

    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer {
        let r = self.to_integer(a);
        let s = self.size();
        if &r * &2.into() > s {
            &r - &s
        } else {
            r
        }
    }

    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Ring>::Element: Copy,
    {
        AlgebraicExtension::galois_field(
            self.poly.ring.clone(),
            new_pow,
            self.poly.variables[0].clone(),
        )
    }

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Ring>::Element {
        larger_field.to_element(e.poly.clone())
    }

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Ring>::Element,
    ) -> Self::Element {
        self.to_element(e.poly.clone())
    }
}

impl<UField: FiniteFieldWorkspace> ConvertToRing for AlgebraicExtension<FiniteField<UField>>
where
    FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
    Integer: ToFiniteField<UField>,
{
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        let mut q = &number % &self.size();
        let mut pow = 0;
        let mut poly = self.poly.zero();
        while !q.is_zero() {
            let (qn, r) = q.quot_rem(&self.poly.ring.size());
            poly.append_monomial(r.to_finite_field(&self.poly.ring), &[pow]);
            pow += 1;
            q = qn;
        }

        AlgebraicNumber { poly }
    }

    fn element_from_coefficient(&self, number: crate::coefficient::Coefficient) -> Self::Element {
        match number {
            crate::coefficient::Coefficient::Rational(r) => {
                let n = self.element_from_integer(r.numerator());
                let d = self.element_from_integer(r.denominator());
                self.div(&n, &d)
            }
            crate::coefficient::Coefficient::Float(_) => {
                panic!("Cannot convert float coefficient to algebraic number")
            }
            crate::coefficient::Coefficient::FiniteField(_, _) => {
                panic!("Cannot convert finite field coefficient to algebraic number")
            }
            crate::coefficient::Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial coefficient to algebraic number")
            }
        }
    }

    fn element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Self::Element {
        match number {
            crate::coefficient::CoefficientView::Natural(n, d) => {
                let n = self.element_from_integer(n.into());
                let d = self.element_from_integer(d.into());
                self.div(&n, &d)
            }
            crate::coefficient::CoefficientView::Large(l) => {
                let r: Rational = l.to_rat();
                let n = self.element_from_integer(r.numerator());
                let d = self.element_from_integer(r.denominator());
                self.div(&n, &d)
            }
            crate::coefficient::CoefficientView::Float(_) => {
                panic!("Cannot convert float coefficient to algebraic number")
            }
            crate::coefficient::CoefficientView::FiniteField(_, _) => {
                panic!("Cannot convert finite field coefficient to algebraic number")
            }
            crate::coefficient::CoefficientView::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial coefficient to algebraic number")
            }
        }
    }
}

impl<UField: FiniteFieldWorkspace> AlgebraicExtension<FiniteField<UField>>
where
    FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    /// Construct the Galois field GF(prime^exp).
    /// The irreducible polynomial is determined automatically.
    pub fn galois_field(prime: FiniteField<UField>, exp: usize, var: Variable) -> Self {
        assert!(exp > 0);

        if exp == 1 {
            let mut poly = MultivariatePolynomial::new(&prime, None, Arc::new(vec![var]));

            poly.append_monomial(prime.one(), &[1]);
            return AlgebraicExtension::new(poly);
        }

        fn is_irreducible<UField: FiniteFieldWorkspace>(
            coeffs: &[u64],
            poly: &mut MultivariatePolynomial<FiniteField<UField>, u16>,
        ) -> bool
        where
            FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
            <FiniteField<UField> as Ring>::Element: Copy,
            AlgebraicExtension<FiniteField<UField>>: PolynomialGCD<u16>,
        {
            poly.clear();
            for (i, c) in coeffs.iter().enumerate() {
                poly.append_monomial(poly.ring.nth((*c).into()), &[i as u16]);
            }

            poly.is_irreducible()
        }

        let mut coeffs = vec![0; exp + 1];
        coeffs[exp] = 1;
        let mut poly = MultivariatePolynomial::new(&prime, Some(coeffs.len()), Arc::new(vec![var]));

        // find the minimal polynomial
        let p = prime.get_prime().to_integer();
        if p == 2 {
            coeffs[0] = 1;

            // try all odd number of non-zero coefficients
            for g in 0..exp / 2 {
                let g = 2 * g + 1;

                let mut c = CombinationIterator::new(exp - 1, g);
                while let Some(comb) = c.next() {
                    for i in 0..g {
                        coeffs[comb[i] + 1] = 1;
                    }

                    if is_irreducible(&coeffs, &mut poly) {
                        return AlgebraicExtension::new(poly);
                    }

                    for i in 0..g {
                        coeffs[comb[i] + 1] = 0;
                    }
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        let sample_max = p.to_i64().unwrap_or(i64::MAX) as u64;
        if exp == 2 {
            for k in 1..sample_max {
                coeffs[0] = k;

                if is_irreducible(&coeffs, &mut poly) {
                    return AlgebraicExtension::new(poly);
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        // try shape x^n+a*x+b for fast division
        for k in 1..sample_max {
            for k2 in 1..sample_max {
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
                *c = r.gen_range(0..sample_max);
            }
            coeffs[exp] = 1;

            if is_irreducible(&coeffs, &mut poly) {
                return AlgebraicExtension::new(poly);
            }
        }
    }
}

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Create a new algebraic extension from a univariate polynomial.
    /// The polynomial should be monic and irreducible.
    pub fn new(poly: MultivariatePolynomial<R, u16>) -> AlgebraicExtension<R> {
        if poly.nvars() == 1 {
            return AlgebraicExtension {
                poly: Arc::new(poly),
            };
        }

        assert_eq!((0..poly.nvars()).filter(|v| poly.degree(*v) > 0).count(), 1);
        let v = (0..poly.nvars()).find(|v| poly.degree(*v) > 0).unwrap();
        let uni = poly.to_univariate_from_univariate(v);

        AlgebraicExtension {
            poly: Arc::new(uni.to_multivariate()),
        }
    }

    pub fn constant(&self, c: R::Element) -> AlgebraicNumber<R> {
        AlgebraicNumber {
            poly: self.poly.constant(c),
        }
    }

    /// Get the minimal polynomial.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
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
            poly: Arc::new(
                self.poly
                    .map_coeff(|c| c.to_finite_field(field), field.clone()),
            ),
        }
    }

    pub fn to_element(&self, mut poly: MultivariatePolynomial<R, u16>) -> <Self as Ring>::Element {
        if poly.nvars() == 0 {
            poly.variables = self.poly.variables.clone();
            poly.exponents = vec![0; poly.coefficients.len()];
        }

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

/// A number in an algebraic number field.
///
/// # Examples
///
/// ```
/// use symbolica::{
///     atom::{Atom, AtomCore},
///     domains::{algebraic_number::AlgebraicExtension, rational::Q, Ring},
/// };
///
/// let extension = AlgebraicExtension::new(Atom::parse("x^2-2").unwrap().to_polynomial(&Q, None));
/// let sqrt_2 = extension.to_element(Atom::parse("x").unwrap().to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.to_element(Atom::parse("2").unwrap().to_polynomial(&Q, None))
/// );
///```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicNumber<R: Ring> {
    pub(crate) poly: MultivariatePolynomial<R, u16>,
}

impl<R: Ring> InternalOrdering for AlgebraicNumber<R> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.poly.internal_cmp(&other.poly)
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
    pub fn mul_coeff(self, c: R::Element) -> Self {
        AlgebraicNumber {
            poly: self.poly.mul_coeff(c),
        }
    }

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

    pub fn into_poly(self) -> MultivariatePolynomial<R, u16> {
        self.poly
    }
}

impl<R: EuclideanDomain> Ring for AlgebraicExtension<R> {
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

    fn nth(&self, n: Integer) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.constant(self.poly.ring.nth(n)),
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
        self.poly.ring.characteristic()
    }

    fn size(&self) -> Integer {
        self.poly.ring.size().pow(self.poly.degree(0) as u64)
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        // solve the linear system (c_0 + c_1*x + c_(d-1)*x^(d-1)) * b % self = a
        // TODO: use the inverse if R is a field (requires specialization)
        let d = self.poly.degree(0) as usize;
        let mut m = vec![self.poly.ring.zero(); d * d];

        let mut f = self.one();

        for e in 0..d {
            let c = self.mul(b, &f);
            for monomial in &c.poly {
                m[monomial.exponents[0] as usize * d + e] = monomial.coefficient.clone();
            }
            f.poly.exponents[0] += 1;
        }

        let mut rhs = vec![self.poly.ring.zero(); d];
        for monomial in &a.poly {
            rhs[monomial.exponents[0] as usize] = monomial.coefficient.clone();
        }

        let m = Matrix::from_linear(m, d as u32, d as u32, self.poly.ring.clone()).unwrap();
        let rhs = Matrix::new_vec(rhs, self.poly.ring.clone());

        if let Ok(s) = m.solve_fraction_free(&rhs) {
            let mut new_poly = self.poly.zero();
            for (p, c) in s.into_vec().into_iter().enumerate() {
                new_poly = &new_poly + &new_poly.monomial(c, vec![p as u16]);
            }

            Some(AlgebraicNumber { poly: new_poly })
        } else {
            None
        }
    }

    /// Sample a polynomial.
    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let coeffs: Vec<_> = (0..self.poly.degree(0))
            .map(|_| self.poly.ring.sample(rng, range))
            .collect();

        let mut poly = self.poly.zero_with_capacity(coeffs.len());
        let mut exp = vec![0];
        for (i, c) in coeffs.into_iter().enumerate() {
            exp[0] = i as u16;
            poly.append_monomial(c, &exp);
        }

        AlgebraicNumber { poly }
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.poly.format(opts, state, f)
    }
}

impl<R: Field + PolynomialGCD<u16>> EuclideanDomain for AlgebraicExtension<R> {
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
            poly: a.poly.constant(a.poly.ring.gcd(&c1, &c2)),
        }
    }
}

impl<R: Field + PolynomialGCD<u16>> Field for AlgebraicExtension<R> {
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

impl<R: Field> AlgebraicExtension<R> {
    /// Create a new minimal field extension that has the algebraic number `x` as a root.
    pub fn simplify(&self, x: &AlgebraicNumber<R>) -> AlgebraicExtension<R> {
        let mut polys = vec![];

        let mut x_i = self.one();
        for _ in 0..=self.poly.degree(0) {
            x_i = self.mul(&x_i, &x);
            polys.push(x_i.clone());

            // solve system c_0 + c_1 x + c_i x^2 + ... + x^i = 0
            let ncols = self.poly.degree(0).to_u32() as usize;

            let mut m = vec![self.poly.ring.zero(); polys.len() * ncols];
            for (row, p) in m.chunks_mut(ncols).zip(&polys) {
                for monomial in &p.poly {
                    row[monomial.exponents[0].to_u32() as usize] = monomial.coefficient.clone();
                }
            }

            let mut rhs = m.split_off((polys.len() - 1) * ncols);
            for e in &mut rhs {
                *e = self.poly.ring.neg(&*e);
            }

            if polys.len() == 1 {
                continue;
            }

            // TODO: recycle matrix
            let mat = Matrix::from_linear(
                m,
                (polys.len() - 1) as u32,
                ncols as u32,
                self.poly.ring.clone(),
            )
            .unwrap()
            .into_transposed();

            let rhs = Matrix::new_vec(rhs, self.poly.ring.clone());

            if let Ok(s) = mat.solve(&rhs) {
                let mut res = s.into_vec();
                res.push(self.poly.ring.one());
                let mut new_poly = self.poly.zero();
                for (p, c) in res.into_iter().enumerate() {
                    new_poly = &new_poly + &new_poly.monomial(c, vec![p as u16]);
                }

                return AlgebraicExtension::new(new_poly);
            }
        }

        unreachable!("Could not simplify algebraic number");
    }
}

impl<R: Field + PolynomialGCD<u16>> AlgebraicExtension<R> {
    /// Extend the current algebraic extension `R[a]` with `b`, whose minimal polynomial
    /// is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.
    ///
    /// `b` must be irreducible over `R` and `R[a]`; this is not checked.
    pub fn extend(
        &self,
        b: &MultivariatePolynomial<AlgebraicExtension<R>>,
    ) -> (
        AlgebraicExtension<R>,
        <AlgebraicExtension<R> as Ring>::Element,
        <AlgebraicExtension<R> as Ring>::Element,
    )
    where
        AlgebraicExtension<R>: PolynomialGCD<u16>,
        MultivariatePolynomial<R>: Factorize,
        MultivariatePolynomial<AlgebraicExtension<R>>: Factorize,
    {
        assert_eq!(self, &b.ring);

        let (_, s, g, r) = b.norm_impl();
        debug_assert!(r.is_irreducible());

        let f: AlgebraicExtension<R> = AlgebraicExtension::new(r.clone());
        let mut g2 = g.to_number_field(&f);
        let mut h = self.poly.to_number_field(&f); // yields constant coeffs

        g2.unify_variables(&mut h);
        let g2 = g2.gcd(&h);

        let a = f.neg(&f.div(&g2.get_constant(), &g2.lcoeff()));
        let y = f.to_element(g2.ring.poly.one().mul_exp(&[1]));
        let b = f.sub(&y, &f.mul(&a, &f.nth(s.into())));

        (f, a, b)
    }
}

impl<R: Field + PolynomialGCD<E>, E: PositiveExponent>
    MultivariatePolynomial<AlgebraicExtension<R>, E>
{
    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    pub fn norm(&self) -> MultivariatePolynomial<R, E> {
        self.norm_impl().3
    }

    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    /// Returns `(v, s, g, r)` where `v` is the shifted variable, `s` is the number of steps,
    /// `g` is the shifted polynomial and `r` is the norm.
    pub(crate) fn norm_impl(
        &self,
    ) -> (
        usize,
        usize,
        MultivariatePolynomial<R, E>,
        MultivariatePolynomial<R, E>,
    ) {
        assert!(!self.is_constant());

        let f = self.from_number_field();

        let alpha = f
            .get_vars()
            .iter()
            .position(|x| x == &self.ring.poly.variables[0])
            .unwrap();

        let mut poly = f.zero();
        let mut exp = vec![E::zero(); f.nvars()];
        for x in self.ring.poly.into_iter() {
            exp[alpha] = E::from_u32(x.exponents[0] as u32);
            poly.append_monomial(x.coefficient.clone(), &exp);
        }

        let poly_uni = poly.to_univariate(alpha);

        let mut s = 0;
        loop {
            for v in 0..f.nvars() {
                if v == alpha || f.degree(v) == E::zero() {
                    continue;
                }

                // construct f(x-s*a)
                let alpha_poly = f.variable(&self.get_vars_ref()[v]).unwrap()
                    - f.variable(&self.ring.poly.variables[0]).unwrap()
                        * &f.constant(f.ring.nth(s.into()));
                let g_multi = f.clone().replace_with_poly(v, &alpha_poly);
                let g_uni = g_multi.to_univariate(alpha);

                let r = g_uni.resultant_prs(&poly_uni);

                let d = r.derivative(v);
                if r.gcd(&d).is_constant() {
                    return (v, s, g_multi, r);
                }
            }

            s += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::{Atom, AtomCore, Symbol};
    use crate::domains::algebraic_number::AlgebraicExtension;
    use crate::domains::finite_field::{PrimeIteratorU64, Zp, Z2};
    use crate::domains::integer::Z;
    use crate::domains::rational::Q;
    use crate::domains::Ring;

    #[test]
    fn gcd_number_field() -> Result<(), String> {
        let ring = Atom::parse("a^3 + 3a^2 - 46*a + 1")?.to_polynomial(&Q, None);
        let ring = AlgebraicExtension::new(ring);

        let a = Atom::parse("x^3-2x^2+(-2a^2+8a+2)x-a^2+11a-1")?
            .to_polynomial::<_, u16>(&Q, None)
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
        for j in 1..10 {
            let _ = AlgebraicExtension::galois_field(Z2, j, Symbol::new("v1").into());
        }

        for i in PrimeIteratorU64::new(2).take(20) {
            for j in 1..10 {
                let _ = AlgebraicExtension::galois_field(
                    Zp::new(i as u32),
                    j,
                    Symbol::new("v1").into(),
                );
            }
        }
    }

    #[test]
    fn norm() {
        let a = Atom::parse("z^4+z^3+(2+a-a^2)z^2+(1+a^2-2a^3)z-2")
            .unwrap()
            .to_polynomial::<_, u8>(&Q, None);
        let f = Atom::parse("a^4-3")
            .unwrap()
            .to_polynomial::<_, u16>(&Q, None);
        let f = AlgebraicExtension::new(f);
        let norm = a.to_number_field(&f).norm();

        let res = Atom::parse("16-32*z-64*z^2-64*z^3-52*z^4-40*z^5-132*z^6-24*z^7-50*z^8+120*z^9+66*z^10+92*z^11+47*z^12+32*z^13+14*z^14+4*z^15+z^16")
        .unwrap()
        .to_polynomial::<_, u8>(&Q, a.variables.clone().into());

        assert_eq!(norm, res);
    }

    #[test]
    fn extend() {
        let a = Atom::parse("x^2-2").unwrap().to_polynomial(&Q, None);
        let ae = AlgebraicExtension::new(a);

        let b = Atom::parse("y^2-3")
            .unwrap()
            .to_polynomial(&Q, None)
            .to_number_field(&ae);

        let (c, rep1, rep2) = ae.extend(&b);

        let rf = Atom::parse("1-10*y^2+y^4").unwrap().to_polynomial(&Q, None);

        assert_eq!(c.poly.as_ref(), &rf);

        let r1 = Atom::parse("-9/2y+1/2y^3")
            .unwrap()
            .to_polynomial::<_, u16>(&Q, None);
        assert_eq!(rep1.poly, r1);

        let r2 = Atom::parse("11/2*y-1/2*y^3")
            .unwrap()
            .to_polynomial::<_, u16>(&Q, None);
        assert_eq!(rep2.poly, r2);
    }

    #[test]
    fn simplify() {
        let poly = AlgebraicExtension::new(
            Atom::parse("13-16v1+28v1^2+2v1^3+11v1^4+v1^6")
                .unwrap()
                .to_polynomial(&Q, None),
        );

        let a = poly.to_element(
            Atom::parse(
                "-295/1882 -2693/1882v1 -237/1882v1^2 -385/941v1^3 -9/1882v1^4  -33/941v1^5",
            )
            .unwrap()
            .to_polynomial::<_, u16>(&Q, None),
        );

        let r = poly.simplify(&a);
        let res = Atom::parse("1+v1+v1^2").unwrap().to_polynomial(&Q, None);
        assert_eq!(*r.poly, res);
    }

    #[test]
    fn try_div() {
        let extension =
            AlgebraicExtension::new(Atom::parse("v1^3-2v1+3").unwrap().to_polynomial(&Z, None));

        let f1 = extension.to_element(Atom::parse("v1^2-2").unwrap().to_polynomial(&Z, None));
        let f2 = extension.to_element(Atom::parse("v1-5").unwrap().to_polynomial(&Z, None));
        let prod = extension.mul(&f1, &f2);

        assert_eq!(extension.try_div(&prod, &f2).unwrap(), f1);
        assert_eq!(extension.try_div(&prod, &f1).unwrap(), f2);
        assert!(extension.try_div(&f2, &f1).is_none());
    }
}
