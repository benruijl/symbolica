use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    slice::Chunks,
};

use crate::{
    domains::{
        integer::Z,
        rational::{Rational, Q},
        EuclideanDomain, Field, Ring,
    },
    printer::{MatrixPrinter, VectorPrinter},
};

/// An n-dimensional vector.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Vector<F: Ring> {
    pub(crate) data: Vec<F::Element>,
    pub(crate) field: F,
}

impl<F: Ring> Vector<F> {
    /// Create a new vector from a list of scalars.
    pub fn new(data: Vec<F::Element>, field: F) -> Vector<F> {
        Vector { data, field }
    }

    /// Create a new zero vector from an existing one.
    pub fn new_zero(&self) -> Vector<F> {
        Vector {
            data: vec![self.field.zero(); self.data.len()],
            field: self.field.clone(),
        }
    }

    /// Create a column vector. This operation is very cheap.
    pub fn into_matrix(self) -> Matrix<F> {
        Matrix {
            nrows: 1,
            ncols: self.data.len() as u32,
            data: self.data,
            field: self.field,
        }
    }

    /// Apply a function `f` to each entry of the vector.
    pub fn map<G: Ring>(&self, f: impl Fn(&F::Element) -> G::Element, field: G) -> Vector<G> {
        Vector {
            data: self.data.iter().map(f).collect(),
            field,
        }
    }

    pub fn norm_squared(&self) -> F::Element {
        let mut res = self.field.zero();
        for e in &self.data {
            self.field.add_mul_assign(&mut res, e, e);
        }
        res
    }

    /// Take the Euclidean scalar product of two column or row vectors.
    pub fn dot(&self, rhs: &Self) -> F::Element {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Vectors do not have equal dimension: {} vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut res = self.field.zero();
        for (e1, e2) in self.data.iter().zip(&rhs.data) {
            self.field.add_mul_assign(&mut res, e1, e2);
        }

        res
    }

    /// Compute the Euclidean cross product in three dimensions.
    pub fn cross_product(&self, rhs: &Self) -> Vector<F> {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Vectors do not have equal dimension: {} vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }
        if self.data.len() != 3 {
            panic!(
                "Vectors must be three-dimensional instead of {}",
                self.data.len(),
            );
        }

        Vector {
            data: vec![
                self.field.sub(
                    &self.field.mul(&self.data[1], &rhs.data[2]),
                    &self.field.mul(&self.data[2], &rhs.data[1]),
                ),
                self.field.sub(
                    &self.field.mul(&self.data[2], &rhs.data[0]),
                    &self.field.mul(&self.data[0], &rhs.data[2]),
                ),
                self.field.sub(
                    &self.field.mul(&self.data[0], &rhs.data[1]),
                    &self.field.mul(&self.data[1], &rhs.data[0]),
                ),
            ],
            field: self.field.clone(),
        }
    }
}

impl<F: Ring> Mul<F::Element> for Vector<F> {
    type Output = Vector<F>;

    fn mul(mut self, rhs: F::Element) -> Self::Output {
        for x in &mut self.data {
            self.field.mul_assign(x, &rhs);
        }
        self
    }
}

impl<F: Ring> Mul<F::Element> for &Vector<F> {
    type Output = Vector<F>;

    fn mul(self, rhs: F::Element) -> Self::Output {
        self.clone() * rhs
    }
}

impl<F: Field> Vector<F> {
    /// Project the vector onto the `target` vector.
    pub fn project(&self, target: &Self) -> Self {
        target.clone() * self.field.div(&self.dot(target), &target.norm_squared())
    }

    /// Use the Gram–Schmidt method to create an orthogonal basis.
    pub fn orthogonalize(system: &[Self]) -> Vec<Vector<F>> {
        let mut res = vec![];

        for s in system {
            let mut new_vec = s.clone();
            for x in &res {
                new_vec -= &new_vec.project(x);
            }

            res.push(new_vec);
        }

        res
    }

    /// Apply the Gram-Schmidt method in-place and compute its factors.
    pub fn gram_schmidt(system: &mut [Self], factor: &mut [F::Element]) {
        for s in 0..system.len() {
            let mut res = system[s].clone();
            for j in 0..s {
                factor[s * system.len() + j] = system[0]
                    .field
                    .div(&system[s].dot(&system[j]), &factor[j * system.len() + j]);
                res -= &(system[j].clone() * factor[s * system.len() + j].clone());
            }

            factor[s * system.len() + s] = res.norm_squared();
            system[s] = res;
        }
    }
}

impl Vector<Z> {
    /// Apply the LLL lattice basis reduction algorithm.
    /// `delta` should be a constant between `1/4` and `1`. The most commonly used value is `3/4`.
    pub fn basis_reduction(system: &[Self], delta: Rational) -> Vec<Vector<Z>> {
        let mut res = system.to_vec();

        let mut mus = vec![0.into(); system.len() * system.len()];
        let mut b: Vec<Vector<Q>> = system
            .iter()
            .map(|v| v.map(|e| e.into(), Q))
            .collect::<Vec<_>>();

        Vector::gram_schmidt(&mut b, &mut mus);

        let mut k = 1;
        while k < system.len() {
            for j in (0..k).rev() {
                if mus[k * system.len() + j].abs() < (1, 2).into() {
                    continue;
                }

                let m = &res[j] * mus[k * system.len() + j].round_to_nearest_integer();
                res[k] -= &m;

                // reset b
                for (bb, rr) in b.iter_mut().zip(res.iter()) {
                    *bb = rr.map(|e| e.into(), Q);
                }

                Vector::gram_schmidt(&mut b, &mut mus); // TODO: do partial GS
            }

            if mus[k * system.len() + k]
                > &mus[(k - 1) * &system.len() + (k - 1)]
                    * &(&delta - &(&mus[k * system.len() + k - 1] * &mus[k * system.len() + k - 1]))
            {
                k += 1;
            } else {
                res.swap(k, k - 1);

                for (bb, rr) in b.iter_mut().zip(res.iter()) {
                    *bb = rr.map(|e| e.into(), Q);
                }
                Vector::gram_schmidt(&mut b, &mut mus);

                k = 1.max(k - 1);
            }
        }

        res
    }
}

impl<F: Ring> Index<u32> for Vector<F> {
    type Output = F::Element;

    /// Get the `i`th entry of the vector.
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl<F: Ring> IndexMut<u32> for Vector<F> {
    /// Get the `i`th entry of the vector.
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut F::Element {
        &mut self.data[index as usize]
    }
}

impl<F: Ring> Display for Vector<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        VectorPrinter::new(self).fmt(f)
    }
}

impl<F: Ring> Add<&Vector<F>> for &Vector<F> {
    type Output = Vector<F>;

    /// Add two vectors.
    fn add(self, rhs: &Vector<F>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot add vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut m = self.new_zero();
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.add(a, b);
        }

        m
    }
}

impl<F: Ring> AddAssign<&Vector<F>> for Vector<F> {
    ///Add two vectors in place.
    fn add_assign(&mut self, rhs: &Vector<F>) {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot add vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.add_assign(a, b);
        }
    }
}

impl<F: Ring> Sub<&Vector<F>> for &Vector<F> {
    type Output = Vector<F>;

    /// Subtract two vectors.
    fn sub(self, rhs: &Vector<F>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot subtract vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut m = self.new_zero();
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.sub(a, b);
        }

        m
    }
}

impl<F: Ring> SubAssign<&Vector<F>> for Vector<F> {
    fn sub_assign(&mut self, rhs: &Vector<F>) {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot subtract vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.sub_assign(a, b);
        }
    }
}

impl<F: Ring> Neg for Vector<F> {
    type Output = Vector<F>;

    /// Negate each entry of the vector.
    fn neg(mut self) -> Self::Output {
        for e in &mut self.data {
            *e = self.field.neg(e);
        }

        self
    }
}

/// A matrix with entries that are elements of a ring `F`.
/// A vector can be represented as a matrix with one row or one column.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Matrix<F: Ring> {
    pub(crate) data: Vec<F::Element>,
    pub(crate) nrows: u32,
    pub(crate) ncols: u32,
    pub(crate) field: F,
}

impl<F: Ring> Matrix<F> {
    /// Create a new zeroed matrix with `nrows` rows and `ncols` columns.
    pub fn new(nrows: u32, ncols: u32, field: F) -> Matrix<F> {
        Matrix {
            data: (0..nrows as usize * ncols as usize)
                .map(|_| field.zero())
                .collect(),
            nrows,
            ncols,
            field,
        }
    }

    /// Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.
    pub fn identity(nrows: u32, field: F) -> Matrix<F> {
        Matrix {
            data: (0..nrows as usize * nrows as usize)
                .map(|i| {
                    if i % nrows as usize == i / nrows as usize {
                        field.one()
                    } else {
                        field.zero()
                    }
                })
                .collect(),
            nrows,
            ncols: nrows,
            field,
        }
    }

    /// Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere.
    pub fn eye(diag: &[F::Element], field: F) -> Matrix<F> {
        let mut m = Matrix::new(diag.len() as u32, diag.len() as u32, field);
        for (i, e) in diag.iter().enumerate() {
            m[(i as u32, i as u32)] = e.clone();
        }
        m
    }

    /// Create a new column vector from a list of scalars.
    pub fn new_vec(data: Vec<F::Element>, field: F) -> Matrix<F> {
        Matrix {
            nrows: data.len() as u32,
            ncols: 1,
            data,
            field,
        }
    }

    /// Convert a linear representation of a matrix to a `Matrix`.
    pub fn from_linear(
        data: Vec<F::Element>,
        nrows: u32,
        ncols: u32,
        field: F,
    ) -> Result<Matrix<F>, String> {
        if data.len() == (nrows * ncols) as usize {
            Ok(Matrix {
                data,
                nrows,
                ncols,
                field,
            })
        } else {
            Err(format!(
                "Data length does not match matrix dimensions: {} vs ({},{})",
                data.len(),
                nrows,
                ncols
            ))
        }
    }

    /// Create a new matrix from a 2-dimensional vector of scalars.
    pub fn from_nested_vec(matrix: Vec<Vec<F::Element>>, field: F) -> Result<Matrix<F>, String> {
        let mut data = vec![];

        let cols = matrix.first().map(|r| r.len()).unwrap_or(0);

        for d in matrix {
            if d.len() != cols {
                return Err("Matrix is not rectangular".to_string());
            }

            data.extend(d);
        }

        Ok(Matrix {
            nrows: (data.len() / cols) as u32,
            ncols: cols as u32,
            data,
            field,
        })
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows as usize
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols as usize
    }

    /// Return the field of the matrix entries.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// Return an iterator over the rows of the matrix.
    pub fn row_iter(&self) -> Chunks<'_, F::Element> {
        self.data.chunks(self.ncols as usize)
    }

    /// Return true iff every entry in the matrix is zero.
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|e| F::is_zero(e))
    }

    /// Return true iff every non- main diagonal entry in the matrix is zero.
    pub fn is_diagonal(&self) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(i, e)| i as u32 % self.ncols == i as u32 / self.ncols || F::is_zero(e))
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix<F> {
        let mut m = Matrix::new(self.ncols, self.nrows, self.field.clone());
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                m[(j, i)] = self[(i, j)].clone();
            }
        }
        m
    }

    /// Transpose the matrix in-place.
    pub fn into_transposed(mut self) -> Matrix<F> {
        if self.nrows == self.ncols {
            for i in 0..self.nrows {
                for j in 0..i {
                    self.data
                        .swap((i * self.ncols + j) as usize, (j * self.ncols + i) as usize);
                }
            }

            (self.nrows, self.ncols) = (self.ncols, self.nrows);
            self
        } else {
            let mut m = Matrix::new(self.ncols, self.nrows, self.field.clone());
            for i in 0..self.nrows {
                for j in 0..self.ncols {
                    m[(j, i)] = std::mem::replace(&mut self[(i, j)], m.field.zero());
                }
            }
            m
        }
    }

    /// Multiply the scalar `e` to each entry of the matrix.
    pub fn mul_scalar(&self, e: &F::Element) -> Matrix<F> {
        Matrix {
            data: self.data.iter().map(|ee| self.field.mul(ee, e)).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }

    /// Apply a function `f` to each entry of the matrix.
    pub fn map<G: Ring>(&self, f: impl Fn(&F::Element) -> G::Element, field: G) -> Matrix<G> {
        Matrix {
            data: self.data.iter().map(f).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field,
        }
    }

    /// Get the squared Euclidean norm of the matrix.
    pub fn norm_squared(&self) -> F::Element {
        let mut norm = self.field.zero();
        for d in &self.data {
            self.field.add_mul_assign(&mut norm, d, d);
        }
        norm
    }

    /// Convert a matrix that is a row or column vector into a [Vector].
    /// Will panic if the input is not vector-like.
    pub fn into_vector(self) -> Vector<F> {
        if self.nrows != 1 && self.ncols != 1 {
            panic!("The matrix is not a vector");
        }

        Vector {
            data: self.data,
            field: self.field,
        }
    }
}

impl<F: Ring> Index<u32> for Matrix<F> {
    type Output = [F::Element];

    /// Get the `index`th row of the matrix.
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize * self.nrows as usize..(index as usize + 1) * self.nrows as usize]
    }
}

impl<F: Ring> Index<(u32, u32)> for Matrix<F> {
    type Output = F::Element;

    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.data[(index.0 * self.ncols + index.1) as usize]
    }
}

impl<F: Ring> IndexMut<(u32, u32)> for Matrix<F> {
    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index_mut(&mut self, index: (u32, u32)) -> &mut F::Element {
        &mut self.data[(index.0 * self.ncols + index.1) as usize]
    }
}

impl<F: Ring> Display for Matrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        MatrixPrinter::new(self).fmt(f)
    }
}

impl<F: Ring> Add<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Add two matrices.
    fn add(self, rhs: &Matrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, self.ncols, self.field.clone());
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.add(a, b);
        }

        m
    }
}

impl<F: Ring> AddAssign<&Matrix<F>> for Matrix<F> {
    ///Add two matrices in place.
    fn add_assign(&mut self, rhs: &Matrix<F>) {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.add_assign(a, b);
        }
    }
}

impl<F: Ring> Sub<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Subtract two matrices.
    fn sub(self, rhs: &Matrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, self.ncols, self.field.clone());
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.sub(a, b);
        }

        m
    }
}

impl<F: Ring> SubAssign<&Matrix<F>> for Matrix<F> {
    ///Add two matrices in place.
    fn sub_assign(&mut self, rhs: &Matrix<F>) {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.sub_assign(a, b);
        }
    }
}

impl<F: Ring> Mul<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Multiply two matrices.
    fn mul(self, rhs: &Matrix<F>) -> Self::Output {
        if self.ncols != rhs.nrows {
            panic!(
                "Cannot multiply matrices because of a dimension mismatch: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, rhs.ncols, self.field.clone());

        for i in 0..self.nrows {
            for j in 0..rhs.ncols {
                let sum = &mut m[(i, j)];
                for k in 0..self.ncols {
                    self.field.add_mul_assign(sum, &self[(i, k)], &rhs[(k, j)]);
                }
            }
        }

        m
    }
}

impl<F: Ring> MulAssign<&Matrix<F>> for Matrix<F> {
    ///Multiply two matrices in place.
    fn mul_assign(&mut self, rhs: &Matrix<F>) {
        *self = &*self * rhs;
    }
}

impl<F: Ring> Neg for Matrix<F> {
    type Output = Matrix<F>;

    /// Negate each entry of the matrix.
    fn neg(mut self) -> Self::Output {
        for e in &mut self.data {
            *e = self.field.neg(e);
        }

        self
    }
}

/// Errors that can occur when performing matrix operations.
#[derive(Debug)]
pub enum MatrixError<F: Ring> {
    Underdetermined {
        min_rank: u32,
        max_rank: u32,
        row_reduced_matrix: Option<Matrix<F>>,
    },
    Inconsistent,
    NotSquare,
    Singular,
    ShapeMismatch,
    RightHandSideIsNotVector,
}

impl<F: Ring> std::fmt::Display for MatrixError<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::Underdetermined {
                min_rank,
                max_rank,
                row_reduced_matrix,
            } => {
                write!(
                    f,
                    "The system is underdetermined. The rank of the matrix is between {} and {}",
                    min_rank, max_rank
                )?;
                if let Some(m) = row_reduced_matrix {
                    write!(f, "\nRow reduced matrix:\n{}", m)?;
                }
                Ok(())
            }
            MatrixError::Inconsistent => write!(f, "The system is inconsistent"),
            MatrixError::NotSquare => write!(f, "The matrix is not square"),
            MatrixError::Singular => write!(f, "The matrix is singular"),
            MatrixError::ShapeMismatch => write!(f, "The shape of the matrix is not compatible"),
            MatrixError::RightHandSideIsNotVector => {
                write!(f, "The right-hand side is not a vector")
            }
        }
    }
}

impl<F: EuclideanDomain> Matrix<F> {
    /// Get the content of the matrix, i.e. the gcd of all entries.
    pub fn content(&self) -> F::Element {
        let mut gcd = self.field.zero();
        for e in &self.data {
            gcd = self.field.gcd(&gcd, e);
        }

        gcd
    }

    /// Divide each entry in the matrix by the scalar `e`.
    pub fn div_scalar(&self, e: &F::Element) -> Matrix<F> {
        Matrix {
            data: self
                .data
                .iter()
                .map(|ee| {
                    let (q, r) = self.field.quot_rem(ee, e);
                    assert_eq!(r, self.field.zero());
                    q
                })
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }

    /// Construct the same matrix, but with the content removed.
    pub fn primitive_part(&self) -> Matrix<F> {
        let c = self.content();
        if self.field.is_one(&c) {
            self.clone()
        } else {
            self.div_scalar(&c)
        }
    }
}

impl<F: Field> Matrix<F> {
    /// Compute the inverse of a square matrix, if it exists.
    /// Otherwise, this function returns `MatrixError::Singular`.
    pub fn inv(&self) -> Result<Self, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        // hardcode options for 2x2 and 3x3 matrices
        let f = &self.field;
        if self.nrows == 2 {
            let d = self.field.sub(
                &f.mul(&self.data[0], &self.data[3]),
                &f.mul(&self.data[1], &self.data[2]),
            );

            if F::is_zero(&d) {
                return Err(MatrixError::Singular);
            }

            let d_inv = self.field.inv(&d);
            return Ok(Matrix {
                nrows: 2,
                ncols: 2,
                data: vec![
                    f.mul(&self.data[3], &d_inv),
                    f.mul(&f.neg(&self.data[1]), &d_inv),
                    f.mul(&f.neg(&self.data[2]), &d_inv),
                    f.mul(&self.data[0], &d_inv),
                ],
                field: f.clone(),
            });
        }

        if self.nrows == 3 {
            #[inline(always)]
            fn sub_mul_mul<F: Field>(
                f: &F,
                r: &[F::Element],
                a: usize,
                b: usize,
                c: usize,
                d: usize,
            ) -> F::Element {
                f.sub(&f.mul(&r[a], &r[b]), &f.mul(&r[c], &r[d]))
            }

            let mut m = Matrix {
                nrows: 3,
                ncols: 3,
                data: vec![
                    sub_mul_mul(f, &self.data, 4, 8, 5, 7),
                    sub_mul_mul(f, &self.data, 2, 7, 1, 8),
                    sub_mul_mul(f, &self.data, 1, 5, 2, 4),
                    sub_mul_mul(f, &self.data, 5, 6, 3, 8),
                    sub_mul_mul(f, &self.data, 0, 8, 2, 6),
                    sub_mul_mul(f, &self.data, 2, 3, 0, 5),
                    sub_mul_mul(f, &self.data, 3, 7, 4, 6),
                    sub_mul_mul(f, &self.data, 1, 6, 0, 7),
                    sub_mul_mul(f, &self.data, 0, 4, 1, 3),
                ],
                field: f.clone(),
            };

            let d = f.add(
                &f.mul(&self.data[0], &m.data[0]),
                &f.add(
                    &f.mul(&self.data[1], &m.data[3]),
                    &f.mul(&self.data[2], &m.data[6]),
                ),
            );

            if F::is_zero(&d) {
                return Err(MatrixError::Singular);
            }

            let d_inv = self.field.inv(&d);
            for e in &mut m.data {
                *e = f.mul(e, &d_inv);
            }
            return Ok(m);
        }

        // use Gaussian elimination with an augmented matrix to find the inverse
        let mut m = Matrix::new(self.nrows, self.nrows * 2, self.field.clone());
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                m[(r, c)] = self[(r, c)].clone();
            }
            m[(r, self.nrows + r)] = self.field.one();
        }

        let rank = m.row_reduce();

        if rank < self.nrows as usize {
            return Err(MatrixError::Singular);
        }

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                m.data[r as usize * self.ncols as usize + c as usize] = std::mem::replace(
                    &mut m.data
                        [r as usize * 2 * self.ncols as usize + self.ncols as usize + c as usize],
                    self.field.zero(),
                );
            }
        }

        m.ncols = self.ncols;
        m.data.truncate(m.nrows as usize * m.ncols as usize);

        Ok(m)
    }

    /// Compute the determinant of the matrix.
    pub fn det(&self) -> Result<F::Element, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        let f = &self.field;
        match self.nrows {
            0 => Err(MatrixError::Singular),
            1 => Ok(self.data[0].clone()),
            2 => Ok(f.sub(
                &f.mul(&self.data[0], &self.data[3]),
                &f.mul(&self.data[1], &self.data[2]),
            )),
            3 => {
                let m0 = f.mul(
                    &self.data[0],
                    &f.sub(
                        &f.mul(&self.data[4], &self.data[8]),
                        &f.mul(&self.data[5], &self.data[7]),
                    ),
                );
                let m1 = f.mul(
                    &self.data[1],
                    &f.sub(
                        &f.mul(&self.data[5], &self.data[6]),
                        &f.mul(&self.data[3], &self.data[8]),
                    ),
                );
                let m2 = f.mul(
                    &self.data[2],
                    &f.sub(
                        &f.mul(&self.data[3], &self.data[7]),
                        &f.mul(&self.data[4], &self.data[6]),
                    ),
                );

                Ok(f.add(&f.add(&m0, &m1), &m2))
            }
            _ => self.clone().det_in_place(),
        }
    }

    /// Compute the determinant of the matrix in-place.
    pub fn det_in_place(&mut self) -> Result<F::Element, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        self.solve_subsystem(self.nrows)?;

        let mut det = self.field.one();
        for x in 0..self.nrows {
            self.field
                .mul_assign(&mut det, &self.data[(x + self.nrows * x) as usize]);
        }

        Ok(det)
    }

    /// Write the matrix in echelon form.
    fn gaussian_elimination(
        &mut self,
        max_col: u32,
        early_return: bool,
    ) -> Result<u32, MatrixError<F>> {
        let zero = self.field.zero();

        let mut i = 0;
        for j in 0..max_col {
            if F::is_zero(&self[(i, j)]) {
                // Select a non-zero pivot.
                for k in i + 1..self.nrows {
                    if !F::is_zero(&self[(k, j)]) {
                        // Swap i-th row and k-th row.
                        for l in j..self.ncols {
                            self.data
                                .swap((self.ncols * i + l) as usize, (self.ncols * k + l) as usize);
                        }
                        break;
                    }
                }
                if F::is_zero(&self[(i, j)]) {
                    if early_return {
                        return Err(MatrixError::Underdetermined {
                            min_rank: i,
                            max_rank: max_col - 1,
                            row_reduced_matrix: None,
                        });
                    } else {
                        continue;
                    }
                }
            }
            let x = self[(i, j)].clone();
            let inv_x = self.field.inv(&x);
            for k in i + 1..self.nrows {
                if !F::is_zero(&self[(k, j)]) {
                    let s = self.field.mul(&self[(k, j)], &inv_x);
                    self[(k, j)] = self.field.zero();
                    for l in j + 1..self.ncols {
                        let mut e = std::mem::replace(&mut self[(k, l)], zero.clone());
                        self.field.sub_mul_assign(&mut e, &self[(i, l)], &s);
                        self[(k, l)] = e;
                    }
                }
            }

            i += 1;
            if i >= self.nrows {
                break;
            }
        }

        Ok(i)
    }

    /// Create a row-reduced matrix from a matrix in echelon form.
    fn back_substitution(&mut self, max_col: u32) {
        let field = self.field.clone();
        for i in (0..self.nrows).rev() {
            if let Some(j) = (0..max_col).find(|&j| !F::is_zero(&self[(i, j)])) {
                if !field.is_one(&self[(i, j)]) {
                    let inv_x = field.inv(&self[(i, j)]);

                    for k in j..self.ncols {
                        field.mul_assign(&mut self[(i, k)], &inv_x);
                    }
                }

                for k in 0..i {
                    if !F::is_zero(&self[(k, j)]) {
                        let scale = std::mem::replace(&mut self[(k, j)], field.zero());
                        for l in j + 1..self.ncols {
                            let mut e = std::mem::replace(&mut self[(k, l)], field.zero());
                            field.sub_mul_assign(&mut e, &self[(i, l)], &scale);
                            self[(k, l)] = e;
                        }
                    }
                }
            }
        }
    }

    /// Solves `A * x = 0` for the first `max_col` columns in x.
    /// The other columns are augmented.
    pub fn solve_subsystem(&mut self, max_col: u32) -> Result<u32, MatrixError<F>> {
        if self.nrows < max_col {
            return Err(MatrixError::Underdetermined {
                min_rank: 0,
                max_rank: self.nrows,
                row_reduced_matrix: None,
            });
        }

        self.gaussian_elimination(max_col, true)
    }

    /// Solve `A * x = b` for `x`, where `A` is `self`.
    pub fn solve(&self, b: &Matrix<F>) -> Result<Matrix<F>, MatrixError<F>> {
        if self.nrows != b.nrows {
            return Err(MatrixError::ShapeMismatch);
        }
        if b.ncols != 1 {
            return Err(MatrixError::RightHandSideIsNotVector);
        }

        let (neqs, nvars) = (self.nrows, self.ncols);

        if neqs < nvars {
            return Err(MatrixError::Underdetermined {
                min_rank: 0,
                max_rank: neqs,
                row_reduced_matrix: None,
            });
        }

        // create the augmented matrix
        let mut m = Matrix::new(neqs, nvars + 1, self.field.clone());
        for r in 0..neqs {
            for c in 0..nvars {
                m[(r, c)] = self[(r, c)].clone();
            }
            m[(r, nvars)] = b.data[r as usize].clone();
        }

        let mut i = match m.solve_subsystem(nvars) {
            Ok(i) => i,
            Err(mut x) => {
                if let MatrixError::Underdetermined {
                    row_reduced_matrix, ..
                } = &mut x
                {
                    *row_reduced_matrix = Some(m);
                }
                return Err(x);
            }
        };
        let rank = i;

        for k in rank..neqs {
            if !F::is_zero(&m[(k, nvars)]) {
                return Err(MatrixError::Inconsistent);
            }
        }

        if rank < nvars {
            return Err(MatrixError::Underdetermined {
                min_rank: rank,
                max_rank: rank,
                row_reduced_matrix: Some(m),
            });
        }
        assert_eq!(rank, nvars);

        // back substitution
        i -= 1;
        for j in (0..nvars).rev() {
            if !m.field.is_one(&m[(i, j)]) {
                let inv_x = m.field.inv(&m[(i, j)]);
                b.field.mul_assign(&mut m[(i, nvars)], &inv_x);
            }
            for k in 0..i {
                if !F::is_zero(&m[(k, j)]) {
                    let mut e = std::mem::replace(&mut m[(k, nvars)], self.field.zero());
                    b.field.sub_mul_assign(&mut e, &m[(i, nvars)], &m[(k, j)]);
                    m[(k, nvars)] = e;
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }

        let result = Matrix {
            nrows: nvars,
            ncols: 1,
            data: (0..nvars).map(|i| m[(i, nvars)].clone()).collect(),
            field: m.field,
        };

        Ok(result)
    }

    /// Row-reduce the matrix in-place using Gaussian elimination and return the rank.
    pub fn row_reduce(&mut self) -> usize {
        let rank = self.gaussian_elimination(self.ncols, false).unwrap() as usize;
        self.back_substitution(self.ncols);
        rank
    }

    /// Get the rank of the matrix.
    pub fn rank(&self) -> usize {
        self.clone()
            .gaussian_elimination(self.ncols, false)
            .unwrap() as usize
    }
}

#[cfg(test)]
mod test {
    use crate::{
        domains::{integer::Z, rational::Q},
        tensors::matrix::{Matrix, Vector},
    };

    #[test]
    fn basics() {
        let a = Matrix::from_linear(
            vec![
                1u64.into(),
                2u64.into(),
                3u64.into(),
                4u64.into(),
                5u64.into(),
                6u64.into(),
            ],
            2,
            3,
            Z,
        )
        .unwrap();

        assert_eq!(a.transpose().data, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!(a.clone().into_transposed().data, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!((-a.clone()).data, vec![-1, -2, -3, -4, -5, -6]);

        assert_eq!((&a - &a).data, vec![0, 0, 0, 0, 0, 0]);

        let b = Matrix::from_nested_vec(
            vec![
                vec![7u64.into(), 8u64.into()],
                vec![9u64.into(), 10u64.into()],
                vec![11u64.into(), 12u64.into()],
            ],
            Z,
        )
        .unwrap();

        let c = &a * &b;

        assert_eq!(c.data, vec![58, 64, 139, 154]);
        assert_eq!(&c[1], &[139, 154]);
        assert_eq!(c[(0, 1)], 64);

        let mut d = a.clone();
        d += &a;

        assert_eq!(d.data, vec![2, 4, 6, 8, 10, 12]);

        let c_m = c.map(|x| x * 2u64, Z);
        assert_eq!(c_m.data, vec![116, 128, 278, 308]);
    }

    #[test]
    fn solve() {
        let a = Matrix::from_linear(
            vec![
                1u64.into(),
                2u64.into(),
                3u64.into(),
                4u64.into(),
                5u64.into(),
                16u64.into(),
                7u64.into(),
                8u64.into(),
                9u64.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        assert_eq!(
            a.inv().unwrap().data,
            vec![
                (-83, 60).into(),
                (1, 10).into(),
                (17, 60).into(),
                (19, 15).into(),
                (-1, 5).into(),
                (-1, 15).into(),
                (-1, 20).into(),
                (1, 10).into(),
                (-1, 20).into()
            ]
        );
        assert_eq!(a.det().unwrap(), 60.into());

        let b = Matrix::from_linear(vec![1u64.into(), 2u64.into(), 3u64.into()], 3, 1, Q).unwrap();

        let r = a.solve(&b).unwrap();
        assert_eq!(r.data, vec![(-1, 3).into(), (2, 3).into(), 0.into()]);
    }

    #[test]
    fn row_reduce() {
        let mut a = Matrix::from_linear(
            vec![
                1u64.into(),
                2u64.into(),
                3u64.into(),
                4u64.into(),
                5u64.into(),
                6u64.into(),
                7u64.into(),
                8u64.into(),
                9u64.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        assert_eq!(a.row_reduce(), 2);

        assert_eq!(
            a.data,
            vec![
                1.into(),
                0.into(),
                (-1).into(),
                0.into(),
                1.into(),
                2.into(),
                0.into(),
                0.into(),
                0.into()
            ]
        );
    }

    #[test]
    fn orthogonalize() {
        let a = vec![
            Vector::new(vec![1u64.into(), 2u64.into(), 3u64.into()], Q),
            Vector::new(vec![4u64.into(), 5u64.into(), 6u64.into()], Q),
            Vector::new(vec![7u64.into(), 8u64.into(), 9u64.into()], Q),
        ];

        let res = Vector::orthogonalize(&a);
        for x in &res {
            for y in &res {
                if x != y {
                    assert_eq!(x.dot(y), (0, 1).into());
                }
            }
        }
    }

    #[test]
    fn inverse() {
        let a = Matrix::from_linear(
            vec![3u64.into(), 2u64.into(), 15u64.into(), 4u64.into()],
            2,
            2,
            Q,
        )
        .unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(2, Q));

        let a = Matrix::from_linear(
            vec![
                3u64.into(),
                2u64.into(),
                15u64.into(),
                4u64.into(),
                9u64.into(),
                6u64.into(),
                7u64.into(),
                8u64.into(),
                17u64.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(3, Q));

        let a = Matrix::from_linear(
            vec![
                3u64.into(),
                2u64.into(),
                15u64.into(),
                4u64.into(),
                9u64.into(),
                6u64.into(),
                7u64.into(),
                8u64.into(),
                17u64.into(),
                45u64.into(),
                23u64.into(),
                12u64.into(),
                13u64.into(),
                14u64.into(),
                15u64.into(),
                16u64.into(),
            ],
            4,
            4,
            Q,
        )
        .unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(4, Q));
    }

    #[test]
    fn basis_reduction() {
        let v1 = Vector::new(vec![1.into(), 0.into(), 0.into(), 31416.into()], Z);
        let v2 = Vector::new(vec![0.into(), 1.into(), 0.into(), 27183.into()], Z);
        let v3 = Vector::new(vec![0.into(), 0.into(), 1.into(), (-320177).into()], Z);

        let basis = Vector::basis_reduction(&[v1, v2, v3], (3, 4).into());

        // 32.0177 = 5 * pi + 6 * e
        assert_eq!(basis[0].data, &[5, 6, 1, 1]);
    }
}
