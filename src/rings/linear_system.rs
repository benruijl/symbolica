use smallvec::SmallVec;
use std::{
    fmt::{Display, Write},
    ops::{Index, IndexMut},
    slice::Chunks,
};

use super::Field;

#[derive(Debug, Eq, PartialEq)]
pub struct Matrix<F: Field> {
    pub shape: (u32, u32),
    pub data: SmallVec<[F::Element; 25]>,
    pub field: F,
}

impl<F: Field> Matrix<F> {
    pub fn new(rows: u32, cols: u32, field: F) -> Matrix<F> {
        Matrix {
            shape: (rows, cols),
            data: (0..rows as usize * cols as usize)
                .map(|_| field.zero())
                .collect(),
            field,
        }
    }

    pub fn rows(&self) -> usize {
        self.shape.0 as usize
    }

    pub fn cols(&self) -> usize {
        self.shape.1 as usize
    }

    pub fn row_iter(&self) -> Chunks<'_, F::Element> {
        self.data.chunks(self.shape.1 as usize)
    }
}

impl<F: Field> Index<(u32, u32)> for Matrix<F> {
    type Output = F::Element;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.data[(index.0 * self.shape.1 + index.1) as usize]
    }
}

impl<F: Field> IndexMut<(u32, u32)> for Matrix<F> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut F::Element {
        &mut self.data[(index.0 * self.shape.1 + index.1) as usize]
    }
}

impl<F: Field> Display for Matrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('{')?;
        for (ri, r) in self.row_iter().enumerate() {
            f.write_char('{')?;
            for (ci, c) in r.iter().enumerate() {
                self.field.fmt_display(c, None, true, f)?;
                if ci + 1 < self.shape.1 as usize {
                    f.write_char(',')?;
                }
            }
            f.write_char('}')?;
            if ri + 1 < self.shape.0 as usize {
                f.write_char(',')?;
            }
        }
        f.write_char('}')
    }
}

/// Error from linear solver.
#[derive(Debug)]
pub enum LinearSolverError<F: Field> {
    Underdetermined {
        min_rank: u32,
        max_rank: u32,
        row_reduced_matrix: Option<Matrix<F>>,
    },
    Inconsistent,
    NotSquare,
}

impl<F: Field> Matrix<F> {
    /// Compute the determinant of the matrix.
    pub fn det(&mut self) -> Result<F::Element, LinearSolverError<F>> {
        if self.shape.0 != self.shape.1 {
            Err(LinearSolverError::NotSquare)?;
        }

        self.solve_subsystem(self.shape.0)?;

        let mut det = self.field.one();
        for x in 0..self.shape.0 {
            self.field
                .mul_assign(&mut det, &self.data[(x + self.shape.0 * x) as usize]);
        }

        Ok(det)
    }

    /// Solves `A * x = 0` for the first `max_col` columns in x.
    /// The other columns are augmented.
    pub fn solve_subsystem(&mut self, max_col: u32) -> Result<u32, LinearSolverError<F>> {
        let (neqs, ncols) = self.shape;

        // A fast check.
        if neqs < max_col {
            return Err(LinearSolverError::Underdetermined {
                min_rank: 0,
                max_rank: neqs,
                row_reduced_matrix: None,
            });
        }

        // Gaussian elimination:
        // First, transform the augmented matrix into the row echelon form.
        let mut i = 0;
        for j in 0..max_col {
            if F::is_zero(&self[(i, j)]) {
                // Select a non-zero pivot.
                for k in i + 1..neqs {
                    if !F::is_zero(&self[(k, j)]) {
                        // Swap i-th row and k-th row.
                        for l in j..ncols {
                            let old = self[(i, l)].clone();
                            self[(i, l)] = self[(k, l)].clone();
                            self[(k, l)] = old;
                        }
                        break;
                    }
                }
                if F::is_zero(&self[(i, j)]) {
                    // NOTE: complete pivoting may give an increase of the rank.
                    return Err(LinearSolverError::Underdetermined {
                        min_rank: i,
                        max_rank: max_col - 1,
                        row_reduced_matrix: None,
                    });
                }
            }
            let x = self[(i, j)].clone();
            let inv_x = self.field.inv(&x);
            for k in i + 1..neqs {
                if !F::is_zero(&self[(k, j)]) {
                    let s = self.field.mul(&self[(k, j)], &inv_x);
                    self[(k, j)] = self.field.zero();
                    for l in j + 1..ncols {
                        self[(k, l)] = self
                            .field
                            .sub(&self[(k, l)], &self.field.mul(&self[(i, l)], &s));
                    }
                }
            }
            i += 1;
            if i >= neqs {
                break;
            }
        }

        // Return the rank
        Ok(i)
    }

    /// Solves `A * x = b` for `x`, where `A` is `self`.
    // TODO: provide vectors to prevent allocation
    pub fn solve(&self, b: &Matrix<F>) -> Result<Matrix<F>, LinearSolverError<F>> {
        assert!(self.shape.0 == b.shape.0 && b.shape.1 == 1 && self.field == b.field);

        let (neqs, nvars) = self.shape;

        // A fast check.
        if neqs < nvars {
            return Err(LinearSolverError::Underdetermined {
                min_rank: 0,
                max_rank: neqs,
                row_reduced_matrix: None,
            });
        }

        // Create the augmented matrix.
        let mut m = Matrix::new(neqs, nvars + 1, self.field);
        for r in 0..neqs {
            for c in 0..nvars {
                m[(r, c)] = self[(r, c)].clone();
            }
            m[(r, nvars)] = b.data[r as usize].clone();
        }

        let mut i = match m.solve_subsystem(nvars) {
            Ok(i) => i,
            Err(mut x) => {
                if let LinearSolverError::Underdetermined {
                    row_reduced_matrix, ..
                } = &mut x
                {
                    *row_reduced_matrix = Some(m);
                }
                return Err(x);
            }
        };
        let rank = i;

        // Check the consistency.
        for k in rank..neqs {
            if !F::is_zero(&m[(k, nvars)]) {
                return Err(LinearSolverError::Inconsistent);
            }
        }

        // Check the rank.
        if rank < nvars {
            return Err(LinearSolverError::Underdetermined {
                min_rank: rank,
                max_rank: rank,
                row_reduced_matrix: Some(m),
            });
        }
        assert_eq!(rank, nvars);

        // Now, do back substitution.
        i -= 1;
        for j in (0..nvars).rev() {
            if !m.field.is_one(&m[(i, j)]) {
                // TODO: check if correct
                let x = m[(i, j)].clone();
                let inv_x = m.field.inv(&x);
                #[cfg(debug_assertions)]
                {
                    m[(i, j)] = m.field.one();
                }
                m[(i, nvars)] = m.field.mul(&m[(i, nvars)], &inv_x);
            }
            for k in 0..i {
                if !F::is_zero(&m[(k, j)]) {
                    m[(k, nvars)] = m
                        .field
                        .sub(&m[(k, nvars)], &m.field.mul(&m[(i, nvars)], &m[(k, j)]));
                    #[cfg(debug_assertions)]
                    {
                        m[(k, j)] = m.field.zero();
                    }
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }

        let result = Matrix {
            shape: (nvars, 1),
            data: (0..nvars).map(|i| m[(i, nvars)].clone()).collect(),
            field: m.field,
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::finite_field::{FiniteField, FiniteFieldCore};
    use crate::rings::rational::{Rational, RationalField};

    #[test]
    fn test_solve_trivial() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (1, 1),
            data: [12].into_iter().map(|n| field.to_element(n)).collect(),
            field,
        };
        let b = Matrix {
            shape: (1, 1),
            data: [7].into_iter().map(|n| field.to_element(n)).collect(),
            field,
        };
        let r = a.solve(&b).unwrap();

        let res: Vec<_> = r
            .data
            .into_iter()
            .map(|i| a.field.from_element(i))
            .collect();
        assert_eq!(&res, &[2]);
    }

    #[test]
    fn test_solve_easy() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (2, 2),
            data: [1, 0, 0, 1]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (2, 1),
            data: [5, 6].into_iter().map(|n| field.to_element(n)).collect(),
            field,
        };
        println!("b={:?}", b.data);
        let r = a.solve(&b).unwrap();

        let res: Vec<_> = r
            .data
            .into_iter()
            .map(|i| a.field.from_element(i))
            .collect();
        assert_eq!(&res, &[5, 6]);
    }

    #[test]
    fn test_solve() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (3, 3),
            data: [1, 1, 2, 3, 4, 3, 16, 5, 5]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (3, 1),
            data: [3, 15, 8]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let r = a.solve(&b).unwrap();

        let res: Vec<_> = r
            .data
            .into_iter()
            .map(|i| a.field.from_element(i))
            .collect();
        assert_eq!(&res, &[2, 3, 16]);
    }

    #[test]
    #[should_panic]
    fn test_solve_bad_shape() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (3, 3),
            data: [1, 1, 2, 3, 4, 3, 16, 5, 5]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (4, 1),
            data: [3, 15, 8, 1]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let _r = a.solve(&b).unwrap();
    }

    #[test]
    fn test_solve_underdetermined1() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (2, 3),
            data: [1, 1, 2, 3, 4, 3]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (2, 1),
            data: [3, 15].into_iter().map(|n| field.to_element(n)).collect(),
            field,
        };
        let r = a.solve(&b);
        assert!(matches!(
            r,
            Err(LinearSolverError::Underdetermined {
                min_rank: 0,
                max_rank: 2,
                ..
            })
        ));
    }

    #[test]
    fn test_solve_underdetermined2() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (4, 4),
            data: [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 2]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (4, 1),
            data: [1, 15, 1, 2]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let r = a.solve(&b);
        assert!(matches!(
            r,
            Err(LinearSolverError::Underdetermined {
                min_rank: 1,
                max_rank: 3,
                ..
            })
        ));
    }

    #[test]
    fn test_solve_underdetermined3() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (3, 3),
            data: [1, 1, 2, 3, 4, 3, 10, 7, 12]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (3, 1),
            data: [3, 15, 12]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let r = a.solve(&b);
        assert!(matches!(
            r,
            Err(LinearSolverError::Underdetermined {
                min_rank: 2,
                max_rank: 2,
                ..
            })
        ));
    }

    #[test]
    fn test_solve_overdetermined() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (5, 3),
            data: [1, 1, 2, 3, 4, 3, 9, 0, 11, 1, 1, 7, 2, 3, 8]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (5, 1),
            data: [3, 15, 7, 6, 6]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let r = a.solve(&b).unwrap();

        let res: Vec<_> = r
            .data
            .into_iter()
            .map(|i| a.field.from_element(i))
            .collect();
        assert_eq!(&res, &[11, 1, 4]);
    }

    #[test]
    fn test_solve_inconsistent() {
        let field = FiniteField::<u32>::new(17);
        let a = Matrix {
            shape: (4, 3),
            data: [1, 1, 2, 3, 4, 3, 16, 5, 5, 14, 2, 4]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (4, 1),
            data: [3, 15, 8, 3]
                .into_iter()
                .map(|n| field.to_element(n))
                .collect(),
            field,
        };
        let r = a.solve(&b);
        assert!(matches!(r, Err(LinearSolverError::Inconsistent)));
    }

    #[test]
    fn test_solve_rational() {
        let field = RationalField::new();
        let a = Matrix {
            shape: (3, 3),
            data: [1, 1, 2, 3, 4, 3, 16, 5, 5]
                .into_iter()
                .map(|n| Rational::Natural(n, 1))
                .collect(),
            field,
        };
        let b = Matrix {
            shape: (3, 1),
            data: [3, 15, 8]
                .into_iter()
                .map(|n| Rational::Natural(n, 1))
                .collect(),
            field,
        };
        let r = a.solve(&b).unwrap();

        assert_eq!(
            r.data.as_slice(),
            &[
                Rational::new(-5, 6),
                Rational::new(47, 10),
                Rational::new(-13, 30)
            ]
        );
    }
}
