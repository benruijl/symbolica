use std::{
    borrow::{Borrow, BorrowMut},
    hash::{Hash, Hasher},
    ops::Neg,
    sync::{Arc, RwLock},
};

use ahash::HashMap;
use once_cell::sync::Lazy;
use pyo3::{
    exceptions, pyclass,
    pyclass::CompareOp,
    pyfunction, pymethods, pymodule,
    types::{PyComplex, PyLong, PyModule, PyTuple, PyType},
    wrap_pyfunction, FromPyObject, IntoPy, PyErr, PyObject, PyRef, PyResult, Python,
};
use rug::Complete;
use self_cell::self_cell;
use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    evaluate::EvaluationFn,
    id::{
        AtomType, Match, MatchStack, Pattern, PatternAtomTreeIterator, PatternRestriction,
        ReplaceIterator,
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, Sample},
    parser::Token,
    poly::{
        evaluate::{
            InstructionEvaluator, InstructionSetMode, InstructionSetModeCPPSettings,
            InstructionSetPrinter,
        },
        polynomial::MultivariatePolynomial,
        Variable, INLINED_EXPONENTS,
    },
    printer::{AtomPrinter, PolynomialPrinter, PrintOptions, RationalPolynomialPrinter},
    representations::{
        default::ListIteratorD, number::Number, Add, Atom, AtomSet, AtomView, Fun, Identifier, Mul,
        Num, OwnedAdd, OwnedFun, OwnedMul, OwnedNum, OwnedPow, OwnedVar, Pow, Var,
    },
    rings::{float::Complex, integer::IntegerRing},
    rings::{
        integer::Integer,
        rational::RationalField,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    state::{FunctionAttribute, ResettableBuffer, State, Workspace, INPUT_ID},
    streaming::TermStreamer,
    transformer::Transformer,
    LicenseManager,
};

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
thread_local!(static WORKSPACE: Workspace = Workspace::new());

macro_rules! get_state {
    () => {
        STATE.read().map_err(|_| {
            exceptions::PyRuntimeError::new_err(
                "A critical error has occurred earlier in Symbolica: the Python interpreter must be restarted.",
            )
        })
    };
}

macro_rules! get_state_mut {
    () => {
        STATE.write().map_err(|_| {
            exceptions::PyRuntimeError::new_err(
                "A critical error has occurred earlier in Symbolica: the Python interpreter must be restarted.",
            )
        })
    };
}

#[pymodule]
fn symbolica(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PythonExpression>()?;
    m.add_class::<PythonFunction>()?;
    m.add_class::<PythonPattern>()?;
    m.add_class::<PythonPolynomial>()?;
    m.add_class::<PythonIntegerPolynomial>()?;
    m.add_class::<PythonRationalPolynomial>()?;
    m.add_class::<PythonRationalPolynomialSmallExponent>()?;
    m.add_class::<PythonNumericalIntegrator>()?;
    m.add_class::<PythonSample>()?;
    m.add_class::<PythonAtomType>()?;
    m.add_class::<PythonAtomTree>()?;
    m.add_class::<PythonInstructionEvaluator>()?;

    m.add_function(wrap_pyfunction!(is_licensed, m)?)?;
    m.add_function(wrap_pyfunction!(set_license_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_hobbyist_license, m)?)?;
    m.add_function(wrap_pyfunction!(request_trial_license, m)?)?;

    Ok(())
}

/// Check if the current Symbolica instance has a valid license key set.
#[pyfunction]
fn is_licensed() -> bool {
    LicenseManager::is_licensed()
}

/// Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions.
#[pyfunction]
fn set_license_key(key: String) -> PyResult<()> {
    LicenseManager::set_license_key(&key).map_err(exceptions::PyException::new_err)
}

/// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
/// `email`.
#[pyfunction]
fn request_hobbyist_license(name: String, email: String) -> PyResult<()> {
    LicenseManager::request_hobbyist_license(&name, &email)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
/// `email`.
#[pyfunction]
fn request_trial_license(name: String, email: String, company: String) -> PyResult<()> {
    LicenseManager::request_trial_license(&name, &email, &company)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Specifies the type of the atom.
#[derive(Clone, Copy)]
#[pyclass(name = "AtomType")]
pub enum PythonAtomType {
    Num,
    Var,
    Fn,
    Add,
    Mul,
    Pow,
}

/// A Python representation of a Symbolica expression.
/// The type of the atom is provided in `atom_type`.
///
/// The `head` contains the string representation of:
/// - a number if the type is `Num`
/// - the variable if the type is `Var`
/// - the function name if the type is `Fn`
/// - otherwise it is `None`.
///
/// The tail contains the child atoms:
/// - the summand for type `Add`
/// - the factors for type `Mul`
/// - the base and exponent for type `Pow`
/// - the function arguments for type `Fn`
#[derive(Clone)]
#[pyclass(name = "AtomTree")]
pub struct PythonAtomTree {
    /// The type of this atom.
    #[pyo3(get)]
    pub atom_type: PythonAtomType,
    /// The string data of this atom.
    #[pyo3(get)]
    pub head: Option<String>,
    /// The list of child atoms of this atom.
    #[pyo3(get)]
    pub tail: Vec<PythonAtomTree>,
}

impl<'a, P: AtomSet> From<AtomView<'a, P>> for PyResult<PythonAtomTree> {
    fn from(atom: AtomView<'a, P>) -> Self {
        let tree = match atom {
            AtomView::Num(_) => PythonAtomTree {
                atom_type: PythonAtomType::Num,
                head: Some(format!("{}", AtomPrinter::new(atom, &&get_state!()?))),
                tail: vec![],
            },
            AtomView::Var(v) => PythonAtomTree {
                atom_type: PythonAtomType::Var,
                head: get_state!()?.get_name(v.get_name()).map(|x| x.to_string()),
                tail: vec![],
            },
            AtomView::Fun(f) => PythonAtomTree {
                atom_type: PythonAtomType::Fn,
                head: get_state!()?.get_name(f.get_name()).map(|x| x.to_string()),
                tail: f.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Add(a) => PythonAtomTree {
                atom_type: PythonAtomType::Add,
                head: None,
                tail: a.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Mul(m) => PythonAtomTree {
                atom_type: PythonAtomType::Mul,
                head: None,
                tail: m.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                PythonAtomTree {
                    atom_type: PythonAtomType::Pow,
                    head: None,
                    tail: vec![
                        <AtomView<P> as Into<PyResult<PythonAtomTree>>>::into(b)?,
                        <AtomView<P> as Into<PyResult<PythonAtomTree>>>::into(e)?,
                    ],
                }
            }
        };

        Ok(tree)
    }
}

#[derive(FromPyObject)]
pub enum ConvertibleToPattern {
    Literal(ConvertibleToExpression),
    Pattern(PythonPattern),
}

impl ConvertibleToPattern {
    pub fn to_pattern(self) -> PyResult<PythonPattern> {
        match self {
            Self::Literal(l) => Ok(PythonPattern {
                expr: Arc::new(Pattern::from_view(
                    l.to_expression().expr.as_view(),
                    &&get_state!()?,
                )),
            }),
            Self::Pattern(e) => Ok(e),
        }
    }
}

/// Operations that transform an expression.
#[pyclass(name = "Transformer")]
#[derive(Clone)]
pub struct PythonPattern {
    pub expr: Arc<Pattern>,
}

#[pymethods]
impl PythonPattern {
    /// Create a new transformer for a term provided by `Expression.map`.
    #[new]
    pub fn new() -> PythonPattern {
        PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Input))),
        }
    }

    /// Create a transformer that expands products and powers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x, x_ = Expression.vars('x', 'x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f((x+1)**2).replace_all(f(x_), x_.transform().expand())
    /// >>> print(e)
    pub fn expand(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Expand(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that computes the product of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2,3).replace_all(f(x_), x_.transform().prod())
    /// >>> print(e)
    pub fn prod(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Product(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that computes the sum of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2,3).replace_all(f(x_), x_.transform().sum())
    /// >>> print(e)
    pub fn sum(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Sum(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that sorts a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(3,2,1).replace_all(f(x_), x_.transform().sort())
    /// >>> print(e)
    pub fn sort(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Sort(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that removes elements from a list if they occur
    /// earlier in the list as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2,1,2).replace_all(f(x_), x_.transform().deduplicate())
    /// >>> print(e)
    ///
    /// Yields `f(1,2)`.
    pub fn deduplicate(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Deduplicate(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that split a sum or product into a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x, x_ = Expression.vars('x', 'x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = (x + 1).replace_all(x_, f(x_.transform().split()))
    /// >>> print(e)
    pub fn split(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Split(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that partitions a list of arguments into named bins of a given length,
    /// returning all partitions and their multiplicity.
    ///
    /// If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
    /// will add all remaining elements to the last bin.
    ///
    /// Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
    /// if possible.
    ///
    /// Note that the functions names to be provided for the bin names must be generated through `Expression.var`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_, f_id, g_id = Expression.vars('x_', 'f', 'g')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2,1,3).replace_all(f(x_), x_.transform().partitions([(f_id, 2), (g_id, 1), (f_id, 1)]))
    /// >>> print(e)
    ///
    /// yields:
    /// ```
    /// 2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)
    /// ```
    #[pyo3(signature = (bins, fill_last = false, repeat = false))]
    pub fn partitions(
        &self,
        bins: Vec<(ConvertibleToPattern, usize)>,
        fill_last: bool,
        repeat: bool,
    ) -> PyResult<PythonPattern> {
        let mut conv_bins = vec![];

        for (x, len) in bins {
            let id = match &*x.to_pattern()?.expr {
                Pattern::Literal(x) => {
                    if let AtomView::Var(x) = x.as_view() {
                        x.get_name()
                    } else {
                        return Err(exceptions::PyValueError::new_err(
                            "Derivative must be taken wrt a variable",
                        ));
                    }
                }
                Pattern::Wildcard(x) => *x,
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ))
                }
            };

            conv_bins.push((id, len));
        }

        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Partition(
                (*self.expr).clone(),
                conv_bins,
                fill_last,
                repeat,
            )))),
        })
    }

    /// Create a transformer that generates all permutations of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_, f_id = Expression.vars('x_', 'f')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2,1,2).replace_all(f(x_), x_.transform().permutations(f_id)
    /// >>> print(e)
    ///
    /// yields:
    /// ```
    /// 4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)
    /// ```
    pub fn permutations(&self, function_name: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let id = match &*function_name.to_pattern()?.expr {
            Pattern::Literal(x) => {
                if let AtomView::Var(x) = x.as_view() {
                    x.get_name()
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            }
            Pattern::Wildcard(x) => *x,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ))
            }
        };

        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Permutations(
                (*self.expr).clone(),
                id,
            )))),
        })
    }

    /// Create a transformer that apply a function `f`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2).replace_all(f(x_), x_.transform().map(lambda r: r**2))
    /// >>> print(e)
    pub fn map(&self, f: PyObject) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Map(
                (*self.expr).clone(),
                Box::new(move |expr, out| {
                    let expr = PythonExpression {
                        expr: Arc::new(expr.into()),
                    };

                    let res = Python::with_gil(|py| {
                        f.call(py, (expr,), None)
                            .expect("Bad callback function")
                            .extract::<ConvertibleToExpression>(py)
                            .expect("Function does not return a pattern")
                    });

                    out.set_from_view(&res.to_expression().expr.as_view());
                }),
            )))),
        })
    }

    /// Create a transformer that derives `self` w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let id = match &*x.to_pattern()?.expr {
            Pattern::Literal(x) => {
                if let AtomView::Var(x) = x.as_view() {
                    x.get_name()
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            }
            Pattern::Wildcard(x) => *x,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ))
            }
        };

        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Derivative(
                (*self.expr).clone(),
                id,
            )))),
        })
    }

    /// Create a transformer that Taylor expands in `x` around `expansion_point` to depth `depth`.
    pub fn taylor_series(
        &self,
        x: ConvertibleToExpression,
        expansion_point: ConvertibleToExpression,
        depth: u32,
    ) -> PyResult<PythonPattern> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_name()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::TaylorSeries(
                (*self.expr).clone(),
                id,
                (*expansion_point.to_expression().expr).clone(),
                depth,
            )))),
        })
    }

    /// Create a transformer that replaces all patterns matching the left-hand side `self` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(3,x)
    /// >>> r = e.transform().replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
    /// >>> print(r)
    pub fn replace_all(
        &self,
        lhs: ConvertibleToPattern,
        rhs: ConvertibleToPattern,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::ReplaceAll(
                (*lhs.to_pattern()?.expr).clone(),
                (*self.expr).clone(),
                (*rhs.to_pattern()?.expr).clone(),
                cond.map(|r| HashMap::clone(&r.restrictions))
                    .unwrap_or(HashMap::default()),
            )))),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let res = WORKSPACE.with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.add(
                &rhs.to_pattern()?.expr,
                workspace,
                get_state!()?.borrow(),
            ))
        })?;

        Ok(PythonPattern {
            expr: Arc::new(res),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        self.__add__(rhs)
    }

    ///  Subtract `other` from this transformer, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        self.__add__(ConvertibleToPattern::Pattern(rhs.to_pattern()?.__neg__()?))
    }

    ///  Subtract this transformer from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        rhs.to_pattern()?
            .__add__(ConvertibleToPattern::Pattern(self.__neg__()?))
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let res = WORKSPACE.with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.mul(
                &rhs.to_pattern()?.expr,
                workspace,
                get_state!()?.borrow(),
            ))
        });

        Ok(PythonPattern {
            expr: Arc::new(res?),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        self.__mul__(rhs)
    }

    /// Divide this transformer by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let res = WORKSPACE.with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.div(
                &rhs.to_pattern()?.expr,
                workspace,
                get_state!()?.borrow(),
            ))
        });

        Ok(PythonPattern {
            expr: Arc::new(res?),
        })
    }

    /// Divide `other` by this transformer, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        rhs.to_pattern()?
            .__truediv__(ConvertibleToPattern::Pattern(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        rhs: ConvertibleToPattern,
        number: Option<i64>,
    ) -> PyResult<PythonPattern> {
        if number.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let res = WORKSPACE.with(|workspace| {
            Ok::<_, PyErr>(self.expr.pow(
                &rhs.to_pattern()?.expr,
                workspace,
                get_state!()?.borrow(),
            ))
        });

        Ok(PythonPattern {
            expr: Arc::new(res?),
        })
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        rhs: ConvertibleToPattern,
        number: Option<i64>,
    ) -> PyResult<PythonPattern> {
        rhs.to_pattern()?
            .__pow__(ConvertibleToPattern::Pattern(self.clone()), number)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: PyObject) -> PyResult<PythonPattern> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: PyObject) -> PyResult<PythonPattern> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current transformer, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonPattern> {
        let res = WORKSPACE
            .with(|workspace| Ok::<Pattern, PyErr>(self.expr.neg(workspace, &&get_state!()?)));

        Ok(PythonPattern {
            expr: Arc::new(res?),
        })
    }
}

/// A Symbolica expression.
///
/// Supports standard arithmetic operations, such
/// as addition and multiplication.
///
/// Examples
/// --------
/// >>> x = Expression.var('x')
/// >>> e = x**2 + 2 - x + 1 / x**4
/// >>> print(e)
#[pyclass(name = "Expression")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonExpression {
    pub expr: Arc<Atom>,
}

/// A restriction on wildcards.
#[pyclass(name = "PatternRestriction")]
#[derive(Clone)]
pub struct PythonPatternRestriction {
    pub restrictions: Arc<HashMap<Identifier, Vec<PatternRestriction>>>,
}

#[pymethods]
impl PythonPatternRestriction {
    /// Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonPatternRestriction {
        let mut res = self.restrictions.as_ref().clone();

        for (id, val) in other.restrictions.as_ref() {
            res.entry(*id).or_insert(vec![]).extend_from_slice(val);
        }

        PythonPatternRestriction {
            restrictions: Arc::new(res),
        }
    }
}

impl<'a> FromPyObject<'a> for ConvertibleToExpression {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            Ok(ConvertibleToExpression(a))
        } else if let Ok(num) = ob.extract::<i64>() {
            Ok(ConvertibleToExpression(PythonExpression {
                expr: Arc::new(Atom::new_num(num)),
            }))
        } else if let Ok(num) = ob.extract::<&PyLong>() {
            let a = format!("{}", num);
            let i = Integer::Large(rug::Integer::parse(&a).unwrap().complete());
            Ok(ConvertibleToExpression(PythonExpression {
                expr: Arc::new(Atom::new_num(i)),
            }))
        } else {
            Err(exceptions::PyValueError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

impl<'a> FromPyObject<'a> for Identifier {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            match a.expr.as_view() {
                AtomView::Var(v) => Ok(v.get_name()),
                e => Err(exceptions::PyValueError::new_err(format!(
                    "Expected variable instead of {:?}",
                    e
                ))),
            }
        } else if let Ok(a) = ob.extract::<PythonFunction>() {
            Ok(a.id)
        } else {
            Err(exceptions::PyValueError::new_err("Not a valid variable"))
        }
    }
}

impl<'a> FromPyObject<'a> for Variable {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        Ok(Variable::Identifier(Identifier::extract(ob)?))
    }
}

pub struct ConvertibleToExpression(PythonExpression);

impl ConvertibleToExpression {
    pub fn to_expression(self) -> PythonExpression {
        self.0
    }
}

impl<'a> FromPyObject<'a> for Complex<f64> {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<f64>() {
            Ok(Complex::new(a, 0.))
        } else if let Ok(a) = ob.extract::<&PyComplex>() {
            Ok(Complex::new(a.real(), a.imag()))
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid complex number",
            ))
        }
    }
}

#[pymethods]
impl PythonExpression {
    /// Create a Symbolica expression that is a single variable.
    ///
    /// Examples
    /// --------
    /// >>> var_x = Expression.var('x')
    /// >>> print(var_x)
    /// x
    ///
    #[classmethod]
    pub fn var(_cls: &PyType, name: &str) -> PyResult<PythonExpression> {
        let mut guard = get_state_mut!()?;
        let state = guard.borrow_mut();
        // TODO: check if the name meets the requirements
        let id = state.get_or_insert_var(name);
        let var = Atom::new_var(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

    /// Create a Symbolica variable for every name in `*names`.
    #[pyo3(signature = (*args,))]
    #[classmethod]
    pub fn vars(_cls: &PyType, args: &PyTuple) -> PyResult<Vec<PythonExpression>> {
        let mut guard = get_state_mut!()?;
        let state = guard.borrow_mut();
        let mut result = Vec::with_capacity(args.len());

        for a in args {
            // TODO: check if the name meets the requirements
            let name = a.extract::<&str>()?;
            let id = state.get_or_insert_var(name);
            let var = Atom::new_var(id);

            result.push(PythonExpression {
                expr: Arc::new(var),
            });
        }

        Ok(result)
    }

    /// Create a new Symbolica function with a given name.
    ///
    /// Examples
    /// --------
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2)
    /// >>> print(e)
    /// f(1,2)
    #[classmethod]
    pub fn fun(_cls: &PyType, name: &str, is_symmetric: Option<bool>) -> PyResult<PythonFunction> {
        PythonFunction::__new__(name, is_symmetric)
    }

    /// Create a Symbolica function for every name in `*names`.
    #[pyo3(signature = (*args,))]
    #[classmethod]
    pub fn funs(_cls: &PyType, args: &PyTuple) -> PyResult<Vec<PythonFunction>> {
        let mut result = Vec::with_capacity(args.len());
        for a in args {
            let name = a.extract::<&str>()?;
            result.push(PythonFunction::__new__(name, None)?);
        }

        Ok(result)
    }

    /// Create a new Symbolica number.
    ///
    /// Examples
    /// --------
    /// >>> e = Expression.num(1) / 2
    /// >>> print(e)
    /// 1/2
    #[classmethod]
    pub fn num(_cls: &PyType, num: &PyLong) -> PyResult<PythonExpression> {
        if let Ok(num) = num.extract::<i64>() {
            Ok(PythonExpression {
                expr: Arc::new(Atom::new_num(num)),
            })
        } else {
            let a = format!("{}", num);
            PythonExpression::parse(_cls, &a)
        }
    }

    /// Parse a Symbolica expression from a string.
    ///
    /// Parameters
    /// ----------
    /// input: str
    ///     An input string. UTF-8 character are allowed.
    ///
    /// Examples
    /// --------
    /// >>> e = Expression.parse('x^2+y+y*4')
    /// >>> print(e)
    /// x^2+5*y
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica expression.
    ///
    #[classmethod]
    pub fn parse(_cls: &PyType, arg: &str) -> PyResult<PythonExpression> {
        let e = WORKSPACE.with(|f| {
            Atom::parse(arg, &mut &mut get_state_mut!()?, f)
                .map_err(exceptions::PyValueError::new_err)
        })?;

        Ok(PythonExpression { expr: Arc::new(e) })
    }

    /// Copy the expression.
    pub fn __copy__(&self) -> PythonExpression {
        PythonExpression {
            expr: Arc::new((*self.expr).clone()),
        }
    }

    /// Convert the expression into a human-readable string.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "{}",
            AtomPrinter::new(self.expr.as_view(), &&get_state!()?)
        ))
    }

    /// Convert the expression into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> a = Expression.parse('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.pretty_str(number_thousands_separator='_', multiplication_operator=' '))
    #[pyo3(signature =
    (terms_on_new_line = false,
        color_top_level_sum = true,
        color_builtin_functions = true,
        print_finite_field = true,
        explicit_rational_polynomial = false,
        number_thousands_separator = None,
        multiplication_operator = '*',
        square_brackets_for_function = false,
        num_exp_as_superscript = true,
        latex = false)
    )]
    pub fn pretty_str(
        &self,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_functions: bool,
        print_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        latex: bool,
    ) -> PyResult<String> {
        Ok(format!(
            "{}",
            AtomPrinter::new_with_options(
                self.expr.as_view(),
                PrintOptions {
                    terms_on_new_line,
                    color_top_level_sum,
                    color_builtin_functions,
                    print_finite_field,
                    explicit_rational_polynomial,
                    number_thousands_separator,
                    multiplication_operator,
                    square_brackets_for_function,
                    num_exp_as_superscript,
                    latex
                },
                &&get_state!()?,
            )
        ))
    }

    /// Convert the expression into a LaTeX string.
    ///
    /// Examples
    /// --------
    /// >>> a = Expression.parse('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.to_latex())
    ///
    /// Yields `$$z^{34}+x^{x+2}+y^{4}+f(x,x^{2})+128378127123 z^{\\frac{2}{3}} w^{2} \\frac{1}{x} \\frac{1}{y}+\\frac{3}{5}$$`.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            AtomPrinter::new_with_options(
                self.expr.as_view(),
                PrintOptions::latex(),
                &&get_state!()?,
            )
        ))
    }

    /// Hash the expression.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the type of the atom.
    pub fn get_type(&self) -> PythonAtomType {
        match self.expr.as_ref() {
            Atom::Num(_) => PythonAtomType::Num,
            Atom::Var(_) => PythonAtomType::Var,
            Atom::Fun(_) => PythonAtomType::Fn,
            Atom::Add(_) => PythonAtomType::Add,
            Atom::Mul(_) => PythonAtomType::Mul,
            Atom::Pow(_) => PythonAtomType::Pow,
            Atom::Empty => unreachable!(),
        }
    }

    /// Convert the expression to a tree.
    pub fn to_atom_tree(&self) -> PyResult<PythonAtomTree> {
        self.expr.as_view().into()
    }

    /// Get the name of a variable or function if the current atom
    /// is a variable or function.
    pub fn get_name(&self) -> PyResult<Option<String>> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(get_state!()?
                .get_name(v.to_var_view().get_name())
                .map(|x| x.to_string())),
            Atom::Fun(f) => Ok(get_state!()?
                .get_name(f.to_fun_view().get_name())
                .map(|x| x.to_string())),
            _ => Ok(None),
        }
    }

    /// Create a wildcard from a variable name by appending a _
    /// if none is present yet.
    #[getter]
    fn get_w(&self) -> PyResult<PythonExpression> {
        let mut guard = get_state_mut!()?;
        let state = guard.borrow_mut();
        let mut var_name = match self.expr.as_view() {
            AtomView::Var(v) => {
                if let Some(true) = state.is_wildcard(v.get_name()) {
                    return Ok(self.clone());
                } else {
                    // create name with underscore
                    state.get_name(v.get_name()).unwrap().to_string()
                }
            }
            AtomView::Fun(f) => {
                if let Some(true) = state.is_wildcard(f.get_name()) {
                    return Ok(self.clone());
                } else {
                    // create name with underscore
                    state.get_name(f.get_name()).unwrap().to_string()
                }
            }
            x => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Cannot convert to wildcard: {:?}",
                    x
                )));
            }
        };

        // create name with underscore
        var_name.push('_');

        // TODO: check if the name meets the requirements
        let id = state.get_or_insert_var(var_name);
        let var = Atom::new_var(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_add();

            a.extend(self.expr.as_view());
            a.extend(rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(ConvertibleToExpression(rhs.to_expression().__neg__()?))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__add__(ConvertibleToExpression(self.__neg__()?))
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_mul();

            a.extend(self.expr.as_view());
            a.extend(rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }

    /// Divide this expression by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut pow = workspace.new_atom();
            let pow_num = pow.to_num();
            pow_num.set_from_number(Number::Natural(-1, 1));

            let mut e = workspace.new_atom();
            let a = e.to_pow();
            a.set_from_base_and_exp(rhs.to_expression().expr.as_view(), pow.get().as_view());
            a.set_dirty(true);

            let mut m = workspace.new_atom();
            let md = m.to_mul();

            md.extend(self.expr.as_view());
            md.extend(e.get().as_view());
            md.set_dirty(true);

            let mut b = Atom::new();
            m.get()
                .as_view()
                .normalize(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Divide `other` by this expression, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__truediv__(ConvertibleToExpression(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        rhs: ConvertibleToExpression,
        number: Option<i64>,
    ) -> PyResult<PythonExpression> {
        if number.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_pow();
            a.set_from_base_and_exp(self.expr.as_view(), rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        rhs: ConvertibleToExpression,
        number: Option<i64>,
    ) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__pow__(ConvertibleToExpression(self.clone()), number)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: PyObject) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: PyObject) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current expression, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonExpression> {
        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_mul();

            let mut sign = workspace.new_atom();
            let sign_num = sign.to_num();
            sign_num.set_from_number(Number::Natural(-1, 1));

            a.extend(self.expr.as_view());
            a.extend(sign.get().as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Return the length of the atom.
    fn __len__(&self) -> usize {
        match self.expr.as_view() {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            _ => 1,
        }
    }

    /// Convert the input to a transformer, on which subsequent transformations can be applied.
    pub fn transform(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(self.expr.into_pattern(&&get_state!()?)),
        })
    }

    /// Create a pattern restriction based on the wildcard length before downcasting.
    pub fn req_len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                let mut h = HashMap::default();
                h.insert(
                    name,
                    vec![PatternRestriction::Length(min_length, max_length)],
                );

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(h),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that tests the type of the atom.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, AtomType
    /// >>> x, x_ = Expression.vars('x', 'x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(x)*f(2)*f(f(3))
    /// >>> e = e.replace_all(f(x_), 1, x_.req_type(AtomType.Num))
    /// >>> print(e)
    ///
    /// Yields `f(x)*f(1)`.
    pub fn req_type(&self, atom_type: PythonAtomType) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                let mut h = HashMap::default();
                h.insert(
                    name,
                    vec![PatternRestriction::IsAtomType(match atom_type {
                        PythonAtomType::Num => AtomType::Num,
                        PythonAtomType::Var => AtomType::Var,
                        PythonAtomType::Add => AtomType::Add,
                        PythonAtomType::Mul => AtomType::Mul,
                        PythonAtomType::Pow => AtomType::Pow,
                        PythonAtomType::Fn => AtomType::Fun,
                    })],
                );

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(h),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that treats the wildcard as a literal variable,
    /// so that it only matches to itself.
    pub fn req_lit(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                let mut h = HashMap::default();
                h.insert(name, vec![PatternRestriction::IsLiteralWildcard(name)]);

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(h),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Compare two expressions.
    fn __richcmp__(&self, other: ConvertibleToExpression, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.expr == other.to_expression().expr),
            CompareOp::Ne => Ok(self.expr != other.to_expression().expr),
            _ => {
                let other = other.to_expression();
                if let n1 @ AtomView::Num(_) = self.expr.as_view() {
                    if let n2 @ AtomView::Num(_) = other.expr.as_view() {
                        return Ok(match op {
                            CompareOp::Eq => n1 == n2,
                            CompareOp::Ge => n1 >= n2,
                            CompareOp::Gt => n1 > n2,
                            CompareOp::Le => n1 <= n2,
                            CompareOp::Lt => n1 < n2,
                            CompareOp::Ne => n1 != n2,
                        });
                    }
                }

                Err(exceptions::PyTypeError::new_err(format!(
                    "Inequalities between expression that are not numbers are not allowed in {} {} {}",
                    self.__str__()?,
                    match op {
                        CompareOp::Eq => "==",
                        CompareOp::Ge => ">=",
                        CompareOp::Gt => ">",
                        CompareOp::Le => "<=",
                        CompareOp::Lt => "<",
                        CompareOp::Ne => "!=",
                    },
                    other.__str__()?,
                )
            ))
            }
        }
    }

    /// Create a new pattern restriction that calls the function `filter_fn` with the matched
    /// atom that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))
    pub fn req(&self, filter_fn: PyObject) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let mut h = HashMap::default();
        h.insert(
            id,
            vec![PatternRestriction::Filter(Box::new(move |m| {
                let data = PythonExpression {
                    expr: Arc::new({
                        let mut a = Atom::new();
                        m.to_atom(&mut a);
                        a
                    }),
                };

                Python::with_gil(|py| {
                    filter_fn
                        .call(py, (data,), None)
                        .expect("Bad callback function")
                        .extract::<bool>(py)
                        .expect("Pattern filter does not return a boolean")
                })
            }))],
        );

        Ok(PythonPatternRestriction {
            restrictions: Arc::new(h),
        })
    }

    /// Create a new pattern restriction that calls the function `cmp_fn` with another the matched
    /// atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = Expression.vars('x_', 'y_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))
    pub fn req_cmp(
        &self,
        other: PythonExpression,
        cmp_fn: PyObject,
    ) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let other_id = match other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !get_state!()?.is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let mut h = HashMap::default();
        h.insert(
            id,
            vec![PatternRestriction::Cmp(
                other_id,
                Box::new(move |m1, m2| {
                    let data1 = PythonExpression {
                        expr: Arc::new({
                            let mut a = Atom::new();
                            m1.to_atom(&mut a);
                            a
                        }),
                    };

                    let data2 = PythonExpression {
                        expr: Arc::new({
                            let mut a = Atom::new();
                            m2.to_atom(&mut a);
                            a
                        }),
                    };

                    Python::with_gil(|py| {
                        cmp_fn
                            .call(py, (data1, data2), None)
                            .expect("Bad callback function")
                            .extract::<bool>(py)
                            .expect("Pattern comparison does not return a boolean")
                    })
                }),
            )],
        );

        Ok(PythonPatternRestriction {
            restrictions: Arc::new(h),
        })
    }

    /// Create an iterator over all atoms in the expression.
    fn __iter__(&self) -> PyResult<PythonAtomIterator> {
        match self.expr.as_view() {
            AtomView::Add(_) | AtomView::Mul(_) | AtomView::Fun(_) => {}
            x => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Non-iterable type: {:?}",
                    x
                )));
            }
        };

        Ok(PythonAtomIterator::from_expr(self.clone()))
    }

    /// Map the transformations to every term in the expression.
    /// The execution happen in parallel.
    ///
    /// No new functions or variables can be defined and no new
    /// expressions can be parsed inside the map. Doing so will
    /// result in a deadlock.
    ///
    /// Examples
    /// --------
    /// >>> x, x_ = Expression.vars('x', 'x_')
    /// >>> e = (1+x)**2
    /// >>> r = e.map(Transformer().expand().replace_all(x, 6))
    /// >>> print(r)
    pub fn map(&self, op: PythonPattern, py: Python) -> PyResult<PythonExpression> {
        let t = match op.expr.as_ref() {
            Pattern::Transformer(t) => t,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Operation must of a transformer".to_string(),
                ));
            }
        };

        // release the GIL as Python functions may be called from
        // within the term mapper
        let mut stream = py.allow_threads(move || {
            // map every term in the expression
            let stream = TermStreamer::new_from((*self.expr).clone());
            let state = get_state!()?;
            let m = stream.map(|workspace, x| {
                let mut out = Atom::new();
                let restrictions = HashMap::default();
                let mut match_stack = MatchStack::new(&restrictions);
                match_stack.insert(INPUT_ID, Match::Single(x.as_view()));

                t.execute(&state.borrow(), workspace, &match_stack, &mut out);

                out
            });
            Ok::<_, PyErr>(m)
        })?;

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| stream.to_expression(workspace, state.borrow()));

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.

    /// Parameters
    /// ----------
    /// vars : List[Expression]
    ///         A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();
        for v in vars {
            match v.expr.as_view() {
                AtomView::Var(v) => var_map.push(v.get_name().into()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {:?}",
                        e
                    )))?;
                }
            }
        }

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr
                .as_view()
                .set_coefficient_ring(&var_map, state.borrow(), workspace, &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Expand the expression.
    pub fn expand(&self) -> PyResult<PythonExpression> {
        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr
                .as_view()
                .expand(workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x))
    ///
    /// yields `x^2+x*(y+5)+5`.
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> exp, coeff = Expression.funs('var', 'coeff')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x, key_map=lambda x: exp(x), coeff_map=lambda x: coeff(x)))
    ///
    /// yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.
    pub fn collect(
        &self,
        x: ConvertibleToExpression,
        key_map: Option<PyObject>,
        coeff_map: Option<PyObject>,
    ) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_name()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function name",
            ));
        };

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr.as_view().collect(
                id,
                workspace,
                state.borrow(),
                if let Some(key_map) = key_map {
                    Some(Box::new(move |key, out| {
                        Python::with_gil(|py| {
                            let key = PythonExpression {
                                expr: Arc::new(key.into()),
                            };

                            out.set_from_view(
                                &key_map
                                    .call(py, (key,), None)
                                    .expect("Bad callback function")
                                    .extract::<PythonExpression>(py)
                                    .expect("Key map should return an expression")
                                    .expr
                                    .as_view(),
                            )
                        });
                    }))
                } else {
                    None
                },
                if let Some(coeff_map) = coeff_map {
                    Some(Box::new(move |coeff, out| {
                        Python::with_gil(|py| {
                            let coeff = PythonExpression {
                                expr: Arc::new(coeff.into()),
                            };

                            out.set_from_view(
                                &coeff_map
                                    .call(py, (coeff,), None)
                                    .expect("Bad callback function")
                                    .extract::<PythonExpression>(py)
                                    .expect("Coeff map should return an expression")
                                    .expr
                                    .as_view(),
                            )
                        });
                    }))
                } else {
                    None
                },
                &mut b,
            );
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Examples
    /// --------
    ///
    /// from symbolica import Expression
    /// >>>
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> for a in e.coefficient_list(x):
    /// >>>     print(a[0], a[1])
    ///
    /// yields
    ///
    /// ```
    /// x y+5
    /// x^2 1
    /// 1 5
    /// ```
    pub fn coefficient_list(
        &self,
        x: ConvertibleToExpression,
    ) -> PyResult<Vec<(PythonExpression, PythonExpression)>> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_name()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Coefficient list must be done wrt a variable or function name",
            ));
        };

        let state = get_state!()?;
        WORKSPACE.with(|workspace| {
            let (list, rest) = self.expr.as_view().coefficient_list(id, workspace, &state);

            let mut py_list: Vec<_> = list
                .into_iter()
                .map(|e| {
                    (
                        PythonExpression {
                            expr: Arc::new(e.0.into()),
                        },
                        PythonExpression {
                            expr: Arc::new(e.1),
                        },
                    )
                })
                .collect();

            if let Atom::Num(n) = &rest {
                if n.to_num_view().is_zero() {
                    return Ok(py_list);
                }
            }

            py_list.push((
                PythonExpression {
                    expr: Arc::new(Atom::new_num(1)),
                },
                PythonExpression {
                    expr: Arc::new(rest),
                },
            ));

            Ok(py_list)
        })
    }

    /// Derive the expression w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_name()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr
                .as_view()
                .derivative(id, workspace, state.borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// Example
    /// -------
    /// >>> from symbolica import Expression
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> f = Expression.fun('f')
    /// >>>
    /// >>> e = 2* x**2 * y + f(x)
    /// >>> e = e.taylor_series(x, 0, 2)
    /// >>>
    /// >>> print(e)
    ///
    /// yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.
    pub fn taylor_series(
        &self,
        x: ConvertibleToExpression,
        expansion_point: ConvertibleToExpression,
        depth: u32,
    ) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_name()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let state = get_state!()?;
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr.as_view().taylor_series(
                id,
                expansion_point.to_expression().expr.as_view(),
                depth,
                workspace,
                state.borrow(),
                &mut b,
            );
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    pub fn to_polynomial(&self, vars: Option<Vec<PythonExpression>>) -> PyResult<PythonPolynomial> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {:?}",
                            e
                        )))?;
                    }
                }
            }
        }

        self.expr
            .as_view()
            .to_polynomial(
                RationalField::new(),
                if var_map.is_empty() {
                    None
                } else {
                    Some(var_map.as_slice())
                },
            )
            .map(|x| PythonPolynomial { poly: Arc::new(x) })
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "Could not convert to polynomial: {:?}",
                    e
                ))
            })
    }

    /// Convert the expression to a rational polynomial, optionally, with the variables and the ordering specified in `vars`.
    /// The latter is useful if it is known in advance that more variables may be added in the future to the
    /// rational polynomial through composition with other rational polynomials.
    ///
    /// Examples
    /// --------
    /// >>> a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
    /// >>> print(a)
    pub fn to_rational_polynomial(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomial> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {:?}",
                            e
                        )))?;
                    }
                }
            }
        }

        WORKSPACE.with(|workspace| {
            self.expr
                .as_view()
                .to_rational_polynomial(
                    workspace,
                    &&get_state!()?,
                    RationalField::new(),
                    IntegerRing::new(),
                    if var_map.is_empty() {
                        None
                    } else {
                        Some(var_map.as_slice())
                    },
                )
                .map(|x| PythonRationalPolynomial { poly: Arc::new(x) })
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!(
                        "Could not convert to polynomial: {:?}",
                        e
                    ))
                })
        })
    }

    /// Similar to [PythonExpression::to_rational_polynomial()], but the power of each variable limited to 255.
    pub fn to_rational_polynomial_small_exponent(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomialSmallExponent> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {:?}",
                            e
                        )))?;
                    }
                }
            }
        }

        WORKSPACE.with(|workspace| {
            self.expr
                .as_view()
                .to_rational_polynomial(
                    workspace,
                    &&get_state!()?,
                    RationalField::new(),
                    IntegerRing::new(),
                    if var_map.is_empty() {
                        None
                    } else {
                        Some(var_map.as_slice())
                    },
                )
                .map(|x| PythonRationalPolynomialSmallExponent { poly: Arc::new(x) })
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!(
                        "Could not convert to polynomial: {:?}",
                        e
                    ))
                })
        })
    }

    /// Return an iterator over the pattern `self` matching to `lhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, x_ = Expression.vars('x','x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(x)*f(1)*f(2)*f(3)
    /// >>> for match in e.match(f(x_)):
    /// >>>    for map in match:
    /// >>>        print(map[0],'=', map[1])
    #[pyo3(name = "r#match")]
    pub fn pattern_match(
        &self,
        lhs: ConvertibleToPattern,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonMatchIterator> {
        let restrictions = cond
            .map(|r| r.restrictions.clone())
            .unwrap_or(Arc::new(HashMap::default()));
        Ok(PythonMatchIterator::new(
            (
                lhs.to_pattern()?.expr,
                self.expr.clone(),
                restrictions,
                get_state!()?.clone(), // FIXME: state is cloned
            ),
            move |(lhs, target, res, state)| {
                PatternAtomTreeIterator::new(lhs, target.as_view(), state, res)
            },
        ))
    }

    /// Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> for r in e.replace(f(x_), f(x_ + 1)):
    /// >>>     print(r)
    ///
    /// Yields:
    /// ```
    /// f(2)*f(2)*f(3)
    /// f(1)*f(3)*f(3)
    /// f(1)*f(2)*f(4)
    /// ```
    pub fn replace(
        &self,
        lhs: ConvertibleToPattern,
        rhs: ConvertibleToPattern,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonReplaceIterator> {
        let restrictions = cond
            .map(|r| r.restrictions.clone())
            .unwrap_or(Arc::new(HashMap::default()));

        Ok(PythonReplaceIterator::new(
            (
                lhs.to_pattern()?.expr,
                self.expr.clone(),
                rhs.to_pattern()?.expr,
                restrictions,
                get_state!()?.clone(), // FIXME: state is cloned
            ),
            move |(lhs, target, rhs, res, state)| {
                ReplaceIterator::new(lhs, target.as_view(), rhs, state, res)
            },
        ))
    }

    /// Replace all atoms matching the pattern `pattern` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(3,x)
    /// >>> r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
    /// >>> print(r)
    pub fn replace_all(
        &self,
        pattern: ConvertibleToPattern,
        rhs: ConvertibleToPattern,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonExpression> {
        let pattern = &pattern.to_pattern()?.expr;
        let rhs = &rhs.to_pattern()?.expr;
        let mut out = Atom::new();

        WORKSPACE.with(|workspace| {
            Ok::<_, PyErr>(
                pattern.replace_all(
                    self.expr.as_view(),
                    rhs,
                    &&get_state!()?,
                    workspace,
                    cond.as_ref()
                        .map(|r| r.restrictions.as_ref())
                        .unwrap_or(&HashMap::default()),
                    &mut out,
                ),
            )
        })?;

        Ok(PythonExpression {
            expr: Arc::new(out),
        })
    }

    /// Solve a linear system in the variables `variables`, where each expression
    /// in the system is understood to yield 0.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = Expression.vars('x', 'y', 'c')
    /// >>> f = Expression.fun('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[classmethod]
    pub fn solve_linear_system(
        _cls: &PyType,
        system: Vec<ConvertibleToExpression>,
        variables: Vec<PythonExpression>,
    ) -> PyResult<Vec<PythonExpression>> {
        let system: Vec<_> = system.into_iter().map(|x| x.to_expression()).collect();
        let system_b: Vec<_> = system.iter().map(|x| x.expr.as_view()).collect();

        let mut vars = vec![];
        for v in variables {
            match v.expr.as_view() {
                AtomView::Var(v) => vars.push(v.get_name().into()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {:?}",
                        e
                    )))?;
                }
            }
        }

        let mut guard = get_state_mut!()?;
        let state = guard.borrow_mut();

        let res = WORKSPACE
            .with(|workspace| {
                AtomView::solve_linear_system::<u16>(&system_b, &vars, workspace, state)
            })
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not solve system: {:?}", e))
            })?;

        Ok(res
            .into_iter()
            .map(|x| PythonExpression { expr: Arc::new(x) })
            .collect())
    }

    /// Evaluate the expression, using a map of all the variables and
    /// user functions to a float.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x = Expression.var('x')
    /// >>> f = Expression.fun('f')
    /// >>> e = Expression.parse('cos(x)')*3 + f(x,2)
    /// >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
    pub fn evaluate(
        &self,
        vars: HashMap<Variable, f64>,
        functions: HashMap<Variable, PyObject>,
    ) -> f64 {
        let mut cache = HashMap::default();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    EvaluationFn::new(Box::new(move |args, _, _, _| {
                        Python::with_gil(|py| {
                            v.call(py, (args.to_vec(),), None)
                                .expect("Bad callback function")
                                .extract::<f64>(py)
                                .expect("Function does not return a float")
                        })
                    })),
                )
            })
            .collect();

        self.expr.as_view().evaluate(&vars, &functions, &mut cache)
    }

    /// Evaluate the expression, using a map of all the variables and
    /// user functions to a complex number.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> e = Expression.parse('sqrt(x)')*y
    /// >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
    pub fn evaluate_complex<'py>(
        &self,
        py: Python<'py>,
        vars: HashMap<Variable, Complex<f64>>,
        functions: HashMap<Variable, PyObject>,
    ) -> &'py PyComplex {
        let mut cache = HashMap::default();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    EvaluationFn::new(Box::new(move |args: &[Complex<f64>], _, _, _| {
                        Python::with_gil(|py| {
                            v.call(
                                py,
                                (args
                                    .iter()
                                    .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                    .collect::<Vec<_>>(),),
                                None,
                            )
                            .expect("Bad callback function")
                            .extract::<Complex<f64>>(py)
                            .expect("Function does not return a complex number")
                        })
                    })),
                )
            })
            .collect();

        let r = self.expr.as_view().evaluate(&vars, &functions, &mut cache);
        PyComplex::from_doubles(py, r.re, r.im)
    }
}

/// A function class for python that constructs an `Expression` when called with arguments.
/// This allows to write:
/// ```python
/// f = Expression.fun("f")
/// e = f(1,2,3)
/// ```
#[pyclass(name = "Function")]
#[derive(Clone)]
pub struct PythonFunction {
    id: Identifier,
}

#[pymethods]
impl PythonFunction {
    /// Create a new function from a `name`. Can be turned into a symmetric function
    /// using `is_symmetric=True`.
    ///
    /// Once attributes are defined on a function, they cannot be redefined later.
    #[new]
    pub fn __new__(name: &str, is_symmetric: Option<bool>) -> PyResult<Self> {
        // TODO: parse and check if this is a valid function name
        let id = if is_symmetric.unwrap_or(false) {
            get_state_mut!()?
                .borrow_mut()
                .get_or_insert_fn(name, Some(vec![FunctionAttribute::Symmetric]))
                .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?
        } else {
            get_state_mut!()?
                .borrow_mut()
                .get_or_insert_fn(name, Some(vec![]))
                .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?
        };
        Ok(PythonFunction { id })
    }

    #[getter]
    fn get_w(&mut self) -> PythonFunction {
        PythonFunction { id: self.id }
    }

    /// Create a Symbolica expression or transformer by calling the function with appropriate arguments.
    ///
    /// Examples
    /// -------
    /// >>> x = Expression.vars('x')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(3,x)
    /// >>> print(e)
    /// f(3,x)
    #[pyo3(signature = (*args,))]
    pub fn __call__(&self, args: &PyTuple, py: Python) -> PyResult<PyObject> {
        let mut fn_args = Vec::with_capacity(args.len());

        for arg in args {
            if let Ok(a) = arg.extract::<ConvertibleToExpression>() {
                fn_args.push(Pattern::Literal((*a.to_expression().expr).clone()));
            } else if let Ok(a) = arg.extract::<ConvertibleToPattern>() {
                fn_args.push((*a.to_pattern()?.expr).clone());
            } else {
                let msg = format!("Unknown type: {}", arg.get_type().name().unwrap());
                return Err(exceptions::PyTypeError::new_err(msg));
            }
        }

        if fn_args.iter().all(|x| matches!(x, Pattern::Literal(_))) {
            // simplify to literal expression
            WORKSPACE.with(|workspace| {
                let mut fun_b = workspace.new_atom();
                let fun = fun_b.to_fun();
                fun.set_from_name(self.id);
                fun.set_dirty(true);

                for x in fn_args {
                    if let Pattern::Literal(a) = x {
                        fun.add_arg(a.as_view());
                    }
                }

                let mut out = Atom::new();
                fun_b
                    .get()
                    .as_view()
                    .normalize(workspace, &&get_state!()?, &mut out);

                Ok(PythonExpression {
                    expr: Arc::new(out),
                }
                .into_py(py))
            })
        } else {
            let p = Pattern::Fn(
                self.id,
                get_state!()?.is_wildcard(self.id).unwrap(),
                fn_args,
            );
            Ok(PythonPattern { expr: Arc::new(p) }.into_py(py))
        }
    }
}

self_cell!(
    #[pyclass]
    pub struct PythonAtomIterator {
        owner: Arc<Atom>,
        #[covariant]
        dependent: ListIteratorD,
    }
);

impl PythonAtomIterator {
    /// Create a self-referential structure for the iterator.
    pub fn from_expr(expr: PythonExpression) -> PythonAtomIterator {
        PythonAtomIterator::new(expr.expr, |expr| match expr.as_view() {
            AtomView::Add(a) => a.iter(),
            AtomView::Mul(m) => m.iter(),
            AtomView::Fun(f) => f.iter(),
            _ => unreachable!(),
        })
    }
}

#[pymethods]
impl PythonAtomIterator {
    fn __next__(&mut self) -> Option<PythonExpression> {
        self.with_dependent_mut(|_, i| {
            i.next().map(|e| PythonExpression {
                expr: Arc::new({
                    let mut owned = Atom::new();
                    owned.set_from_view(&e);
                    owned
                }),
            })
        })
    }
}

type OwnedMatch = (
    Arc<Pattern>,
    Arc<Atom>,
    Arc<HashMap<Identifier, Vec<PatternRestriction>>>,
    State,
);
type MatchIterator<'a> = PatternAtomTreeIterator<'a, 'a, crate::representations::default::Linear>;

self_cell!(
    /// An iterator over matches.
    #[pyclass]
    pub struct PythonMatchIterator {
        owner: OwnedMatch,
        #[not_covariant]
        dependent: MatchIterator,
    }
);

#[pymethods]
impl PythonMatchIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next match.
    fn __next__(&mut self) -> Option<HashMap<PythonExpression, PythonExpression>> {
        self.with_dependent_mut(|_, i| {
            i.next().map(|(_, _, _, matches)| {
                matches
                    .into_iter()
                    .map(|m| {
                        (
                            PythonExpression {
                                expr: Arc::new(Atom::new_var(m.0)),
                            },
                            PythonExpression {
                                expr: Arc::new({
                                    let mut a = Atom::new();
                                    m.1.to_atom(&mut a);
                                    a
                                }),
                            },
                        )
                    })
                    .collect()
            })
        })
    }
}

type OwnedReplace = (
    Arc<Pattern>,
    Arc<Atom>,
    Arc<Pattern>,
    Arc<HashMap<Identifier, Vec<PatternRestriction>>>,
    State,
);
type ReplaceIteratorOne<'a> = ReplaceIterator<'a, 'a, crate::representations::default::Linear>;

self_cell!(
    /// An iterator over all single replacements.
    #[pyclass]
    pub struct PythonReplaceIterator {
        owner: OwnedReplace,
        #[not_covariant]
        dependent: ReplaceIteratorOne,
    }
);

#[pymethods]
impl PythonReplaceIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next replacement.
    fn __next__(&mut self) -> PyResult<Option<PythonExpression>> {
        self.with_dependent_mut(|_, i| {
            WORKSPACE.with(|workspace| {
                let mut out = Atom::new();

                if i.next(&&get_state!()?, workspace, &mut out).is_none() {
                    Ok(None)
                } else {
                    Ok::<_, PyErr>(Some(PythonExpression {
                        expr: Arc::new(out),
                    }))
                }
            })
        })
    }
}

#[pyclass(name = "Polynomial")]
#[derive(Clone)]
pub struct PythonPolynomial {
    pub poly: Arc<MultivariatePolynomial<RationalField, u32>>,
}

#[pymethods]
impl PythonPolynomial {
    /// Parse a polynomial with rational coefficients from a string.
    /// The input must be written in an expanded format and a list of all
    /// the variables must be provided.
    ///
    /// If these requirements are too strict, use `Expression.to_polynomial()` or
    /// `RationalPolynomial.parse()` instead.
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica polynomial.
    #[classmethod]
    pub fn parse(_cls: &PyType, arg: &str, vars: Vec<&str>) -> PyResult<Self> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        {
            let mut state = get_state_mut!()?;
            for v in vars {
                let id = state.get_or_insert_var(v);
                var_map.push(id.into());
                var_name_map.push(v.into());
            }
        }

        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(RationalField::new(), &var_map, &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }

    /// Convert the polynomial to a polynomial with integer coefficients, if possible.
    pub fn to_integer_polynomial(&self) -> PyResult<PythonIntegerPolynomial> {
        let mut poly_int = MultivariatePolynomial::new(
            self.poly.nvars,
            IntegerRing::new(),
            Some(self.poly.nterms),
            self.poly.var_map.as_ref().map(|x| x.as_slice()),
        );

        let mut new_exponent = SmallVec::<[u8; 5]>::new();

        for t in self.poly.into_iter() {
            if !t.coefficient.is_integer() {
                Err(exceptions::PyValueError::new_err(format!(
                    "Coefficient {} is not an integer",
                    t.coefficient
                )))?;
            }

            new_exponent.clear();
            for e in t.exponents {
                if *e > u8::MAX as u32 {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Exponent {} is too large",
                        e
                    )))?;
                }
                new_exponent.push(*e as u8);
            }

            poly_int.append_monomial(t.coefficient.numerator(), &new_exponent);
        }

        Ok(PythonIntegerPolynomial {
            poly: Arc::new(poly_int),
        })
    }

    #[pyo3(signature =
        (iterations = 1000,
        to_file = None)
    )]
    /// Optimize the polynomial for evaluation using `iterations` number of iterations.
    /// The optimized output can be exported in a C++ format using `to_file`.
    ///
    /// Returns an evaluator for the polynomial.
    pub fn optimize(
        &self,
        iterations: usize,
        to_file: Option<String>,
    ) -> PyResult<PythonInstructionEvaluator> {
        let o = self.poly.optimize(iterations);
        if let Some(file) = to_file.as_ref() {
            std::fs::write(
                file,
                format!(
                    "{}",
                    InstructionSetPrinter {
                        instr: &o,
                        state: &&get_state!()?,
                        mode: InstructionSetMode::CPP(InstructionSetModeCPPSettings {
                            write_header_and_test: true,
                            always_pass_output_array: false,
                        })
                    }
                ),
            )
            .unwrap();
        }

        let o_f64 = o.convert::<f64>();
        Ok(PythonInstructionEvaluator {
            instr: Arc::new(o_f64.evaluator()),
        })
    }
}

#[pyclass(name = "Evaluator")]
#[derive(Clone)]
pub struct PythonInstructionEvaluator {
    pub instr: Arc<InstructionEvaluator<f64>>,
}

#[pymethods]
impl PythonInstructionEvaluator {
    /// Evaluate the polynomial for multiple inputs and return the result.
    fn evaluate(&self, inputs: Vec<Vec<f64>>) -> Vec<f64> {
        let mut eval = (*self.instr).clone();

        inputs.iter().map(|s| eval.evaluate(s)[0]).collect()
    }
}

#[pyclass(name = "IntegerPolynomial")]
#[derive(Clone)]
pub struct PythonIntegerPolynomial {
    pub poly: Arc<MultivariatePolynomial<IntegerRing, u8>>,
}

#[pymethods]
impl PythonIntegerPolynomial {
    /// Parse a polynomial with integer coefficients from a string.
    /// The input must be written in an expanded format and a list of all
    /// the variables must be provided.
    ///
    /// If these requirements are too strict, use `Expression.to_polynomial()` or
    /// `RationalPolynomial.parse()` instead.
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'])
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica polynomial.
    #[classmethod]
    pub fn parse(_cls: &PyType, arg: &str, vars: Vec<&str>) -> PyResult<Self> {
        let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        {
            let mut state = get_state_mut!()?;
            for v in vars {
                let id = state.get_or_insert_var(v);
                var_map.push(id.into());
                var_name_map.push(v.into());
            }
        }

        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(IntegerRing::new(), &var_map, &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }
}

macro_rules! generate_methods {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            /// Copy the polynomial.
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            /// Print the polynomial in a human-readable format.
            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly,
                        state: &&get_state!()?,
                        opts: PrintOptions::default()
                    }
                ))
            }

            /// Convert the polynomial into a LaTeX string.
            pub fn to_latex(&self) -> PyResult<String> {
                Ok(format!(
                    "$${}$$",
                    PolynomialPrinter::new_with_options(
                        &self.poly,
                        &&get_state!()?,
                        PrintOptions::latex(),
                    )
                ))
            }

            /// Print the polynomial in a debug representation.
            pub fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:?}", self.poly))
            }

            /// Get the list of variables in the internal ordering of the polynomial.
            pub fn get_var_list(&self) -> PyResult<Vec<PythonExpression>> {
                let mut var_list = vec![];

                let vars = self
                    .poly
                    .var_map
                    .as_ref()
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable map missing",
                    )))?;

                for x in vars {
                    match x {
                        Variable::Identifier(x) => {
                            var_list.push(PythonExpression {
                                expr: Arc::new(Atom::new_var(*x)),
                            });
                        }
                        _ => {
                            Err(exceptions::PyValueError::new_err(format!(
                                "Temporary variable in polynomial",
                            )))?;
                        }
                    }
                }

                Ok(var_list)
            }

            /// Add two polynomials `self and `rhs`, returning the result.
            pub fn __add__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new((*self.poly).clone() + (*rhs.poly).clone()),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self + new_rhs),
                    }
                }
            }

            /// Subtract polynomials `rhs` from `self`, returning the result.
            pub fn __sub__(&self, rhs: Self) -> Self {
                self.__add__(rhs.__neg__())
            }

            /// Multiply two polynomials `self and `rhs`, returning the result.
            pub fn __mul__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self * &new_rhs),
                    }
                }
            }

            /// Divide the polynomial `self` by `rhs` if possible, returning the result.
            pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
                let (q, r) = if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    self.poly.quot_rem(&rhs.poly, false)
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);

                    new_self.quot_rem(&new_rhs, false)
                };

                if r.is_zero() {
                    Ok(Self { poly: Arc::new(q) })
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "The division has a remainder: {}",
                        r
                    )))
                }
            }

            /// Divide `self` by `rhs`, returning the quotient and remainder.
            pub fn quot_rem(&self, rhs: Self) -> (Self, Self) {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    let (q, r) = self.poly.quot_rem(&rhs.poly, false);

                    (Self { poly: Arc::new(q) }, Self { poly: Arc::new(r) })
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);

                    let (q, r) = new_self.quot_rem(&new_rhs, false);

                    (Self { poly: Arc::new(q) }, Self { poly: Arc::new(r) })
                }
            }

            /// Negate the polynomial.
            pub fn __neg__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone().neg()),
                }
            }

            /// Compute the greatest common divisor (GCD) of two polynomials.
            pub fn gcd(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(MultivariatePolynomial::gcd(&self.poly, &rhs.poly)),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(MultivariatePolynomial::gcd(&new_self, &new_rhs)),
                    }
                }
            }
        }
    };
}

generate_methods!(PythonPolynomial);
generate_methods!(PythonIntegerPolynomial);

/// A Symbolica rational polynomial.
#[pyclass(name = "RationalPolynomial")]
#[derive(Clone)]
pub struct PythonRationalPolynomial {
    pub poly: Arc<RationalPolynomial<IntegerRing, u32>>,
}

#[pymethods]
impl PythonRationalPolynomial {
    /// Create a new rational polynomial from a numerator and denominator polynomial.
    #[new]
    pub fn __new__(num: &PythonPolynomial, den: &PythonPolynomial) -> Self {
        Self {
            poly: Arc::new(RationalPolynomial::from_num_den(
                (*num.poly).clone(),
                (*den.poly).clone(),
                IntegerRing::new(),
                true,
            )),
        }
    }
}

/// A Symbolica rational polynomial with variable powers limited to 255.
#[pyclass(name = "RationalPolynomialSmallExponent")]
#[derive(Clone)]
pub struct PythonRationalPolynomialSmallExponent {
    pub poly: Arc<RationalPolynomial<IntegerRing, u8>>,
}

// TODO: unify with polynomial methods
macro_rules! generate_rat_methods {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            /// Parse a rational polynomial from a string.
            /// The list of all the variables must be provided.
            ///
            /// If this requirements is too strict, use `Expression.to_polynomial()` instead.
            ///
            ///
            /// Examples
            /// --------
            /// >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If the input is not a valid Symbolica rational polynomial.
            #[classmethod]
            pub fn parse(_cls: &PyType, arg: &str, vars: Vec<&str>) -> PyResult<Self> {
                let mut var_map: SmallVec<[Variable; INLINED_EXPONENTS]> = SmallVec::new();
                let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
                    SmallVec::new();

                let mut state = get_state_mut!()?;
                for v in vars {
                    let id = state.get_or_insert_var(v);
                    var_map.push(id.into());
                    var_name_map.push(v.into());
                }

                let e = WORKSPACE.with(|workspace| {
                    Token::parse(arg)
                        .map_err(exceptions::PyValueError::new_err)?
                        .to_rational_polynomial(
                            workspace,
                            &mut state,
                            RationalField::new(),
                            IntegerRing::new(),
                            &var_map,
                            &var_name_map,
                        )
                        .map_err(exceptions::PyValueError::new_err)
                })?;

                Ok(Self { poly: Arc::new(e) })
            }

            /// Copy the rational polynomial.
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            /// Get the list of variables in the internal ordering of the polynomial.
            pub fn get_var_list(&self) -> PyResult<Vec<PythonExpression>> {
                let mut var_list = vec![];

                let vars = self.poly.numerator.var_map.as_ref().ok_or(
                    exceptions::PyValueError::new_err(format!("Variable map missing",)),
                )?;

                for x in vars {
                    match x {
                        Variable::Identifier(x) => {
                            var_list.push(PythonExpression {
                                expr: Arc::new(Atom::new_var(*x)),
                            });
                        }
                        _ => {
                            Err(exceptions::PyValueError::new_err(format!(
                                "Temporary variable in polynomial",
                            )))?;
                        }
                    }
                }

                Ok(var_list)
            }

            /// Print the rational polynomial in a human-readable format.
            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    RationalPolynomialPrinter {
                        poly: &self.poly,
                        state: &&get_state!()?,
                        opts: PrintOptions::default(),
                        add_parentheses: false,
                    }
                ))
            }

            /// Convert the rational polynomial into a LaTeX string.
            pub fn to_latex(&self) -> PyResult<String> {
                Ok(format!(
                    "$${}$$",
                    RationalPolynomialPrinter::new_with_options(
                        &self.poly,
                        &&get_state!()?,
                        PrintOptions::latex(),
                    )
                ))
            }

            /// Print the rational polynomial in a debug representation.
            pub fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:?}", self.poly))
            }

            /// Add two rational polynomials `self and `rhs`, returning the result.
            pub fn __add__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(&*self.poly + &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self + &new_rhs),
                    }
                }
            }

            /// Subtract rational polynomials `rhs` from `self`, returning the result.
            pub fn __sub__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(&*self.poly - &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self - &new_rhs),
                    }
                }
            }

            /// Multiply two rational polynomials `self and `rhs`, returning the result.
            pub fn __mul__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self * &new_rhs),
                    }
                }
            }

            /// Divide the rational polynomial `self` by `rhs` if possible, returning the result.
            pub fn __truediv__(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self / &new_rhs),
                    }
                }
            }

            /// Negate the rational polynomial.
            pub fn __neg__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone().neg()),
                }
            }

            /// Compute the greatest common divisor (GCD) of two rational polynomials.
            pub fn gcd(&self, rhs: Self) -> Self {
                if self.poly.get_var_map() == rhs.poly.get_var_map() {
                    Self {
                        poly: Arc::new(self.poly.gcd(&rhs.poly)),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_var_map(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self.gcd(&new_rhs)),
                    }
                }
            }
        }
    };
}

generate_rat_methods!(PythonRationalPolynomial);
generate_rat_methods!(PythonRationalPolynomialSmallExponent);

/// A sample from the Symbolica integrator. It could consist of discrete layers,
/// accessible with `d` (empty when there are not discrete layers), and the final continous layer `c` if it is present.
#[pyclass(name = "Sample")]
#[derive(Clone)]
pub struct PythonSample {
    #[pyo3(get)]
    /// The weights the integrator assigned to this sample point, given in descending order:
    /// first the discrete layer weights and then the continuous layer weight.
    weights: Vec<f64>,
    #[pyo3(get)]
    /// A sample point per (nested) discrete layer. Empty if not present.
    d: Vec<usize>,
    #[pyo3(get)]
    /// A sample in the continuous layer. Empty if not present.
    c: Vec<f64>,
}

impl PythonSample {
    fn into_sample(self) -> Sample<f64> {
        assert_eq!(
            self.weights.len(),
            self.d.len() + if self.c.is_empty() { 0 } else { 1 }
        );
        let mut weight_index = self.weights.len() - 1;

        let mut sample = if !self.c.is_empty() {
            let s = Some(Sample::Continuous(self.weights[weight_index], self.c));
            weight_index -= 1;
            s
        } else {
            None
        };

        for dd in self.d.iter().rev() {
            sample = Some(Sample::Discrete(
                self.weights[weight_index],
                *dd,
                sample.map(Box::new),
            ));
            weight_index -= 1;
        }

        sample.unwrap()
    }

    fn from_sample(mut sample: &Sample<f64>) -> PythonSample {
        let mut weights = vec![];
        let mut d = vec![];
        let mut c = vec![];

        loop {
            match sample {
                Sample::Continuous(w, cs) => {
                    weights.push(*w);
                    c.extend_from_slice(cs);
                    break;
                }
                Sample::Discrete(w, i, s) => {
                    weights.push(*w);
                    d.push(*i);
                    if let Some(ss) = s {
                        sample = ss;
                    } else {
                        break;
                    }
                }
            }
        }

        PythonSample { weights, d, c }
    }
}

#[pyclass(name = "NumericalIntegrator")]
#[derive(Clone)]
struct PythonNumericalIntegrator {
    grid: Grid<f64>,
}

#[pymethods]
impl PythonNumericalIntegrator {
    /// Create a new continuous grid for the numerical integrator.
    #[classmethod]
    #[pyo3(signature =
        (n_dims, n_bins = 128,
        min_samples_for_update = 100,
        bin_number_evolution = None,
        train_on_avg = false)
    )]
    pub fn continuous(
        _cls: &PyType,
        n_dims: usize,
        n_bins: usize,
        min_samples_for_update: usize,
        bin_number_evolution: Option<Vec<usize>>,
        train_on_avg: bool,
    ) -> PythonNumericalIntegrator {
        PythonNumericalIntegrator {
            grid: Grid::Continuous(ContinuousGrid::new(
                n_dims,
                n_bins,
                min_samples_for_update,
                bin_number_evolution,
                train_on_avg,
            )),
        }
    }

    /// Create a new discrete grid for the numerical integrator.
    /// Each bin can have a sub-grid.
    ///
    /// Examples
    /// --------
    /// >>> def integrand(samples: list[Sample]):
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         if sample.d[0] == 0:
    /// >>>             res.append(sample.c[0]**2)
    /// >>>         else:
    /// >>>             res.append(sample.c[0]**1/2)
    /// >>>     return res
    /// >>>
    /// >>> integrator = NumericalIntegrator.discrete(
    /// >>>     [NumericalIntegrator.continuous(1), NumericalIntegrator.continuous(1)])
    /// >>> integrator.integrate(integrand, True, 10, 10000)
    #[classmethod]
    #[pyo3(signature =
        (bins,
        max_prob_ratio = 100.,
        train_on_avg = false)
    )]
    pub fn discrete(
        _cls: &PyType,
        bins: Vec<Option<PythonNumericalIntegrator>>,
        max_prob_ratio: f64,
        train_on_avg: bool,
    ) -> PythonNumericalIntegrator {
        let bins = bins.into_iter().map(|b| b.map(|bb| bb.grid)).collect();

        PythonNumericalIntegrator {
            grid: Grid::Discrete(DiscreteGrid::new(bins, max_prob_ratio, train_on_avg)),
        }
    }

    /// Sample `num_samples` points from the grid.
    pub fn sample(&mut self, num_samples: usize) -> Vec<PythonSample> {
        let mut rng = rand::thread_rng();
        let mut sample = Sample::new();

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            self.grid.sample(&mut rng, &mut sample);
            samples.push(PythonSample::from_sample(&sample));
        }

        samples
    }

    /// Add the samples and their corresponding function evaluations to the grid.
    /// Call `update` after to update the grid and to obtain the new expected value for the integral.
    fn add_training_samples(
        &mut self,
        samples: Vec<PythonSample>,
        evals: Vec<f64>,
    ) -> PyResult<()> {
        if evals.len() != samples.len() {
            return PyResult::Err(pyo3::exceptions::PyAssertionError::new_err(
                "Number of returned values does not equal number of samples",
            ));
        }

        for (s, f) in samples.into_iter().zip(evals) {
            self.grid
                .add_training_sample(&s.into_sample(), f)
                .map_err(pyo3::exceptions::PyAssertionError::new_err)?;
        }

        Ok(())
    }

    /// Update the grid using the `learning_rate`.
    /// Examples
    /// --------
    /// >>> from symbolica import NumericalIntegrator, Sample
    /// >>>
    /// >>> def integrand(samples: list[Sample]):
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         res.append(sample.c[0]**2+sample.c[1]**2)
    /// >>>     return res
    /// >>>
    /// >>> integrator = NumericalIntegrator.continuous(2)
    /// >>> for i in range(10):
    /// >>>     samples = integrator.sample(10000 + i * 1000)
    /// >>>     res = integrand(samples)
    /// >>>     integrator.add_training_samples(samples, res)
    /// >>>     avg, err, chi_sq = integrator.update(1.5)
    /// >>>     print('Iteration {}: {:.6} +- {:.6}, chi={:.6}'.format(i+1, avg, err, chi_sq))
    fn update(&mut self, learing_rate: f64) -> PyResult<(f64, f64, f64)> {
        self.grid.update(learing_rate);

        let stats = self.grid.get_statistics();
        Ok((stats.avg, stats.err, stats.chi_sq))
    }

    /// Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
    /// The return value is the average, the statistical error, and chi-squared of the integral.
    ///
    /// With `show_stats=True`, intermediate statistics will be printed. `max_n_iter` determines the number
    /// of iterations and `n_samples_per_iter` determine the number of samples per iteration. This is
    /// the same amount of samples that the integrand function will be called with.
    ///
    /// For more flexibility, use `sample`, `add_training_samples` and `update`. See `update` for an example.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import NumericalIntegrator, Sample
    /// >>>
    /// >>> def integrand(samples: list[Sample]):
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         res.append(sample.c[0]**2+sample.c[1]**2)
    /// >>>     return res
    /// >>>
    /// >>> avg, err = NumericalIntegrator.continuous(2).integrate(integrand, True, 10, 100000)
    /// >>> print('Result: {} +- {}'.format(avg, err))
    #[pyo3(signature =
        (integrand,
        max_n_iter = 10_000_000,
        min_error = 0.01,
        n_samples_per_iter = 10_000,
        show_stats = true)
    )]
    pub fn integrate(
        &mut self,
        py: Python,
        integrand: PyObject,
        max_n_iter: usize,
        min_error: f64,
        n_samples_per_iter: usize,
        show_stats: bool,
    ) -> PyResult<(f64, f64, f64)> {
        let mut rng = rand::thread_rng();

        let mut samples = vec![Sample::new(); n_samples_per_iter];
        for iteration in 1..=max_n_iter {
            for sample in &mut samples {
                self.grid.sample(&mut rng, sample);
            }

            let p_samples: Vec<_> = samples.iter().map(PythonSample::from_sample).collect();

            let res = integrand
                .call(py, (p_samples,), None)?
                .extract::<Vec<f64>>(py)?;

            if res.len() != n_samples_per_iter {
                return Err(exceptions::PyValueError::new_err(
                    "Wrong number of arguments returned for integration function.",
                ));
            }

            for (s, r) in samples.iter().zip(res) {
                self.grid.add_training_sample(s, r).unwrap();
            }

            self.grid.update(1.5);

            let stats = self.grid.get_statistics();
            if show_stats {
                println!(
                    "Iteration {:2}: {}  {:.2} χ²",
                    iteration,
                    stats.format_uncertainty(),
                    stats.chi_sq / stats.cur_iter as f64
                );
            }

            if stats.avg != 0. && stats.err / stats.avg.abs() <= min_error {
                break;
            }
        }

        let stats = self.grid.get_statistics();
        Ok((stats.avg, stats.err, stats.chi_sq / stats.cur_iter as f64))
    }
}
