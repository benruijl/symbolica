use std::{
    borrow::{Borrow, BorrowMut},
    hash::{Hash, Hasher},
    ops::Neg,
    sync::{Arc, RwLock},
};

use ahash::{HashMap, HashMapExt};
use once_cell::sync::Lazy;
use pyo3::{
    exceptions, pyclass,
    pyclass::CompareOp,
    pymethods, pymodule,
    types::{PyModule, PyTuple, PyType},
    FromPyObject, IntoPy, PyObject, PyRef, PyResult, Python,
};
use self_cell::self_cell;
use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    id::{Match, MatchStack, Pattern, PatternAtomTreeIterator, PatternRestriction},
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, Sample},
    parser::Token,
    poly::{polynomial::MultivariatePolynomial, INLINED_EXPONENTS},
    printer::{AtomPrinter, PolynomialPrinter, PrintOptions, RationalPolynomialPrinter},
    representations::{
        default::ListIteratorD,
        number::{BorrowedNumber, Number},
        Add, Atom, AtomView, Fun, Identifier, Mul, Num, OwnedAdd, OwnedFun, OwnedMul, OwnedNum,
        OwnedPow, Var,
    },
    rings::integer::IntegerRing,
    rings::{
        rational::RationalField,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    state::{FunctionAttribute, ResettableBuffer, State, Workspace, INPUT_ID},
    streaming::TermStreamer,
    transformer::Transformer,
};

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
thread_local!(static WORKSPACE: Workspace = Workspace::new());

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

    Ok(())
}

#[derive(FromPyObject)]
pub enum ConvertibleToPattern {
    Literal(ConvertibleToExpression),
    Pattern(PythonPattern),
}

impl ConvertibleToPattern {
    pub fn to_pattern(self) -> PythonPattern {
        match self {
            Self::Literal(l) => PythonPattern {
                expr: Arc::new(Pattern::from_view(
                    l.to_expression().expr.as_view(),
                    &STATE.read().unwrap(),
                )),
            },
            Self::Pattern(e) => e,
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

    /// Expand products and powers.
    pub fn expand(&self) -> PyResult<PythonPattern> {
        Ok(PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new(Transformer::Expand(
                (*self.expr).clone(),
            )))),
        })
    }

    /// Create a transformer that derives `self` w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let id = match &*x.to_pattern().expr {
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
                (*lhs.to_pattern().expr).clone(),
                (*self.expr).clone(),
                (*rhs.to_pattern().expr).clone(),
                cond.map(|r| r.convert()).unwrap_or(HashMap::default()),
            )))),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        let res = WORKSPACE.with(|workspace| {
            self.expr.add(
                &rhs.to_pattern().expr,
                workspace,
                STATE.read().unwrap().borrow(),
            )
        });

        PythonPattern {
            expr: Arc::new(res),
        }
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        self.__add__(rhs)
    }

    ///  Subtract `other` from this transformer, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        self.__add__(ConvertibleToPattern::Pattern(rhs.to_pattern().__neg__()))
    }

    ///  Subtract this transformer from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        rhs.to_pattern()
            .__add__(ConvertibleToPattern::Pattern(self.__neg__()))
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        let res = WORKSPACE.with(|workspace| {
            self.expr.mul(
                &rhs.to_pattern().expr,
                workspace,
                STATE.read().unwrap().borrow(),
            )
        });

        PythonPattern {
            expr: Arc::new(res),
        }
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        self.__mul__(rhs)
    }

    /// Divide this transformer by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        let res = WORKSPACE.with(|workspace| {
            self.expr.div(
                &rhs.to_pattern().expr,
                workspace,
                STATE.read().unwrap().borrow(),
            )
        });

        PythonPattern {
            expr: Arc::new(res),
        }
    }

    /// Divide `other` by this transformer, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToPattern) -> PythonPattern {
        rhs.to_pattern()
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
            self.expr.pow(
                &rhs.to_pattern().expr,
                workspace,
                STATE.read().unwrap().borrow(),
            )
        });

        Ok(PythonPattern {
            expr: Arc::new(res),
        })
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        rhs: ConvertibleToPattern,
        number: Option<i64>,
    ) -> PyResult<PythonPattern> {
        rhs.to_pattern()
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
    pub fn __neg__(&self) -> PythonPattern {
        let res =
            WORKSPACE.with(|workspace| self.expr.neg(workspace, STATE.read().unwrap().borrow()));

        PythonPattern {
            expr: Arc::new(res),
        }
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

/// A subset of pattern restrictions that can be used in Python.
#[derive(Debug, Clone, Copy)]
pub enum SimplePatternRestriction {
    Length(Identifier, usize, Option<usize>), // min-max range
    IsVar(Identifier),
    IsNumber(Identifier),
    IsLiteralWildcard(Identifier), // matches x_ to x_ only
    NumberCmp(Identifier, CompareOp, i64),
}

/// A restriction on wildcards.
#[pyclass(name = "PatternRestriction")]
#[derive(Debug, Clone)]
pub struct PythonPatternRestriction {
    pub restrictions: Arc<Vec<SimplePatternRestriction>>,
}

impl PythonPatternRestriction {
    fn convert(&self) -> HashMap<Identifier, Vec<PatternRestriction>> {
        let mut restrictions = HashMap::with_capacity(self.restrictions.len());

        for r in &*self.restrictions {
            match *r {
                SimplePatternRestriction::IsVar(name) => {
                    restrictions
                        .entry(name)
                        .or_insert(vec![])
                        .push(PatternRestriction::IsVar);
                }
                SimplePatternRestriction::IsLiteralWildcard(name) => {
                    restrictions
                        .entry(name)
                        .or_insert(vec![])
                        .push(PatternRestriction::IsLiteralWildcard(name));
                }
                SimplePatternRestriction::IsNumber(name) => {
                    restrictions
                        .entry(name)
                        .or_insert(vec![])
                        .push(PatternRestriction::IsNumber);
                }
                SimplePatternRestriction::Length(name, min, max) => {
                    restrictions
                        .entry(name)
                        .or_insert(vec![])
                        .push(PatternRestriction::Length(min, max));
                }
                SimplePatternRestriction::NumberCmp(name, op, ref_num) => {
                    restrictions
                        .entry(name)
                        .or_insert(vec![])
                        .push(PatternRestriction::Filter(Box::new(
                            move |v: &Match| match v {
                                Match::Single(AtomView::Num(n)) => {
                                    let num = n.get_number_view();
                                    let ordering = num.cmp(&BorrowedNumber::Natural(ref_num, 1));
                                    match op {
                                        CompareOp::Lt => ordering.is_lt(),
                                        CompareOp::Le => ordering.is_le(),
                                        CompareOp::Eq => ordering.is_eq(),
                                        CompareOp::Ne => ordering.is_ne(),
                                        CompareOp::Gt => ordering.is_gt(),
                                        CompareOp::Ge => ordering.is_ge(),
                                    }
                                }
                                _ => false,
                            },
                        )));
                }
            }
        }

        restrictions
    }
}

#[pymethods]
impl PythonPatternRestriction {
    /// Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonPatternRestriction {
        PythonPatternRestriction {
            restrictions: Arc::new(
                self.restrictions
                    .iter()
                    .chain(other.restrictions.iter())
                    .cloned()
                    .collect(),
            ),
        }
    }
}

#[derive(FromPyObject)]
pub enum ConvertibleToExpression {
    Int(i64),
    Expression(PythonExpression),
}

impl ConvertibleToExpression {
    pub fn to_expression(self) -> PythonExpression {
        match self {
            ConvertibleToExpression::Int(i) => {
                let num = Atom::new_num(Number::Natural(i, 1));
                PythonExpression {
                    expr: Arc::new(num),
                }
            }
            ConvertibleToExpression::Expression(e) => e,
        }
    }
}

#[pymethods]
impl PythonExpression {
    /// Creates a Symbolica expression that is a single variable.
    ///
    /// Examples
    /// --------
    /// >>> var_x = Expression.var('x')
    /// >>> print(var_x)
    /// x
    ///
    #[classmethod]
    pub fn var(_cls: &PyType, name: &str) -> PyResult<PythonExpression> {
        let mut guard = STATE.write().unwrap();
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
        let mut guard = STATE.write().unwrap();
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

    /// Creates a new Symbolica function with a given name.
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
            Atom::parse(arg, STATE.write().unwrap().borrow_mut(), f)
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
            AtomPrinter::new(self.expr.as_view(), &STATE.read().unwrap())
        ))
    }

    /// Hash the expression.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a wildcard from a variable name by appending a _
    /// if none is present yet.
    #[getter]
    fn get_w(&self) -> PyResult<PythonExpression> {
        let mut guard = STATE.write().unwrap();
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
    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_add();

            a.extend(self.expr.as_view());
            a.extend(rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });

        PythonExpression { expr: Arc::new(b) }
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        self.__add__(ConvertibleToExpression::Expression(
            rhs.to_expression().__neg__(),
        ))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        rhs.to_expression()
            .__add__(ConvertibleToExpression::Expression(self.__neg__()))
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_mul();

            a.extend(self.expr.as_view());
            a.extend(rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });

        PythonExpression { expr: Arc::new(b) }
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        self.__mul__(rhs)
    }

    /// Divide this expression by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
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
                .normalize(workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });
        PythonExpression { expr: Arc::new(b) }
    }

    /// Divide `other` by this expression, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        rhs.to_expression()
            .__truediv__(ConvertibleToExpression::Expression(self.clone()))
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

        let b = WORKSPACE.with(|workspace| {
            let mut e = workspace.new_atom();
            let a = e.to_pow();
            a.set_from_base_and_exp(self.expr.as_view(), rhs.to_expression().expr.as_view());
            a.set_dirty(true);

            let mut b = Atom::new();
            e.get()
                .as_view()
                .normalize(workspace, STATE.read().unwrap().borrow(), &mut b);
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
            .__pow__(ConvertibleToExpression::Expression(self.clone()), number)
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
    pub fn __neg__(&self) -> PythonExpression {
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
                .normalize(workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });

        PythonExpression { expr: Arc::new(b) }
    }

    /// Return the number of terms in this expression.
    fn __len__(&self) -> usize {
        match self.expr.as_view() {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            _ => 1,
        }
    }

    /// Convert the input to a transformer, on which subsequent transformations can be applied.
    pub fn transform(&self) -> PythonPattern {
        PythonPattern {
            expr: Arc::new(self.expr.into_pattern(&STATE.read().unwrap())),
        }
    }

    /// Create a pattern restriction based on the length.
    pub fn len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(vec![SimplePatternRestriction::Length(
                        name, min_length, max_length,
                    )]),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that tests if a wildcard is a variable.
    pub fn is_var(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(vec![SimplePatternRestriction::IsVar(name)]),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that tests if a wildcard is a number.
    pub fn is_num(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(vec![SimplePatternRestriction::IsNumber(name)]),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that treats the wildcard as a literal variable,
    /// so that it only matches to itself.
    pub fn is_lit(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(vec![SimplePatternRestriction::IsLiteralWildcard(name)]),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction based on a comparison of a wildcard with a number.
    fn __richcmp__(
        &self,
        other: ConvertibleToExpression,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        match other {
            ConvertibleToExpression::Int(i) => {
                // when comparing an expression with an int, construct a pattern restriction
                // TODO: find another way to construct a pattern restriction
                match self.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_name();
                        if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }

                        Ok(PythonPatternRestriction {
                            restrictions: Arc::new(vec![SimplePatternRestriction::NumberCmp(
                                name, op, i,
                            )]),
                        }
                        .into_py(py))
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                }
            }
            ConvertibleToExpression::Expression(e) => match op {
                CompareOp::Eq => Ok((self.expr == e.expr).into_py(py)),
                CompareOp::Ne => Ok((self.expr != e.expr).into_py(py)),
                _ => Err(exceptions::PyTypeError::new_err(
                    "Inequalities between expression are not allowed",
                )),
            },
        }
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
    ///     The execution happen in parallel.
    ///
    ///     Examples
    ///     --------
    ///     >>> x, x_ = Expression.vars('x', 'x_')
    ///     >>> e = (1+x)**2
    ///     >>> r = e.map(Transformer().expand().replace_all(x, 6))
    ///     >>> print(r)
    pub fn map(&self, op: PythonPattern) -> PyResult<PythonExpression> {
        let t = match op.expr.as_ref() {
            Pattern::Transformer(t) => t,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Operation must of a transformer".to_string(),
                ));
            }
        };

        let mut stream = TermStreamer::new_from((*self.expr).clone());

        // map every term in the expression
        stream = stream.map(|workspace, x| {
            let mut out = Atom::new();
            let restrictions = HashMap::default();
            let mut match_stack = MatchStack::new(&restrictions);
            match_stack.insert(INPUT_ID, Match::Single(x.as_view()));

            t.execute(
                STATE.read().unwrap().borrow(),
                workspace,
                &match_stack,
                &mut out,
            );
            out
        });

        let b = WORKSPACE
            .with(|workspace| stream.to_expression(workspace, STATE.read().unwrap().borrow()));

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.

    /// Parameters
    /// ----------
    /// vars : List[Expression]
    ///         A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();
        for v in vars {
            match v.expr.as_view() {
                AtomView::Var(v) => var_map.push(v.get_name()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {:?}",
                        e
                    )))?;
                }
            }
        }

        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr.as_view().set_coefficient_ring(
                &var_map,
                STATE.read().unwrap().borrow(),
                workspace,
                &mut b,
            );
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Expand the expression.
    pub fn expand(&self) -> PythonExpression {
        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr
                .as_view()
                .expand(workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });

        PythonExpression { expr: Arc::new(b) }
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

        let b = WORKSPACE.with(|workspace| {
            let mut b = Atom::new();
            self.expr
                .as_view()
                .derivative(id, workspace, STATE.read().unwrap().borrow(), &mut b);
            b
        });

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    pub fn to_polynomial(&self, vars: Option<Vec<PythonExpression>>) -> PyResult<PythonPolynomial> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name()),
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
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name()),
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
                    &STATE.read().unwrap(),
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
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_name()),
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
                    &STATE.read().unwrap(),
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
    ) -> PythonMatchIterator {
        let restrictions = cond.map(|r| r.convert()).unwrap_or(HashMap::default());
        PythonMatchIterator::new(
            (
                lhs.to_pattern().expr,
                self.expr.clone(),
                restrictions,
                STATE.read().unwrap().clone(), // FIXME: state is cloned
            ),
            move |(lhs, target, res, state)| {
                PatternAtomTreeIterator::new(lhs, target.as_view(), state, res)
            },
        )
    }

    /// Replace all patterns matching the left-hand side `lhs` by the right-hand side `rhs`.
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
        lhs: ConvertibleToPattern,
        rhs: ConvertibleToPattern,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonExpression> {
        let pattern = &lhs.to_pattern().expr;
        let rhs = &rhs.to_pattern().expr;
        let restrictions = cond.map(|r| r.convert()).unwrap_or(HashMap::default());
        let mut out = Atom::new();

        WORKSPACE.with(|workspace| {
            pattern.replace_all(
                self.expr.as_view(),
                rhs,
                &STATE.read().unwrap(),
                workspace,
                &restrictions,
                &mut out,
            );
        });

        Ok(PythonExpression {
            expr: Arc::new(out),
        })
    }
}

/// A function class for python that constructs an `Expression` when called with arguments.
/// This allows to write:
/// ```python
/// f = Expression.fun("f")
/// e = f(1,2,3)
/// ```
#[pyclass(name = "Function")]
pub struct PythonFunction {
    id: Identifier,
}

#[pymethods]
impl PythonFunction {
    /// Create a new function from a `name`. Can be turned into a symmetric function
    /// using `is_symmetric=True`.
    #[new]
    pub fn __new__(name: &str, is_symmetric: Option<bool>) -> PyResult<Self> {
        // TODO: parse and check if this is a valid function name
        let id = if is_symmetric.unwrap_or(false) {
            STATE
                .write()
                .unwrap()
                .borrow_mut()
                .get_or_insert_fn(name, Some(vec![FunctionAttribute::Symmetric]))
        } else {
            STATE
                .write()
                .unwrap()
                .borrow_mut()
                .get_or_insert_fn(name, None)
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
                fn_args.push((*a.to_pattern().expr).clone());
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
                    .normalize(workspace, &STATE.read().unwrap(), &mut out);

                Ok(PythonExpression {
                    expr: Arc::new(out),
                }
                .into_py(py))
            })
        } else {
            let p = Pattern::Fn(
                self.id,
                STATE.read().unwrap().is_wildcard(self.id).unwrap(),
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
                    owned.from_view(&e);
                    owned
                }),
            })
        })
    }
}

type OwnedMatch = (
    Arc<Pattern>,
    Arc<Atom>,
    HashMap<Identifier, Vec<PatternRestriction>>,
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
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        {
            let mut state = STATE.write().unwrap();
            for v in vars {
                let id = state.get_or_insert_var(v);
                var_map.push(id);
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
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        {
            let mut state = STATE.write().unwrap();
            for v in vars {
                let id = state.get_or_insert_var(v);
                var_map.push(id);
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
                        state: &STATE.read().unwrap(),
                        opts: PrintOptions::default()
                    }
                ))
            }

            /// Print the polynomial in a debug representation.
            pub fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:?}", self.poly))
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
                let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();
                let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
                    SmallVec::new();

                let mut state = STATE.write().unwrap();
                for v in vars {
                    let id = state.get_or_insert_var(v);
                    var_map.push(id);
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

            /// Print the rational polynomial in a human-readable format.
            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    RationalPolynomialPrinter {
                        poly: &self.poly,
                        state: &STATE.read().unwrap(),
                        opts: PrintOptions::default()
                    }
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

            if stats.avg != 0. && stats.err / stats.avg <= min_error {
                break;
            }
        }

        let stats = self.grid.get_statistics();
        Ok((stats.avg, stats.err, stats.chi_sq / stats.cur_iter as f64))
    }
}
