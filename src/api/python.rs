use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    ops::Neg,
    sync::Arc,
};

use ahash::HashMap;
use pyo3::{
    exceptions::{self, PyIndexError},
    pyclass,
    pyclass::CompareOp,
    pyfunction, pymethods, pymodule,
    types::{PyBytes, PyComplex, PyLong, PyModule, PyTuple, PyType},
    wrap_pyfunction, FromPyObject, IntoPy, PyErr, PyObject, PyRef, PyResult, Python,
};
use rug::Complete;
use self_cell::self_cell;
use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    domains::{
        finite_field::{ToFiniteField, Zp},
        float::Complex,
        integer::{Integer, IntegerRing, Z},
        rational::{Rational, RationalField, Q},
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
        Ring,
    },
    evaluate::EvaluationFn,
    id::{
        AtomType, Condition, Match, MatchSettings, MatchStack, Pattern, PatternAtomTreeIterator,
        PatternRestriction, ReplaceIterator, WildcardAndRestriction,
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample},
    parser::Token,
    poly::{
        evaluate::{
            InstructionEvaluator, InstructionSetMode, InstructionSetModeCPPSettings,
            InstructionSetPrinter,
        },
        factor::Factorize,
        groebner::GroebnerBasis,
        polynomial::MultivariatePolynomial,
        GrevLexOrder, LexOrder, Variable, INLINED_EXPONENTS,
    },
    printer::{
        AtomPrinter, MatrixPrinter, PolynomialPrinter, PrintOptions, RationalPolynomialPrinter,
    },
    representations::{Atom, AtomView, ListIterator, Symbol},
    state::{FunctionAttribute, RecycledAtom, State, Workspace},
    streaming::TermStreamer,
    tensors::matrix::Matrix,
    transformer::{StatsOptions, Transformer, TransformerError},
    LicenseManager,
};

#[pymodule]
fn symbolica(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PythonExpression>()?;
    m.add_class::<PythonFunction>()?;
    m.add_class::<PythonPattern>()?;
    m.add_class::<PythonPolynomial>()?;
    m.add_class::<PythonIntegerPolynomial>()?;
    m.add_class::<PythonFiniteFieldPolynomial>()?;
    m.add_class::<PythonRationalPolynomial>()?;
    m.add_class::<PythonRationalPolynomialSmallExponent>()?;
    m.add_class::<PythonFiniteFieldRationalPolynomial>()?;
    m.add_class::<PythonMatrix>()?;
    m.add_class::<PythonNumericalIntegrator>()?;
    m.add_class::<PythonSample>()?;
    m.add_class::<PythonAtomType>()?;
    m.add_class::<PythonAtomTree>()?;
    m.add_class::<PythonInstructionEvaluator>()?;
    m.add_class::<PythonRandomNumberGenerator>()?;

    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(is_licensed, m)?)?;
    m.add_function(wrap_pyfunction!(set_license_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_hobbyist_license, m)?)?;
    m.add_function(wrap_pyfunction!(request_trial_license, m)?)?;
    m.add_function(wrap_pyfunction!(request_sublicense, m)?)?;
    m.add_function(wrap_pyfunction!(get_offline_license_key, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Get the current Symbolica version.
#[pyfunction]
fn get_version() -> String {
    LicenseManager::get_version().to_string()
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

/// Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
/// The key will be sent to the e-mail address `email`.
#[pyfunction]
fn request_sublicense(
    name: String,
    email: String,
    company: String,
    super_license: String,
) -> PyResult<()> {
    LicenseManager::request_sublicense(&name, &email, &company, &super_license)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Get a license key for offline use, generated from a licensed Symbolica session. The key will remain valid for 24 hours.
#[pyfunction]
fn get_offline_license_key() -> PyResult<String> {
    LicenseManager::get_offline_license_key().map_err(exceptions::PyValueError::new_err)
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

impl<'a> From<AtomView<'a>> for PyResult<PythonAtomTree> {
    fn from(atom: AtomView<'a>) -> Self {
        let tree = match atom {
            AtomView::Num(_) => PythonAtomTree {
                atom_type: PythonAtomType::Num,
                head: Some(format!("{}", AtomPrinter::new(atom))),
                tail: vec![],
            },
            AtomView::Var(v) => PythonAtomTree {
                atom_type: PythonAtomType::Var,
                head: Some(State::get_name(v.get_symbol()).to_string()),
                tail: vec![],
            },
            AtomView::Fun(f) => PythonAtomTree {
                atom_type: PythonAtomType::Fn,
                head: Some(State::get_name(f.get_symbol()).to_string()),
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
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(b)?,
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(e)?,
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
                expr: Arc::new(l.to_expression().expr.as_view().into_pattern()),
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

macro_rules! append_transformer {
    ($self:ident,$t:expr) => {
        if let Pattern::Transformer(b) = $self.expr.borrow() {
            let mut t = b.clone();
            t.1.push($t);
            Ok(PythonPattern {
                expr: Arc::new(Pattern::Transformer(t)),
            })
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Pattern must be a transformer",
            ));
        }
    };
}

#[pymethods]
impl PythonPattern {
    /// Create a new transformer for a term provided by `Expression.map`.
    #[new]
    pub fn new() -> PythonPattern {
        PythonPattern {
            expr: Arc::new(Pattern::Transformer(Box::new((None, vec![])))),
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
        return append_transformer!(self, Transformer::Expand);
    }

    /// Create a transformer that computes the product of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x__ = Expression.var('x__')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2,3).replace_all(f(x__), x__.transform().prod())
    /// >>> print(e)
    pub fn prod(&self) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::Product);
    }

    /// Create a transformer that computes the sum of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x__ = Expression.var('x__')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2,3).replace_all(f(x__), x__.transform().sum())
    /// >>> print(e)
    pub fn sum(&self) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::Sum);
    }

    /// Create a transformer that returns the number of arguments.
    /// If the argument is not a function, return 0.
    ///
    /// If `only_for_arg_fun` is `True`, only count the number of arguments
    /// in the `arg()` function and return 1 if the input is not `arg`.
    /// This is useful for obtaining the length of a range during pattern matching.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x__ = Expression.var('x__')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(2,3,4).replace_all(f(x__), x__.transform().nargs())
    /// >>> print(e)
    #[pyo3(signature = (only_for_arg_fun = false))]
    pub fn nargs(&self, only_for_arg_fun: bool) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::ArgCount(only_for_arg_fun));
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
        return append_transformer!(self, Transformer::Sort);
    }

    /// Create a transformer that removes elements from a list if they occur
    /// earlier in the list as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x__ = Expression.var('x__')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2,1,2).replace_all(f(x__), x__.transform().deduplicate())
    /// >>> print(e)
    ///
    /// Yields `f(1,2)`.
    pub fn deduplicate(&self) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::Deduplicate);
    }

    /// Create a transformer that extracts a rational polynomial from a coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Function
    /// >>> e = Function.COEFF((x^2+1)/y^2).transform().from_coeff()
    /// >>> print(e)
    pub fn from_coeff(&self) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::FromNumber);
    }

    /// Create a transformer that split a sum or product into a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x, x__ = Expression.vars('x', 'x__')
    /// >>> f = Expression.fun('f')
    /// >>> e = (x + 1).replace_all(x__, f(x__.transform().split()))
    /// >>> print(e)
    pub fn split(&self) -> PyResult<PythonPattern> {
        return append_transformer!(self, Transformer::Split);
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
                        x.get_symbol()
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

        return append_transformer!(self, Transformer::Partition(conv_bins, fill_last, repeat));
    }

    /// Create a transformer that generates all permutations of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x_, f_id = Expression.vars('x_', 'f')
    /// >>> f = Expression.fun('f')
    /// >>> e = f(1,2,1,2).replace_all(f(x_), x_.transform().permutations(f_id))
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
                    x.get_symbol()
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

        return append_transformer!(self, Transformer::Permutations(id));
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
        let transformer = Transformer::Map(Box::new(move |expr, out| {
            let expr = PythonExpression {
                expr: Arc::new(expr.to_owned()),
            };

            let res = Python::with_gil(|py| {
                f.call(py, (expr,), None)
                    .map_err(|e| {
                        TransformerError::ValueError(format!("Bad callback function: {}", e))
                    })?
                    .extract::<ConvertibleToExpression>(py)
                    .map_err(|e| {
                        TransformerError::ValueError(format!(
                            "Function does not return a pattern, but {}",
                            e,
                        ))
                    })
            });

            match res {
                Ok(res) => {
                    out.set_from_view(&res.to_expression().expr.as_view());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }));

        return append_transformer!(self, transformer);
    }

    /// Create a transformer that checks for a Python interrupt,
    /// such as ctrl-c and aborts the current transformer.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> f(10).transform().repeat(Transformer().replace_all(
    /// >>> f(x_), f(x_+1)).check_interrupt()).execute()
    pub fn check_interrupt(&self) -> PyResult<PythonPattern> {
        let transformer = Transformer::Map(Box::new(move |expr, out| {
            out.set_from_view(&expr);
            Python::with_gil(|py| py.check_signals()).map_err(|_| TransformerError::Interrupt)
        }));

        return append_transformer!(self, transformer);
    }

    /// Create a transformer that keeps executing the transformer chain until the input equals the output.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = Expression.parse("f(5)")
    /// >>> e = e.transform().repeat(
    /// >>>     Transformer().expand(),
    /// >>>     Transformer().replace_all(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ).execute()
    #[pyo3(signature = (*transformers))]
    pub fn repeat(&self, transformers: &PyTuple) -> PyResult<PythonPattern> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonPattern>()?;

            let Pattern::Transformer(t) = p.expr.borrow() else {
                return Err(exceptions::PyValueError::new_err(
                    "Argument must be a transformer",
                ));
            };

            if t.0.is_some() {
                return Err(exceptions::PyValueError::new_err(
                    "Transformers in a repeat must be unbound. Use Transformer() to create it.",
                ));
            }

            rep_chain.extend_from_slice(&t.1);
        }

        return append_transformer!(self, Transformer::Repeat(rep_chain));
    }

    /// Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
    /// where `A`, `B`, `C` are transformers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = Expression.parse("f(5)")
    /// >>> e = e.transform().repeat(
    /// >>>     Transformer().expand(),
    /// >>>     Transformer().replace_all(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ).execute()
    #[pyo3(signature = (*transformers))]
    pub fn chain(&self, transformers: &PyTuple) -> PyResult<PythonPattern> {
        if let Pattern::Transformer(b) = self.expr.borrow() {
            let mut ts = b.clone();

            for r in transformers {
                let p = r.extract::<PythonPattern>()?;

                let Pattern::Transformer(t) = p.expr.borrow() else {
                    return Err(exceptions::PyValueError::new_err(
                        "Argument must be a transformer",
                    ));
                };

                if t.0.is_some() {
                    return Err(exceptions::PyValueError::new_err(
                        "Transformers in a repeat must be unbound. Use Transformer() to create it.",
                    ));
                }

                ts.1.extend_from_slice(&t.1);
            }

            Ok(PythonPattern {
                expr: Arc::new(Pattern::Transformer(ts)),
            })
        } else {
            Err(exceptions::PyValueError::new_err(
                "Pattern must be a transformer",
            ))
        }
    }

    /// Execute the transformer.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x = Expression.var('x')
    /// >>> e = (x+1)**5
    /// >>> e = e.transform().expand().execute()
    /// >>> print(e)
    pub fn execute(&self) -> PyResult<PythonExpression> {
        let mut out = Atom::default();
        Workspace::get_local()
            .with(|workspace| {
                self.expr.substitute_wildcards(
                    workspace,
                    &mut out,
                    &MatchStack::new(&Condition::default(), &MatchSettings::default()),
                )
            })
            .map_err(|e| match e {
                TransformerError::Interrupt => {
                    exceptions::PyKeyboardInterrupt::new_err("Interrupted by user")
                }
                TransformerError::ValueError(v) => exceptions::PyValueError::new_err(v),
            })?;

        Ok(PythonExpression {
            expr: Arc::new(out),
        })
    }

    /// Create a transformer that derives `self` w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let id = match &*x.to_pattern()?.expr {
            Pattern::Literal(x) => {
                if let AtomView::Var(x) = x.as_view() {
                    x.get_symbol()
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

        return append_transformer!(self, Transformer::Derivative(id));
    }

    /// Create a transformer that Taylor expands in `x` around `expansion_point` to depth `depth`.
    pub fn taylor_series(
        &self,
        x: ConvertibleToExpression,
        expansion_point: ConvertibleToExpression,
        depth: u32,
    ) -> PyResult<PythonPattern> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        return append_transformer!(
            self,
            Transformer::TaylorSeries(id, (*expansion_point.to_expression().expr).clone(), depth,)
        );
    }

    /// Create a transformer that replaces all patterns matching the left-hand side `self` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`. The settings `non_greedy_wildcards` can be used to specify
    /// wildcards that try to match as little as possible.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
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
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
    ) -> PyResult<PythonPattern> {
        let mut settings = MatchSettings::default();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }

        return append_transformer!(
            self,
            Transformer::ReplaceAll(
                (*lhs.to_pattern()?.expr).clone(),
                (*rhs.to_pattern()?.expr).clone(),
                cond.map(|r| r.condition.as_ref().clone())
                    .unwrap_or_default(),
                settings,
            )
        );
    }

    /// Create a transformer that prints the expression.
    ///
    /// Examples
    /// --------
    /// >>> Expression.parse('f(10)').transform().print(terms_on_new_line = True).execute()
    #[pyo3(signature =
        (terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_finite_field = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            latex = false)
        )]
    pub fn print(
        &self,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_finite_field: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        latex: bool,
    ) -> PyResult<PythonPattern> {
        return append_transformer!(
            self,
            Transformer::Print(PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_finite_field,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                square_brackets_for_function,
                num_exp_as_superscript,
                latex
            },)
        );
    }

    /// Print statistics of a transformer, tagging it with `tag`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun('f')
    /// >>> e = Expression.parse("f(5)")
    /// >>> e = e.transform().stats('replace', Transformer().replace_all(f(x_), 1)).execute()
    ///
    /// yields
    /// ```log
    /// Stats for replace:
    ///     In  │ 1 │  10.00 B │
    ///     Out │ 1 │   3.00 B │ ⧗ 40.15µs
    /// ```
    #[pyo3(signature =
        (tag,
            transformer,
            color_medium_change_threshold = 10.,
            color_large_change_threshold = 100.)
        )]
    pub fn stats(
        &self,
        tag: String,
        transformer: PythonPattern,
        color_medium_change_threshold: Option<f64>,
        color_large_change_threshold: Option<f64>,
    ) -> PyResult<PythonPattern> {
        let Pattern::Transformer(t) = transformer.expr.borrow() else {
            return Err(exceptions::PyValueError::new_err(
                "Argument must be a transformer",
            ));
        };

        return append_transformer!(
            self,
            Transformer::Stats(
                StatsOptions {
                    tag,
                    color_medium_change_threshold,
                    color_large_change_threshold,
                },
                t.1.clone()
            )
        );
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonPattern> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.add(&rhs.to_pattern()?.expr, workspace))
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
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.mul(&rhs.to_pattern()?.expr, workspace))
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
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.div(&rhs.to_pattern()?.expr, workspace))
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

        let res = Workspace::get_local()
            .with(|workspace| Ok::<_, PyErr>(self.expr.pow(&rhs.to_pattern()?.expr, workspace)));

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
        let res =
            Workspace::get_local().with(|workspace| Ok::<Pattern, PyErr>(self.expr.neg(workspace)));

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
    pub condition: Arc<Condition<WildcardAndRestriction>>,
}

#[pymethods]
impl PythonPatternRestriction {
    /// Create a new pattern restriction that is the logical 'and' operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonPatternRestriction {
        PythonPatternRestriction {
            condition: Arc::new(self.condition.as_ref().clone() & other.condition.as_ref().clone()),
        }
    }

    /// Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).
    pub fn __or__(&self, other: Self) -> PythonPatternRestriction {
        PythonPatternRestriction {
            condition: Arc::new(self.condition.as_ref().clone() | other.condition.as_ref().clone()),
        }
    }

    /// Create a new pattern restriction that takes the logical 'not' of the current restriction.
    pub fn __invert__(&self) -> PythonPatternRestriction {
        PythonPatternRestriction {
            condition: Arc::new(!self.condition.as_ref().clone()),
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
            let i = Integer::from_large(rug::Integer::parse(&a).unwrap().complete());
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

impl<'a> FromPyObject<'a> for Symbol {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            match a.expr.as_view() {
                AtomView::Var(v) => Ok(v.get_symbol()),
                e => Err(exceptions::PyValueError::new_err(format!(
                    "Expected variable instead of {}",
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
        Ok(Variable::Symbol(Symbol::extract(ob)?))
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

macro_rules! req_cmp {
    ($self:ident,$num:ident,$cmp_any_atom:ident,$c:ident) => {{
        let num = $num.to_expression();

        if !$cmp_any_atom && !matches!(num.expr.as_view(), AtomView::Num(_)) {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare to number",
            ));
        };

        match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: Arc::new(
                        (
                            name,
                            PatternRestriction::Filter(Box::new(move |v: &Match| {
                                let k = num.expr.as_view();

                                if let Match::Single(m) = v {
                                    if !$cmp_any_atom {
                                        if let AtomView::Num(_) = m {
                                            return m.cmp(&k).$c();
                                        }
                                    } else {
                                        return m.cmp(&k).$c();
                                    }
                                }

                                false
                            })),
                        )
                            .into(),
                    ),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }};
}

macro_rules! req_wc_cmp {
    ($self:ident,$other:ident,$cmp_any_atom:ident,$c:ident) => {{
        let id = match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
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

        let other_id = match $other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
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

        Ok(PythonPatternRestriction {
            condition: Arc::new(
                (
                    id,
                    PatternRestriction::Cmp(
                        other_id,
                        Box::new(move |m1: &Match, m2: &Match| {
                            if let Match::Single(a1) = m1 {
                                if let Match::Single(a2) = m2 {
                                    if !$cmp_any_atom {
                                        if let AtomView::Num(_) = a1 {
                                            if let AtomView::Num(_) = a2 {
                                                return a1.cmp(a2).$c();
                                            }
                                        }
                                    } else {
                                        return a1.cmp(a2).$c();
                                    }
                                }
                            }
                            false
                        }),
                    ),
                )
                    .into(),
            ),
        })
    }};
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
        // TODO: check if the name meets the requirements
        let id = State::get_symbol(name);
        let var = Atom::new_var(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

    /// Create a Symbolica variable for every name in `*names`.
    #[pyo3(signature = (*args,))]
    #[classmethod]
    pub fn vars(_cls: &PyType, args: &PyTuple) -> PyResult<Vec<PythonExpression>> {
        let mut result = Vec::with_capacity(args.len());

        for a in args {
            // TODO: check if the name meets the requirements
            let name = a.extract::<&str>()?;
            let id = State::get_symbol(name);
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
    ///
    /// Define a symmetric function:
    /// >>> f = Expression.fun('f', is_symmetric=True)
    /// >>> e = f(2,1)
    /// >>> print(e)
    /// f(1,2)
    ///
    /// Define a linear and symmetric function:
    /// >>> p1, p2, p3, p4 = Expression.vars('p1', 'p2', 'p3', 'p4')
    /// >>> dot = Expression.fun('dot', is_symmetric=True, is_linear=True)
    /// >>> e = dot(p2+2*p3,p1+3*p2-p3)
    /// dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)
    #[classmethod]
    pub fn fun(
        _cls: &PyType,
        name: &str,
        is_symmetric: Option<bool>,
        is_antisymmetric: Option<bool>,
        is_linear: Option<bool>,
    ) -> PyResult<PythonFunction> {
        PythonFunction::__new__(name, is_symmetric, is_antisymmetric, is_linear)
    }

    /// Create a Symbolica function for every name in `*names`.
    #[pyo3(signature = (*args,))]
    #[classmethod]
    pub fn funs(_cls: &PyType, args: &PyTuple) -> PyResult<Vec<PythonFunction>> {
        let mut result = Vec::with_capacity(args.len());
        for a in args {
            let name = a.extract::<&str>()?;
            result.push(PythonFunction::__new__(name, None, None, None)?);
        }

        Ok(result)
    }

    /// Create a new Symbolica number from an int or a float.
    /// A floating point number is converted to its rational number equivalent,
    /// but it can also be truncated by specifying the maximal denominator value.
    ///
    /// Examples
    /// --------
    /// >>> e = Expression.num(1) / 2
    /// >>> print(e)
    /// 1/2
    ///
    /// >>> print(Expression.num(0.33))
    /// >>> print(Expression.num(0.33, 5))
    /// 5944751508129055/18014398509481984
    /// 1/3
    #[classmethod]
    pub fn num(
        _cls: &PyType,
        py: Python,
        num: PyObject,
        max_denom: Option<usize>,
    ) -> PyResult<PythonExpression> {
        if let Ok(num) = num.extract::<i64>(py) {
            Ok(PythonExpression {
                expr: Arc::new(Atom::new_num(num)),
            })
        } else if let Ok(num) = num.extract::<&PyLong>(py) {
            let a = format!("{}", num);
            PythonExpression::parse(_cls, &a)
        } else if let Ok(f) = num.extract::<f64>(py) {
            if !f.is_finite() {
                return Err(exceptions::PyValueError::new_err("Number must be finite"));
            }

            let mut r: Rational = f.into();
            if let Some(max_denom) = max_denom {
                r = r.truncate_denominator(&(max_denom as u64).into())
            }

            Ok(PythonExpression {
                expr: Arc::new(Atom::new_num(r)),
            })
        } else {
            Err(exceptions::PyValueError::new_err("Not a valid number"))
        }
    }

    /// Euler's number `e`.
    #[classattr]
    #[pyo3(name = "E")]
    pub fn e() -> PythonExpression {
        PythonExpression {
            expr: Arc::new(Atom::new_var(State::E)),
        }
    }

    /// The mathematical constant `π`.
    #[classattr]
    #[pyo3(name = "PI")]
    pub fn pi() -> PythonExpression {
        PythonExpression {
            expr: Arc::new(Atom::new_var(State::PI)),
        }
    }

    /// The mathematical constant `i`, where
    /// `i^2 = -1`.
    #[classattr]
    #[pyo3(name = "I")]
    pub fn i() -> PythonExpression {
        PythonExpression {
            expr: Arc::new(Atom::new_var(State::I)),
        }
    }

    /// Return all defined symbol names (function names and variables).
    #[classmethod]
    pub fn get_all_symbol_names(_cls: &PyType) -> PyResult<Vec<String>> {
        Ok(State::symbol_iter().map(|x| x.to_string()).collect())
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
        let e = Atom::parse(arg).map_err(exceptions::PyValueError::new_err)?;

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
        Ok(format!("{}", AtomPrinter::new(self.expr.as_view())))
    }

    /// Get the number of bytes that this expression takes up in memory.
    pub fn get_byte_size(&self) -> usize {
        self.expr.as_view().get_byte_size()
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
        color_builtin_symbols = true,
        print_finite_field = true,
        symmetric_representation_for_finite_field = false,
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
        color_builtin_symbols: bool,
        print_finite_field: bool,
        symmetric_representation_for_finite_field: bool,
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
                    color_builtin_symbols,
                    print_finite_field,
                    symmetric_representation_for_finite_field,
                    explicit_rational_polynomial,
                    number_thousands_separator,
                    multiplication_operator,
                    square_brackets_for_function,
                    num_exp_as_superscript,
                    latex
                },
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
            AtomPrinter::new_with_options(self.expr.as_view(), PrintOptions::latex(),)
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
            Atom::Var(v) => Ok(Some(State::get_name(v.get_symbol()).to_string())),
            Atom::Fun(f) => Ok(Some(State::get_name(f.get_symbol()).to_string())),
            _ => Ok(None),
        }
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok(PythonExpression {
            expr: Arc::new(self.expr.as_ref() + rhs.expr.as_ref()),
        })
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
        let rhs = rhs.to_expression();
        Ok(PythonExpression {
            expr: Arc::new(self.expr.as_ref() * rhs.expr.as_ref()),
        })
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }

    /// Divide this expression by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok(PythonExpression {
            expr: Arc::new(self.expr.as_ref() / rhs.expr.as_ref()),
        })
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

        let rhs = rhs.to_expression();
        Ok(PythonExpression {
            expr: Arc::new(self.expr.pow(&rhs.expr)),
        })
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
        Ok(PythonExpression {
            expr: Arc::new(-self.expr.as_ref()),
        })
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
            expr: Arc::new(Pattern::Transformer(Box::new((
                Some(self.expr.into_pattern()),
                vec![],
            )))),
        })
    }

    /// Get the `idx`th component of the expression.
    fn __getitem__(&self, idx: isize) -> PyResult<PythonExpression> {
        let slice = match self.expr.as_view() {
            AtomView::Add(a) => a.to_slice(),
            AtomView::Mul(m) => m.to_slice(),
            AtomView::Fun(f) => f.to_slice(),
            AtomView::Pow(p) => p.to_slice(),
            _ => Err(PyIndexError::new_err("Cannot access child of leaf node"))?,
        };

        if idx.unsigned_abs() < slice.len() {
            Ok(PythonExpression {
                expr: Arc::new(if idx < 0 {
                    slice.get(slice.len() - idx.abs() as usize).to_owned()
                } else {
                    slice.get(idx as usize).to_owned()
                }),
            })
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the atom only has {} children.",
                idx,
                slice.len(),
            )))
        }
    }

    /// Create a pattern restriction based on the wildcard length before downcasting.
    pub fn req_len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: Arc::new(
                        (name, PatternRestriction::Length(min_length, max_length)).into(),
                    ),
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
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: Arc::new(
                        (
                            name,
                            PatternRestriction::IsAtomType(match atom_type {
                                PythonAtomType::Num => AtomType::Num,
                                PythonAtomType::Var => AtomType::Var,
                                PythonAtomType::Add => AtomType::Add,
                                PythonAtomType::Mul => AtomType::Mul,
                                PythonAtomType::Pow => AtomType::Pow,
                                PythonAtomType::Fn => AtomType::Fun,
                            }),
                        )
                            .into(),
                    ),
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
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: Arc::new((name, PatternRestriction::IsLiteralWildcard(name)).into()),
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

    /// Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_), 1, x_.req_lt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_lt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_cmp!(self, other, cmp_any_atom, is_lt);
    }

    /// Create a pattern restriction that passes when the wildcard is greater than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_), 1, x_.req_gt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_gt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_cmp!(self, other, cmp_any_atom, is_gt);
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_), 1, x_.req_le(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_le(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_cmp!(self, other, cmp_any_atom, is_le);
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = Expression.var('x_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace_all(f(x_), 1, x_.req_ge(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_ge(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_cmp!(self, other, cmp_any_atom, is_ge);
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
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
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

        Ok(PythonPatternRestriction {
            condition: Arc::new(
                (
                    id,
                    PatternRestriction::Filter(Box::new(move |m| {
                        let data = PythonExpression {
                            expr: Arc::new({
                                let mut a = Atom::default();
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
                    })),
                )
                    .into(),
            ),
        })
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = Expression.var('x_', 'y_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_lt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_lt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_wc_cmp!(self, other, cmp_any_atom, is_lt);
    }

    /// Create a pattern restriction that passes when the wildcard is greater than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = Expression.var('x_', 'y_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_gt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_gt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_wc_cmp!(self, other, cmp_any_atom, is_gt);
    }

    /// Create a pattern restriction that passes when the wildcard is less than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = Expression.var('x_', 'y_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_le(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_le(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_wc_cmp!(self, other, cmp_any_atom, is_le);
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = Expression.var('x_', 'y_')
    /// >>> f = Expression.fun("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_ge(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_ge(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        return req_wc_cmp!(self, other, cmp_any_atom, is_ge);
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
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
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
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
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

        Ok(PythonPatternRestriction {
            condition: Arc::new(
                (
                    id,
                    PatternRestriction::Cmp(
                        other_id,
                        Box::new(move |m1, m2| {
                            let data1 = PythonExpression {
                                expr: Arc::new({
                                    let mut a = Atom::default();
                                    m1.to_atom(&mut a);
                                    a
                                }),
                            };

                            let data2 = PythonExpression {
                                expr: Arc::new({
                                    let mut a = Atom::default();
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
                    ),
                )
                    .into(),
            ),
        })
    }

    /// Create an iterator over all atoms in the expression.
    fn __iter__(&self) -> PyResult<PythonAtomIterator> {
        match self.expr.as_view() {
            AtomView::Add(_) | AtomView::Mul(_) | AtomView::Fun(_) | AtomView::Pow(_) => {}
            x => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Non-iterable type: {}",
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
            Pattern::Transformer(t) => {
                if t.0.is_some() {
                    return Err(exceptions::PyValueError::new_err(
                        "Transformer is bound to expression. Use Transformer() instead."
                            .to_string(),
                    ));
                }
                &t.1
            }
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
            let m = stream.map(|workspace, x| {
                let mut out = Atom::default();
                // TODO: capture and abort the parallel run
                Transformer::execute(x.as_view(), &t, workspace, &mut out).unwrap_or_else(|e| {
                    panic!("Transformer failed during parallel execution: {:?}", e)
                });
                out
            });
            Ok::<_, PyErr>(m)
        })?;

        let b = stream.to_expression();

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.

    /// Parameters
    /// ----------
    /// vars : List[Expression]
    ///         A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map = vec![];
        for v in vars {
            match v.expr.as_view() {
                AtomView::Var(v) => var_map.push(v.get_symbol().into()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {}",
                        e
                    )))?;
                }
            }
        }

        let b = self.expr.as_view().set_coefficient_ring(&Arc::new(var_map));

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Expand the expression.
    pub fn expand(&self) -> PyResult<PythonExpression> {
        let b = self.expr.as_view().expand();
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
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function name",
            ));
        };

        let b = self.expr.as_view().collect(
            id,
            if let Some(key_map) = key_map {
                Some(Box::new(move |key, out| {
                    Python::with_gil(|py| {
                        let key = PythonExpression {
                            expr: Arc::new(key.to_owned()),
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
                            expr: Arc::new(coeff.to_owned()),
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
        );

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Collect terms involving the literal occurrence of `x`.
    ///
    /// Examples
    /// --------
    ///
    /// from symbolica import Expression
    /// >>>
    /// >>> x, y = Expression.vars('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + y*x**2
    /// >>> print(e.coefficient(x**2))
    ///
    /// yields
    ///
    /// ```
    /// y + 1
    /// ```
    pub fn coefficient(&self, x: ConvertibleToExpression) -> PythonExpression {
        let r = self.expr.coefficient(x.to_expression().expr.as_view());
        PythonExpression { expr: Arc::new(r) }
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
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Coefficient list must be done wrt a variable or function name",
            ));
        };

        let (list, rest) = self.expr.coefficient_list(id);

        let mut py_list: Vec<_> = list
            .into_iter()
            .map(|e| {
                (
                    PythonExpression {
                        expr: Arc::new(e.0.to_owned()),
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
    }

    /// Derive the expression w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let b = self.expr.derivative(id);

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
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let b = self
            .expr
            .taylor_series(id, expansion_point.to_expression().expr.as_view(), depth);

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    /// Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    /// All non-polynomial elements will be converted to new independent variables.
    pub fn to_polynomial(&self, vars: Option<Vec<PythonExpression>>) -> PyResult<PythonPolynomial> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_symbol().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {}",
                            e
                        )))?;
                    }
                }
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        Ok(PythonPolynomial {
            poly: Arc::new(self.expr.to_polynomial(&Q, var_map)),
        })
    }

    /// Convert the expression to a rational polynomial, optionally, with the variable ordering specified in `vars`.
    /// The latter is useful if it is known in advance that more variables may be added in the future to the
    /// rational polynomial through composition with other rational polynomials.
    ///
    /// All non-rational polynomial parts will automatically be converted to new independent variables.
    ///
    /// Examples
    /// --------
    /// >>> a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
    /// >>> print(a)
    pub fn to_rational_polynomial(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomial> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_symbol().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {}",
                            e
                        )))?;
                    }
                }
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        Ok(PythonRationalPolynomial {
            poly: Arc::new(self.expr.to_rational_polynomial(&Q, &Z, var_map)),
        })
    }

    /// Similar to [PythonExpression::to_rational_polynomial()], but the power of each variable limited to 255.
    pub fn to_rational_polynomial_small_exponent(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomialSmallExponent> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                match v.expr.as_view() {
                    AtomView::Var(v) => var_map.push(v.get_symbol().into()),
                    e => {
                        Err(exceptions::PyValueError::new_err(format!(
                            "Expected variable instead of {}",
                            e
                        )))?;
                    }
                }
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        Ok(PythonRationalPolynomialSmallExponent {
            poly: Arc::new(self.expr.to_rational_polynomial(&Q, &Z, var_map)),
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
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
    ) -> PyResult<PythonMatchIterator> {
        let conditions = cond
            .map(|r| r.condition.clone())
            .unwrap_or(Arc::new(Condition::default()));
        let settings = Arc::new(MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            ..MatchSettings::default()
        });
        Ok(PythonMatchIterator::new(
            (
                lhs.to_pattern()?.expr,
                self.expr.clone(),
                conditions,
                settings,
            ),
            move |(lhs, target, res, settings)| {
                PatternAtomTreeIterator::new(lhs, target.as_view(), res, settings)
            },
        ))
    }

    /// Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
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
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
    ) -> PyResult<PythonReplaceIterator> {
        let conditions = cond
            .map(|r| r.condition.clone())
            .unwrap_or(Arc::new(Condition::default()));
        let settings = Arc::new(MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            ..MatchSettings::default()
        });

        Ok(PythonReplaceIterator::new(
            (
                lhs.to_pattern()?.expr,
                self.expr.clone(),
                rhs.to_pattern()?.expr,
                conditions,
                settings,
            ),
            move |(lhs, target, rhs, res, settings)| {
                ReplaceIterator::new(lhs, target.as_view(), rhs, res, settings)
            },
        ))
    }

    /// Replace all atoms matching the pattern `pattern` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    ///
    /// The entire operation can be repeated until there are no more matches using `repeat=True`.
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
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        repeat: Option<bool>,
    ) -> PyResult<PythonExpression> {
        let pattern = &pattern.to_pattern()?.expr;
        let rhs = &rhs.to_pattern()?.expr;

        let mut settings = MatchSettings::default();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }

        let mut expr_ref = self.expr.as_view();

        let mut out = RecycledAtom::new();
        let mut out2 = RecycledAtom::new();
        while pattern.replace_all_into(
            expr_ref,
            rhs,
            cond.as_ref().map(|r| r.condition.as_ref()),
            Some(&settings),
            &mut out,
        ) {
            if !repeat.unwrap_or(false) {
                break;
            }

            std::mem::swap(&mut out, &mut out2);
            expr_ref = out2.as_view();
        }

        Ok(PythonExpression {
            expr: Arc::new(out.into_inner()),
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
                AtomView::Var(v) => vars.push(v.get_symbol().into()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {}",
                        e
                    )))?;
                }
            }
        }

        let res = AtomView::solve_linear_system::<u16>(&system_b, &vars).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not solve system: {}", e))
        })?;

        Ok(res
            .into_iter()
            .map(|x| PythonExpression { expr: Arc::new(x) })
            .collect())
    }

    /// Evaluate the expression, using a map of all the constants and
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
        constants: HashMap<PythonExpression, f64>,
        functions: HashMap<Variable, PyObject>,
    ) -> PyResult<f64> {
        let mut cache = HashMap::default();

        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let Variable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {:?}",
                        k
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args, _, _, _| {
                        Python::with_gil(|py| {
                            v.call(py, (args.to_vec(),), None)
                                .expect("Bad callback function")
                                .extract::<f64>(py)
                                .expect("Function does not return a float")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        Ok(self
            .expr
            .as_view()
            .evaluate(&constants, &functions, &mut cache))
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
        constants: HashMap<PythonExpression, Complex<f64>>,
        functions: HashMap<Variable, PyObject>,
    ) -> PyResult<&'py PyComplex> {
        let mut cache = HashMap::default();

        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let Variable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {:?}",
                        k
                    )))?
                };

                Ok((
                    id,
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
                ))
            })
            .collect::<PyResult<_>>()?;

        let r = self
            .expr
            .as_view()
            .evaluate(&constants, &functions, &mut cache);
        Ok(PyComplex::from_doubles(py, r.re, r.im))
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
    id: Symbol,
}

#[pymethods]
impl PythonFunction {
    /// Create a new function from a `name`. Can be turned into a symmetric function
    /// using `is_symmetric=True` or into an antisymmetric function using `is_antisymmetric=True`.
    /// The function can be made multilinear using `is_linear=True`.
    ///
    /// Once attributes are defined on a function, they cannot be redefined later.
    #[new]
    pub fn __new__(
        name: &str,
        is_symmetric: Option<bool>,
        is_antisymmetric: Option<bool>,
        is_linear: Option<bool>,
    ) -> PyResult<Self> {
        if is_symmetric.is_none() && is_antisymmetric.is_none() && is_linear.is_none() {
            return Ok(PythonFunction {
                id: State::get_symbol(name),
            });
        }

        if is_symmetric == Some(true) && is_antisymmetric == Some(true) {
            Err(exceptions::PyValueError::new_err(
                "Function cannot be both symmetric and antisymmetric",
            ))?;
        }

        let mut opts = vec![];

        if let Some(true) = is_symmetric {
            opts.push(FunctionAttribute::Symmetric);
        }

        if let Some(true) = is_antisymmetric {
            opts.push(FunctionAttribute::Antisymmetric);
        }

        if let Some(true) = is_linear {
            opts.push(FunctionAttribute::Linear);
        }

        let id = State::get_symbol_with_attributes(name, opts)
            .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

        Ok(PythonFunction { id })
    }

    /// The built-in function that converts a rational polynomial to a coefficient.
    #[classattr]
    #[pyo3(name = "COEFF")]
    pub fn coeff() -> PythonFunction {
        PythonFunction { id: State::COEFF }
    }

    /// The built-in cosine function.
    #[classattr]
    #[pyo3(name = "COS")]
    pub fn cos() -> PythonFunction {
        PythonFunction { id: State::COS }
    }

    /// The built-in sine function.
    #[classattr]
    #[pyo3(name = "SIN")]
    pub fn sin() -> PythonFunction {
        PythonFunction { id: State::SIN }
    }

    /// The built-in exponential function.
    #[classattr]
    #[pyo3(name = "EXP")]
    pub fn exp() -> PythonFunction {
        PythonFunction { id: State::EXP }
    }

    /// The built-in logarithm function.
    #[classattr]
    #[pyo3(name = "LOG")]
    pub fn log() -> PythonFunction {
        PythonFunction { id: State::LOG }
    }

    /// Returns `True` iff this function is symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.id.is_symmetric()
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
        pub enum ExpressionOrTransformer {
            Expression(PythonExpression),
            Transformer(ConvertibleToPattern),
        }

        let mut fn_args = Vec::with_capacity(args.len());

        for arg in args {
            if let Ok(a) = arg.extract::<ConvertibleToExpression>() {
                fn_args.push(ExpressionOrTransformer::Expression(a.to_expression()));
            } else if let Ok(a) = arg.extract::<ConvertibleToPattern>() {
                fn_args.push(ExpressionOrTransformer::Transformer(a));
            } else {
                let msg = format!("Unknown type: {}", arg.get_type().name().unwrap());
                return Err(exceptions::PyTypeError::new_err(msg));
            }
        }

        if fn_args
            .iter()
            .all(|x| matches!(x, ExpressionOrTransformer::Expression(_)))
        {
            // simplify to literal expression
            Workspace::get_local().with(|workspace| {
                let mut fun_b = workspace.new_atom();
                let fun = fun_b.to_fun(self.id);

                for x in fn_args {
                    if let ExpressionOrTransformer::Expression(a) = x {
                        fun.add_arg(a.expr.as_view());
                    }
                }

                let mut out = Atom::default();
                fun_b.as_view().normalize(workspace, &mut out);

                Ok(PythonExpression {
                    expr: Arc::new(out),
                }
                .into_py(py))
            })
        } else {
            // convert all wildcards back from literals
            let mut transformer_args = Vec::with_capacity(args.len());
            for arg in fn_args {
                match arg {
                    ExpressionOrTransformer::Transformer(t) => {
                        transformer_args.push(t.to_pattern()?.expr.as_ref().clone());
                    }
                    ExpressionOrTransformer::Expression(a) => {
                        transformer_args.push(a.expr.as_view().into_pattern().clone());
                    }
                }
            }

            let p = Pattern::Fn(self.id, transformer_args);
            Ok(PythonPattern { expr: Arc::new(p) }.into_py(py))
        }
    }
}

self_cell!(
    #[pyclass]
    pub struct PythonAtomIterator {
        owner: Arc<Atom>,
        #[covariant]
        dependent: ListIterator,
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
                    let mut owned = Atom::default();
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
    Arc<Condition<WildcardAndRestriction>>,
    Arc<MatchSettings>,
);
type MatchIterator<'a> = PatternAtomTreeIterator<'a, 'a>;

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
                                    let mut a = Atom::default();
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
    Arc<Condition<WildcardAndRestriction>>,
    Arc<MatchSettings>,
);
type ReplaceIteratorOne<'a> = ReplaceIterator<'a, 'a>;

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
            let mut out = Atom::default();

            if i.next(&mut out).is_none() {
                Ok(None)
            } else {
                Ok::<_, PyErr>(Some(PythonExpression {
                    expr: Arc::new(out),
                }))
            }
        })
    }
}

#[pyclass(name = "Polynomial")]
#[derive(Clone)]
pub struct PythonPolynomial {
    pub poly: Arc<MultivariatePolynomial<RationalField, u16>>,
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
        let mut var_map = vec![];
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        for v in vars {
            let id = State::get_symbol(v);
            var_map.push(id.into());
            var_name_map.push(v.into());
        }

        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(&Q, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }

    /// Convert the polynomial to a polynomial with integer coefficients, if possible.
    pub fn to_integer_polynomial(&self) -> PyResult<PythonIntegerPolynomial> {
        let mut poly_int =
            MultivariatePolynomial::new(&Z, Some(self.poly.nterms()), self.poly.variables.clone());

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
                if *e > u8::MAX as u16 {
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

    /// Convert the coefficients of the polynomial to a finite field with prime `prime`.
    pub fn to_finite_field(&self, prime: u32) -> PythonFiniteFieldPolynomial {
        let f = Zp::new(prime);
        PythonFiniteFieldPolynomial {
            poly: Arc::new(self.poly.map_coeff(|c| c.to_finite_field(&f), f.clone())),
        }
    }

    /// Optimize the polynomial for evaluation using `iterations` number of iterations.
    /// The optimized output can be exported in a C++ format using `to_file`.
    ///
    /// Returns an evaluator for the polynomial.
    #[pyo3(signature = (iterations = 1000, to_file = None))]
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
                        name: "evaluate".to_string(),
                        instr: &o,
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

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    ///
    /// Examples
    /// --------
    /// >>> basis = Polynomial.groebner_basis(
    /// >>>     [Expression.parse("a b c d - 1").to_polynomial(),
    /// >>>      Expression.parse("a b c + a b d + a c d + b c d").to_polynomial(),
    /// >>>      Expression.parse("a b + b c + a d + c d").to_polynomial(),
    /// >>>      Expression.parse("a + b + c + d").to_polynomial()],
    /// >>>     grevlex=True,
    /// >>>     print_stats=True
    /// >>> )
    /// >>> for p in basis:
    /// >>>     print(p)
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &PyType,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: Arc::new(p.reorder::<LexOrder>()),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.as_ref().clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system
                .into_iter()
                .map(|p| Self { poly: Arc::new(p) })
                .collect()
        }
    }

    /// Convert the polynomial to an expression.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = Expression.var('x')
    /// >>> e = Expression.parse('x*y+2*x+x^2')
    /// >>> p = e.to_polynomial()
    /// >>> print(e - p.to_expression())
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(PythonExpression {
            expr: Arc::new(self.poly.to_expression()),
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

        inputs
            .iter()
            .map(|s| eval.evaluate_with_input(s)[0])
            .collect()
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
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        for v in vars {
            let id = State::get_symbol(v);
            var_map.push(id.into());
            var_name_map.push(v.into());
        }

        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(&Z, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }

    /// Convert the polynomial to an expression.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> e = Expression.parse('x*y+2*x+x^2')
    /// >>> p = e.to_polynomial()
    /// >>> print((e - p.to_expression()).expand())
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(PythonExpression {
            expr: Arc::new(self.poly.to_expression()),
        })
    }
}

/// A Symbolica polynomial over finite fields.
#[pyclass(name = "FiniteFieldPolynomial")]
#[derive(Clone)]
pub struct PythonFiniteFieldPolynomial {
    pub poly: Arc<MultivariatePolynomial<Zp, u16>>,
}

#[pymethods]
impl PythonFiniteFieldPolynomial {
    /// Parse a polynomial with integer coefficients from a string.
    /// The input must be written in an expanded format and a list of all
    /// the variables must be provided.
    ///
    /// If these requirements are too strict, use `Expression.to_polynomial()` or
    /// `RationalPolynomial.parse()` instead.
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'], 5)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica polynomial.
    #[classmethod]
    pub fn parse(_cls: &PyType, arg: &str, vars: Vec<&str>, prime: u32) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        for v in vars {
            let id = State::get_symbol(v);
            var_map.push(id.into());
            var_name_map.push(v.into());
        }

        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(&Zp::new(prime), &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &PyType,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: Arc::new(p.reorder::<LexOrder>()),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.as_ref().clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system
                .into_iter()
                .map(|p| Self { poly: Arc::new(p) })
                .collect()
        }
    }
}

macro_rules! generate_methods {
    ($type:ty, $exp_type:ty) => {
        #[pymethods]
        impl $type {
            /// Compare two polynomials.
            fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
                match op {
                    CompareOp::Eq => Ok(self.poly == other.poly),
                    CompareOp::Ne => Ok(self.poly != other.poly),
                    _ => {
                        if self.poly.is_constant() && other.poly.is_constant() {
                            return Ok(match op {
                                CompareOp::Eq => self.poly == other.poly,
                                CompareOp::Ge => self.poly.lcoeff() >= other.poly.lcoeff(),
                                CompareOp::Gt => self.poly.lcoeff() > other.poly.lcoeff(),
                                CompareOp::Le => self.poly.lcoeff() <= other.poly.lcoeff(),
                                CompareOp::Lt => self.poly.lcoeff() < other.poly.lcoeff(),
                                CompareOp::Ne => self.poly != other.poly,
                            });
                        }

                        Err(exceptions::PyTypeError::new_err(format!(
                            "Inequalities between polynomials that are not numbers are not allowed in {} {} {}",
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

            /// Copy the polynomial.
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            /// Convert the polynomial into a human-readable string, with tunable settings.
            ///
            /// Examples
            /// --------
            /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
            /// >>> print(p.pretty_str(symmetric_representation_for_finite_field=True))
            #[pyo3(signature =
                (terms_on_new_line = false,
                    color_top_level_sum = true,
                    color_builtin_symbols = true,
                    print_finite_field = true,
                    symmetric_representation_for_finite_field = false,
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
                    color_builtin_symbols: bool,
                    print_finite_field: bool,
                    symmetric_representation_for_finite_field: bool,
                    explicit_rational_polynomial: bool,
                    number_thousands_separator: Option<char>,
                    multiplication_operator: char,
                    square_brackets_for_function: bool,
                    num_exp_as_superscript: bool,
                    latex: bool,
                ) -> PyResult<String> {
                    Ok(format!(
                        "{}",
                        PolynomialPrinter::new_with_options(
                            &self.poly,
                            PrintOptions {
                                terms_on_new_line,
                                color_top_level_sum,
                                color_builtin_symbols,
                                print_finite_field,
                                symmetric_representation_for_finite_field,
                                explicit_rational_polynomial,
                                number_thousands_separator,
                                multiplication_operator,
                                square_brackets_for_function,
                                num_exp_as_superscript,
                                latex
                            },
                        )
                    ))
                }

            /// Print the polynomial in a human-readable format.
            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly,
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
                        PrintOptions::latex(),
                    )
                ))
            }

            /// Get the number of terms.
            pub fn nterms(&self) -> usize {
                self.poly.nterms()
            }

            /// Get the list of variables in the internal ordering of the polynomial.
            pub fn get_var_list(&self) -> PyResult<Vec<PythonExpression>> {
                let mut var_list = vec![];

                for x in self.poly.get_vars_ref() {
                    match x {
                        Variable::Symbol(x) => {
                            var_list.push(PythonExpression {
                                expr: Arc::new(Atom::new_var(*x)),
                            });
                        }
                        Variable::Temporary(_) => {
                            Err(exceptions::PyValueError::new_err(format!(
                                "Temporary variable in polynomial",
                            )))?;
                        }
                        Variable::Function(_, a) | Variable::Other(a) => {
                            var_list.push(PythonExpression {
                                expr: a.clone(),
                            });
                        }
                    }
                }

                Ok(var_list)
            }

            /// Add two polynomials `self and `rhs`, returning the result.
            pub fn __add__(&self, rhs: Self) -> Self {
                if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    Self {
                        poly: Arc::new((*self.poly).clone() + (*rhs.poly).clone()),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
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
                if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self * &new_rhs),
                    }
                }
            }

            /// Divide the polynomial `self` by `rhs` if possible, returning the result.
            pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
                let (q, r) = if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    self.poly.quot_rem(&rhs.poly, false)
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);

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
                if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    let (q, r) = self.poly.quot_rem(&rhs.poly, false);

                    (Self { poly: Arc::new(q) }, Self { poly: Arc::new(r) })
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);

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
                if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    Self {
                        poly: Arc::new(self.poly.gcd(&rhs.poly)),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self.gcd(&new_rhs)),
                    }
                }
            }

            /// Compute the resultant of two polynomials with respect to the variable `var`.
            pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
                let x = self.poly.get_vars_ref().iter().position(|v| match (v, var.expr.as_view()) {
                    (Variable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (Variable::Function(_, f) | Variable::Other(f), a) => f.as_view() == a,
                    _ => false,
                }).ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    var.__str__()?
                )))?;


                if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
                    let self_uni = self.poly.to_univariate(x);
                    let rhs_uni = rhs.poly.to_univariate(x);

                    Ok(Self {
                        poly: Arc::new(self_uni.resultant_prs(&rhs_uni)),
                    })
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);

                    let self_uni = new_self.to_univariate(x);
                    let rhs_uni = new_rhs.to_univariate(x);

                    Ok(Self {
                        poly: Arc::new(self_uni.resultant_prs(&rhs_uni)),
                    })
                }
            }

            /// Compute the square-free factorization of the polynomial.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> p = Expression.parse('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
            /// >>> print('Square-free factorization of {}:'.format(p))
            /// >>> for f, exp in p.factor_square_free():
            /// >>>     print('\t({})^{}'.format(f, exp))
            pub fn factor_square_free(&self) -> Vec<(Self, usize)> {
                self.poly
                    .square_free_factorization()
                    .into_iter()
                    .map(|(f, p)| (Self { poly: Arc::new(f) }, p))
                    .collect()
            }

            /// Factorize the polynomial.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> p = Expression.parse('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
            /// >>> print('Factorization of {}:'.format(p))
            /// >>> for f, exp in p.factor():
            /// >>>     print('\t({})^{}'.format(f, exp))
            pub fn factor(&self) -> Vec<(Self, usize)> {
                self.poly
                    .factor()
                    .into_iter()
                    .map(|(f, p)| (Self { poly: Arc::new(f) }, p))
                    .collect()
            }

            /// Take a derivative in `x`.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> x = Expression.var('x')
            /// >>> p = Expression.parse('x^2+2').to_polynomial()
            /// >>> print(p.derivative(x))
            pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
                let x = self.poly.get_vars_ref().iter().position(|v| match (v, x.expr.as_view()) {
                    (Variable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (Variable::Function(_, f) | Variable::Other(f), a) => f.as_view() == a,
                    _ => false,
                }).ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    x.__str__()?
                )))?;

                Ok(Self { poly: Arc::new(self.poly.derivative(x))})
            }

            /// Get the content, i.e., the GCD of the coefficients.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
            /// >>> print(p.content())
            pub fn content(&self) -> PyResult<Self> {
                Ok(Self { poly: Arc::new(self.poly.constant(self.poly.content()))})
            }

            /// Get the coefficient list in `x`.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> x = Expression.var('x')
            /// >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
            /// >>> for n, pp in p.coefficient_list(x):
            /// >>>     print(n, pp)
            pub fn coefficient_list(&self, var: PythonExpression) -> PyResult<Vec<(usize, Self)>> {
                let x = self.poly.get_vars_ref().iter().position(|v| match (v, var.expr.as_view()) {
                    (Variable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (Variable::Function(_, f) | Variable::Other(f), a) => f.as_view() == a,
                    _ => false,
                }).ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    var.__str__()?
                )))?;

                Ok(self.poly.to_univariate_polynomial_list(x).into_iter()
                    .map(|(f, p)| (p as usize, Self { poly: Arc::new(f) })).collect())
            }

            /// Replace the variable `x` with a polynomial `v`.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> x = Expression.var('x')
            /// >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
            /// >>> r = Expression.parse('y+1').to_polynomial())
            /// >>> p.replace(x, r)
            pub fn replace(&self, x: PythonExpression, v: Self) -> PyResult<Self> {
                let id = match x.expr.as_view() {
                    AtomView::Var(x) => {
                        x.get_symbol()
                    }
                    _ => {
                        return Err(exceptions::PyValueError::new_err(
                            "Derivative must be taken wrt a variable",
                        ))
                    }
                };

                let x = self.poly.get_vars_ref().iter().position(|x| match x {
                    Variable::Symbol(y) => *y == id,
                    _ => false,
                }).ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    x.__str__()?
                )))?;

                if self.poly.get_vars_ref() == v.poly.get_vars_ref() {
                    Ok(Self {
                        poly: Arc::new(self.poly.replace_with_poly(x, &v.poly))
                    })
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*v.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Ok(Self {
                        poly: Arc::new(new_self.replace_with_poly(x, &new_rhs))
                    })
                }
            }
        }
    };
}

generate_methods!(PythonPolynomial, u16);
generate_methods!(PythonIntegerPolynomial, u8);
generate_methods!(PythonFiniteFieldPolynomial, u16);

/// A Symbolica rational polynomial.
#[pyclass(name = "RationalPolynomial")]
#[derive(Clone)]
pub struct PythonRationalPolynomial {
    pub poly: Arc<RationalPolynomial<IntegerRing, u16>>,
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
                &Z,
                true,
            )),
        }
    }

    /// Convert the coefficients to finite fields with prime `prime`.
    pub fn to_finite_field(&self, prime: u32) -> PythonFiniteFieldRationalPolynomial {
        PythonFiniteFieldRationalPolynomial {
            poly: Arc::new(self.poly.to_finite_field(&Zp::new(prime))),
        }
    }

    /// Get the numerator.
    pub fn numerator(&self) -> PythonPolynomial {
        PythonPolynomial {
            poly: Arc::new((&self.poly.numerator).into()),
        }
    }

    /// Get the denominator.
    pub fn denominator(&self) -> PythonPolynomial {
        PythonPolynomial {
            poly: Arc::new((&self.poly.denominator).into()),
        }
    }
}

/// A Symbolica rational polynomial with variable powers limited to 255.
#[pyclass(name = "RationalPolynomialSmallExponent")]
#[derive(Clone)]
pub struct PythonRationalPolynomialSmallExponent {
    pub poly: Arc<RationalPolynomial<IntegerRing, u8>>,
}

macro_rules! generate_rat_parse {
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
                let mut var_map = vec![];
                let mut var_name_map = vec![];

                for v in vars {
                    let id = State::get_symbol(v);
                    var_map.push(id.into());
                    var_name_map.push(v.into());
                }

                let e = Token::parse(arg)
                    .map_err(exceptions::PyValueError::new_err)?
                    .to_rational_polynomial(&Q, &Z, &Arc::new(var_map), &var_name_map)
                    .map_err(exceptions::PyValueError::new_err)?;

                Ok(Self { poly: Arc::new(e) })
            }

            /// Convert the rational polynomial to an expression.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> e = Expression.parse('(x*y+2*x+x^2)/(1+y^2+x^7)')
            /// >>> p = e.to_rational_polynomial()
            /// >>> print((e - p.to_expression()).expand())
            pub fn to_expression(&self) -> PyResult<PythonExpression> {
                Ok(PythonExpression {
                    expr: Arc::new(self.poly.to_expression()),
                })
            }
        }
    };
}

generate_rat_parse!(PythonRationalPolynomial);
generate_rat_parse!(PythonRationalPolynomialSmallExponent);

/// A Symbolica rational polynomial over finite fields.
#[pyclass(name = "FiniteFieldRationalPolynomial")]
#[derive(Clone)]
pub struct PythonFiniteFieldRationalPolynomial {
    pub poly: Arc<RationalPolynomial<Zp, u16>>,
}

#[pymethods]
impl PythonFiniteFieldRationalPolynomial {
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
    pub fn parse(_cls: &PyType, arg: &str, vars: Vec<&str>, prime: u32) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        for v in vars {
            let id = State::get_symbol(v);
            var_map.push(id.into());
            var_name_map.push(v.into());
        }

        let field = Zp::new(prime);
        let e = Token::parse(arg)
            .map_err(exceptions::PyValueError::new_err)?
            .to_rational_polynomial(&field, &field, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: Arc::new(e) })
    }
}

// TODO: unify with polynomial methods
macro_rules! generate_rat_methods {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            /// Copy the rational polynomial.
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            /// Compare two polynomials.
            fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
                match op {
                    CompareOp::Eq => Ok(self.poly == other.poly),
                    CompareOp::Ne => Ok(self.poly != other.poly),
                    _ => {
                        Err(exceptions::PyTypeError::new_err(format!(
                            "Inequalities between polynomials that are not numbers are not allowed in {} {} {}",
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

            /// Get the list of variables in the internal ordering of the polynomial.
            pub fn get_var_list(&self) -> PyResult<Vec<PythonExpression>> {
                let mut var_list = vec![];

                for x in self.poly.get_variables().iter() {
                    match x {
                        Variable::Symbol(x) => {
                            var_list.push(PythonExpression {
                                expr: Arc::new(Atom::new_var(*x)),
                            });
                        }
                        Variable::Temporary(_) => {
                            Err(exceptions::PyValueError::new_err(format!(
                                "Temporary variable in polynomial",
                            )))?;
                        }
                        Variable::Function(_, a) | Variable::Other(a) => {
                            var_list.push(PythonExpression {
                                expr: a.clone(),
                            });
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
                        PrintOptions::latex(),
                    )
                ))
            }

            /// Add two rational polynomials `self and `rhs`, returning the result.
            pub fn __add__(&self, rhs: Self) -> Self {
                if self.poly.get_variables() == rhs.poly.get_variables() {
                    Self {
                        poly: Arc::new(&*self.poly + &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self + &new_rhs),
                    }
                }
            }

            /// Subtract rational polynomials `rhs` from `self`, returning the result.
            pub fn __sub__(&self, rhs: Self) -> Self {
                if self.poly.get_variables() == rhs.poly.get_variables() {
                    Self {
                        poly: Arc::new(&*self.poly - &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self - &new_rhs),
                    }
                }
            }

            /// Multiply two rational polynomials `self and `rhs`, returning the result.
            pub fn __mul__(&self, rhs: Self) -> Self {
                if self.poly.get_variables() == rhs.poly.get_variables() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(&new_self * &new_rhs),
                    }
                }
            }

            /// Divide the rational polynomial `self` by `rhs` if possible, returning the result.
            pub fn __truediv__(&self, rhs: Self) -> Self {
                if self.poly.get_variables() == rhs.poly.get_variables() {
                    Self {
                        poly: Arc::new(&*self.poly * &*rhs.poly),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
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
                if self.poly.get_variables() == rhs.poly.get_variables() {
                    Self {
                        poly: Arc::new(self.poly.gcd(&rhs.poly)),
                    }
                } else {
                    let mut new_self = (*self.poly).clone();
                    let mut new_rhs = (*rhs.poly).clone();
                    new_self.unify_variables(&mut new_rhs);
                    Self {
                        poly: Arc::new(new_self.gcd(&new_rhs)),
                    }
                }
            }

            /// Compute the partial fraction decomposition in `x`.
            ///
            /// Examples
            /// --------
            ///
            /// >>> from symbolica import Expression
            /// >>> x = Expression.var('x')
            /// >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
            /// >>> for pp in p.apart(x):
            /// >>>     print(pp)
            pub fn apart(&self, x: PythonExpression) -> PyResult<Vec<Self>> {
                let id = match x.expr.as_view() {
                    AtomView::Var(x) => {
                        x.get_symbol()
                    }
                    _ => {
                        return Err(exceptions::PyValueError::new_err(
                            "Invalid variable specified.",
                        ))
                    }
                };

                let x = self.poly.get_variables().iter().position(|x| match x {
                    Variable::Symbol(y) => *y == id,
                    _ => false,
                }).ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    x.__str__()?
                )))?;

                Ok(self.poly.apart(x).into_iter()
                    .map(|f| Self { poly: Arc::new(f) }).collect())
            }
        }
    };
}

generate_rat_methods!(PythonRationalPolynomial);
generate_rat_methods!(PythonRationalPolynomialSmallExponent);
generate_rat_methods!(PythonFiniteFieldRationalPolynomial);

#[derive(FromPyObject)]
pub enum ConvertibleToRationalPolynomial {
    Literal(PythonRationalPolynomial),
    Expression(ConvertibleToExpression),
}

impl ConvertibleToRationalPolynomial {
    pub fn to_rational_polynomial(self) -> PyResult<PythonRationalPolynomial> {
        match self {
            Self::Literal(l) => Ok(l),
            Self::Expression(e) => {
                let expr = &e.to_expression().expr;

                let poly = expr.to_rational_polynomial(&Q, &Z, None);

                Ok(PythonRationalPolynomial {
                    poly: Arc::new(poly),
                })
            }
        }
    }
}

#[derive(FromPyObject)]
pub enum ScalarOrMatrix {
    Scalar(ConvertibleToRationalPolynomial),
    Matrix(PythonMatrix),
}

/// A Symbolica matrix with rational polynomial coefficients.
#[pyclass(name = "Matrix")]
#[derive(Clone)]
pub struct PythonMatrix {
    pub matrix: Arc<Matrix<RationalPolynomialField<IntegerRing, u16>>>,
}

impl PythonMatrix {
    fn unify(&self, rhs: &PythonMatrix) -> (PythonMatrix, PythonMatrix) {
        if self.matrix.field == rhs.matrix.field {
            return (self.clone(), rhs.clone());
        }

        let mut new_self = self.matrix.as_ref().clone();
        let mut new_rhs = rhs.matrix.as_ref().clone();

        let mut zero = self.matrix.field.zero();

        zero.unify_variables(&mut new_rhs[(0, 0)]);
        new_self.field = RationalPolynomialField::new(Z, zero.numerator.get_vars());
        new_rhs.field = new_self.field.clone();

        // now update every element
        for e in &mut new_self.data {
            zero.unify_variables(e);
        }
        for e in &mut new_rhs.data {
            zero.unify_variables(e);
        }

        (
            PythonMatrix {
                matrix: Arc::new(new_self),
            },
            PythonMatrix {
                matrix: Arc::new(new_rhs),
            },
        )
    }

    fn unify_scalar(
        &self,
        rhs: &PythonRationalPolynomial,
    ) -> (PythonMatrix, PythonRationalPolynomial) {
        if self.matrix.field == RationalPolynomialField::new(Z, rhs.poly.numerator.get_vars()) {
            return (self.clone(), rhs.clone());
        }

        let mut new_self = self.matrix.as_ref().clone();
        let mut new_rhs = rhs.poly.as_ref().clone();

        let mut zero = self.matrix.field.zero();

        zero.unify_variables(&mut new_rhs);
        new_self.field = RationalPolynomialField::new(Z, zero.numerator.get_vars());

        // now update every element
        for e in &mut new_self.data {
            zero.unify_variables(e);
        }

        (
            PythonMatrix {
                matrix: Arc::new(new_self),
            },
            PythonRationalPolynomial {
                poly: Arc::new(new_rhs),
            },
        )
    }
}

#[pymethods]
impl PythonMatrix {
    /// Create a new zeroed matrix with `nrows` rows and `ncols` columns.
    #[new]
    pub fn new(nrows: u32, ncols: u32) -> PyResult<PythonMatrix> {
        if nrows == 0 || ncols == 0 {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        Ok(PythonMatrix {
            matrix: Arc::new(Matrix::new(
                nrows,
                ncols,
                RationalPolynomialField::new(Z, Arc::new(vec![])),
            )),
        })
    }

    /// Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.
    #[classmethod]
    pub fn identity(_cls: &PyType, nrows: u32) -> PyResult<PythonMatrix> {
        if nrows == 0 {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        Ok(PythonMatrix {
            matrix: Arc::new(Matrix::identity(
                nrows,
                RationalPolynomialField::new(Z, Arc::new(vec![])),
            )),
        })
    }

    /// Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere.
    #[classmethod]
    pub fn eye(
        _cls: &PyType,
        diag: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if diag.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The diagonal must have at least one entry",
            ));
        }

        let mut diag: Vec<_> = diag
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.as_ref().clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = diag.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z, first.numerator.get_vars());

        Ok(PythonMatrix {
            matrix: Arc::new(Matrix::eye(&diag, field)),
        })
    }

    /// Create a new row vector from a list of scalars.
    #[classmethod]
    pub fn vec(
        _cls: &PyType,
        entries: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let mut entries: Vec<_> = entries
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.as_ref().clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = entries.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z, first.numerator.get_vars());

        Ok(PythonMatrix {
            matrix: Arc::new(Matrix::new_vec(entries, field)),
        })
    }

    /// Create a new row vector from a list of scalars.
    #[classmethod]
    pub fn from_linear(
        _cls: &PyType,
        nrows: u32,
        ncols: u32,
        entries: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let mut entries: Vec<_> = entries
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.as_ref().clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = entries.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z, first.numerator.get_vars());

        Ok(PythonMatrix {
            matrix: Arc::new(
                Matrix::from_linear(entries, nrows, ncols, field).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid matrix: {}", e))
                })?,
            ),
        })
    }

    /// Create a new matrix from a 2-dimensional vector of scalars.
    #[classmethod]
    pub fn from_nested(
        cls: &PyType,
        entries: Vec<Vec<ConvertibleToRationalPolynomial>>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() || entries.iter().any(|x| x.is_empty()) {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let nrows = entries.len() as u32;
        let ncols = entries[0].len() as u32;

        if entries.iter().any(|x| x.len() != ncols as usize) {
            return Err(exceptions::PyValueError::new_err(
                "The matrix is not rectangular",
            ));
        }

        let entries: Vec<_> = entries.into_iter().flatten().collect();

        Self::from_linear(cls, nrows, ncols, entries)
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.matrix.nrows()
    }

    /// Return true iff every entry in the matrix is zero.
    pub fn is_zero(&self) -> bool {
        self.matrix.is_zero()
    }

    /// Return true iff every non- main diagonal entry in the matrix is zero.
    pub fn is_diagonal(&self) -> bool {
        self.matrix.is_diagonal()
    }

    /// Return the transpose of the matrix.
    pub fn transpose(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: Arc::new(self.matrix.transpose()),
        }
    }

    /// Return the inverse of the matrix, if it exists.
    pub fn inv(&self) -> PyResult<PythonMatrix> {
        Ok(PythonMatrix {
            matrix: Arc::new(
                self.matrix
                    .inv()
                    .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
            ),
        })
    }

    /// Return the determinant of the matrix.
    pub fn det(&self) -> PyResult<PythonRationalPolynomial> {
        Ok(PythonRationalPolynomial {
            poly: Arc::new(
                self.matrix
                    .det()
                    .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
            ),
        })
    }

    /// Solve `A * x = b` for `x`, where `A` is the current matrix.
    pub fn solve(&self, b: PythonMatrix) -> PyResult<PythonMatrix> {
        let (new_self, new_rhs) = self.unify(&b);
        Ok(PythonMatrix {
            matrix: Arc::new(
                new_self
                    .matrix
                    .solve(&new_rhs.matrix)
                    .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
            ),
        })
    }

    /// Get the content of the matrix, i.e. the gcd of all entries.
    pub fn content(&self) -> PythonRationalPolynomial {
        PythonRationalPolynomial {
            poly: Arc::new(self.matrix.content()),
        }
    }

    /// Construct the same matrix, but with the content removed.
    pub fn primitive_part(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: Arc::new(self.matrix.primitive_part()),
        }
    }

    /// Apply a function `f` to every entry of the matrix.
    pub fn map(&self, f: PyObject) -> PyResult<PythonMatrix> {
        let data = self
            .matrix
            .data
            .iter()
            .map(|x| {
                let expr = PythonRationalPolynomial {
                    poly: Arc::new(x.clone()),
                };

                Python::with_gil(|py| {
                    Ok(f.call1(py, (expr,))
                        .map_err(|e| e)?
                        .extract::<ConvertibleToRationalPolynomial>(py)?
                        .to_rational_polynomial()?
                        .poly
                        .as_ref()
                        .clone())
                })
            })
            .collect::<PyResult<_>>()?;

        Ok(PythonMatrix {
            matrix: Arc::new(
                Matrix::from_linear(
                    data,
                    self.matrix.nrows,
                    self.matrix.ncols,
                    self.matrix.field.clone(),
                )
                .unwrap(),
            ),
        })
    }

    fn __getitem__(&self, mut idx: (isize, isize)) -> PyResult<PythonRationalPolynomial> {
        if idx.0 < 0 {
            idx.0 += self.matrix.nrows() as isize;
        }
        if idx.1 < 0 {
            idx.1 += self.matrix.ncols() as isize;
        }

        if idx.0 as usize >= self.matrix.nrows() || idx.1 as usize >= self.matrix.ncols() {
            return Err(exceptions::PyIndexError::new_err("Index out of bounds"));
        }

        Ok(PythonRationalPolynomial {
            poly: Arc::new(self.matrix[(idx.0 as u32, idx.1 as u32)].clone()),
        })
    }

    /// Convert the matrix into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            MatrixPrinter::new_with_options(&self.matrix, PrintOptions::latex(),)
        ))
    }

    /// Compare two matrices.
    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.matrix == other.matrix),
            CompareOp::Ne => Ok(self.matrix != other.matrix),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between matrices are not supported",
            ))),
        }
    }

    /// Copy the matrix.
    pub fn __copy__(&self) -> Self {
        Self {
            matrix: Arc::new((*self.matrix).clone()),
        }
    }

    /// Convert the matrix into a human-readable string.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.matrix))
    }

    /// Add this matrix to `rhs`, returning the result.
    pub fn __add__(&self, rhs: PythonMatrix) -> PythonMatrix {
        let (new_self, new_rhs) = self.unify(&rhs);
        PythonMatrix {
            matrix: Arc::new(&*new_self.matrix + &*new_rhs.matrix),
        }
    }

    ///  Subtract `rhs` from this matrix, returning the result.
    pub fn __sub__(&self, rhs: PythonMatrix) -> PythonMatrix {
        self.__add__(rhs.__neg__())
    }

    /// Add this matrix to `rhs`, returning the result.
    pub fn __mul__(&self, rhs: ScalarOrMatrix) -> PyResult<PythonMatrix> {
        match rhs {
            ScalarOrMatrix::Scalar(s) => {
                let (new_self, new_rhs) = self.unify_scalar(&s.to_rational_polynomial()?);

                Ok(Self {
                    matrix: Arc::new(new_self.matrix.mul_scalar(&new_rhs.poly)),
                })
            }
            ScalarOrMatrix::Matrix(m) => {
                let (new_self, new_rhs) = self.unify(&m);
                Ok(PythonMatrix {
                    matrix: Arc::new(&*new_self.matrix * &*new_rhs.matrix),
                })
            }
        }
    }

    /// Add this matrix to `rhs`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToRationalPolynomial) -> PyResult<PythonMatrix> {
        self.__mul__(ScalarOrMatrix::Scalar(rhs))
    }

    /// Divide the matrix by the scalar, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToRationalPolynomial) -> PyResult<PythonMatrix> {
        Ok(PythonMatrix {
            matrix: Arc::new(self.matrix.div_scalar(&rhs.to_rational_polynomial()?.poly)),
        })
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: PyObject) -> PyResult<PythonMatrix> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor a matrix. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: PyObject) -> PyResult<PythonMatrix> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor a matrix. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the matrix, returning the result.
    pub fn __neg__(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: Arc::new(-self.matrix.as_ref().clone()),
        }
    }
}

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

/// A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
/// A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).
///
/// Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
/// which is an instance counter starting at 0.
#[pyclass(name = "RandomNumberGenerator")]
struct PythonRandomNumberGenerator {
    state: MonteCarloRng,
}

#[pymethods]
impl PythonRandomNumberGenerator {
    /// Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
    /// each thread or instance generating samples should use the same `seed` but a different `stream_id`.
    #[new]
    fn new(seed: u64, stream_id: usize) -> Self {
        Self {
            state: MonteCarloRng::new(seed, stream_id),
        }
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

    /// Create a new random number generator, suitable for use with the integrator.
    /// Each thread of instance of the integrator should have its own random number generator,
    /// that is initialized with the same seed but with a different stream id.
    #[classmethod]
    pub fn rng(_cls: &PyType, seed: u64, stream_id: usize) -> PythonRandomNumberGenerator {
        PythonRandomNumberGenerator::new(seed, stream_id)
    }

    /// Sample `num_samples` points from the grid using the random number generator
    /// `rng`. See `rng()` for how to create a random number generator.
    pub fn sample(
        &mut self,
        num_samples: usize,
        rng: &mut PythonRandomNumberGenerator,
    ) -> Vec<PythonSample> {
        let mut sample = Sample::new();

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            self.grid.sample(&mut rng.state, &mut sample);
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

    /// Import an exported grid from another thread or machine.
    /// Use `export_grid` to export the grid.
    #[classmethod]
    fn import_grid(_cls: &PyType, grid: &[u8]) -> PyResult<Self> {
        let grid = bincode::deserialize(grid)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(PythonNumericalIntegrator { grid })
    }

    /// Export the grid, so that it can be sent to another thread or machine.
    /// Use `import_grid` to load the grid.
    fn export_grid<'p>(&self, py: Python<'p>) -> PyResult<&'p PyBytes> {
        bincode::serialize(&self.grid)
            .map(|a| PyBytes::new(py, &a))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get the estamate of the average, error, chi-squared, maximum negative and positive evaluations, and the number of processed samples
    /// for the current iteration, including the points submitted in the current iteration.
    fn get_live_estimate(&self) -> PyResult<(f64, f64, f64, f64, f64, usize)> {
        match &self.grid {
            Grid::Continuous(cs) => {
                let mut a = cs.accumulator.shallow_copy();
                a.update_iter();
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                ))
            }
            Grid::Discrete(ds) => {
                let mut a = ds.accumulator.shallow_copy();
                a.update_iter();
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                ))
            }
        }
    }

    /// Add the accumulated training samples from the grid `other` to the current grid.
    /// The grid structure of `self` and `other` must be equivalent.
    fn merge(&mut self, other: &PythonNumericalIntegrator) -> PyResult<()> {
        self.grid
            .merge(&other.grid)
            .map_err(|e| pyo3::exceptions::PyAssertionError::new_err(e))
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
        Ok((stats.avg, stats.err, stats.chi_sq / stats.cur_iter as f64))
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
        seed = 0,
        show_stats = true)
    )]
    pub fn integrate(
        &mut self,
        py: Python,
        integrand: PyObject,
        max_n_iter: usize,
        min_error: f64,
        n_samples_per_iter: usize,
        seed: u64,
        show_stats: bool,
    ) -> PyResult<(f64, f64, f64)> {
        let mut rng = MonteCarloRng::new(seed, 0);

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
