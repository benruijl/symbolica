use std::{
    borrow::{Borrow, BorrowMut},
    ops::Neg,
    sync::{Arc, RwLock},
};

use ahash::HashMap;
use once_cell::sync::Lazy;
use pyo3::{
    exceptions, pyclass,
    pyclass::CompareOp,
    pymethods, pymodule,
    types::{PyModule, PyTuple, PyType},
    FromPyObject, PyResult, Python,
};
use self_cell::self_cell;
use smallvec::SmallVec;

use crate::{
    id::{Match, Pattern, PatternRestriction},
    parser::parse,
    poly::{polynomial::MultivariatePolynomial, INLINED_EXPONENTS},
    printer::{AtomPrinter, PolynomialPrinter, PrintMode, RationalPolynomialPrinter},
    representations::{
        default::{
            DefaultRepresentation, ListIteratorD, OwnedAddD, OwnedFunD, OwnedMulD, OwnedNumD,
            OwnedPowD, OwnedVarD,
        },
        number::{BorrowedNumber, Number},
        Add, AtomView, Fun, Identifier, Mul, Num, OwnedAdd, OwnedAtom, OwnedFun, OwnedMul,
        OwnedNum, OwnedPow, OwnedVar, Var,
    },
    rings::integer::IntegerRing,
    rings::{
        rational::RationalField,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    state::{ResettableBuffer, State, Workspace},
};

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
const WORKSPACE: Lazy<Workspace<DefaultRepresentation>> = Lazy::new(|| Workspace::new());

#[pymodule]
fn symbolica(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PythonExpression>()?;
    m.add_class::<PythonFunction>()?;
    m.add_class::<PythonPolynomial>()?;
    m.add_class::<PythonIntegerPolynomial>()?;
    m.add_class::<PythonRationalPolynomial>()?;
    m.add_class::<PythonRationalPolynomialSmallExponent>()?;

    Ok(())
}

#[pyclass(name = "Expression")]
#[derive(Clone)]
pub struct PythonExpression {
    pub expr: Arc<OwnedAtom<DefaultRepresentation>>,
}

/// A subset of pattern restrictions that can be used in Python.
#[derive(Debug, Clone, Copy)]
pub enum SimplePatternRestriction {
    Length(Identifier, usize, Option<usize>), // min-max range
    IsVar(Identifier),
    IsNumber(Identifier),
    NumberCmp(Identifier, CompareOp, i64),
}

#[pyclass(name = "PatternRestriction")]
#[derive(Debug, Clone)]
pub struct PythonPatternRestriction {
    pub restrictions: Arc<Vec<SimplePatternRestriction>>,
}

#[pymethods]
impl PythonPatternRestriction {
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
                let mut num = OwnedAtom::new();
                let num_d: &mut OwnedNumD = num.transform_to_num();
                num_d.from_number(Number::Natural(i, 1));
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
    #[classmethod]
    pub fn var(_cls: &PyType, name: &str) -> PyResult<PythonExpression> {
        let mut guard = STATE.write().unwrap();
        let state = guard.borrow_mut();
        // TODO: check if the name meets the requirements
        let id = state.get_or_insert_var(name);
        let mut var = OwnedAtom::new();
        let o: &mut OwnedVarD = var.transform_to_var();
        o.from_id(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

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
            let mut var = OwnedAtom::new();
            let o: &mut OwnedVarD = var.transform_to_var();
            o.from_id(id);

            result.push(PythonExpression {
                expr: Arc::new(var),
            });
        }

        Ok(result)
    }

    #[classmethod]
    pub fn fun(_cls: &PyType, name: &str) -> PyResult<PythonFunction> {
        PythonFunction::__new__(name)
    }

    #[pyo3(signature = (*args,))]
    #[classmethod]
    pub fn funs(_cls: &PyType, args: &PyTuple) -> PyResult<Vec<PythonFunction>> {
        let mut result = Vec::with_capacity(args.len());
        for a in args {
            let name = a.extract::<&str>()?;
            result.push(PythonFunction::__new__(name)?);
        }

        Ok(result)
    }

    #[classmethod]
    pub fn parse(_cls: &PyType, arg: &str) -> PyResult<PythonExpression> {
        let e = parse(arg)
            .map_err(|m| exceptions::PyValueError::new_err(m))?
            .to_atom(STATE.write().unwrap().borrow_mut(), &WORKSPACE)
            .map_err(|m| exceptions::PyValueError::new_err(m))?;

        Ok(PythonExpression { expr: Arc::new(e) })
    }

    pub fn __copy__(&self) -> PythonExpression {
        PythonExpression {
            expr: Arc::new((*self.expr).clone()),
        }
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "{}",
            AtomPrinter {
                atom: self.expr.to_view(),
                state: &STATE.read().unwrap(),
                print_mode: PrintMode::default(),
            }
        ))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.expr))
    }

    /// Create a wildcard from a variable name.
    ///
    #[getter]
    fn get_w(&self) -> PyResult<PythonExpression> {
        let mut guard = STATE.write().unwrap();
        let state = guard.borrow_mut();
        let mut var_name = match self.expr.to_view() {
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
        let mut var = OwnedAtom::new();
        let o: &mut OwnedVarD = var.transform_to_var();
        o.from_id(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let workspace = WORKSPACE;
        let mut e = workspace.new_atom();
        let a: &mut OwnedAddD = e.transform_to_add();

        a.extend(self.expr.to_view());
        a.extend(rhs.to_expression().expr.to_view());
        a.set_dirty(true);

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        PythonExpression { expr: Arc::new(b) }
    }

    pub fn __sub__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        self.__add__(ConvertibleToExpression::Expression(
            rhs.to_expression().__neg__(),
        ))
    }

    pub fn __mul__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let workspace = WORKSPACE;
        let mut e = workspace.new_atom();
        let a: &mut OwnedMulD = e.transform_to_mul();

        a.extend(self.expr.to_view());
        a.extend(rhs.to_expression().expr.to_view());
        a.set_dirty(true);

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        PythonExpression { expr: Arc::new(b) }
    }

    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let workspace = WORKSPACE;
        let mut pow = workspace.new_atom();
        let pow_num = pow.transform_to_num();
        pow_num.from_number(Number::Natural(-1, 1));

        let mut e = workspace.new_atom();
        let a: &mut OwnedPowD = e.transform_to_pow();
        a.from_base_and_exp(rhs.to_expression().expr.to_view(), pow.get().to_view());
        a.set_dirty(true);

        let mut m = workspace.new_atom();
        let md: &mut OwnedMulD = m.transform_to_mul();

        md.extend(self.expr.to_view());
        md.extend(e.get().to_view());
        md.set_dirty(true);

        let mut b = OwnedAtom::new();
        m.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);
        PythonExpression { expr: Arc::new(b) }
    }

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

        let workspace = WORKSPACE;
        let mut e = workspace.new_atom();
        let a: &mut OwnedPowD = e.transform_to_pow();

        a.from_base_and_exp(self.expr.to_view(), rhs.to_expression().expr.to_view());
        a.set_dirty(true);

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    pub fn __neg__(&self) -> PythonExpression {
        let workspace = WORKSPACE;
        let mut e = workspace.new_atom();
        let a: &mut OwnedMulD = e.transform_to_mul();

        let mut sign = workspace.new_atom();
        let sign_num = sign.transform_to_num();
        sign_num.from_number(Number::Natural(-1, 1));

        a.extend(self.expr.to_view());
        a.extend(sign.get().to_view());
        a.set_dirty(true);

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        PythonExpression { expr: Arc::new(b) }
    }

    fn __len__(&self) -> usize {
        match self.expr.to_view() {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            _ => 1,
        }
    }

    /// Create a pattern restriction based on the length.
    pub fn len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.to_view() {
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

    pub fn is_var(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.to_view() {
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

    pub fn is_num(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.to_view() {
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

    fn __richcmp__(&self, other: i64, op: CompareOp) -> PyResult<PythonPatternRestriction> {
        match self.expr.to_view() {
            AtomView::Var(v) => {
                let name = v.get_name();
                if !STATE.read().unwrap().is_wildcard(name).unwrap_or(false) {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    restrictions: Arc::new(vec![SimplePatternRestriction::NumberCmp(
                        name, op, other,
                    )]),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    fn __iter__(&self) -> PyResult<PythonAtomIterator> {
        match self.expr.to_view() {
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

    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();
        for v in vars {
            match v.expr.to_view() {
                AtomView::Var(v) => var_map.push(v.get_name()),
                e => {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected variable instead of {:?}",
                        e
                    )))?;
                }
            }
        }

        let mut b = OwnedAtom::new();
        self.expr.to_view().set_coefficient_ring(
            &var_map,
            &STATE.read().unwrap().borrow(),
            &WORKSPACE,
            &mut b,
        );

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    pub fn expand(&self) -> PythonExpression {
        let mut b = OwnedAtom::new();
        self.expr
            .to_view()
            .expand(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        PythonExpression { expr: Arc::new(b) }
    }

    pub fn to_polynomial(&self, vars: Option<Vec<PythonExpression>>) -> PyResult<PythonPolynomial> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.to_view() {
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
            .to_view()
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

    pub fn to_rational_polynomial(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomial> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.to_view() {
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
            .to_view()
            .to_rational_polynomial(
                &WORKSPACE,
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
    }

    // TODO: use macro as the body is the same as for to_rational_polynomial
    pub fn to_rational_polynomial_small_exponent(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomialSmallExponent> {
        let mut var_map: SmallVec<[Identifier; INLINED_EXPONENTS]> = SmallVec::new();

        if let Some(vm) = vars {
            for v in vm {
                match v.expr.to_view() {
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
            .to_view()
            .to_rational_polynomial(
                &WORKSPACE,
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
    }

    pub fn replace_all(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToExpression,
        cond: Option<PythonPatternRestriction>,
    ) -> PyResult<PythonExpression> {
        let pattern =
            Pattern::from_view(lhs.to_expression().expr.to_view(), &STATE.read().unwrap());
        let mut restrictions = HashMap::default();
        let rhs = Pattern::from_view(rhs.to_expression().expr.to_view(), &STATE.read().unwrap());
        let mut out = OwnedAtom::new();

        if let Some(rs) = cond {
            for r in &*rs.restrictions {
                match r {
                    &SimplePatternRestriction::IsVar(name) => {
                        restrictions
                            .entry(name)
                            .or_insert(vec![])
                            .push(PatternRestriction::<DefaultRepresentation>::IsVar);
                    }
                    &SimplePatternRestriction::IsNumber(name) => {
                        restrictions
                            .entry(name)
                            .or_insert(vec![])
                            .push(PatternRestriction::IsNumber);
                    }
                    &SimplePatternRestriction::Length(name, min, max) => {
                        restrictions
                            .entry(name)
                            .or_insert(vec![])
                            .push(PatternRestriction::Length(min, max));
                    }
                    &SimplePatternRestriction::NumberCmp(name, op, ref_num) => {
                        restrictions.entry(name).or_insert(vec![]).push(
                            PatternRestriction::Filter(Box::new(
                                move |v: &Match<DefaultRepresentation>| match v {
                                    Match::Single(v) => {
                                        if let AtomView::Num(n) = v {
                                            let num = n.get_number_view();
                                            let ordering =
                                                num.cmp(&BorrowedNumber::Natural(ref_num, 1));
                                            match op {
                                                CompareOp::Lt => ordering.is_lt(),
                                                CompareOp::Le => ordering.is_le(),
                                                CompareOp::Eq => ordering.is_eq(),
                                                CompareOp::Ne => ordering.is_ne(),
                                                CompareOp::Gt => ordering.is_gt(),
                                                CompareOp::Ge => ordering.is_ge(),
                                            }
                                        } else {
                                            false
                                        }
                                    }
                                    _ => false,
                                },
                            )),
                        );
                    }
                }
            }
        }

        pattern.replace_all(
            self.expr.to_view(),
            &rhs,
            &STATE.read().unwrap(),
            &WORKSPACE,
            &restrictions,
            &mut out,
        );

        Ok(PythonExpression {
            expr: Arc::new(out),
        })
    }
}

/// A function class for python that constructs an `Expression` when called with arguments.
/// This allows to write:
/// ```python
/// f = Function("f")
/// e = f(1,2,3)
/// ```
#[pyclass(name = "Function")]
pub struct PythonFunction {
    id: Identifier,
}

#[pymethods]
impl PythonFunction {
    #[new]
    pub fn __new__(name: &str) -> PyResult<Self> {
        // TODO: parse and check if this is a valid function name
        let id = STATE.write().unwrap().borrow_mut().get_or_insert_var(name);
        Ok(PythonFunction { id })
    }

    #[getter]
    fn get_w(&mut self) -> PythonFunction {
        PythonFunction { id: self.id }
    }

    #[pyo3(signature = (*args,))]
    pub fn __call__(&self, args: &PyTuple) -> PyResult<PythonExpression> {
        let b = WORKSPACE;
        let mut fun_b = b.new_atom();
        let fun: &mut OwnedFunD = fun_b.transform_to_fun();
        fun.from_name(self.id);

        for arg in args {
            if let Ok(a) = arg.extract::<ConvertibleToExpression>() {
                fun.add_arg(a.to_expression().expr.to_view());
            } else {
                let msg = format!("Unknown type: {:?}", arg.get_type().name());
                return Err(exceptions::PyTypeError::new_err(msg));
            }
        }

        let mut out = OwnedAtom::new();
        fun_b
            .get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap(), &mut out);

        Ok(PythonExpression {
            expr: Arc::new(out),
        })
    }
}

self_cell!(
    #[pyclass]
    pub struct PythonAtomIterator {
        owner: Arc<OwnedAtom<DefaultRepresentation>>,
        #[covariant]
        dependent: ListIteratorD,
    }
);

impl PythonAtomIterator {
    /// Create a self-referential structure for the iterator.
    pub fn from_expr(expr: PythonExpression) -> PythonAtomIterator {
        PythonAtomIterator::new(expr.expr.clone(), |expr| match expr.to_view() {
            AtomView::Add(a) => a.into_iter(),
            AtomView::Mul(m) => m.into_iter(),
            AtomView::Fun(f) => f.into_iter(),
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
                    let mut owned = OwnedAtom::new();
                    owned.from_view(&e);
                    owned
                }),
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

macro_rules! generate_methods {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly,
                        state: &STATE.read().unwrap(),
                        print_mode: PrintMode::default()
                    }
                ))
            }

            pub fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:?}", self.poly))
            }

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

            pub fn __sub__(&self, rhs: Self) -> Self {
                self.__add__(rhs.__neg__())
            }

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

            pub fn __neg__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone().neg()),
                }
            }

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

#[pyclass(name = "RationalPolynomial")]
#[derive(Clone)]
pub struct PythonRationalPolynomial {
    pub poly: Arc<RationalPolynomial<IntegerRing, u32>>,
}

#[pymethods]
impl PythonRationalPolynomial {
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
            pub fn __copy__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone()),
                }
            }

            pub fn __str__(&self) -> PyResult<String> {
                Ok(format!(
                    "{}",
                    RationalPolynomialPrinter {
                        poly: &self.poly,
                        state: &STATE.read().unwrap(),
                        print_mode: PrintMode::default()
                    }
                ))
            }

            pub fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:?}", self.poly))
            }

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

            pub fn __neg__(&self) -> Self {
                Self {
                    poly: Arc::new((*self.poly).clone().neg()),
                }
            }

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
