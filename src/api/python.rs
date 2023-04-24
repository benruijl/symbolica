use std::{
    borrow::{Borrow, BorrowMut},
    sync::{Arc, RwLock},
};

use ahash::HashMap;
use once_cell::sync::Lazy;
use ouroboros::self_referencing;
use pyo3::{
    exceptions, pyclass, pymethods, pymodule,
    types::{PyModule, PyTuple, PyType},
    FromPyObject, PyResult, Python,
};

use crate::{
    id::Pattern,
    parser::parse,
    printer::{AtomPrinter, PrintMode},
    representations::{
        default::{
            DefaultRepresentation, ListIteratorD, OwnedAddD, OwnedFunD, OwnedMulD, OwnedNumD,
            OwnedPowD, OwnedVarD,
        },
        number::Number,
        Add, AtomView, Fun, Identifier, Mul, OwnedAdd, OwnedAtom, OwnedFun, OwnedMul, OwnedNum,
        OwnedPow, OwnedVar, Var,
    },
    state::{ResettableBuffer, State, Workspace},
};

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
const WORKSPACE: Lazy<Workspace<DefaultRepresentation>> = Lazy::new(|| Workspace::new());

#[pymodule]
fn symbolica(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PythonExpression>()?;
    m.add_class::<PythonFunction>()?;

    Ok(())
}

#[pyclass(name = "Expression")]
#[derive(Clone)]
pub struct PythonExpression {
    pub expr: Arc<OwnedAtom<DefaultRepresentation>>,
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
    #[staticmethod]
    pub fn var(name: &str) -> PyResult<PythonExpression> {
        // TODO: check if the name meets the requirements
        let id = STATE.write().unwrap().borrow_mut().get_or_insert_var(name);
        let mut var = OwnedAtom::new();
        let o: &mut OwnedVarD = var.transform_to_var();
        o.from_id(id);

        Ok(PythonExpression {
            expr: Arc::new(var),
        })
    }

    #[staticmethod]
    pub fn fun(name: &str) -> PyResult<PythonFunction> {
        PythonFunction::__new__(name)
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
                print_mode: PrintMode::Form
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
        let state = STATE.borrow().read().unwrap();
        match self.expr.to_view() {
            AtomView::Var(v) => {
                if let Some(true) = state.is_wildcard(v.get_name()) {
                    Ok(self.clone())
                } else {
                    // create name with underscore
                    let mut name = state.get_name(v.get_name()).unwrap().to_string();
                    name.push('_');
                    drop(state);
                    PythonExpression::var(&name)
                }
            }
            AtomView::Fun(f) => {
                if let Some(true) = state.is_wildcard(f.get_name()) {
                    Ok(self.clone())
                } else {
                    // create name with underscore
                    let mut name = state.get_name(f.get_name()).unwrap().to_string();
                    name.push('_');
                    drop(state);
                    PythonExpression::var(&name)
                }
            }
            x => Err(exceptions::PyValueError::new_err(format!(
                "Cannot convert to wildcard: {:?}",
                x
            ))),
        }
    }

    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PythonExpression {
        let workspace = WORKSPACE;
        let mut e = workspace.new_atom();
        let a: &mut OwnedAddD = e.get_mut().transform_to_add();

        a.extend(self.expr.to_view());
        a.extend(rhs.to_expression().expr.to_view());

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
        let a: &mut OwnedMulD = e.get_mut().transform_to_mul();

        a.extend(self.expr.to_view());
        a.extend(rhs.to_expression().expr.to_view());

        let mut b = OwnedAtom::new();
        e.get()
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
        let a: &mut OwnedPowD = e.get_mut().transform_to_pow();

        a.from_base_and_exp(self.expr.to_view(), rhs.to_expression().expr.to_view());

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        Ok(PythonExpression { expr: Arc::new(b) })
    }

    pub fn __neg__(&self) -> PythonExpression {
        let b = WORKSPACE;
        let mut e = b.new_atom();
        let a: &mut OwnedMulD = e.get_mut().transform_to_mul();

        let mut sign = b.new_atom();
        let sign_num = sign.get_mut().transform_to_num();
        sign_num.from_number(Number::Natural(-1, 1));

        a.extend(self.expr.to_view());
        a.extend(sign.get().to_view());

        let mut b = OwnedAtom::new();
        e.get()
            .to_view()
            .normalize(&WORKSPACE, &STATE.read().unwrap().borrow(), &mut b);

        PythonExpression { expr: Arc::new(b) }
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

    pub fn replace_all(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToExpression,
    ) -> PyResult<PythonExpression> {
        let pattern =
            Pattern::from_view(lhs.to_expression().expr.to_view(), &STATE.read().unwrap());
        let restrictions = HashMap::default();
        let rhs = Pattern::from_view(rhs.to_expression().expr.to_view(), &STATE.read().unwrap());
        let mut out = OwnedAtom::new();

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
        let fun: &mut OwnedFunD = fun_b.get_mut().transform_to_fun();
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

#[pyclass]
#[self_referencing]
pub struct PythonAtomIterator {
    expr: Arc<OwnedAtom<DefaultRepresentation>>,
    #[borrows(expr)]
    #[covariant]
    iter: ListIteratorD<'this>,
}

impl PythonAtomIterator {
    /// Create a self-referential structure for the iterator.
    pub fn from_expr(expr: PythonExpression) -> PythonAtomIterator {
        PythonAtomIteratorBuilder {
            expr: expr.expr.clone(),
            iter_builder: |expr| match expr.to_view() {
                AtomView::Add(a) => a.into_iter(),
                AtomView::Mul(m) => m.into_iter(),
                AtomView::Fun(f) => f.into_iter(),
                _ => unreachable!(),
            },
        }
        .build()
    }
}

#[pymethods]
impl PythonAtomIterator {
    fn __next__(&mut self) -> Option<PythonExpression> {
        self.with_iter_mut(|i| {
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
