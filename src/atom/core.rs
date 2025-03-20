//! Provide the basic operations on general expressions.
//!
//! See [AtomCore] for the possible operations.

use ahash::{HashMap, HashSet};
use rayon::ThreadPool;

use crate::{
    coefficient::{Coefficient, CoefficientView, ConvertToRing},
    domains::{
        atom::AtomField,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        float::{Real, SingleFloat},
        integer::Z,
        rational::Rational,
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
        EuclideanDomain, InternalOrdering,
    },
    evaluate::{EvalTree, EvaluationFn, ExpressionEvaluator, FunctionMap, OptimizationSettings},
    id::{
        BorrowReplacement, Condition, ConditionResult, Context, MatchSettings, Pattern,
        PatternAtomTreeIterator, PatternRestriction, ReplaceBuilder,
    },
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, series::Series,
        Exponent, PositiveExponent, Variable,
    },
    printer::{AtomPrinter, PrintOptions, PrintState},
    state::Workspace,
    tensors::matrix::Matrix,
    utils::BorrowedOrOwned,
};
use std::sync::Arc;

use super::{
    representation::{InlineNum, InlineVar},
    Atom, AtomOrView, AtomView, KeyLookup, Symbol,
};

/// All core features of expressions, such as expansion and
/// pattern matching that leave the expression unchanged.
pub trait AtomCore {
    /// Take a view of the atom.
    fn as_atom_view(&self) -> AtomView;

    /// Export the atom and state to a binary stream. It can be loaded
    /// with [Atom::import].
    fn export<W: std::io::Write>(&self, dest: W) -> Result<(), std::io::Error> {
        self.as_atom_view().export(dest)
    }

    /// Get the symbol of a variable or function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let atom = parse!("f(x)").unwrap();
    /// assert_eq!(atom.get_symbol(), Some(symbol!("f")));
    /// ```
    #[inline(always)]
    fn get_symbol(&self) -> Option<Symbol> {
        match self.as_atom_view() {
            AtomView::Var(v) => Some(v.get_symbol()),
            AtomView::Fun(f) => Some(f.get_symbol()),
            _ => None,
        }
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Use [collect_symbol](AtomCore::collect_symbol) to collect using the name of a function only.
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2").unwrap();
    /// let x = parse!("x").unwrap();
    /// let collected = expr.collect::<u8>(x, None, None);
    /// assert_eq!(collected, parse!("x * (1 + y) + x^2").unwrap());
    /// ```
    fn collect<E: Exponent>(
        &self,
        x: impl AtomCore,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view().collect::<E, _>(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of variables or functions with the name `x`, e.g.
    ///
    /// ```math
    /// collect_symbol(f(1,2) + x*f*(1,2), f) = (1+x)*f(1,2)
    /// ```
    ///
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1,2) + x*f(1,2)").unwrap();
    /// let collected = expr.collect_symbol::<u8>(symbol!("f"), None, None);
    /// assert_eq!(collected, parse!("(1+x)*f(1,2)").unwrap());
    /// ```
    fn collect_symbol<E: Exponent>(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view()
            .collect_symbol::<E>(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2 + z + z^2").unwrap();
    /// let x = parse!("x").unwrap();
    /// let z = parse!("z").unwrap();
    /// let collected = expr.collect_multiple::<u8>(&[x, z], None, None);
    /// assert_eq!(collected, parse!("x * (1 + y) + x^2 + z + z^2").unwrap());
    /// ```
    fn collect_multiple<E: Exponent>(
        &self,
        xs: &[impl AtomCore],
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view()
            .collect_multiple::<E, _>(xs, key_map, coeff_map)
    }

    /// Collect common factors from (nested) sums.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x*(x+y*x+x^2+y*(x+x^2))").unwrap();
    /// let collected = expr.collect_factors();
    /// assert_eq!(collected, parse!("x^2*(1+x+y+y*(1+x))").unwrap());
    /// ```
    fn collect_factors(&self) -> Atom {
        self.as_atom_view().collect_factors()
    }

    /// Collect terms involving the same power of `x` in `xs`, where `xs` is a list of indeterminates.
    /// Return the list of key-coefficient pairs
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2 + z + z^2").unwrap();
    /// let x = parse!("x").unwrap();
    /// let z = parse!("z").unwrap();
    /// let coeff_list = expr.coefficient_list::<u8>(&[x, z]);
    /// assert_eq!(coeff_list.len(), 4);
    /// ```
    fn coefficient_list<E: Exponent>(&self, xs: &[impl AtomCore]) -> Vec<(Atom, Atom)> {
        self.as_atom_view().coefficient_list::<E, _>(xs)
    }

    /// Collect terms involving the literal occurrence of `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2").unwrap();
    /// let x = parse!("x").unwrap();
    /// let coeff = expr.coefficient(x);
    /// let r = parse!("1+y").unwrap();
    /// assert_eq!(coeff, coeff);
    /// ```
    fn coefficient<T: AtomCore>(&self, x: T) -> Atom {
        Workspace::get_local().with(|ws| {
            self.as_atom_view()
                .coefficient_with_ws(x.as_atom_view(), ws)
        })
    }

    /// Write the expression over a common denominator.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("1/x + 1/y").unwrap();
    /// let together = expr.together();
    /// let r = parse!("(x + y) / (x * y)").unwrap();
    /// assert_eq!(together, r);
    /// ```
    fn together(&self) -> Atom {
        self.as_atom_view().together()
    }

    /// Write the expression as a sum of terms with minimal denominators.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("(x + y) / (x * y)").unwrap();
    /// let apart = expr.apart(symbol!("x"));
    /// let r = parse!("1 / y + 1 / x").unwrap();
    /// assert_eq!(apart, r);
    /// ```
    fn apart(&self, x: Symbol) -> Atom {
        self.as_atom_view().apart(x)
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x^2 - 1) / (x - 1)").unwrap();
    /// let canceled = expr.cancel();
    /// let r = parse!("x+1").unwrap();
    /// assert_eq!(canceled, r);
    /// ```
    fn cancel(&self) -> Atom {
        self.as_atom_view().cancel()
    }

    /// Factor the expression over the rationals.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x^2 - 1").unwrap();
    /// let factored = expr.factor();
    /// let r = parse!("(x - 1) * (x + 1)").unwrap();
    /// assert_eq!(factored, r);
    /// ```
    fn factor(&self) -> Atom {
        self.as_atom_view().factor()
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("-2*x + 4*x^2 + 6*x^3").unwrap();
    /// let collected_num = expr.collect_num();
    /// let r = parse!("-2 * (x - 2 * x^2 - 3 * x^3)").unwrap();
    /// assert_eq!(collected_num, r);
    /// ```
    fn collect_num(&self) -> Atom {
        self.as_atom_view().collect_num()
    }

    /// Expand an expression. The function [AtomCore::expand_via_poly] may be faster.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x + 1)^2").unwrap();
    /// let expanded = expr.expand();
    /// let r = parse!("x^2 + 2 * x + 1").unwrap();
    /// assert_eq!(expanded, r);
    /// ```
    fn expand(&self) -> Atom {
        self.as_atom_view().expand()
    }

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("(x + 1)^2").unwrap();
    /// let expanded = expr.expand_via_poly::<u8, Atom>(None);
    /// let r = parse!("x^2 + 2 * x + 1").unwrap();
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_via_poly<E: Exponent, T: AtomCore>(&self, var: impl Into<Option<T>>) -> Atom {
        self.as_atom_view()
            .expand_via_poly::<E>(var.into().as_ref().map(|x| x.as_atom_view()))
    }

    /// Expand an expression in the variable `var`. The function [AtomCore::expand_via_poly] may be faster.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x + 1)^2").unwrap();
    /// let x = parse!("x").unwrap();
    /// let expanded = expr.expand_in(x);
    /// let r = parse!("x^2 + 2 * x + 1").unwrap();
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_in<T: AtomCore>(&self, var: T) -> Atom {
        self.as_atom_view().expand_in(var.as_atom_view())
    }

    /// Expand an expression in the variable `var`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("(x + 1)^2").unwrap();
    /// let expanded = expr.expand_in_symbol(symbol!("x"));
    /// let r = parse!("x^2 + 2 * x + 1").unwrap();
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_in_symbol(&self, var: Symbol) -> Atom {
        self.as_atom_view()
            .expand_in(InlineVar::from(var).as_view())
    }

    /// Expand an expression, returning `true` iff the expression changed.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("(x + 1)^2").unwrap();
    /// let mut out = Atom::new();
    /// let changed = expr.expand_into::<Atom>(None, &mut out);
    /// let r = parse!("x^2 + 2 * x + 1").unwrap();
    /// assert!(changed);
    /// assert_eq!(out, r);
    /// ```
    fn expand_into<T: AtomCore>(&self, var: impl Into<Option<T>>, out: &mut Atom) -> bool {
        self.as_atom_view()
            .expand_into(var.into().as_ref().map(|x| x.as_atom_view()), out)
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("2*(x+y)").unwrap();
    /// let expanded_num = expr.expand_num();
    /// let r = parse!("2 * x + 2 * y").unwrap();
    /// assert_eq!(expanded_num, r);
    /// ```
    fn expand_num(&self) -> Atom {
        self.as_atom_view().expand_num()
    }

    /// Check if the expression is expanded, optionally in only the variable or function `var`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("x^2 + 2*x + 1").unwrap();
    /// let is_expanded = expr.is_expanded::<Atom>(None);
    /// assert!(is_expanded);
    /// ```
    fn is_expanded<T: AtomCore>(&self, var: Option<T>) -> bool {
        self.as_atom_view()
            .is_expanded(var.as_ref().map(|x| x.as_atom_view()))
    }

    /// Take a derivative of the expression with respect to `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + 2*x + 1").unwrap();
    /// let derivative = expr.derivative(symbol!("x"));
    /// let r = parse!("2 * x + 2").unwrap();
    /// assert_eq!(derivative, r);
    /// ```
    fn derivative(&self, x: Symbol) -> Atom {
        self.as_atom_view().derivative(x)
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + 2*x + 1").unwrap();
    /// let mut out = Atom::new();
    /// let non_zero = expr.derivative_into(symbol!("x"), &mut out);
    /// assert!(non_zero);
    /// assert_eq!(out, parse!("2 * x + 2").unwrap());
    /// ```
    fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        self.as_atom_view().derivative_into(x, out)
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("exp(x)").unwrap();
    /// let series = expr
    ///     .series(symbol!("x"), Atom::new_num(0), (4, 1).into(), true)
    ///     .unwrap();
    /// assert_eq!(
    ///     series.to_atom(),
    ///     parse!("1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24").unwrap()
    /// );
    /// ```
    fn series<T: AtomCore>(
        &self,
        x: Symbol,
        expansion_point: T,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> Result<Series<AtomField>, &'static str> {
        self.as_atom_view()
            .series(x, expansion_point.as_atom_view(), depth, depth_is_absolute)
    }

    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 - 2").unwrap();
    /// let root = expr.nsolve(symbol!("x"), 1.0, 1e-7, 100).unwrap();
    /// assert!((root - 1.414213562373095).abs() < 1e-7);
    /// ```
    fn nsolve<N: SingleFloat + Real + PartialOrd>(
        &self,
        x: Symbol,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<N, String> {
        self.as_atom_view().nsolve(x, init, prec, max_iterations)
    }

    /// Solve a non-linear system numerically over the reals using Newton's method.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::domains::float::F64;
    /// let expr1 = parse!("x^2 + y^2 - 1").unwrap();
    /// let expr2 = parse!("x^2 - y").unwrap();
    /// let system = &[expr1, expr2];
    /// let vars = &[symbol!("x"), symbol!("y")];
    /// let init = &[F64::from(0.5), F64::from(0.5)];
    /// let roots = Atom::nsolve_system(system, vars, init, 1e-7.into(), 100).unwrap();
    /// assert!((roots[0].into_inner() - 0.786151377757424).abs() < 1e-7);
    /// assert!((roots[1].into_inner() - 0.6180339887498941).abs() < 1e-7);
    /// ```
    fn nsolve_system<
        N: SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
        T: AtomCore,
    >(
        system: &[T],
        vars: &[Symbol],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        AtomView::nsolve_system(system, vars, init, prec, max_iterations)
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr1 = parse!("2*x + y - 1").unwrap();
    /// let expr2 = parse!("x + y + 1").unwrap();
    /// let system = &[expr1, expr2];
    /// let vars = &[parse!("x").unwrap(), parse!("y").unwrap()];
    /// let solution = Atom::solve_linear_system::<u8, _, _>(system, vars).unwrap();
    /// assert_eq!(solution[0], Atom::new_num(2));
    /// assert_eq!(solution[1], Atom::new_num(-3));
    /// ```
    fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, String> {
        AtomView::solve_linear_system::<E, T1, T2>(system, vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::domains::Ring;
    /// let expr1 = parse!("2*x + y - 1").unwrap();
    /// let expr2 = parse!("x - y + 1").unwrap();
    /// let system = &[expr1, expr2];
    /// let vars = &[parse!("x").unwrap(), parse!("y").unwrap()];
    /// let (matrix, rhs) = Atom::system_to_matrix::<u8, _, _>(system, vars).unwrap();
    /// let one = matrix.field().one();
    /// assert_eq!(
    ///     matrix.into_vec(),
    ///     [&one + &one, one.clone(), one.clone(), -one.clone()]
    /// );
    /// assert_eq!(rhs.into_vec(), [one.clone(), -one]);
    /// ```
    fn system_to_matrix<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        AtomView::system_to_matrix::<E, T1, T2>(system, vars)
    }

    /// Evaluate a (nested) expression a single time.
    /// For repeated evaluations, use [Self::evaluator()] and convert
    /// to an optimized version or generate a compiled version of your expression.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use ahash::HashMap;
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let x = parse!("x").unwrap();
    /// let y = parse!("y").unwrap();
    /// let mut const_map = HashMap::default();
    /// const_map.insert(x.clone(), 1.0);
    /// const_map.insert(y.clone(), 2.0);
    /// let result = expr
    ///     .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
    ///     .unwrap();
    /// assert_eq!(result, 3.0);
    /// ```
    fn evaluate<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) -> Result<T, String> {
        self.as_atom_view()
            .evaluate(coeff_map, const_map, function_map)
    }

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// Consider using [AtomCore::evaluator] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::FunctionMap;
    /// let expr = parse!("x + y").unwrap();
    /// let x = parse!("x").unwrap();
    /// let y = parse!("y").unwrap();
    /// let fn_map = FunctionMap::new();
    /// let params = vec![x.clone(), y.clone()];
    /// let mut tree = expr.to_evaluation_tree(&fn_map, &params).unwrap();
    /// tree.common_subexpression_elimination();
    /// let e = tree.optimize(1, 1, None, false);
    /// let mut e = e.map_coeff(&|c| c.to_f64());
    /// let r = e.evaluate_single(&[0.5, 0.3]);
    /// assert_eq!(r, 0.8);
    /// ```
    fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap<Rational>,
        params: &[Atom],
    ) -> Result<EvalTree<Rational>, String> {
        self.as_atom_view().to_evaluation_tree(fn_map, params)
    }

    /// Create an efficient evaluator for a (nested) expression.
    /// All free parameters must appear in `params` and all other variables
    /// and user functions in the expression must occur in the function map.
    /// The function map may have nested expressions.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let expr = parse!("x + y").unwrap();
    /// let x = parse!("x").unwrap();
    /// let y = parse!("y").unwrap();
    /// let fn_map = FunctionMap::new();
    /// let params = vec![x.clone(), y.clone()];
    /// let optimization_settings = OptimizationSettings::default();
    /// let mut evaluator = expr
    ///     .evaluator(&fn_map, &params, optimization_settings)
    ///     .unwrap()
    ///     .map_coeff(&|x| x.to_f64());
    /// assert_eq!(evaluator.evaluate_single(&[1.0, 2.0]), 3.0);
    /// ```
    fn evaluator(
        &self,
        fn_map: &FunctionMap<Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Rational>, String> {
        let mut tree = self.to_evaluation_tree(fn_map, params)?;
        Ok(tree.optimize(
            optimization_settings.horner_iterations,
            optimization_settings.n_cores,
            optimization_settings.hot_start.clone(),
            optimization_settings.verbose,
        ))
    }

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let expr1 = parse!("x + y").unwrap();
    /// let expr2 = parse!("x - y").unwrap();
    /// let x = parse!("x").unwrap();
    /// let y = parse!("y").unwrap();
    /// let fn_map = FunctionMap::new();
    /// let params = vec![x.clone(), y.clone()];
    /// let evaluator = Atom::evaluator_multiple(
    ///     &[expr1, expr2],
    ///     &fn_map,
    ///     &params,
    ///     OptimizationSettings::default(),
    /// )
    /// .unwrap();
    /// let mut evaluator = evaluator.map_coeff(&|c| c.to_f64());
    /// let mut out = vec![0., 0.];
    /// evaluator.evaluate(&[1.0, 2.0], &mut out);
    /// assert_eq!(out, &[3.0, -1.0]);
    /// ```
    fn evaluator_multiple<A: AtomCore>(
        exprs: &[A],
        fn_map: &FunctionMap<Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Rational>, String> {
        let mut tree = AtomView::to_eval_tree_multiple(exprs, fn_map, params)?;
        Ok(tree.optimize(
            optimization_settings.horner_iterations,
            optimization_settings.n_cores,
            optimization_settings.hot_start.clone(),
            optimization_settings.verbose,
        ))
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::ConditionResult;
    /// let expr = parse!("(x+1)^2 - x^2 - 2x - 1").unwrap();
    /// let result = expr.zero_test(100, 1e-7);
    /// assert_eq!(result, ConditionResult::Inconclusive);
    /// ```
    fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        self.as_atom_view().zero_test(iterations, tolerance)
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x*y + x^2*y + y/(1+x)").unwrap();
    /// let vars = Arc::new(vec![symbol!("x").into()]);
    /// let result = expr.set_coefficient_ring(&vars);
    /// let r = result.set_coefficient_ring(&Arc::new(vec![]));
    /// assert_eq!(r, parse!("y*(x+1)^-1*(x+2*x^2+x^3+1)").unwrap());
    /// ```
    fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Atom {
        self.as_atom_view().set_coefficient_ring(vars)
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("1/3").unwrap();
    /// let result = expr.coefficients_to_float(2);
    /// assert_eq!(result.to_string(), "3.3e-1");
    /// ```
    fn coefficients_to_float(&self, decimal_prec: u32) -> Atom {
        let mut a = Atom::new();
        self.as_atom_view()
            .coefficients_to_float_into(decimal_prec, &mut a);
        a
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("1/3").unwrap();
    /// let mut out = Atom::new();
    /// expr.coefficients_to_float_into(2, &mut out);
    /// assert_eq!(out.to_string(), "3.3e-1");
    /// ```
    fn coefficients_to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
        self.as_atom_view()
            .coefficients_to_float_into(decimal_prec, out);
    }

    /// Map all coefficients using a given function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::coefficient::{Coefficient, CoefficientView};
    /// use symbolica::domains::rational::Rational;
    /// let expr = parse!("0.33*x + 3").unwrap();
    /// let out = expr.map_coefficient(|c| match c {
    ///     CoefficientView::Natural(r, d) => {
    ///         Coefficient::Float(Rational::from((r, d)).to_multi_prec_float(53))
    ///     }
    ///     _ => c.to_owned(),
    /// });
    /// assert_eq!(
    ///     out,
    ///     parse!("3.30000000000000e-1*x+3.00000000000000").unwrap()
    /// );
    /// ```
    fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(&self, f: F) -> Atom {
        self.as_atom_view().map_coefficient(f)
    }

    /// Map all coefficients using a given function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::coefficient::{Coefficient, CoefficientView};
    /// use symbolica::domains::rational::Rational;
    /// let expr = parse!("0.33*x + 3").unwrap();
    /// let mut out = Atom::new();
    /// expr.map_coefficient_into(|c| match c {
    ///     CoefficientView::Natural(r, d) => {
    ///         Coefficient::Float(Rational::from((r, d)).to_multi_prec_float(53))
    ///     }
    ///     _ => c.to_owned(),
    /// }, &mut out);
    /// assert_eq!(
    ///     out,
    ///     parse!("3.30000000000000e-1*x+3.00000000000000").unwrap()
    /// );
    /// ```
    fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        self.as_atom_view().map_coefficient_into(f, out);
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("0.333").unwrap();
    /// let result = expr.rationalize_coefficients(&(1, 100).into());
    /// assert_eq!(result, Atom::new_num((1, 3)));
    /// ```
    fn rationalize_coefficients(&self, relative_error: &Rational) -> Atom {
        self.as_atom_view().rationalize_coefficients(relative_error)
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1").unwrap();
    /// let poly = expr.to_polynomial::<_,u8>(&Q, None);
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1").unwrap());
    /// ```
    ///
    /// With explicit variable ordering:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1").unwrap();
    /// let var_map = Arc::new(vec![symbol!("x").into()]);
    /// let poly = expr.to_polynomial::<_,u8>(&Q, Some(var_map));
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1").unwrap());
    /// ```
    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: impl Into<Option<Arc<Vec<Variable>>>>,
    ) -> MultivariatePolynomial<R, E> {
        self.as_atom_view().to_polynomial(field, var_map.into())
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + y*x + x + 1").unwrap();
    /// let var_map = Arc::new(vec![symbol!("x").into()]);
    /// let poly = expr.to_polynomial_in_vars::<u8>(&var_map);
    /// assert_eq!(
    ///     poly.flatten(false),
    ///     parse!("x^2 + (1+y)*x + 1").unwrap()
    /// );
    /// ```
    fn to_polynomial_in_vars<E: Exponent>(
        &self,
        var_map: &Arc<Vec<Variable>>,
    ) -> MultivariatePolynomial<AtomField, E> {
        self.as_atom_view().to_polynomial_in_vars(var_map)
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};;
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)").unwrap();
    /// let rat_poly = expr.to_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(rat_poly.to_expression(), parse!("1+x").unwrap());
    /// ```
    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<Variable>>>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.as_atom_view()
            .to_rational_polynomial(field, out_field, var_map.into())
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)").unwrap();
    /// let fact_rat_poly = expr.to_factorized_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(
    ///     fact_rat_poly.numerator.to_expression(),
    ///     parse!("x+1").unwrap()
    /// );
    /// ```
    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<Variable>>>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.as_atom_view()
            .to_factorized_rational_polynomial(field, out_field, var_map.into())
    }

    /// Format the atom.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::printer::{PrintOptions, PrintState};
    /// let expr = parse!("x + y").unwrap();
    /// let mut output = String::new();
    /// expr.format(&mut output, &PrintOptions::file_no_namespace(), PrintState::default()).unwrap();
    /// assert_eq!(output, "x+y");
    /// ```
    fn format<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, std::fmt::Error> {
        self.as_atom_view().format(fmt, opts, print_state)
    }

    /// Construct a printer for the atom with special options.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::printer::PrintOptions;
    /// let expr = parse!("x^2").unwrap();
    /// let opts = PrintOptions {
    ///     double_star_for_exponentiation: true,
    ///     hide_all_namespaces: true,
    ///    ..Default::default()
    /// };
    /// let printer = expr.printer(opts);
    /// assert_eq!(printer.to_string(), "x**2");
    /// ```
    fn printer(&self, opts: PrintOptions) -> AtomPrinter {
        AtomPrinter::new_with_options(self.as_atom_view(), opts)
    }

    /// Print the atom in a form that is unique and independent of any implementation details.
    ///
    /// Anti-symmetric functions are not supported.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let canonical_str = expr.to_canonical_string();
    /// assert_eq!(canonical_str, "symbolica::x+symbolica::y");
    /// ```
    fn to_canonical_string(&self) -> String {
        self.as_atom_view().to_canonical_string()
    }

    /// Map the function `f` over all terms.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let result = expr.map_terms_single_core(|term| term.expand());
    /// assert_eq!(result, parse!("x + y").unwrap());
    /// ```
    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom) -> Atom {
        self.as_atom_view().map_terms_single_core(f)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let result = expr.map_terms(|term| term.expand(), 4);
    /// assert_eq!(result, parse!("x + y").unwrap());
    /// ```
    fn map_terms(&self, f: impl Fn(AtomView) -> Atom + Send + Sync, n_cores: usize) -> Atom {
        self.as_atom_view().map_terms(f, n_cores)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    /// let result = expr.map_terms_with_pool(|term| term.expand(), &pool);
    /// assert_eq!(result, parse!("x + y").unwrap());
    /// ```
    fn map_terms_with_pool(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync,
        p: &ThreadPool,
    ) -> Atom {
        self.as_atom_view().map_terms_with_pool(f, p)
    }

    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices as the arguments.
    /// Subexpressions, constants and open indices are supported.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide a group marker as the second element in the tuple of the index
    /// specification.
    /// This makes sure that an index will not be renamed to an index from a different group.
    ///
    /// Example
    /// -------
    /// ```
    /// # use symbolica::{atom::{Atom, AtomCore}, symbol, parse};
    /// #
    /// # fn main() {
    /// let _ = symbol!("fs"; Symmetric).unwrap();
    /// let _ = symbol!("fc"; Cyclesymmetric).unwrap();
    /// let a = parse!("fs(mu2,mu3)*fc(mu4,mu2,k1,mu4,k1,mu3)").unwrap();
    ///
    /// let mu1 = parse!("mu1").unwrap();
    /// let mu2 = parse!("mu2").unwrap();
    /// let mu3 = parse!("mu3").unwrap();
    /// let mu4 = parse!("mu4").unwrap();
    ///
    /// let r = a.canonize_tensors(&[(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)]).unwrap();
    /// println!("{}", r);
    /// # }
    /// ```
    /// yields `fs(mu1,mu2)*fc(mu1,k1,mu3,k1,mu2,mu3)`.
    fn canonize_tensors<T: AtomCore, G: Ord + std::hash::Hash>(
        &self,
        indices: &[(T, G)],
    ) -> Result<Atom, String> {
        let indices = indices
            .iter()
            .map(|(i, g)| (i.as_atom_view(), g))
            .collect::<Vec<_>>();
        self.as_atom_view().canonize_tensors(&indices)
    }

    fn to_pattern(&self) -> Pattern {
        Pattern::from_view(self.as_atom_view(), true)
    }

    /// Get all symbols in the expression, optionally including function symbols.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + y").unwrap();
    /// let symbols = expr.get_all_symbols(true);
    /// assert!(symbols.contains(&symbol!("x")));
    /// assert!(symbols.contains(&symbol!("y")));
    /// ```
    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        self.as_atom_view()
            .get_all_symbols(include_function_symbols)
    }

    /// Get all variables and functions in the expression.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + f(x)").unwrap();
    /// let indeterminates = expr.get_all_indeterminates(true);
    /// assert!(indeterminates.contains(&Atom::new_var(symbol!("x")).as_view()));
    /// assert!(indeterminates.contains(&parse!("f(x)").unwrap().as_view()));
    /// ```
    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView> {
        self.as_atom_view().get_all_indeterminates(enter_functions)
    }

    /// Returns true iff `self` contains the symbol `s`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + y").unwrap();
    /// let contains_x = expr.contains_symbol(symbol!("x"));
    /// assert!(contains_x);
    /// ```
    fn contains_symbol(&self, s: Symbol) -> bool {
        self.as_atom_view().contains_symbol(s)
    }

    /// Returns true iff `self` contains `a` literally.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let x = parse!("x").unwrap();
    /// let contains_x = expr.contains(x);
    /// assert!(contains_x);
    /// ```
    fn contains<T: AtomCore>(&self, s: T) -> bool {
        self.as_atom_view().contains(s.as_atom_view())
    }

    /// Check if the expression can be considered a polynomial in some variables, including
    /// redefinitions. For example `f(x)+y` is considered a polynomial in `f(x)` and `y`, whereas
    /// `f(x)+x` is not a polynomial.
    ///
    /// Rational powers or powers in variables are not rewritten, e.g. `x^(2y)` is not considered
    /// polynomial in `x^y`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("f(x) + y").unwrap();
    /// let is_poly = expr.is_polynomial(true, false);
    /// assert!(is_poly.is_some());
    /// ```
    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        self.as_atom_view()
            .is_polynomial(allow_not_expanded, allow_negative_powers)
    }

    /// Replace all occurrences of the pattern. The right-hand side is
    /// either another pattern, or a function that maps the matched wildcards to a new expression.
    ///
    /// # Examples
    ///
    /// Replace all occurrences of `x` with `z`:
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::Pattern;
    /// let expr = parse!("x + y").unwrap();
    /// let pattern = parse!("x").unwrap();
    /// let replacement = parse!("z").unwrap();
    /// let result = expr.replace(pattern).with(replacement);
    /// assert_eq!(result, parse!("z + y").unwrap());
    /// ```
    ///
    /// Set a condition `x_ > 1` (conditions can be chained with `&` and `|`):
    /// ```
    /// use symbolica::id::Pattern;
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1) + f(2) + f(3)").unwrap();
    /// let out = expr
    ///     .replace(parse!("f(x_)").unwrap())
    ///     .when(symbol!("x_").filter(|x| x.to_atom() > 1))
    ///     .with(parse!("f(x_ - 1)").unwrap());
    /// assert_eq!(out, parse!("2*f(1) + f(2)").unwrap());
    /// ```
    ///
    /// Use a map as a right-hand side:
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, function, parse, printer::PrintOptions, symbol};
    /// let (f, x_) = symbol!("f", "x_");
    /// let a = function!(f, 1) * function!(f, 3);
    /// let p = function!(f, x_);
    ///
    /// let r = a.replace(p).with_map(move |m| {
    ///     function!(
    ///         f,
    ///         parse!(&format!(
    ///             "p{}",
    ///             m.get(x_)
    ///                 .unwrap()
    ///                 .to_atom()
    ///                 .printer(PrintOptions::file()),
    ///         ))
    ///         .unwrap()
    ///     )
    /// });
    /// let res = parse!("f(p1)*f(p3)").unwrap();
    /// assert_eq!(r, res);
    /// ```
    ///
    /// Access the match stack to filter for an ascending order of `x`, `y`, `z`:
    /// ```
    /// use symbolica::id::{Condition, ConditionResult, Pattern};
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1, 2, 3)").unwrap();
    /// let out = expr
    ///     .replace(parse!("f(x_,y_,z_)").unwrap())
    ///     .when(Condition::match_stack(|m| {
    ///         if let Some(x) = m.get(symbol!("x")) {
    ///             if let Some(y) = m.get(symbol!("y")) {
    ///                 if x.to_atom() > y.to_atom() {
    ///                     return ConditionResult::False;
    ///                 }
    ///                 if let Some(z) = m.get(symbol!("z")) {
    ///                     if y.to_atom() > z.to_atom() {
    ///                         return ConditionResult::False;
    ///                     }
    ///                 }
    ///                 return ConditionResult::True;
    ///             }
    ///         }
    ///         ConditionResult::Inconclusive
    ///     }))
    ///     .with(parse!("1").unwrap());
    /// assert_eq!(out, parse!("1").unwrap());
    /// ```
    fn replace<'b, P: Into<BorrowedOrOwned<'b, Pattern>>>(
        &self,
        pattern: P,
    ) -> ReplaceBuilder<'_, 'b> {
        self.as_atom_view().replace(pattern)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// To repeatedly replace multiple patterns, wrap the call in [Atom::replace_map].
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::{Pattern, Replacement};
    /// let expr = parse!("x + y").unwrap();
    /// let pattern1 = parse!("x").unwrap().to_pattern();
    /// let replacement1 = parse!("y").unwrap().to_pattern();
    /// let pattern2 = parse!("y").unwrap().to_pattern();
    /// let replacement2 = parse!("x").unwrap().to_pattern();
    /// let result = expr.replace_multiple(&[
    ///     Replacement::new(pattern1, replacement1),
    ///     Replacement::new(pattern2, replacement2),
    /// ]);
    /// assert_eq!(result, parse!("x + y").unwrap());
    /// ```
    fn replace_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Atom {
        self.as_atom_view().replace_multiple(replacements)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// Returns `true` iff a match was found.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::id::{Pattern, Replacement};
    /// let expr = parse!("x + y").unwrap();
    /// let pattern1 = parse!("x").unwrap().to_pattern();
    /// let replacement1 = parse!("y").unwrap().to_pattern();
    /// let pattern2 = parse!("y").unwrap().to_pattern();
    /// let replacement2 = parse!("x").unwrap().to_pattern();
    /// let mut out = Atom::new();
    /// let replacements = [
    ///     Replacement::new(pattern1, replacement1),
    ///     Replacement::new(pattern2, replacement2),
    /// ];
    /// let changed = expr.replace_multiple_into(&replacements, &mut out);
    /// assert!(changed);
    /// assert_eq!(out, parse!("x + y").unwrap());
    /// ```
    fn replace_multiple_into<T: BorrowReplacement>(
        &self,
        replacements: &[T],
        out: &mut Atom,
    ) -> bool {
        self.as_atom_view().replace_multiple_into(replacements, out)
    }

    /// Replace part of an expression by calling the map `m` on each subexpression.
    /// The function `m`  must return `true` if the expression was replaced and must write the new expression to `out`.
    /// A [Context] object is passed to the function, which contains information about the current position in the expression.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y").unwrap();
    /// let result = expr.replace_map(|term, _ctx, out| {
    ///     if term.to_string() == "symbolica::x" {
    ///         *out = parse!("z").unwrap();
    ///         true
    ///     } else {
    ///         false
    ///     }
    /// });
    /// assert_eq!(result, parse!("z + y").unwrap());
    /// ```
    fn replace_map<F: FnMut(AtomView, &Context, &mut Atom) -> bool>(&self, m: F) -> Atom {
        self.as_atom_view().replace_map(m)
    }

    /// Return an iterator over matched expressions.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::id::Pattern;
    /// let expr = parse!("f(1) + f(2)").unwrap();
    /// let pattern = parse!("f(x_)").unwrap().to_pattern();
    /// let mut iter = expr.pattern_match(&pattern, None, None);
    /// let result = iter.next().unwrap();
    /// assert_eq!(
    ///     result.get(&symbol!("x_")).unwrap(),
    ///     &Atom::new_num(1)
    /// );
    /// ```
    fn pattern_match<
        'a: 'b,
        'b,
        C: Into<Option<&'b Condition<PatternRestriction>>>,
        S: Into<Option<&'b MatchSettings>>,
    >(
        &'a self,
        pattern: &'b Pattern,
        conditions: C,
        settings: S,
    ) -> PatternAtomTreeIterator<'a, 'b> {
        PatternAtomTreeIterator::new(
            pattern,
            self.as_atom_view(),
            conditions.into(),
            settings.into(),
        )
    }
}

impl<'a> AtomCore for AtomView<'a> {
    fn as_atom_view(&self) -> AtomView<'a> {
        *self
    }
}

impl AtomCore for InlineVar {
    fn as_atom_view(&self) -> AtomView {
        self.as_view()
    }
}

impl AtomCore for InlineNum {
    fn as_atom_view(&self) -> AtomView {
        self.as_view()
    }
}

impl<T: AsRef<Atom>> AtomCore for T {
    fn as_atom_view(&self) -> AtomView {
        self.as_ref().as_view()
    }
}

impl<'a> AtomCore for AtomOrView<'a> {
    fn as_atom_view(&self) -> AtomView {
        self.as_view()
    }
}
