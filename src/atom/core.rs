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
        BorrowPatternOrMap, BorrowReplacement, Condition, ConditionResult, Context, MatchSettings,
        Pattern, PatternAtomTreeIterator, PatternRestriction, ReplaceIterator,
    },
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, series::Series,
        Exponent, PositiveExponent, Variable,
    },
    printer::{AtomPrinter, PrintOptions, PrintState},
    state::Workspace,
    tensors::matrix::Matrix,
};
use std::sync::Arc;

use super::{
    representation::{InlineNum, InlineVar},
    Atom, AtomOrView, AtomView, Symbol,
};

/// All core features of expressions, such as expansion and
/// pattern matching that leave the expression unchanged.
pub trait AtomCore {
    /// Take a view of the atom.
    fn as_atom_view(&self) -> AtomView;

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    fn collect<E: Exponent, T: AtomCore>(
        &self,
        x: T,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view().collect::<E, T>(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    fn collect_multiple<E: Exponent, T: AtomCore>(
        &self,
        xs: &[T],
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view()
            .collect_multiple::<E, T>(xs, key_map, coeff_map)
    }

    /// Collect terms involving the same power of `x` in `xs`, where `xs` is a list of indeterminates.
    /// Return the list of key-coefficient pairs
    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        self.as_atom_view().coefficient_list::<E, T>(xs)
    }

    /// Collect terms involving the literal occurrence of `x`.
    fn coefficient<T: AtomCore>(&self, x: T) -> Atom {
        Workspace::get_local().with(|ws| {
            self.as_atom_view()
                .coefficient_with_ws(x.as_atom_view(), ws)
        })
    }

    /// Write the expression over a common denominator.
    fn together(&self) -> Atom {
        self.as_atom_view().together()
    }

    /// Write the expression as a sum of terms with minimal denominators.
    fn apart(&self, x: Symbol) -> Atom {
        self.as_atom_view().apart(x)
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    fn cancel(&self) -> Atom {
        self.as_atom_view().cancel()
    }

    /// Factor the expression over the rationals.
    fn factor(&self) -> Atom {
        self.as_atom_view().factor()
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    fn collect_num(&self) -> Atom {
        self.as_atom_view().collect_num()
    }

    /// Expand an expression. The function [AtomCore::expand_via_poly] may be faster.
    fn expand(&self) -> Atom {
        self.as_atom_view().expand()
    }

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    fn expand_via_poly<E: Exponent, T: AtomCore>(&self, var: Option<T>) -> Atom {
        self.as_atom_view()
            .expand_via_poly::<E>(var.as_ref().map(|x| x.as_atom_view()))
    }

    /// Expand an expression in the variable `var`. The function [AtomCore::expand_via_poly] may be faster.
    fn expand_in<T: AtomCore>(&self, var: T) -> Atom {
        self.as_atom_view().expand_in(var.as_atom_view())
    }

    /// Expand an expression in the variable `var`.
    fn expand_in_symbol(&self, var: Symbol) -> Atom {
        self.as_atom_view()
            .expand_in(InlineVar::from(var).as_view())
    }

    /// Expand an expression, returning `true` iff the expression changed.
    fn expand_into(&self, var: Option<AtomView>, out: &mut Atom) -> bool {
        self.as_atom_view().expand_into(var, out)
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    fn expand_num(&self) -> Atom {
        self.as_atom_view().expand_num()
    }

    /// Check if the expression is expanded, optionally in only the variable or function `var`.
    fn is_expanded(&self, var: Option<AtomView>) -> bool {
        self.as_atom_view().is_expanded(var)
    }

    /// Take a derivative of the expression with respect to `x`.
    fn derivative(&self, x: Symbol) -> Atom {
        self.as_atom_view().derivative(x)
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        self.as_atom_view().derivative_into(x, out)
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
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
    fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, String> {
        AtomView::solve_linear_system::<E, T1, T2>(system, vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
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
    fn evaluate<'b, T: Real, F: Fn(&Rational) -> T + Copy>(
        &'b self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        cache: &mut HashMap<AtomView<'b>, T>,
    ) -> Result<T, String> {
        self.as_atom_view()
            .evaluate(coeff_map, const_map, function_map, cache)
    }

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTree<Rational>, String> {
        self.as_atom_view().to_evaluation_tree(fn_map, params)
    }

    /// Create an efficient evaluator for a (nested) expression.
    /// All free parameters must appear in `params` and all other variables
    /// and user functions in the expression must occur in the function map.
    /// The function map may have nested expressions.
    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
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
    fn evaluator_multiple<'a>(
        exprs: &[AtomView<'a>],
        fn_map: &FunctionMap<'a, Rational>,
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
    fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        self.as_atom_view().zero_test(iterations, tolerance)
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Atom {
        self.as_atom_view().set_coefficient_ring(vars)
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    fn coefficients_to_float(&self, decimal_prec: u32) -> Atom {
        let mut a = Atom::new();
        self.as_atom_view()
            .coefficients_to_float_into(decimal_prec, &mut a);
        a
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    fn coefficients_to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
        self.as_atom_view()
            .coefficients_to_float_into(decimal_prec, out);
    }

    /// Map all coefficients using a given function.
    fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(&self, f: F) -> Atom {
        self.as_atom_view().map_coefficient(f)
    }

    /// Map all coefficients using a given function.
    fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        self.as_atom_view().map_coefficient_into(f, out);
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    fn rationalize_coefficients(&self, relative_error: &Rational) -> Atom {
        self.as_atom_view().rationalize_coefficients(relative_error)
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> MultivariatePolynomial<R, E> {
        self.as_atom_view().to_polynomial(field, var_map)
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
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
    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.as_atom_view()
            .to_rational_polynomial(field, out_field, var_map)
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.as_atom_view()
            .to_factorized_rational_polynomial(field, out_field, var_map)
    }

    // Format the atom.
    fn format<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, std::fmt::Error> {
        self.as_atom_view().format(fmt, opts, print_state)
    }

    /// Construct a printer for the atom with special options.
    fn printer<'a>(&'a self, opts: PrintOptions) -> AtomPrinter<'a> {
        AtomPrinter::new_with_options(self.as_atom_view(), opts)
    }

    /// Print the atom in a form that is unique and independent of any implementation details.
    ///     
    /// Anti-symmetric functions are not supported.
    fn to_canonical_string(&self) -> String {
        self.as_atom_view().to_canonical_string()
    }

    /// Map the function `f` over all terms.
    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom) -> Atom {
        self.as_atom_view().map_terms_single_core(f)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    fn map_terms(&self, f: impl Fn(AtomView) -> Atom + Send + Sync, n_cores: usize) -> Atom {
        self.as_atom_view().map_terms(f, n_cores)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    fn map_terms_with_pool(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync,
        p: &ThreadPool,
    ) -> Atom {
        self.as_atom_view().map_terms_with_pool(f, p)
    }

    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices are the arguments.
    /// The repeated indices should be provided in `contracted_indices`.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide an optional group marker for each index using `index_group`.
    /// This makes sure that an index will not be renamed to an index from a different group.
    ///
    /// Example
    /// -------
    /// ```
    /// # use symbolica::{atom::{Atom, AtomCore}, state::{FunctionAttribute, State}};
    /// #
    /// # fn main() {
    /// let _ = State::get_symbol_with_attributes("fs", &[FunctionAttribute::Symmetric]).unwrap();
    /// let _ = State::get_symbol_with_attributes("fc", &[FunctionAttribute::Cyclesymmetric]).unwrap();
    /// let a = Atom::parse("fs(mu2,mu3)*fc(mu4,mu2,k1,mu4,k1,mu3)").unwrap();
    ///
    /// let mu1 = Atom::parse("mu1").unwrap();
    /// let mu2 = Atom::parse("mu2").unwrap();
    /// let mu3 = Atom::parse("mu3").unwrap();
    /// let mu4 = Atom::parse("mu4").unwrap();
    ///
    /// let r = a.canonize_tensors(&[mu1.as_view(), mu2.as_view(), mu3.as_view(), mu4.as_view()], None).unwrap();
    /// println!("{}", r);
    /// # }
    /// ```
    /// yields `fs(mu1,mu2)*fc(mu1,k1,mu3,k1,mu2,mu3)`.
    fn canonize_tensors(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView]>,
    ) -> Result<Atom, String> {
        self.as_atom_view()
            .canonize_tensors(contracted_indices, index_group)
    }

    fn to_pattern(&self) -> Pattern {
        Pattern::from_view(self.as_atom_view(), true)
    }

    /// Get all symbols in the expression, optionally including function symbols.
    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        self.as_atom_view()
            .get_all_symbols(include_function_symbols)
    }

    /// Get all variables and functions in the expression.
    fn get_all_indeterminates<'a>(&'a self, enter_functions: bool) -> HashSet<AtomView<'a>> {
        self.as_atom_view().get_all_indeterminates(enter_functions)
    }

    /// Returns true iff `self` contains the symbol `s`.
    fn contains_symbol(&self, s: Symbol) -> bool {
        self.as_atom_view().contains_symbol(s)
    }

    /// Returns true iff `self` contains `a` literally.
    fn contains<T: AtomCore>(&self, s: T) -> bool {
        self.as_atom_view().contains(s.as_atom_view())
    }

    /// Check if the expression can be considered a polynomial in some variables, including
    /// redefinitions. For example `f(x)+y` is considered a polynomial in `f(x)` and `y`, whereas
    /// `f(x)+x` is not a polynomial.
    ///
    /// Rational powers or powers in variables are not rewritten, e.g. `x^(2y)` is not considered
    /// polynomial in `x^y`.
    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        self.as_atom_view()
            .is_polynomial(allow_not_expanded, allow_negative_powers)
    }

    /// Replace all occurrences of the pattern.
    fn replace_all<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Atom {
        self.as_atom_view()
            .replace_all(pattern, rhs, conditions, settings)
    }

    /// Replace all occurrences of the pattern.
    fn replace_all_into<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
        out: &mut Atom,
    ) -> bool {
        self.as_atom_view()
            .replace_all_into(pattern, rhs, conditions, settings, out)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    fn replace_all_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Atom {
        self.as_atom_view().replace_all_multiple(replacements)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// Returns `true` iff a match was found.
    fn replace_all_multiple_into<T: BorrowReplacement>(
        &self,
        replacements: &[T],
        out: &mut Atom,
    ) -> bool {
        self.as_atom_view()
            .replace_all_multiple_into(replacements, out)
    }

    /// Replace part of an expression by calling the map `m` on each subexpression.
    /// The function `m`  must return `true` if the expression was replaced and must write the new expression to `out`.
    /// A [Context] object is passed to the function, which contains information about the current position in the expression.
    fn replace_map<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&self, m: &F) -> Atom {
        self.as_atom_view().replace_map(m)
    }

    /// Return an iterator that replaces the pattern in the target once.
    fn replace_iter<'a, R: BorrowPatternOrMap>(
        &'a self,
        pattern: &'a Pattern,
        rhs: &'a R,
        conditions: Option<&'a Condition<PatternRestriction>>,
        settings: Option<&'a MatchSettings>,
    ) -> ReplaceIterator<'a, 'a> {
        ReplaceIterator::new(
            pattern,
            self.as_atom_view(),
            rhs.borrow(),
            conditions,
            settings,
        )
    }

    /// Return an iterator over matched expressions.
    fn pattern_match<'a>(
        &'a self,
        pattern: &'a Pattern,
        conditions: Option<&'a Condition<PatternRestriction>>,
        settings: Option<&'a MatchSettings>,
    ) -> PatternAtomTreeIterator<'a, 'a> {
        PatternAtomTreeIterator::new(pattern, self.as_atom_view(), conditions, settings)
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
