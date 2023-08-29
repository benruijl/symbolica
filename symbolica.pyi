from __future__ import annotations
from typing import Callable, overload, Iterator, Optional, Sequence, Tuple, List


class Expression:
    @classmethod
    def var(_cls, name: str) -> Expression:
        """
        Creates a Symbolica expression that is a single variable.

        Examples
        --------
        >>> var_x = Expression.var('x')
        >>> print(var_x)
        x
        """

    @classmethod
    def vars(_cls, *names: str) -> Sequence[Expression]:
        """
        Create a Symbolica variable for every name in `*names`.
        """

    @classmethod
    def fun(_cls, name: str, is_symmetric: bool = False) -> Function:
        """
        Creates a new Symbolica function with a given name.

        Examples
        --------
        >>> f = Expression.fun('f')
        >>> e = f(1,2)
        >>> print(e)
        f(1,2)
        """

    @classmethod
    def funs(_cls, *names: str) -> Sequence[Function]:
        """
        Create a Symbolica function for every name in `*names`.
        """

    @classmethod
    def parse(_cls, input: str) -> Expression:
        """
        Parse a Symbolica expression from a string.

        Parameters
        ----------
        input: str
            An input string. UTF-8 character are allowed.

        Examples
        --------
        >>> e = Expression.parse('x^2+y+y*4')
        >>> print(e)
        x^2+5*y

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica expression.
        """

    def __copy__(self) -> Expression:
        """
        Copy the expression.
        """

    def __str__(self) -> str:
        """
        Convert the expression into a human-readable string.
        """

    def __repr__(self) -> str:
        """
        Convert the expression into a debug string.
        """

    def __add__(self, other: Expression | int) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __radd__(self, other: Expression | int) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __sub__(self, other: Expression | int) -> Expression:
        """
        Subtract `other` from this expression, returning the result.
        """

    def __rsub__(self, other: Expression | int) -> Expression:
        """
        Subtract this expression from `other`, returning the result.
        """

    def __mul__(self, other: Expression | int) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __rmul__(self, other: Expression | int) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __truediv__(self, other: Expression | int) -> Expression:
        """
        Divide this expression by `other`, returning the result.
        """

    def __rtruediv__(self, other: Expression | int) -> Expression:
        """
        Divide `other` by this expression, returning the result.
        """

    def __pow__(self, exp: Expression | int) -> Expression:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: Expression | int) -> Expression:
        """
        Take `base` to power `self`, returning the result.
        """

    def __neg__(self) -> Expression:
        """
        Negate the current expression, returning the result.
        """

    def __len__(self) -> int:
        """
        Return the number of terms in this expression.
        """

    def transform(self) -> Transformer:
        """
        Convert the input to a transformer, on which subsequent
        transformations can be applied.
        """

    def set_coefficient_ring(self, vars: Sequence[Expression]) -> Expression:
        """
        Set the coefficient ring to contain the variables in the `vars` list.
        This will move all variables into a rational polynomial function.

        Parameters
        ----------
        vars : List[Expression]
                A list of variables
        """

    def len(self, min_length: int, max_length: int | None) -> PatternRestriction:
        """
        Create a pattern restriction on a pattern based on length.
        """

    def is_var(self) -> PatternRestriction:
        """
        Create a pattern restriction that tests if a wildcard is a variable.
        """

    def is_lit(self) -> PatternRestriction:
        """
        Create a pattern restriction that treats the wildcard as a literal variable,
        so that it only matches to itself.
        """

    def is_num(self) -> PatternRestriction:
        """
        Create a pattern restriction that tests if a wildcard is a number.
        """

    def __richcmp__(self, other: int, op: CompareOp) -> PatternRestriction:
        """
        Create a pattern restriction based on a comparison of a wildcard with a number.
        """

    def __iter__(self) -> Iterator[Expression]:
        """
        Create an iterator over all atoms in the expression.
        """

    def map(
        self,
        transformations: Transformer,
    ) -> Expression:
        """
        Map the transformations to every term in the expression.
        The execution happen in parallel.

        Examples
        --------
        >>> x, x_ = Expression.vars('x', 'x_')
        >>> e = (1+x)**2
        >>> r = e.map(Transformer().expand().replace_all(x, 6))
        >>> print(r)
        """

    def expand(self) -> Expression:
        """
        Expand the expression.
        """

    def derivative(self, x: Expression) -> Expression:
        """ Derive the expression w.r.t the variable `x`. """

    def to_rational_polynomial(
        self,
        vars: Optional[Sequence[Expression]] = None,
    ) -> RationalPolynomial:
        """
        Convert the expression to a rational polynomial, optionally, with the variables specified in `vars`.
        The latter is useful if it is known in advance that more variables may be added in the future to the
        rational polynomial through composition with other rational polynomials.

        Examples
        --------
        >>> a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
        >>> print(a)
        """

    def match(
        self,
        lhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
    ) -> MatchIterator:
        """
        Return an iterator over the pattern `self` matching to `lhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> x, x_ = Expression.vars('x','x_')
        >>> f = Expression.fun('f')
        >>> e = f(x)*f(1)*f(2)*f(3)
        >>> for match in e.match(f(x_)):
        >>>    for map in match:
        >>>        print(map[0],'=', map[1])
        """

    def replace_all(
        self,
        lhs: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
    ) -> Expression:
        """
        Replace all patterns matching the left-hand side `lhs` by the right-hand side `rhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
        >>> print(r)
        """


class PatternRestriction:
    """ A restriction on wildcards. """

    def __and__(self, other: PatternRestriction) -> PatternRestriction:
        """ Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold)."""


class CompareOp:
    """ One of the following comparison operators: `<`,`>`,`<=`,`>=`,`==`,`!=`."""


class Function:
    """ A Symbolica function. Will turn into an expression or a transformer when called with arguments."""

    @overload
    def __call__(self, *args: Expression | int) -> Expression:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x = Expression.vars('x')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> print(e)
        f(3,x)
        """

    @overload
    def __call__(self, *args: Transformer) -> Transformer:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x = Expression.vars('x')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> print(e)
        f(3,x)
        """


class Transformer:
    """ Transform an expression. """

    def __init__() -> Transformer:
        """ Create a new transformer for a term provided by `Expression.map`. """

    def expand(self) -> Transformer:
        """ Expand products and powers. """

    def derivative(self, x: Transformer | Expression) -> Transformer:
        """ Create a transformer that derives `self` w.r.t the variable `x`. """

    def replace_all(
        self,
        pat: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
    ) -> Transformer:
        """
        Create a transformer that replaces all patterns matching the left-hand side `self` by the right-hand side `rhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> r = e.transform().replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
        >>> print(r)
        """

    def __add__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __radd__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __sub__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Subtract `other` from this transformer, returning the result.
        """

    def __rsub__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Subtract this transformer from `other`, returning the result.
        """

    def __mul__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __rmul__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __truediv__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Divide this transformer by `other`, returning the result.
        """

    def __rtruediv__(self, other: Transformer | Expression | int) -> Transformer:
        """
        Divide `other` by this transformer, returning the result.
        """

    def __pow__(self, exp: Transformer | Expression | int) -> Transformer:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: Transformer | Expression | int) -> Transformer:
        """
        Take `base` to power `self`, returning the result.
        """

    def __neg__(self) -> Transformer:
        """
        Negate the current transformer, returning the result.
        """


class MatchIterator:
    """ An iterator over matches. """

    def __iter__(self) -> MatchIterator:
        """ Create the iterator. """

    def __next__(self) -> Sequence[Tuple[Expression, Expression]]:
        """ Return the next match. """


class RationalPolynomial:
    """ A Symbolica rational polynomial. """

    def __copy__(self) -> RationalPolynomial:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def __repr__(self) -> str:
        """Print the rational polynomial in a debug representation."""

    def __add__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Add two rational polynomials `self and `rhs`, returning the result."""

    def __sub__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Multiply two rational polynomials `self and `rhs`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> RationalPolynomial:
        """Negate the rational polynomial."""

    def gcd(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""


class NumericalIntegrator:
    def continuous(n_dims: int, n_bins: int = 128,
                   min_samples_for_update: int = 100,
                   bin_number_evolution: List[int] = None,
                   train_on_avg: bool = False) -> NumericalIntegrator:
        """ Create a new continuous grid for the numerical integrator."""

    def discrete(bins: List[Optional[NumericalIntegrator]],
                 max_prob_ratio: float = 100.,
                 train_on_avg: bool = False) -> NumericalIntegrator:
        """ Create a new discrete grid for the numerical integrator. Each
        bin can have a sub-grid.

        Examples
        --------
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         if sample.d[0] == 0:
        >>>             res.append(sample.c[0]**2)
        >>>         else:
        >>>             res.append(sample.c[0]**1/2)
        >>>     return res
        >>> 
        >>> integrator = NumericalIntegrator.discrete(
        >>>     [NumericalIntegrator.continuous(1), NumericalIntegrator.continuous(1)])
        >>> integrator.integrate(integrand, True, 10, 10000)
        """

    def sample(self, num_samples: int) -> List[Sample]:
        """Sample `num_samples` points from the grid."""

    def add_training_samples(self, samples: List[Sample], evals: List[float]):
        """Add the samples and their corresponding function evaluations to the grid.
        Call `update` after to update the grid and to obtain the new expected value for the integral."""

    def update(self, learning_rate: float) -> Tuple[float, float, float]:
        """Update the grid using the `learning_rate`. 
        Examples
        --------
        >>> from symbolica import NumericalIntegrator, Sample
        >>>
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         res.append(sample.c[0]**2+sample.c[1]**2)
        >>>     return res
        >>> 
        >>> integrator = NumericalIntegrator.continuous(2)
        >>> for i in range(10):
        >>>     samples = integrator.sample(10000 + i * 1000)
        >>>     res = integrand(samples)
        >>>     integrator.add_training_samples(samples, res)
        >>>     avg, err, chi_sq = integrator.update(1.5)
        >>>     print('Iteration {}: {:.6} +- {:.6}, chi={:.6}'.format(i+1, avg, err, chi_sq))
        """

    def integrate(self,
                  integrand: Callable[[List[Sample]], List[float]],
                  max_n_iter: int = 10000000,
                  min_error: float = 0.01,
                  n_samples_per_iter: int = 10000, show_stats: bool = True) -> Tuple[float, float, float]:
        """ Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
        The return value is the average, the statistical error, and chi-squared of the integral.

        With `show_stats=True`, intermediate statistics will be printed. `max_n_iter` determines the number
        of iterations and `n_samples_per_iter` determine the number of samples per iteration. This is
        the same amount of samples that the integrand function will be called with.

        For more flexibility, use `sample`, `add_training_samples` and `update`. See `update` for an example.

        Examples
        --------
        >>> from symbolica import NumericalIntegrator, Sample
        >>>
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         res.append(sample.c[0]**2+sample.c[1]**2)
        >>>     return res
        >>>
        >>> avg, err = NumericalIntegrator.continuous(2).integrate(integrand, True, 10, 100000)
        >>> print('Result: {} +- {}'.format(avg, err))
        """


class Sample:
    """ A sample from the Symbolica integrator. It could consist of discrete layers,
     accessible with `d` (empty when there are not discrete layers), and the final continous layer `c` if it is present. """

    """ The weights the integrator assigned to this sample point, given in descending order:
    first the discrete layer weights and then the continuous layer weight."""
    weights: List[float]
    d: List[int]
    """ A sample point per (nested) discrete layer. Empty if not present."""
    c: List[float]
    """ A sample in the continuous layer. Empty if not present."""
