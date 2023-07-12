from __future__ import annotations
from typing import overload, Iterator, Optional, Sequence, Tuple


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
    def fun(_cls, name: str) -> Function:
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

    def expand(self) -> Expression:
        """
        Expand the expression.
        """

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

    @classmethod
    def expand(_cls, expr: Transformer | Expression | int) -> Transformer:
        """ Expand products and powers. """

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
