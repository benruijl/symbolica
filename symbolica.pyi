"""
Symbolica Python API.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, overload, Iterator, Optional, Sequence, Tuple, List


def is_licensed() -> bool:
    """Check if the current Symbolica instance has a valid license key set."""


def set_license_key(key: str) -> None:
    """Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions."""


def request_hobbyist_license(name: str, email: str) -> None:
    """Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address `email`."""


def request_trial_license(name: str, email: str, company: str) -> None:
    """Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address `email`."""


class AtomType(Enum):
    """Specifies the type of the atom."""
    Num = 1,
    Var = 2,
    Fn = 3,
    Add = 4,
    Mul = 5,
    Pow = 6,


class AtomTree:
    """
    A Python representation of a Symbolica expression.
    The type of the atom is provided in `atom_type`.

    The `head` contains the string representation of:
    - a number if the type is `Num`
    - the variable if the type is `Var`
    - the function name if the type is `Fn`
    - otherwise it is `None`.

    The tail contains the child atoms:
    - the summand for type `Add`
    - the factors for type `Mul`
    - the base and exponent for type `Pow`
    - the function arguments for type `Fn`
    """

    atom_type: AtomType
    """ The type of this atom."""
    head: Optional[str]
    """The string data of this atom."""
    tail: List[AtomTree]
    """The list of child atoms of this atom."""


class Expression:
    """
    A Symbolica expression.

    Supports standard arithmetic operations, such
    as addition and multiplication.

    Examples
    --------
    >>> x = Expression.var('x')
    >>> e = x**2 + 2 - x + 1 / x**4
    >>> print(e)
    """

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

    def pretty_str(self,
                   terms_on_new_line: bool = False,
                   color_top_level_sum: bool = True,
                   color_builtin_functions: bool = True,
                   print_finite_field: bool = True,
                   explicit_rational_polynomial: bool = False,
                   number_thousands_separator: Optional[str] = None,
                   multiplication_operator: str = '*',
                   square_brackets_for_function: bool = False,
                   num_exp_as_superscript: bool = True,
                   latex: bool = False
                   ) -> str:
        """
        Convert the expression into a human-readable string, with tunable settings.

        Examples
        --------
        >>> a = Expression.parse('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
        >>> print(a.pretty_str(number_thousands_separator='_', multiplication_operator=' '))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def to_latex(self) -> str:
        """
        Convert the expression into a LaTeX string.

        Examples
        --------
        >>> a = Expression.parse('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
        >>> print(a.to_latex())

        Yields `$$z^{34}+x^{x+2}+y^{4}+f(x,x^{2})+128378127123 z^{\\frac{2}{3}} w^{2} \\frac{1}{x} \\frac{1}{y}+\\frac{3}{5}$$`.
        """

    def __repr__(self) -> str:
        """
        Convert the expression into a debug string.
        """

    def __hash__(self) -> str:
        """
        Hash the expression.
        """

    def get_type(self) -> AtomType:
        """ Get the type of the atom."""

    def to_atom_tree(self) -> AtomTree:
        """Convert the expression to a tree."""

    def get_name(self) -> Optional[str]:
        """
        Get the name of a variable or function if the current atom
        is a variable or function.
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

    def __xor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
        """

    def __rxor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
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

    def req_len(self, min_length: int, max_length: int | None) -> PatternRestriction:
        """
        Create a pattern restriction based on the wildcard length before downcasting.
        """

    def req_type(self, atom_type: AtomType) -> PatternRestriction:
        """
        Create a pattern restriction that tests the type of the atom.

        Examples
        --------
        >>> from symbolica import Expression, AtomType
        >>> x, x_ = Expression.vars('x', 'x_')
        >>> f = Expression.fun("f")
        >>> e = f(x)*f(2)*f(f(3))
        >>> e = e.replace_all(f(x_), 1, x_.req_type(AtomType.Num))
        >>> print(e)

        Yields `f(x)*f(1)`.
        """

    def req_lit(self) -> PatternRestriction:
        """
        Create a pattern restriction that treats the wildcard as a literal variable,
        so that it only matches to itself.
        """

    def req(self, filter_fn: Callable[[Expression], bool],) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `filter_fn` with the matched
        atom that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun("f")
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))
        """

    def req_cmp(
        self,
        other: Expression | int,
        cmp_fn: Callable[[Expression, Expression], bool],
    ) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `cmp_fn` with another the matched
        atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.vars('x_', 'y_')
        >>> f = Expression.fun("f")
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))
        """

    def __eq__(self, other: Expression | int) -> bool:
        """
        Compare two expressions.
        """

    def __neq__(self, other: Expression | int) -> bool:
        """
        Compare two expressions.
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

    def set_coefficient_ring(self, vars: Sequence[Expression]) -> Expression:
        """
        Set the coefficient ring to contain the variables in the `vars` list.
        This will move all variables into a rational polynomial function.

        Parameters
        ----------
        vars : List[Expression]
                A list of variables
        """

    def expand(self) -> Expression:
        """
        Expand the expression.
        """

    def derivative(self, x: Expression) -> Expression:
        """ Derive the expression w.r.t the variable `x`. """

    def to_polynomial(self,  vars: Optional[Sequence[Expression]] = None) -> Polynomial:
        """Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`."""

    def to_rational_polynomial(
        self,
        vars: Optional[Sequence[Expression]] = None,
    ) -> RationalPolynomial:
        """
        Convert the expression to a rational polynomial, optionally, with the variables and the ordering specified in `vars`.
        The latter is useful if it is known in advance that more variables may be added in the future to the
        rational polynomial through composition with other rational polynomials.

        Examples
        --------
        >>> a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
        >>> print(a)
        """

    def to_rational_polynomial_small_exponent(
        self,
        vars: Optional[Sequence[Expression]] = None,
    ) -> RationalPolynomial:
        """Similar to `to_rational_polynomial()`, but the power of each variable is limited to 255."""

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

    def replace(
        self,
        lhs: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
    ) -> ReplaceIterator:
        """
        Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(1)*f(2)*f(3)
        >>> for r in e.replace(f(x_), f(x_ + 1)):
        >>>     print(r)

        Yields:
        ```
        f(2)*f(2)*f(3)
        f(1)*f(3)*f(3)
        f(1)*f(2)*f(4)
        ```
        """

    def replace_all(
        self,
        pattern: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
    ) -> Expression:
        """
        Replace all atoms matching the pattern `pattern` by the right-hand side `rhs`.
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

    @staticmethod
    def __new__(name: str, is_symmetric: Optional[bool]) -> Function:
        """ 
        Create a new function from a `name`. Can be turned into a symmetric function
        using `is_symmetric=True`.
        """

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
    """ Operations that transform an expression. """

    def __init__() -> Transformer:
        """ Create a new transformer for a term provided by `Expression.map`. """

    def expand(self) -> Transformer:
        """ Create a transformer that expands products and powers.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x, x_ = Expression.vars('x', 'x_')
        >>> f = Expression.fun('f')
        >>> e = f((x+1)**2).replace_all(f(x_), x_.transform().expand())
        >>> print(e)
        """

    def prod(self) -> Transformer:
        """ Create a transformer that computes the product of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(2,3).replace_all(f(x_), x_.transform().prod())
        >>> print(e)
        """

    def sum(self) -> Transformer:
        """ Create a transformer that computes the sum of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(2,3).replace_all(f(x_), x_.transform().sum())
        >>> print(e)
        """

    def sort(self) -> Transformer:
        """ Create a transformer that sorts a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(3,2,1).replace_all(f(x_), x_.transform().sort())
        >>> print(e)
        """

    def split(self) -> Transformer:
        """ Create a transformer that split a sum or product into a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x, x_ = Expression.vars('x', 'x_')
        >>> f = Expression.fun('f')
        >>> e = (x + 1).replace_all(x_, f(x_.transform().split()))
        >>> print(e)
        """

    def partitions(
        self,
        bins: Sequence[Tuple[Transformer | Expression, int]],
        fill_last: bool = False,
        repeat: bool = False,
    ) -> Transformer:
        """ Create a transformer partition a list of arguments into bins of sets with a given tag and the length of the set,
        returning all partitions and their multiplicity.
        
        If the set `elements` is larger than the bins, setting the flag `fill_last`
        will add all remaining elements to the last set.
        
        Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
        if possible.
        
        Note that the functions names to be provided for the bin names must be generated through `Expression.var`.
        
        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_, f_id, g_id = Expression.vars('x_', 'f', 'g')
        >>> f = Expression.fun('f')
        >>> e = f(1,2,1,3).replace_all(f(x_), x_.transform().partitions([(f_id, 2), (g_id, 1), (f_id, 1)]))
        >>> print(e)
        
        yields:
        ```
        2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)
        ```
        """

    def map(self, f: Callable[[Expression | int], Expression]) -> Transformer:
        """ Create a transformer that expands products and powers.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(2).replace_all(f(x_), x_.transform().map(lambda r: r**2))
        >>> print(e)
        """

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

    def __xor__(self, a: Any) -> Transformer:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
        """

    def __rxor__(self, a: Any) -> Transformer:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
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


class ReplaceIterator:
    """ An iterator over single replacments. """

    def __iter__(self) -> ReplaceIterator:
        """ Create the iterator. """

    def __next__(self) -> Expression:
        """ Return the next replacement. """


class Polynomial:
    """ A Symbolica polynomial with rational coefficients. """

    @classmethod
    def parse(_cls, input: str, vars: List[str]) -> Polynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> Polynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def __repr__(self) -> str:
        """Print the polynomial in a debug representation."""

    def __add__(self, rhs: Polynomial) -> Polynomial:
        """Add two polynomials `self and `rhs`, returning the result."""

    def __sub__(self, rhs: Polynomial) -> Polynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: Polynomial) -> Polynomial:
        """Multiply two polynomials `self and `rhs`, returning the result."""

    def __truediv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: Polynomial) -> Polynomial:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __neg__(self) -> Polynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: Polynomial) -> Polynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def to_integer_polynomial(self) -> IntegerPolynomial:
        """Convert the polynomial to a polynomial with integer coefficients, if possible."""

    def optimize(self, iterations: int = 1000, to_file: str | None = None) -> Evaluator:
        """
        Optimize the polynomial for evaluation using `iterations` number of iterations.
        The optimized output can be exported in a C++ format using `to_file`.

        Returns an evaluator for the polynomial.
        """


class IntegerPolynomial:
    """ A Symbolica polynomial with integer coefficients. """

    @classmethod
    def parse(_cls, input: str, vars: List[str]) -> Polynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> IntegerPolynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def __repr__(self) -> str:
        """Print the polynomial in a debug representation."""

    def __add__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Add two polynomials `self and `rhs`, returning the result."""

    def __sub__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Multiply two polynomials `self and `rhs`, returning the result."""

    def __truediv__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: Polynomial) -> Polynomial:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __neg__(self) -> IntegerPolynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""


class RationalPolynomial:
    """ A Symbolica rational polynomial. """

    @staticmethod
    def __new__(num: Polynomial, den: Polynomial) -> RationalPolynomial:
        """Create a new rational polynomial from a numerator and denominator polynomial."""

    @classmethod
    def parse(_cls, input: str, vars: List[str]) -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __copy__(self) -> RationalPolynomial:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the rational polynomial into a LaTeX string."""

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


class RationalPolynomialSmallExponent:
    """ A Symbolica rational polynomial with variable powers limited to 255. """

    @classmethod
    def parse(_cls, input: str, vars: List[str]) -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __copy__(self) -> RationalPolynomialSmallExponent:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the rational polynomial into a LaTeX string."""

    def __repr__(self) -> str:
        """Print the rational polynomial in a debug representation."""

    def __add__(self, rhs: RationalPolynomialSmallExponent) -> RationalPolynomialSmallExponent:
        """Add two rational polynomials `self and `rhs`, returning the result."""

    def __sub__(self, rhs: RationalPolynomialSmallExponent) -> RationalPolynomialSmallExponent:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: RationalPolynomialSmallExponent) -> RationalPolynomialSmallExponent:
        """Multiply two rational polynomials `self and `rhs`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomialSmallExponent) -> RationalPolynomialSmallExponent:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> RationalPolynomialSmallExponent:
        """Negate the rational polynomial."""

    def gcd(self, rhs: RationalPolynomialSmallExponent) -> RationalPolynomialSmallExponent:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

class Evaluator:
    def evaluate(self, inputs: List[List[float]]) -> List[float]:
        """Evaluate the polynomial for multiple inputs and return the result."""

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
