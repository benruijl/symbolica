"""
Symbolica Python API.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, overload, Iterator, Optional, Sequence, Tuple, List
from decimal import Decimal


def get_version() -> str:
    """Get the current Symbolica version."""


def is_licensed() -> bool:
    """Check if the current Symbolica instance has a valid license key set."""


def set_license_key(key: str) -> None:
    """Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions."""


def request_hobbyist_license(name: str, email: str) -> None:
    """Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address `email`."""


def request_trial_license(name: str, email: str, company: str) -> None:
    """Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address `email`."""


def request_sublicense(name: str, email: str, company: str, super_licence: str) -> None:
    """Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    The key will be sent to the e-mail address `email`."""


def get_license_key(email: str) -> str:
    """Get the license key for the account registered with the provided email address."""


@overload
def S(name: str,
      is_symmetric: Optional[bool] = None,
      is_antisymmetric: Optional[bool] = None,
      is_cyclesymmetric: Optional[bool] = None,
      is_linear: Optional[bool] = None,
      custom_normalization: Optional[Transformer] = None) -> Expression:
    """
    Create new symbols from `names`. Symbols can have attributes,
    such as symmetries. If no attributes
    are specified and the symbol was previously defined, the attributes are inherited.
    Once attributes are defined on a symbol, they cannot be redefined later.

    Examples
    --------
    Define a regular symbol and use it as a variable:
    >>> x = S('x')
    >>> e = x**2 + 5
    >>> print(e)
    x**2 + 5

    Define a regular symbol and use it as a function:
    >>> f = S('f')
    >>> e = f(1,2)
    >>> print(e)
    f(1,2)


    Define a symmetric function:
    >>> f = S('f', is_symmetric=True)
    >>> e = f(2,1)
    >>> print(e)
    f(1,2)


    Define a linear and symmetric function:
    >>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
    >>> dot = S('dot', is_symmetric=True, is_linear=True)
    >>> e = dot(p2+2*p3,p1+3*p2-p3)
    dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

    Define a custom normalization function:
    >>> e = S('real_log', custom_normalization=Transformer().replace_all(E("x_(exp(x1_))"), E("x1_")))
    >>> E("real_log(exp(x)) + real_log(5)")

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : Optional[bool]
        Set to true if the symbol is symmetric.
    is_antisymmetric : Optional[bool]
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : Optional[bool]
        Set to true if the symbol is cyclesymmetric.
    is_linear : Optional[bool]
        Set to true if the symbol is linear.
    custom_normalization : Optional[Transformer]
        A transformer that is called after every normalization. Note that the symbol
        name cannot be used in the transformer as this will lead to a definition of the 
        symbol. Use a wildcard with the same attributes instead.
    """


@overload
def S(*names: str,
      is_symmetric: Optional[bool] = None,
      is_antisymmetric: Optional[bool] = None,
      is_cyclesymmetric: Optional[bool] = None,
      is_linear: Optional[bool] = None,
      custom_normalization: Optional[Transformer] = None) -> Sequence[Expression]:
    """
    Create new symbols from `names`. Symbols can have attributes,
    such as symmetries. If no attributes
    are specified and the symbol was previously defined, the attributes are inherited.
    Once attributes are defined on a symbol, they cannot be redefined later.

    Examples
    --------
    Define two regular symbols:
    >>> x, y = S('x', 'y')

    Define two symmetric functions:
    >>> f, g = S('f', 'g', is_symmetric=True)
    >>> e = f(2,1)
    >>> print(e)
    f(1,2)

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : Optional[bool]
        Set to true if the symbol is symmetric.
    is_antisymmetric : Optional[bool]
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : Optional[bool]
        Set to true if the symbol is cyclesymmetric.
    is_linear : Optional[bool]
        Set to true if the symbol is multilinear.
    custom_normalization : Optional[Transformer]
        A transformer that is called after every normalization. Note that the symbol
        name cannot be used in the transformer as this will lead to a definition of the 
        symbol. Use a wildcard with the same attributes instead.
    """


def N(num: int | float | str | Decimal, relative_error: Optional[float] = None) -> Expression:
    """Create a new Symbolica number from an int, a float, or a string.
    A floating point number is kept as a float with the same precision as the input,
    but it can also be converted to the smallest rational number given a `relative_error`.

    Examples
    --------
    >>> e = N(1) / 2
    >>> print(e)
    1/2

    >>> print(N(1/3))
    >>> print(N(0.33, 0.1))
    >>> print(N('0.333`3'))
    >>> print(N(Decimal('0.1234')))
    3.3333333333333331e-1
    1/3
    3.33e-1
    1.2340e-1
    """


def E(input: str) -> Expression:
    """
    Parse a Symbolica expression from a string.

    Parameters
    ----------
    input: str
        An input string. UTF-8 character are allowed.

    Examples
    --------
    >>> e = E('x^2+y+y*4')
    >>> print(e)
    x^2+5*y

    Raises
    ------
    ValueError
        If the input is not a valid Symbolica expression.
    """


class AtomType(Enum):
    """Specifies the type of the atom."""

    Num = 1
    Var = 2
    Fn = 3
    Add = 4
    Mul = 5
    Pow = 6


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
    >>> x = Expression.symbol('x')
    >>> e = x**2 + 2 - x + 1 / x**4
    >>> print(e)
    """

    E: Expression
    """Euler's number `e`."""

    PI: Expression
    """The mathematical constant `π`."""

    I: Expression
    """The mathematical constant `i`, where `i^2 = -1`."""

    COEFF: Expression
    """The built-in function that convert a rational polynomials to a coefficient."""

    COS: Expression
    """The built-in cosine function."""

    SIN: Expression
    """The built-in sine function."""

    EXP: Expression
    """The built-in exponential function."""

    LOG: Expression
    """The built-in logarithm function."""

    @overload
    @classmethod
    def symbol(_cls,
               name: str,
               is_symmetric: Optional[bool] = None,
               is_antisymmetric: Optional[bool] = None,
               is_cyclesymmetric: Optional[bool] = None,
               is_linear: Optional[bool] = None,
               custom_normalization: Optional[Transformer] = None) -> Expression:
        """
        Create new symbols from `names`. Symbols can have attributes,
        such as symmetries. If no attributes
        are specified and the symbol was previously defined, the attributes are inherited.
        Once attributes are defined on a symbol, they cannot be redefined later.

        Examples
        --------
        Define a regular symbol and use it as a variable:
        >>> x = Expression.symbol('x')
        >>> e = x**2 + 5
        >>> print(e)
        x**2 + 5

        Define a regular symbol and use it as a function:
        >>> f = Expression.symbol('f')
        >>> e = f(1,2)
        >>> print(e)
        f(1,2)


        Define a symmetric function:
        >>> f = Expression.symbol('f', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)
        f(1,2)


        Define a linear and symmetric function:
        >>> p1, p2, p3, p4 = Expression.symbol('p1', 'p2', 'p3', 'p4')
        >>> dot = Expression.symbol('dot', is_symmetric=True, is_linear=True)
        >>> e = dot(p2+2*p3,p1+3*p2-p3)
        dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

        Define a custom normalization function:
        >>> e = S('real_log', custom_normalization=Transformer().replace_all(E("x_(exp(x1_))"), E("x1_")))
        >>> E("real_log(exp(x)) + real_log(5)")

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : Optional[bool]
            Set to true if the symbol is symmetric.
        is_antisymmetric : Optional[bool]
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : Optional[bool]
            Set to true if the symbol is cyclesymmetric.
        is_linear : Optional[bool]
            Set to true if the symbol is linear.
        custom_normalization : Optional[Transformer]
            A transformer that is called after every normalization. Note that the symbol
            name cannot be used in the transformer as this will lead to a definition of the 
            symbol. Use a wildcard with the same attributes instead.
        """

    @overload
    @classmethod
    def symbol(_cls,
               *names: str,
               is_symmetric: Optional[bool] = None,
               is_antisymmetric: Optional[bool] = None,
               is_cyclesymmetric: Optional[bool] = None,
               is_linear: Optional[bool] = None,
               custom_normalization: Optional[Transformer] = None) -> Sequence[Expression]:
        """
        Create new symbols from `names`. Symbols can have attributes,
        such as symmetries. If no attributes
        are specified and the symbol was previously defined, the attributes are inherited.
        Once attributes are defined on a symbol, they cannot be redefined later.

        Examples
        --------
        Define two regular symbols:
        >>> x, y = Expression.symbol('x', 'y')

        Define two symmetric functions:
        >>> f, g = Expression.symbol('f', 'g', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)
        f(1,2)

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : Optional[bool]
            Set to true if the symbol is symmetric.
        is_antisymmetric : Optional[bool]
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : Optional[bool]
            Set to true if the symbol is cyclesymmetric.
        is_linear : Optional[bool]
            Set to true if the symbol is multilinear.
        custom_normalization : Optional[Transformer]
            A transformer that is called after every normalization. Note that the symbol
            name cannot be used in the transformer as this will lead to a definition of the 
            symbol. Use a wildcard with the same attributes instead.
        """

    @overload
    def __call__(self, *args: Expression | int | float | Decimal) -> Expression:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x, f = Expression.symbol('x', 'f')
        >>> e = f(3,x)
        >>> print(e)
        f(3,x)
        """

    @overload
    def __call__(self, *args: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x, f = Expression.symbol('x', 'f')
        >>> e = f(3,x)
        >>> print(e)
        f(3,x)
        """

    @classmethod
    def num(_cls, num: int | float | str | Decimal, relative_error: Optional[float] = None) -> Expression:
        """Create a new Symbolica number from an int, a float, or a string.
        A floating point number is kept as a float with the same precision as the input,
        but it can also be converted to the smallest rational number given a `relative_error`.

        Examples
        --------
        >>> e = Expression.num(1) / 2
        >>> print(e)
        1/2

        >>> print(Expression.num(1/3))
        >>> print(Expression.num(0.33, 0.1))
        >>> print(Expression.num('0.333`3'))
        >>> print(Expression.num(Decimal('0.1234')))
        3.3333333333333331e-1
        1/3
        3.33e-1
        1.2340e-1
        """

    @classmethod
    def get_all_symbol_names(_cls) -> list[str]:
        """Return all defined symbol names (function names and variables)."""

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

    def __new__(cls) -> Expression:
        """Create a new expression that represents 0."""

    def __copy__(self) -> Expression:
        """
        Copy the expression.
        """

    def __str__(self) -> str:
        """
        Convert the expression into a human-readable string.
        """

    @classmethod
    def load(_cls, filename: str, conflict_fn: Optional[Callable[[str], str]] = None) -> Expression:
        """Load an expression and its state from a file. The state will be merged
        with the current one. If a symbol has conflicting attributes, the conflict
        can be resolved using the renaming function `conflict_fn`.

        Expressions can be saved using `Expression.save`.

        Examples
        --------
        If `export.dat` contains a serialized expression: `f(x)+f(y)`:
        >>> e = Expression.load('export.dat')

        whill yield `f(x)+f(y)`.

        If we have defined symbols in a different order:
        >>> y, x = S('y', 'x')
        >>> e = Expression.load('export.dat')

        we get `f(y)+f(x)`.

        If we define a symbol with conflicting attributes, we can resolve the conflict
        using a renaming function:

        >>> x = S('x', is_symmetric=True)
        >>> e = Expression.load('export.dat', lambda x: x + '_new')
        print(e)

        will yield `f(x_new)+f(y)`.
        """

    def save(self, filename: str, compression_level: int = 9):
        """Save the expression and its state to a binary file.
        The data is compressed and the compression level can be set between 0 and 11.

        The data can be loaded using `Expression.load`.

        Examples
        --------
        >>> e = E("f(x)+f(y)").expand()
        >>> e.save('export.dat')
        """

    def get_byte_size(self) -> int:
        """ Get the number of bytes that this expression takes up in memory."""

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the expression into a human-readable string, with tunable settings.

        Examples
        --------
        >>> a = Expression.parse('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
        >>> print(a.format(number_thousands_separator='_', multiplication_operator=' '))

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

    def to_sympy(self) -> str:
        """Convert the expression into a sympy-parsable string.

        Examples
        --------
        >>> from sympy import *
        >>> s = sympy.parse_expr(Expression.parse('x^2+f((1+x)^y)').to_sympy())
        """

    def __hash__(self) -> str:
        """
        Hash the expression.
        """

    def get_type(self) -> AtomType:
        """Get the type of the atom."""

    def to_atom_tree(self) -> AtomTree:
        """Convert the expression to a tree."""

    def get_name(self) -> Optional[str]:
        """
        Get the name of a variable or function if the current atom
        is a variable or function.
        """

    def __add__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __radd__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __sub__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Subtract `other` from this expression, returning the result.
        """

    def __rsub__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Subtract this expression from `other`, returning the result.
        """

    def __mul__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.
        """

    def __rmul__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.
        """

    def __truediv__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Divide this expression by `other`, returning the result.
        """

    def __rtruediv__(self, other: Expression | int | float | Decimal) -> Expression:
        """
        Divide `other` by this expression, returning the result.
        """

    def __pow__(self, exp: Expression | int | float | Decimal) -> Expression:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: Expression | int | float | Decimal) -> Expression:
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

    def contains(self, a: Transformer | Expression | int | float | Decimal) -> Condition:
        """Returns true iff `self` contains `a` literally.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, z = Expression.symbol('x', 'y', 'z')
        >>> e = x * y * z
        >>> e.contains(x) # True
        >>> e.contains(x*y*z) # True
        >>> e.contains(x*y) # False
        """

    def get_all_symbols(self, include_function_symbols: bool = True) -> Sequence[Expression]:
        """ Get all symbols in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.
        """

    def get_all_indeterminates(self, enter_functions: bool = True) -> Sequence[Expression]:
        """ Get all symbols and functions in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.
        """

    def coefficients_to_float(self, decimal_prec: int) -> Expression:
        """Convert all coefficients to floats with a given precision `decimal_prec`.
        The precision of floating point coefficients in the input will be truncated to `decimal_prec`."""

    def rationalize_coefficients(self, relative_error: float) -> Expression:
        """Map all floating point and rational coefficients to the best rational approximation
        in the interval `[self*(1-relative_error),self*(1+relative_error)]`."""

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
        >>> x, x_ = Expression.symbol('x', 'x_')
        >>> f = Expression.symbol('f')
        >>> e = f(x)*f(2)*f(f(3))
        >>> e = e.replace_all(f(x_), 1, x_.req_type(AtomType.Num))
        >>> print(e)

        Yields `f(x)*f(1)`.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the expression is of a certain type.
        """

    def req_contains(self, a: Expression) -> PatternRestriction:
        """
        Create a pattern restriction that filters for expressions that contain `a`.
        """

    def req_lit(self) -> PatternRestriction:
        """
        Create a pattern restriction that treats the wildcard as a literal variable,
        so that it only matches to itself.
        """

    def req(
        self,
        filter_fn: Callable[[Expression], bool | Condition],
    ) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `filter_fn` with the matched
        atom that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))
        """

    def req_cmp(
        self,
        other: Expression | int | float | Decimal,
        cmp_fn: Callable[[Expression, Expression], bool | Condition],
    ) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `cmp_fn` with another the matched
        atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.symbol('x_', 'y_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))
        """

    def req_lt(self, num: Expression | int | float | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_lt(2))
        """

    def req_gt(self, num: Expression | int | float | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_gt(2))
        """

    def req_le(self, num: Expression | int | float | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_le(2))
        """

    def req_ge(self, num: Expression | int | float | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_ge(2))
        """

    def req_cmp_lt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.symbol('x_', 'y_')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2)
        >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_lt(y_))
        """

    def req_cmp_gt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.symbol('x_', 'y_')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2)
        >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_gt(y_))
        """

    def req_cmp_le(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than or equal to another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.symbol('x_', 'y_')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2)
        >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_le(y_))
        """

    def req_cmp_ge(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_, y_ = Expression.symbol('x_', 'y_')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2)
        >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_ge(y_))
        """

    def __eq__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions.
        """

    def __neq__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions.
        """

    def __lt__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __le__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __gt__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __ge__(self, other: Expression | int | float | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __iter__(self) -> Iterator[Expression]:
        """
        Create an iterator over all atoms in the expression.
        """

    def __getitem__(self, idx: int) -> Expression:
        """Get the `idx`th component of the expression."""

    def map(
        self,
        transformations: Transformer,
        n_cores: Optional[int] = 1,
    ) -> Expression:
        """
        Map the transformations to every term in the expression.
        The execution happens in parallel using `n_cores`.

        Examples
        --------
        >>> x, x_ = Expression.symbol('x', 'x_')
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
        vars : Sequence[Expression]
                A list of variables
        """

    def expand(self, var: Optional[Expression] = None, via_poly: Optional[bool] = None) -> Expression:
        """
        Expand the expression. Optionally, expand in `var` only.

        Using `via_poly=True` may give a significant speedup for large expressions.
        """

    def expand_num(self) -> Expression:
        """ Distribute numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 3*(x+y)*(4*x+5*y)
        >>> print(e.expand_num())

        yields

        ```
        (3*x+3*y)*(4*x+5*y)
        ```
        """

    def collect(
        self,
        *x: Expression,
        key_map: Optional[Callable[[Expression], Expression]] = None,
        coeff_map: Optional[Callable[[Expression], Expression]] = None,
    ) -> Expression:
        """
        Collect terms involving the same power of the indeterminate(s) `x`.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Both the key (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` respectively.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x))

        yields `x^2+x*(y+5)+5`.

        >>> from symbolica import Expression
        >>> x, y = Expression.symbol('x', 'y')
        >>> var, coeff = Expression.funs('var', 'coeff')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x, key_map=lambda x: var(x), coeff_map=lambda x: coeff(x)))

        yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.

        Parameters
        ----------
        *x: Expression
            The variable(s) or function(s) to collect terms in
        key_map
            A function to be applied to the quantity collected in
        coeff_map
            A function to be applied to the coefficient
        """

    def collect_num(self) -> Expression:
        """Collect numerical factors by removing the content from additions.
        For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.

        The first argument of the addition is normalized to a positive quantity.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = (-3*x+6*y)*(2*x+2*y)
        >>> print(e.collect_num())

        yields

        ```
        -6*(x+y)*(x-2*y)
        ```
        """

    def coefficient_list(
        self, *x: Expression
    ) -> Sequence[Tuple[Expression, Expression]]:
        """Collect terms involving the same power of `x`, where `x` are variables or functions.
        Return the list of key-coefficient pairs.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> for a in e.coefficient_list(x):
        >>>     print(a[0], a[1])

        yields
        ```
        x y+5
        x^2 1
        1 5
        ```
        """

    def coefficient(self, x: Expression) -> Expression:
        """Collect terms involving the literal occurrence of `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 5*x + x * y + x**2 + y*x**2
        >>> print(e.coefficient(x**2))

        yields

        ```
        y + 1
        ```
        """

    def derivative(self, x: Expression) -> Expression:
        """Derive the expression w.r.t the variable `x`."""

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Series:
        """Series expand in `x` around `expansion_point` to depth `depth`."""

    def apart(self, x: Expression) -> Expression:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))')
        >>> print(p.apart(x))
        """

    def together(self) -> Expression:
        """Write the expression over a common denominator.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('v1^2/2+v1^3/v4*v2+v3/(1+v4)')
        >>> print(p.together())
        """

    def cancel(self) -> Expression:
        """Cancel common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('1+(y+1)^10*(x+1)/(x^2+2x+1)')
        >>> print(p.cancel())
        1+(y+1)**10/(x+1)
        """

    def factor(self) -> Expression:
        """Factor the expression over the rationals.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('(6 + x)/(7776 + 6480*x + 2160*x^2 + 360*x^3 + 30*x^4 + x^5)')
        >>> print(p.factor())
        (x+6)**-4
        """

    @overload
    def to_polynomial(self, vars: Optional[Sequence[Expression]] = None) -> Polynomial:
        """Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
        All non-polynomial parts will be converted to new, independent variables.
        """

    @overload
    def to_polynomial(self, minimal_poly: Expression, vars: Optional[Sequence[Expression]] = None,
                      ) -> NumberFieldPolynomial:
        """Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
        The minimal polynomial must be a monic, irreducible univariate polynomial.
        """

    @overload
    def to_polynomial(self,
                      modulus: int,
                      power: Optional[Tuple[int, Expression]] = None,
                      minimal_poly: Optional[Expression] = None,
                      vars: Optional[Sequence[Expression]] = None,
                      ) -> FiniteFieldPolynomial:
        """Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to finite field elements modulo `modulus`.
        If on top an `extension` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
        `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

        If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.
        """

    def to_rational_polynomial(
        self,
        vars: Optional[Sequence[Expression]] = None,
    ) -> RationalPolynomial:
        """
        Convert the expression to a rational polynomial, optionally, with the variable ordering specified in `vars`.
        The latter is useful if it is known in advance that more variables may be added in the future to the
        rational polynomial through composition with other rational polynomials.

        All non-rational polynomial parts are converted to new, independent variables.

        Examples
        --------
        >>> a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
        >>> print(a)
        """

    def match(
        self,
        lhs: Transformer | Expression | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> MatchIterator:
        """
        Return an iterator over the pattern `self` matching to `lhs`.
        Restrictions on the pattern can be supplied through `cond`.

        The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
        The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
        depending on `level_is_tree_depth`.

        Examples
        --------

        >>> x, x_ = Expression.symbol('x','x_')
        >>> f = Expression.symbol('f')
        >>> e = f(x)*f(1)*f(2)*f(3)
        >>> for match in e.match(f(x_)):
        >>>    for map in match:
        >>>        print(map[0],'=', map[1])
        """

    def matches(
        self,
        lhs: Transformer | Expression | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> Condition:
        """
        Test whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.

        Examples
        --------

        >>> f = Expression.symbol('f')
        >>> if f(1).matches(f(2)):
        >>>    print('match')
        """

    def replace(
        self,
        lhs: Transformer | Expression | int | float | Decimal,
        rhs: Transformer | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> ReplaceIterator:
        """
        Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(1)*f(2)*f(3)
        >>> for r in e.replace(f(x_), f(x_ + 1)):
        >>>     print(r)

        Yields:
        ```
        f(2)*f(2)*f(3)
        f(1)*f(3)*f(3)
        f(1)*f(2)*f(4)
        ```

        Parameters
        ----------
        lhs:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond:
            Conditions on the pattern.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth:
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs:
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        """

    def replace_all(
        self,
        pattern: Transformer | Expression | int | float | Decimal,
        rhs: Transformer | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
        rhs_cache_size: Optional[int] = None,
        repeat: Optional[bool] = False,
    ) -> Expression:
        """
        Replace all subexpressions matching the pattern `pattern` by the right-hand side `rhs`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.symbol('x','w1_','w2_')
        >>> f = Expression.symbol('f')
        >>> e = f(3,x)
        >>> r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
        >>> print(r)

        Parameters
        ----------
        self:
            The expression to match and replace on.
        pattern:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond:
            Conditions on the pattern.
        non_greedy_wildcards:
            Wildcards that try to match as little as possible.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth: bool, optional
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs: bool, optional
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        repeat: bool, optional
            If set to `True`, the entire operation will be repeated until there are no more matches.
        """

    def replace_all_multiple(self, replacements: Sequence[Replacement],  repeat: Optional[bool] = False) -> Expression:
        """
        Replace all atoms matching the patterns. See `replace_all` for more information.

        The entire operation can be repeated until there are no more matches using `repeat=True`.

        Examples
        --------

        >>> x, y, f = Expression.symbol('x', 'y', 'f')
        >>> e = f(x,y)
        >>> r = e.replace_all_multiple([Replacement(x, y), Replacement(y, x)])
        >>> print(r)
        f(y,x)

        Parameters
        ----------
        replacements: Sequence[Replacement]
            The list of replacements to apply.
        repeat: bool, optional
            If set to `True`, the entire operation will be repeated until there are no more matches.
       """

    @classmethod
    def solve_linear_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
    ) -> Sequence[Expression]:
        """Solve a linear system in the variables `variables`, where each expression
        in the system is understood to yield 0.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y, c = Expression.symbol('x', 'y', 'c')
        >>> f = Expression.symbol('f')
        >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
        >>> print('x =', x_r, ', y =', y_r)
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: float,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> float:
        """Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"), 1., 0.0001, 1000000)
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: Decimal,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Decimal:
        """Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"),
                      Decimal("1.000000000000000000000000000000000000000000000000000000000000000000000000"), 1e-74, 1000000)
        """

    @overload
    @classmethod
    def nsolve_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
        init: Sequence[float],
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Sequence[float]:
        """Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [1., 1.], 1e-20, 1000000)
        """

    @overload
    @classmethod
    def nsolve_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
        init: Sequence[Decimal],
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Sequence[Decimal]:
        """Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [Decimal("1.00000000000000000"), Decimal("1.00000000000000000")], 1e-20, 1000000)
        """

    def evaluate(
        self, constants: dict[Expression, float], funs: dict[Expression, Callable[[Sequence[float]], float]]
    ) -> float:
        """Evaluate the expression, using a map of all the variables and
        user functions to a float.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> f = Expression.symbol('f')
        >>> e = Expression.parse('cos(x)')*3 + f(x,2)
        >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
        """

    def evaluate_with_prec(
        self,
        constants: dict[Expression, float | str | Decimal],
        funs: dict[Expression, Callable[[Sequence[Decimal]], float | str | Decimal]],
        decimal_digit_precision: int
    ) -> Decimal:
        """Evaluate the expression, using a map of all the constants and
        user functions using arbitrary precision arithmetic.
        The user has to specify the number of decimal digits of precision
        and provide all input numbers as floats, strings or `Decimal`.

        Examples
        --------
        >>> from symbolica import *
        >>> from decimal import Decimal, getcontext
        >>> x = Expression.symbol('x', 'f')
        >>> e = Expression.parse('cos(x)')*3 + f(x, 2)
        >>> getcontext().prec = 100
        >>> a = e.evaluate_with_prec({x: Decimal('1.123456789')}, {
        >>>                         f: lambda args: args[0] + args[1]}, 100)
        """

    def evaluate_complex(
        self, constants: dict[Expression, float | complex], funs: dict[Expression, Callable[[Sequence[complex]], float | complex]]
    ) -> complex:
        """Evaluate the expression, using a map of all the variables and
        user functions to a complex number.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = Expression.parse('sqrt(x)')*y
        >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
        """

    def evaluator(
        self,
        constants: dict[Expression, Expression],
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 100,
        n_cores: int = 4,
        verbose: bool = False,
    ) -> Evaluator:
        """Create an evaluator that can evaluate (nested) expressions in an optimized fashion.
        All constants and functions should be provided as dictionaries, where the function
        dictionary has a key `(name, printable name, arguments)` and the value is the function
        body. For example the function `f(x,y)=x^2+y` should be provided as
        `{(f, "f", (x, y)): x**2 + y}`. All free parameters should be provided in the `params` list.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, z, pi, f, g = Expression.symbol(
        >>>     'x', 'y', 'z', 'pi', 'f', 'g')
        >>>
        >>> e1 = Expression.parse("x + pi + cos(x) + f(g(x+1),x*2)")
        >>> fd = Expression.parse("y^2 + z^2*y^2")
        >>> gd = Expression.parse("y + 5")
        >>>
        >>> ev = e1.evaluator({pi: Expression.num(22)/7},
        >>>              {(f, "f", (y, z)): fd, (g, "g", (y, )): gd}, [x])
        >>> res = ev.evaluate([[1.], [2.], [3.]])  # evaluate at x=1, x=2, x=3
        >>> print(res)

        Parameters
        ----------
        constants: dict[Expression, Expression]
            A map of expressions to constants. The constants should be numerical expressions.
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression]
            A dictionary of functions. The key is a tuple of the function name, printable name and the argument variables.
            The value is the function body.
        params: Sequence[Expression]
            A list of free parameters.
        iterations: int, optional
            The number of optimization iterations to perform.
        n_cores: int, optional
            The number of cores to use for the optimization.
        verbose: bool, optional
            If set to `True`, print the progress of the optimization.
        """

    @classmethod
    def evaluator_multiple(
        _cls,
        exprs: Sequence[Expression],
        constants: dict[Expression, Expression],
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 100,
        n_cores: int = 4,
        verbose: bool = False,
    ) -> Evaluator:
        """Create an evaluator that can jointly evaluate (nested) expressions in an optimized fashion.
        See `Expression.evaluator()` for more information.

        Examples
        --------
        >>> from symbolica import *
        >>> x = Expression.symbol('x')
        >>> e1 = Expression.parse("x^2 + 1")
        >>> e2 = Expression.parse("x^2 + 2)
        >>> ev = Expression.evaluator_multiple([e1, e2], {}, {}, [x])

        will recycle the `x^2`
        """

    def canonize_tensors(self,
                         contracted_indices: Sequence[Tuple[Expression | int, Expression | int]]) -> Expression:
        """Canonize (products of) tensors in the expression by relabeling repeated indices.
        The tensors must be written as functions, with its indices as the arguments.
        Subexpressions, constants and open indices are supported.

        If the contracted indices are distinguishable (for example in their dimension),
        you can provide a group marker as the second element in the tuple of the index
        specification.
        This makes sure that an index will not be renamed to an index from a different group.

        Examples
        --------
        >>> g = Expression.symbol('g', is_symmetric=True)
        >>> fc = Expression.symbol('fc', is_cyclesymmetric=True)
        >>> mu1, mu2, mu3, mu4, k1 = Expression.symbol('mu1', 'mu2', 'mu3', 'mu4', 'k1')
        >>> e = g(mu2, mu3)*fc(mu4, mu2, k1, mu4, k1, mu3)
        >>> print(e.canonize_tensors([(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)]))

        yields `g(mu1, mu2)*fc(mu1, mu3, mu2, k1, mu3, k1)`.
        """


class Replacement:
    """A replacement of a pattern by a right-hand side."""

    def __new__(
            cls,
            pattern: Transformer | Expression | int | float | Decimal,
            rhs: Transformer | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | Decimal,
            cond: Optional[PatternRestriction | Condition] = None,
            non_greedy_wildcards: Optional[Sequence[Expression]] = None,
            level_range: Optional[Tuple[int, Optional[int]]] = None,
            level_is_tree_depth: Optional[bool] = False,
            allow_new_wildcards_on_rhs: Optional[bool] = False,
            rhs_cache_size: Optional[int] = None) -> Replacement:
        """Create a new replacement. See `replace_all` for more information."""


class PatternRestriction:
    """A restriction on wildcards."""

    def __and__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold)."""

    def __or__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold)."""

    def __invert__(self) -> PatternRestriction:
        """Create a new pattern restriction that takes the logical 'not' of the current restriction."""

    @classmethod
    def req_matches(_cls, match_fn: Callable[[dict[Expression, Expression]], int]) -> PatternRestriction:
        """Create a pattern restriction based on the current matched variables.
        `match_fn` is a Python function that takes a dictionary of wildcards and their matched values
        and should return an integer. If the integer is less than 0, the restriction is false.
        If the integer is 0, the restriction is inconclusive.
        If the integer is greater than 0, the restriction is true.

        If your pattern restriction cannot decide if it holds since not all the required variables
        have been matched, it should return inclusive (0).

        Examples
        --------
        >>> from symbolica import *
        >>> f, x_, y_, z_ = S('f', 'x_', 'y_', 'z_')
        >>>
        >>> def filter(m: dict[Expression, Expression]) -> int:
        >>>    if x_ in m and y_ in m:
        >>>        if m[x_] > m[y_]:
        >>>            return -1  # no match
        >>>        if z_ in m:
        >>>            if m[y_] > m[z_]:
        >>>                return -1
        >>>            return 1  # match
        >>>
        >>>    return 0  # inconclusive
        >>>
        >>>
        >>> e = f(1, 2, 3).replace_all(f(x_, y_, z_), 1,
        >>>         PatternRestriction.req_matches(filter))
        """


class Condition:
    """Relations that evaluate to booleans"""

    def eval(self) -> bool:
        """Evaluate the condition."""

    def __repr__(self) -> str:
        """Return a string representation of the condition."""

    def __str__(self) -> str:
        """Return a string representation of the condition."""

    def __bool__(self) -> bool:
        """Return the boolean value of the condition."""

    def __and__(self, other:  Condition) -> Condition:
        """Create a condition that is the logical and operation between two conditions (i.e., both should hold)."""

    def __or__(self, other:  Condition) -> Condition:
        """Create a condition that is the logical 'or' operation between two conditions (i.e., at least one of the two should hold)."""

    def __invert__(self) -> Condition:
        """Create a condition that takes the logical 'not' of the current condition."""

    def to_req(self) -> PatternRestriction:
        """Convert the condition to a pattern restriction."""


class CompareOp:
    """One of the following comparison operators: `<`,`>`,`<=`,`>=`,`==`,`!=`."""


class Transformer:
    """Operations that transform an expression."""

    def __new__(_cls) -> Transformer:
        """Create a new transformer for a term provided by `Expression.map`."""

    def __call__(self, expr: Expression | int | float | Decimal) -> Expression:
        """Execute an unbound transformer on the given expression. If the transformer
        is bound, use `execute()` instead.

        Examples
        --------
        >>> x = Expression.symbol('x')
        >>> e = Transformer().expand()((1+x)**2)
        """

    def __eq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.
        """

    def __neq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.
        """

    def __lt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __le__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __gt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __ge__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the transformed expression is of a certain type.
        """

    def contains(self, element: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Create a transformer that checks if the expression contains the given `element`.
        """

    def matches(
        self,
        lhs: Transformer | Expression | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> Condition:
        """
        Create a transformer that tests whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.
        """

    def if_then(self, condition: Condition, if_block: Transformer, else_block: Optional[Transformer] = None) -> Transformer:
        """Evaluate the condition and apply the `if_block` if the condition is true, otherwise apply the `else_block`.
        The expression that is the input of the transformer is the input for the condition, the `if_block` and the `else_block`.

        Examples
        --------
        >>> t = T.map_terms(T.if_then(T.contains(x), T.print()))
        >>> t(x + y + 4)

        prints `x`.
        """

    def if_changed(self, condition: Transformer, if_block: Transformer, else_block: Optional[Transformer] = None) -> Transformer:
        """Execute the `condition` transformer. If the result of the `condition` transformer is different from the input expression,
        apply the `if_block`, otherwise apply the `else_block`. The input expression of the `if_block` is the output
        of the `condition` transformer.

        Examples
        --------
        >>> t = T.map_terms(T.if_changed(T.replace_all(x, y), T.print()))
        >>> print(t(x + y + 4))

        prints
        ```
        y
        2*y+4
        ```
        """

    def break_chain(self) -> Transformer:
        """Break the current chain and all higher-level chains containing `if` transformers.

        Examples
        --------
        >>> from symbolica import *
        >>> t = T.map_terms(T.repeat(
        >>>     T.replace_all(y, 4),
        >>>     T.if_changed(T.replace_all(x, y),
        >>>                 T.break_chain()),
        >>>     T.print()  # print of y is never reached
        >>> ))
        >>> print(t(x))
        """

    def expand(self, var: Optional[Expression] = None, via_poly: Optional[bool] = None) -> Transformer:
        """Create a transformer that expands products and powers. Optionally, expand in `var` only.

        Using `via_poly=True` may give a significant speedup for large expressions.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x, x_ = Expression.symbol('x', 'x_')
        >>> f = Expression.symbol('f')
        >>> e = f((x+1)**2).replace_all(f(x_), x_.transform().expand())
        >>> print(e)
        """

    def expand_num(self) -> Expression:
        """Create a transformer that distributes numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 3*(x+y)*(4*x+5*y)
        >>> print(Transformer().expand_num()(e))

        yields

        ```
        (3*x+3*y)*(4*x+5*y)
        ```
        """

    def prod(self) -> Transformer:
        """Create a transformer that computes the product of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(2,3).replace_all(f(x__), x__.transform().prod())
        >>> print(e)
        """

    def sum(self) -> Transformer:
        """Create a transformer that computes the sum of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(2,3).replace_all(f(x__), x__.transform().sum())
        >>> print(e)
        """

    def nargs(self, only_for_arg_fun: bool = False) -> Transformer:
        """Create a transformer that returns the number of arguments.
        If the argument is not a function, return 0.

        If `only_for_arg_fun` is `True`, only count the number of arguments
        in the `arg()` function and return 1 if the input is not `arg`.
        This is useful for obtaining the length of a range during pattern matching.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(2,3,4).replace_all(f(x__), x__.transform().nargs())
        >>> print(e)
        """

    def sort(self) -> Transformer:
        """Create a transformer that sorts a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(3,2,1).replace_all(f(x__), x__.transform().sort())
        >>> print(e)
        """

    def cycle_symmetrize(self) -> Transformer:
        """ Create a transformer that cycle-symmetrizes a function.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2,4,1,2,3).replace_all(f(x__), x_.transform().cycle_symmetrize())
        >>> print(e)

        Yields `f(1,2,3,1,2,4)`.
        """

    def deduplicate(self) -> Transformer:
        """Create a transformer that removes elements from a list if they occur
        earlier in the list as well.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.symbol('x__')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2,1,2).replace_all(f(x__), x__.transform().deduplicate())
        >>> print(e)

        Yields `f(1,2)`.
        """

    def from_coeff(self) -> Transformer:
        """Create a transformer that extracts a rational polynomial from a coefficient.

        Examples
        --------
        >>> from symbolica import Expression, Function
        >>> e = Function.COEFF((x^2+1)/y^2).transform().from_coeff()
        >>> print(e)
        """

    def split(self) -> Transformer:
        """Create a transformer that split a sum or product into a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x, x__ = Expression.symbol('x', 'x__')
        >>> f = Expression.symbol('f')
        >>> e = (x + 1).replace_all(x__, f(x_.transform().split()))
        >>> print(e)
        """

    def linearize(self, symbols: Optional[Sequence[Expression]]) -> Transformer:
        """Create a transformer that linearizes a function, optionally extracting `symbols`
        as well.

        Examples
        --------

        >>> from symbolica import Expression, Transformer
        >>> x, y, z, w, f, x__ = Expression.symbol('x', 'y', 'z', 'w', 'f', 'x__')
        >>> e = f(x+y, 4*z*w+3).replace_all(f(x__), f(x__).transform().linearize([z]))
        >>> print(e)

        yields `f(x,3)+f(y,3)+4*z*f(x,w)+4*z*f(y,w)`.
        """

    def partitions(
        self,
        bins: Sequence[Tuple[Transformer | Expression, int]],
        fill_last: bool = False,
        repeat: bool = False,
    ) -> Transformer:
        """Create a transformer that partitions a list of arguments into named bins of a given length,
        returning all partitions and their multiplicity.

        If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
        will add all remaining elements to the last bin.

        Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
        if possible.

        Note that the functions names to be provided for the bin names must be generated through `Expression.var`.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_, f_id, g_id = Expression.symbol('x__', 'f', 'g')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2,1,3).replace_all(f(x_), x_.transform().partitions([(f_id, 2), (g_id, 1), (f_id, 1)]))
        >>> print(e)

        yields:
        ```
        2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)
        ```
        """

    def permutations(self, function_name: Transformer | Expression) -> Transformer:
        """Create a transformer that generates all permutations of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_, f_id = Expression.symbol('x__', 'f')
        >>> f = Expression.symbol('f')
        >>> e = f(1,2,1,2).replace_all(f(x_), x_.transform().permutations(f_id)
        >>> print(e)

        yields:
        ```
        4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)
        ```
        """

    def map(self, f: Callable[[Expression], Expression | int | float | Decimal]) -> Transformer:
        """Create a transformer that applies a Python function.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = f(2).replace_all(f(x_), x_.transform().map(lambda r: r**2))
        >>> print(e)
        """

    def map_terms(self, *transformers: Transformer, n_cores: int = 1) -> Transformer:
        """Map a chain of transformer over the terms of the expression, optionally using multiple cores.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> t = Transformer().map_terms(Transformer().print(), n_cores=2)
        >>> e = t(x + y)
        """

    def for_each(self, *transformers: Transformer) -> Transformer:
        """Create a transformer that applies a transformer chain to every argument of the `arg()` function.
        If the input is not `arg()`, the transformer is applied to the input.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> f = Expression.symbol('f')
        >>> e = (1+x).transform().split().for_each(Transformer().map(f)).execute()
        """

    def check_interrupt(self) -> Transformer:
        """Create a transformer that checks for a Python interrupt,
        such as ctrl-c and aborts the current transformer.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> f(10).transform().repeat(Transformer().replace_all(
        >>> f(x_), f(x_+1)).check_interrupt()).execute()
        """

    def repeat(self, *transformers: Transformer) -> Transformer:
        """Create a transformer that repeatedly executes the arguments in order
        until there are no more changes.
        The output from one transformer is inserted into the next.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = Expression.parse("f(5)")
        >>> e = e.transform().repeat(
        >>>     Transformer().expand(),
        >>>     Transformer().replace_all(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
        >>> ).execute()
        """

    def chain(self, *transformers: Transformer) -> Transformer:
        """Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
        where `A`, `B`, `C` are transformers.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = Expression.parse("f(5)")
        >>> e = e.transform().chain(
        >>>     Transformer().expand(),
        >>>     Transformer().replace_all(f(x_), f(5))
        >>> ).execute()
        """

    def execute(self) -> Expression:
        """Execute a bound transformer.  If the transformer is unbound,
        you can call it with an expression as an argument.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> e = (x+1)**5
        >>> e = e.transform().expand().execute()
        >>> print(e)
        """

    def derivative(self, x: Transformer | Expression) -> Transformer:
        """Create a transformer that derives `self` w.r.t the variable `x`."""

    def set_coefficient_ring(self, vars: Sequence[Expression]) -> Transformer:
        """
        Create a transformer that sets the coefficient ring to contain the variables in the `vars` list.
        This will move all variables into a rational polynomial function.

        Parameters
        ----------
        vars : Sequence[Expression]
                A list of variables
        """

    def collect(
        self,
        *x: Expression,
        key_map: Optional[Transformer] = None,
        coeff_map: Optional[Transformer] = None,
    ) -> Transformer:
        """
        Create a transformer that collects terms involving the same power of the indeterminate(s) `x`.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Both the key (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` transformers respectively.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.transform().collect(x).execute())

        yields `x^2+x*(y+5)+5`.

        >>> from symbolica import Expression
        >>> x, y, x_, var, coeff = Expression.symbol('x', 'y', 'x_', 'var', 'coeff')
        >>> e = 5*x + x * y + x**2 + 5
        >>> print(e.collect(x, key_map=Transformer().replace_all(x_, var(x_)),
                coeff_map=Transformer().replace_all(x_, coeff(x_))))

        yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.

        Parameters
        ----------
        *x: Expression
            The variable(s) or function(s) to collect terms in
        key_map: Transformer
            A transformer to be applied to the quantity collected in
        coeff_map: Transformer
            A transformer to be applied to the coefficient
        """

    def collect_num(self) -> Expression:
        """Create a transformer that collects numerical factors by removing the content from additions.
        For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.

        The first argument of the addition is normalized to a positive quantity.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.symbol('x', 'y')
        >>> e = (-3*x+6*y)*(2*x+2*y)
        >>> print(Transformer().collect_num()(e))

        yields

        ```
        -6*(x+y)*(x-2*y)
        ```
        """

    def coefficient(self, x: Expression) -> Transformer:
        """Create a transformer that collects terms involving the literal occurrence of `x`.
        """

    def apart(self, x: Expression) -> Transformer:
        """Create a transformer that computes the partial fraction decomposition in `x`.
        """

    def together(self) -> Transformer:
        """Create a transformer that writes the expression over a common denominator.
        """

    def cancel(self) -> Transformer:
        """Create a transformer that cancels common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.
        """

    def factor(self) -> Transformer:
        """Create a transformer that factors the expression over the rationals."""

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Transformer:
        """Create a transformer that series expands in `x` around `expansion_point` to depth `depth`.

        Examples
        -------
        >>> from symbolica import Expression
        >>> x, y = Expression.symbol('x', 'y')
        >>> f = Expression.symbol('f')
        >>>
        >>> e = 2* x**2 * y + f(x)
        >>> e = e.series(x, 0, 2)
        >>>
        >>> print(e)

        yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.
        """

    def replace_all(
        self,
        pat: Transformer | Expression | int | float | Decimal,
        rhs: Transformer | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False
    ) -> Transformer:
        """
        Create a transformer that replaces all subexpressions matching the pattern `pat` by the right-hand side `rhs`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.symbol('x','w1_','w2_')
        >>> f = Expression.symbol('f')
        >>> e = f(3,x)
        >>> r = e.transform().replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
        >>> print(r)

        Parameters
        ----------
        pat:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond:
            Conditions on the pattern.
        non_greedy_wildcards:
            Wildcards that try to match as little as possible.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth:
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs:
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        repeat:
            If set to `True`, the entire operation will be repeated until there are no more matches.
        """

    def replace_all_multiple(self, replacements: Sequence[Replacement]) -> Transformer:
        """
        Create a transformer that replaces all atoms matching the patterns. See `replace_all` for more information.

        Examples
        --------

        >>> x, y, f = Expression.symbol('x', 'y', 'f')
        >>> e = f(x,y)
        >>> r = e.transform().replace_all_multiple([Replacement(x, y), Replacement(y, x)])
        >>> print(r)
        """

    def print(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        show_namespaces: bool = False,
    ) -> Transformer:
        """
        Create a transformer that prints the expression.

        Examples
        --------
        >>> Expression.parse('f(10)').transform().print(terms_on_new_line = True).execute()
        """

    def stats(
        self,
        tag: str,
        transformer: Transformer,
        color_medium_change_threshold: Optional[float] = 10.,
        color_large_change_threshold: Optional[float] = 100.,
    ) -> Transformer:
        """
        Print statistics of a transformer, tagging it with `tag`.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.symbol('x_')
        >>> f = Expression.symbol('f')
        >>> e = Expression.parse("f(5)")
        >>> e = e.transform().stats('replace', Transformer().replace_all(f(x_), 1)).execute()

        yields
        ```
        Stats for replace:
            In  │ 1 │  10.00 B │
            Out │ 1 │   3.00 B │ ⧗ 40.15µs
        ```
        """

    def __add__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __radd__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __sub__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Subtract `other` from this transformer, returning the result.
        """

    def __rsub__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Subtract this transformer from `other`, returning the result.
        """

    def __mul__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __rmul__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Add this transformer to `other`, returning the result.
        """

    def __truediv__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Divide this transformer by `other`, returning the result.
        """

    def __rtruediv__(self, other: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Divide `other` by this transformer, returning the result.
        """

    def __pow__(self, exp: Transformer | Expression | int | float | Decimal) -> Transformer:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: Transformer | Expression | int | float | Decimal) -> Transformer:
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


class Series:
    """
    A series expansion class.

    Supports standard arithmetic operations, such
    as addition and multiplication.

    Examples
    --------
    >>> x = Expression.symbol('x')
    >>> s = Expression.parse("(1-cos(x))/sin(x)").series(x, 0, 4) * x
    >>> print(s)
    """

    def __str__(self) -> str:
        """Print the series in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the series into a LaTeX string."""

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the series into a human-readable string.
        """

    def __add__(self, other: Series | Expression) -> Series:
        """Add another series or expression to this series, returning the result."""

    def __radd__(self, other: Expression) -> Series:
        """Add two series together, returning the result."""

    def __sub__(self, other: Series | Expression) -> Series:
        """Subtract `other` from `self`, returning the result."""

    def __rsub__(self, other: Expression) -> Series:
        """Subtract `self` from `other`, returning the result."""

    def __mul__(self, other: Series | Expression) -> Series:
        """Multiply another series or expression to this series, returning the result."""

    def __rmul__(self, other: Expression) -> Series:
        """Multiply two series together, returning the result."""

    def __truediv__(self, other: Series | Expression) -> Series:
        """Divide `self` by `other`, returning the result."""

    def __rtruediv__(self, other: Expression) -> Series:
        """Divide `other` by `self`, returning the result."""

    def __pow__(self, exp: int) -> Series:
        """Raise the series to the power of `exp`, returning the result."""

    def __neg__(self) -> Series:
        """Negate the series."""

    def sin(self) -> Series:
        """Compute the sine of the series, returning the result."""

    def cos(self) -> Series:
        """Compute the cosine of the series, returning the result."""

    def exp(self) -> Series:
        """Compute the exponential of the series, returning the result."""

    def log(self) -> Series:
        """Compute the natural logarithm of the series, returning the result."""

    def pow(self, num: int, den: int) -> Series:
        """Raise the series to the power of `num/den`, returning the result."""

    def spow(self, exp: Series) -> Series:
        """Raise the series to the power of `exp`, returning the result."""

    def shift(self, e: int) -> Series:
        """Shift the series by `e` units of the ramification."""

    def get_ramification(self) -> int:
        """Get the ramification."""

    def get_trailing_exponent(self) -> Tuple[int, int]:
        """Get the trailing exponent; the exponent of the first non-zero term."""

    def get_relative_order(self) -> Tuple[int, int]:
        """Get the relative order."""

    def get_absolute_order(self) -> Tuple[int, int]:
        """Get the absolute order."""

    def to_expression(self) -> Expression:
        """Convert the series to an expression"""


class TermStreamer:
    """
    A term streamer that can handle large expressions, by
    streaming terms to and from disk.
    """

    def __new__(_cls, path: Optional[str] = None,
                max_mem_bytes: Optional[int] = None,
                n_cores: Optional[int] = None) -> TermStreamer:
        """Create a new term streamer with a given path for its files,
           the maximum size of the memory buffer and the number of cores.
        """

    def __add__(self, other: TermStreamer) -> TermStreamer:
        """Add two term streamers together, returning the result."""

    def __iadd__(self, other: TermStreamer) -> None:
        """Add another term streamer to this one."""

    def get_byte_size(self) -> int:
        """Get the byte size of the term stream."""

    def get_num_terms(self) -> int:
        """Get the number of terms in the stream."""

    def fits_in_memory(self) -> bool:
        """Check if the term stream fits in memory."""

    def push(self, expr: Expression) -> None:
        """Push an expression to the term stream."""

    def normalize(self) -> None:
        """Sort and fuse all terms in the stream."""

    def to_expression(self) -> Expression:
        """Convert the term stream into an expression. This may exceed the available memory."""

    def map(self, f: Transformer) -> TermStreamer:
        """Apply a transformer to all terms in the stream."""

    def map_single_thread(self, f: Transformer) -> TermStreamer:
        """Apply a transformer to all terms in the stream using a single thread."""


class MatchIterator:
    """An iterator over matches."""

    def __iter__(self) -> MatchIterator:
        """Create the iterator."""

    def __next__(self) -> dict[Expression, Expression]:
        """Return the next match."""


class ReplaceIterator:
    """An iterator over single replacements."""

    def __iter__(self) -> ReplaceIterator:
        """Create the iterator."""

    def __next__(self) -> Expression:
        """Return the next replacement."""


class Polynomial:
    """A Symbolica polynomial with rational coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str]) -> Polynomial:
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

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: Polynomial) -> Polynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: Polynomial) -> Polynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: Polynomial) -> Polynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: Polynomial) -> Tuple[Polynomial, Polynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: Polynomial) -> Polynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> Polynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: Polynomial) -> Polynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def resultant(self, rhs: Polynomial, var: Expression) -> Polynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def to_integer_polynomial(self) -> IntegerPolynomial:
        """Convert the polynomial to a polynomial with integer coefficients, if possible."""

    def to_finite_field(self, prime: int) -> FiniteFieldPolynomial:
        """Convert the coefficients of the polynomial to a finite field with prime `prime`."""

    def isolate_roots(self, refine: Optional[float | Decimal] = None) -> list[Tuple[Expression, Expression, int]]:
        """Isolate the real roots of the polynomial. The result is a list of intervals with rational bounds that contain exactly one root,
        and the multiplicity of that root. Optionally, the intervals can be refined to a given precision.

        Examples
        --------
        >>> from symbolica import Expression
        >>> p = Expression.parse('2016+5808*x+5452*x^2+1178*x^3+-753*x^4+-232*x^5+41*x^6').to_polynomial()
        >>> for a, b, n in p.isolate_roots():
        >>>     print('({},{}): {}'.format(a, b, n))

        yields
        ```
        (-56/45,-77/62): 1
        (-98/79,-119/96): 1
        (-119/96,-21/17): 1
        (-7/6,0): 1
        (0,6): 1
        (6,12): 1
        ```
        """

    def approximate_roots(self, max_iterations: int, tolerance: float) -> list[Tuple[complex, int]]:
        """Approximate all complex roots of a univariate polynomial, given a maximal number of iterations
        and a given tolerance. Returns the roots and their multiplicity.

        Examples
        --------

        >>> p = E('x^10+9x^7+4x^3+2x+1').to_polynomial()
        >>> for (r, m) in p.approximate_roots(1000, 1e-10):
        >>>     print(r, m)
        """

    def factor_square_free(self) -> list[Tuple[Polynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[Polynomial, int]]:
        """Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> Polynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> Polynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def content(self) -> Polynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]]) -> list[Tuple[list[int], Polynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: list[Polynomial], grevlex: bool = True, print_stats: bool = False) -> list[Polynomial]:
        """Compute the Groebner basis of a polynomial system.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        If `print_stats=True` intermediate statistics will be printed.

        Examples
        --------
        >>> basis = Polynomial.groebner_basis(
        >>>     [Expression.parse("a b c d - 1").to_polynomial(),
        >>>     Expression.parse("a b c + a b d + a c d + b c d").to_polynomial(),
        >>>     Expression.parse("a b + b c + a d + c d").to_polynomial(),
        >>>     Expression.parse("a + b + c + d").to_polynomial()],
        >>>     grevlex=True,
        >>>     print_stats=True
        >>> )
        >>> for p in basis:
        >>>     print(p)
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import Expression
        >>> e = Expression.parse('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def replace(self, x: Expression, v: Polynomial) -> Polynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """

    @classmethod
    def interpolate(_cls, x: Expression, sample_points: Sequence[Expression | int], values: Sequence[Polynomial]) -> Polynomial:
        """Perform Newton interpolation in the variable `x` given the sample points
        `sample_points` and the values `values`.

        Examples
        --------
        >>> x, y = S('x', 'y')
        >>> a = Polynomial.interpolate(
        >>>         x, [4, 5], [(y**2+5).to_polynomial(), (y**3).to_polynomial()])
        >>> print(a)

        yields `25-5*x+5*y^2-y^2*x-4*y^3+y^3*x`.
        """


class IntegerPolynomial:
    """A Symbolica polynomial with integer coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str]) -> IntegerPolynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = IntegerPolynomial.parse('3*x^2+y+y*4', ['x', 'y'])

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

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: IntegerPolynomial) -> Tuple[IntegerPolynomial, IntegerPolynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> IntegerPolynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def resultant(self, rhs: IntegerPolynomial, var: Expression) -> IntegerPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def factor_square_free(self) -> list[Tuple[IntegerPolynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial().to_integer_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[IntegerPolynomial, int]]:
        """Factorize the polynomial.

        The polynomial must be univariate.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial().to_integer_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> IntegerPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial().to_integer_polynomial()
        >>> print(p.derivative(x))
        """

    def content(self) -> IntegerPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial().to_integer_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]]) -> list[Tuple[list[int], IntegerPolynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial().to_integer_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import Expression
        >>> e = Expression.parse('x*y+2*x+x^2')
        >>> p = e.to_polynomial().to_integer_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def replace(self, x: Expression, v: Polynomial) -> Polynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """


class NumberFieldPolynomial:
    """A Symbolica polynomial with rational coefficients."""

    def __copy__(self) -> NumberFieldPolynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldNumberFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: NumberFieldPolynomial) -> Tuple[NumberFieldPolynomial, NumberFieldPolynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> NumberFieldPolynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def resultant(self, rhs: NumberFieldPolynomial, var: Expression) -> NumberFieldPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def factor_square_free(self) -> list[Tuple[NumberFieldPolynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[NumberFieldPolynomial, int]]:
        """Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> NumberFieldPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> NumberFieldPolynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def content(self) -> NumberFieldPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]]) -> list[Tuple[list[int], NumberFieldPolynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: list[NumberFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[NumberFieldPolynomial]:
        """Compute the Groebner basis of a polynomial system.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        If `print_stats=True` intermediate statistics will be printed.
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import Expression
        >>> e = Expression.parse('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def replace(self, x: Expression, v: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """

    def get_minimal_polynomial(self) -> Polynomial:
        """Get the minimal polynomial of the algebraic extension."""

    def extend(self, b: NumberFieldPolynomial) -> Tuple[NumberFieldPolynomial, Polynomial, Polynomial]:
        """Extend the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
        is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.

        `b`  must be irreducible over `R` and `R[a]`; this is not checked."""


class FiniteFieldPolynomial:
    """A Symbolica polynomial with finite field coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int) -> FiniteFieldPolynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = FiniteFieldPolynomial.parse('18*x^2+y+y*4', ['x', 'y'], 17)

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> FiniteFieldPolynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def format(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: FiniteFieldPolynomial) -> Tuple[FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> FiniteFieldPolynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def resultant(self, rhs: FiniteFieldPolynomial, var: Expression) -> FiniteFieldPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def factor_square_free(self) -> list[Tuple[FiniteFieldPolynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial().to_finite_field(7)
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[FiniteFieldPolynomial, int]]:
        """Factorize the polynomial.

        The polynomial must be univariate.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial().to_finite_field(7)
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> FiniteFieldPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> FiniteFieldPolynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def content(self) -> FiniteFieldPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]]) -> list[Tuple[list[int], FiniteFieldPolynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: list[FiniteFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[FiniteFieldPolynomial]:
        """Compute the Groebner basis of a polynomial system.

        Examples
        --------
        >>> basis = Polynomial.groebner_basis(
        >>>     [Expression.parse("a b c d - 1").to_polynomial(),
        >>>     Expression.parse("a b c + a b d + a c d + b c d").to_polynomial(),
        >>>     Expression.parse("a b + b c + a d + c d").to_polynomial(),
        >>>     Expression.parse("a + b + c + d").to_polynomial()],
        >>>     grevlex=True,
        >>>     print_stats=True
        >>> )
        >>> for p in basis:
        >>>     print(p)

        Parameters
        ----------
        grevlex: bool
            If `True`, reverse graded lexicographical ordering is used, otherwise the ordering is lexicographical.
        print_stats: bool
            If `True`, intermediate statistics will be printed.
        """

    def replace(self, x: Expression, v: Polynomial) -> Polynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression."""


class RationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: Polynomial, den: Polynomial) -> RationalPolynomial:
        """Create a new rational polynomial from a numerator and denominator polynomial."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str]) -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = RationalPolynomial.parse('(3/4*x^2+y+y*4)/(1+x)', ['x', 'y'])

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

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def numerator(self) -> Polynomial:
        """Get the numerator."""

    def denominator(self) -> Polynomial:
        """Get the denominator."""

    def __add__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Add two rational polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Multiply two rational polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> RationalPolynomial:
        """Negate the rational polynomial."""

    def gcd(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

    def to_finite_field(self, prime: int) -> FiniteFieldRationalPolynomial:
        """Convert the coefficients of the rational polynomial to a finite field with prime `prime`."""

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import Expression
        >>> e = Expression.parse('(x*y+2*x+x^2)/(x^7+y+1)')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def derivative(self, x: Expression) -> RationalPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))
        """

    def apart(self, x: Expression) -> List[RationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class FiniteFieldRationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: FiniteFieldPolynomial, den: FiniteFieldPolynomial) -> FiniteFieldRationalPolynomial:
        """Create a new rational polynomial from a numerator and denominator polynomial."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int) -> FiniteFieldRationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = FiniteFieldRationalPolynomial.parse('3*x^2+y+y*4', ['x', 'y'], 17)

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __copy__(self) -> FiniteFieldRationalPolynomial:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the rational polynomial into a LaTeX string."""

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Add two rational polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Multiply two rational polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> FiniteFieldRationalPolynomial:
        """Negate the rational polynomial."""

    def gcd(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

    def derivative(self, x: Expression) -> RationalPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))
        """

    def apart(self, x: Expression) -> List[FiniteFieldRationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.symbol('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class Matrix:
    """A matrix with rational polynomial coefficients."""

    def __new__(cls, nrows: int, ncols: int) -> Matrix:
        """Create a new zeroed matrix with `nrows` rows and `ncols` columns."""

    @classmethod
    def identity(cls, nrows: int) -> Matrix:
        """Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere."""

    @classmethod
    def eye(cls, diag: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere."""

    @classmethod
    def vec(cls, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new column vector from a list of scalars."""

    @classmethod
    def from_linear(cls, nrows: int, ncols: int, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new matrix from a 1-dimensional vector of scalars."""

    @classmethod
    def from_nested(cls, entries: Sequence[Sequence[RationalPolynomial | Polynomial | Expression | int]]) -> Matrix:
        """Create a new matrix from a 2-dimensional vector of scalars."""

    def nrows(self) -> int:
        """Get the number of rows in the matrix."""

    def ncols(self) -> int:
        """Get the number of columns in the matrix."""

    def is_zero(self) -> bool:
        """Return true iff every entry in the matrix is zero."""

    def is_diagonal(self) -> bool:
        """Return true iff every non- main diagonal entry in the matrix is zero."""

    def transpose(self) -> Matrix:
        """Return the transpose of the matrix."""

    def inv(self) -> Matrix:
        """Return the inverse of the matrix, if it exists."""

    def det(self) -> RationalPolynomial:
        """Return the determinant of the matrix."""

    def solve(self, b: Matrix) -> Matrix:
        """Solve `A * x = b` for `x`, where `A` is the current matrix."""

    def solve_any(self, b: Matrix) -> Matrix:
        """Solve `A * x = b` for `x`, where `A` is the current matrix and return any solution if the
        system is underdetermined.
        """

    def row_reduce(self, max_col: int) -> int:
        """Row-reduce the first `max_col` columns of the matrix in-place using Gaussian elimination and return the rank."""

    def augment(self, b: Matrix) -> Matrix:
        """Augment the matrix with another matrix, e.g. create `[A B]` from matrix `A` and `B`.

        Returns an error when the matrices do not have the same number of rows.
        """

    def split_col(self, col: int) -> Tuple[Matrix, Matrix]:
        """Split the matrix into two matrices at column `col`."""

    def content(self) -> RationalPolynomial:
        """Get the content, i.e., the GCD of the coefficients."""

    def primitive_part(self) -> Matrix:
        """Construct the same matrix, but with the content removed."""

    def map(self, f: Callable[[RationalPolynomial], RationalPolynomial]) -> Matrix:
        """Apply a function `f` to every entry of the matrix."""

    def format(
        self,
        pretty_matrix=True,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
        precision: Optional[int] = None,
    ) -> str:
        """
        Convert the matrix into a human-readable string, with tunable settings.
        """

    def to_latex(self) -> str:
        """Convert the matrix into a LaTeX string."""

    def __copy__(self) -> Matrix:
        """Copy the matrix."""

    def __getitem__(self, key: Tuple[int, int]) -> RationalPolynomial:
        """Get the entry at position `key` in the matrix."""

    def __str__(self) -> str:
        """Print the matrix in a human-readable format."""

    def __eq__(self, other: Matrix) -> bool:
        """Compare two matrices."""

    def __neq__(self, other: Matrix) -> bool:
        """Compare two matrices."""

    def __add__(self, rhs: Matrix) -> Matrix:
        """Add two matrices `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: Matrix) -> Matrix:
        """Subtract matrix `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply `self` and `rhs`, returning the result."""

    def __rmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply  `rhs` and `self`, returning the result."""

    def __matmul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply `self` and `rhs`, returning the result."""

    def __rmatmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply  `rhs` and `self`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Divide this matrix by scalar `rhs` and return the result."""

    def __neg__(self) -> Matrix:
        """Negate the matrix, returning the result."""


class Evaluator:
    """An optimized evaluator of an expression."""

    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        inline_asm: str = 'default',
        optimization_level: int = 3,
        compiler_path: Optional[str] = None,
    ) -> CompiledEvaluator:
        """Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
        The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
        """

    def evaluate(self, inputs: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the expression for multiple inputs and return the result."""

    def evaluate_flat(self, inputs: Sequence[float]) -> List[float]:
        """Evaluate the expression for multiple inputs that are flattened and return the flattened result.
        This method has less overhead than `evaluate`."""

    def evaluate_complex(self, inputs: Sequence[Sequence[complex]]) -> List[List[complex]]:
        """Evaluate the expression for multiple inputs and return the result."""

    def evaluate_complex_flat(self, inputs: Sequence[complex]) -> List[complex]:
        """Evaluate the expression for multiple inputs that are flattened and return the flattened result.
        This method has less overhead than `evaluate_complex`."""


class CompiledEvaluator:
    """An compiled evaluator of an expression. This will give the highest performance of 
    all evaluators."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
    ) -> CompiledEvaluator:
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the expression for multiple inputs and return the result."""

    def evaluate_flat(self, inputs: Sequence[float]) -> List[float]:
        """Evaluate the expression for multiple inputs that are flattened and return the flattened result.
        This method has less overhead than `evaluate`."""

    def evaluate_complex(self, inputs: Sequence[Sequence[complex]]) -> List[List[complex]]:
        """Evaluate the expression for multiple inputs and return the result."""

    def evaluate_complex_flat(self, inputs: Sequence[complex]) -> List[complex]:
        """Evaluate the expression for multiple inputs that are flattened and return the flattened result.
        This method has less overhead than `evaluate_complex`."""


class NumericalIntegrator:
    """A numerical integrator for high-dimensional integrals."""

    @classmethod
    def continuous(
        _cls,
        n_dims: int,
        n_bins: int = 128,
        min_samples_for_update: int = 100,
        bin_number_evolution: Optional[Sequence[int]] = None,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """Create a new continuous grid for the numerical integrator."""

    @classmethod
    def discrete(
        _cls,
        bins: Sequence[Optional[NumericalIntegrator]],
        max_prob_ratio: float = 100.0,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """Create a new discrete grid for the numerical integrator. Each
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

    @classmethod
    def rng(
        _cls,
        seed: int,
        stream_id: int
    ) -> RandomNumberGenerator:
        """Create a new random number generator, suitable for use with the integrator.
        Each thread of instance of the integrator should have its own random number generator,
        that is initialized with the same seed but with a different stream id."""

    @classmethod
    def import_grid(
        _cls,
        grid: bytes
    ) -> NumericalIntegrator:
        """Import an exported grid from another thread or machine.
        Use `export_grid` to export the grid."""

    def export_grid(
        self,
    ) -> bytes:
        """Export the grid, so that it can be sent to another thread or machine.
        Use `import_grid` to load the grid."""

    def get_live_estimate(
        self,
    ) -> Tuple[float, float, float, float, float, int]:
        """Get the estamate of the average, error, chi-squared, maximum negative and positive evaluations, and the number of processed samples
        for the current iteration, including the points submitted in the current iteration."""

    def sample(self, num_samples: int, rng: RandomNumberGenerator) -> List[Sample]:
        """Sample `num_samples` points from the grid using the random number generator
        `rng`. See `rng()` for how to create a random number generator."""

    def merge(self, other: NumericalIntegrator) -> None:
        """Add the accumulated training samples from the grid `other` to the current grid.
        The grid structure of `self` and `other` must be equivalent."""

    def add_training_samples(self, samples: Sequence[Sample], evals: Sequence[float]) -> None:
        """Add the samples and their corresponding function evaluations to the grid.
        Call `update` after to update the grid and to obtain the new expected value for the integral."""

    def update(self, discrete_learning_rate: float, continous_learning_rate: float) -> Tuple[float, float, float]:
        """Update the grid using the `discrete_learning_rate` and `continuous_learning_rate`.
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
        >>>     avg, err, chi_sq = integrator.update(1.5, 1.5)
        >>>     print('Iteration {}: {:.6} +- {:.6}, chi={:.6}'.format(i+1, avg, err, chi_sq))
        """

    def integrate(
        self,
        integrand: Callable[[Sequence[Sample]], List[float]],
        max_n_iter: int = 10000000,
        min_error: float = 0.01,
        n_samples_per_iter: int = 10000,
        seed: int = 0,
        show_stats: bool = True,
    ) -> Tuple[float, float, float]:
        """Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
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
    """A sample from the Symbolica integrator. It could consist of discrete layers,
    accessible with `d` (empty when there are not discrete layers), and the final continuous layer `c` if it is present."""

    """ The weights the integrator assigned to this sample point, given in descending order:
    first the discrete layer weights and then the continuous layer weight."""
    weights: List[float]
    d: List[int]
    """ A sample point per (nested) discrete layer. Empty if not present."""
    c: List[float]
    """ A sample in the continuous layer. Empty if not present."""


class RandomNumberGenerator:
    """A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
    A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).

    Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
    which is an instance counter starting at 0."""

    def __new__(_cls, seed: int, stream_id: int):
        """Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
        each thread or instance generating samples should use the same `seed` but a different `stream_id`."""


class Graph:
    """A graph that supported directional edges, parallel edges, self-edges and expression data on the nodes and edges.

    Warning: modifying the graph if it is contained in a `dict` or `set` will invalidate the hash.
    """

    def __new__(_cls):
        """Create a new empty graph."""

    def __str__(self) -> str:
        """Print the graph in a human-readable format."""

    def __hash__(self) -> int:
        """Hash the graph."""

    def __copy__(self) -> Graph:
        """Copy the graph."""

    def __len__(self) -> int:
        """Get the number of nodes in the graph."""

    def __eq__(self, other: Graph) -> bool:
        """Compare two graphs."""

    def __neq__(self, other: Graph) -> bool:
        """Compare two graphs."""

    def __getitem__(self, idx: int) -> tuple[Sequence[int], Expression]:
        """Get the `idx`th node, consisting of the edge indices and the data."""

    @classmethod
    def generate(_cls,
                 external_nodes: Sequence[tuple[Expression | int, Tuple[Optional[bool], Expression | int]]],
                 vertex_signatures: Sequence[Sequence[Tuple[Optional[bool], Expression | int]]],
                 max_vertices: Optional[int] = None,
                 max_loops: Optional[int] = None,
                 max_bridges: Optional[int] = None,
                 allow_self_loops: bool = False,) -> dict[Graph, Expression]:
        """Generate all connected graphs with `external_edges` half-edges and the given allowed list
        of vertex connections. The vertex signatures are given in terms of an edge direction (or `None` if
        there is no direction) and edge data.

        Returns the canonical form of the graph and the size of its automorphism group (including edge permutations).

        Examples
        --------
        >>> from symbolica import *
        >>> g, q, gh = S('g', 'q', 'gh')
        >>> gp, qp, qbp, ghp, ghbp = (None, g), (True, q), (False, q), (True, gh), (False, gh)
        >>> graphs = Graph.generate([(1, gp), (2, gp)],
        >>>                         [[gp, gp, gp], [gp, gp, gp, gp],
        >>>                         [qp, qbp, gp], [ghp, ghbp, gp]], max_loops=2)
        >>> for (g, sym) in graphs.items():
        >>>     print(f'Symmetry factor = 1/{sym}:')
        >>>     print(g.to_dot())

        generates all connected graphs up to 2 loops with the specified vertices.

        Parameters
        ----------
        external_nodes: Sequence[tuple[Expression | int, Tuple[Optional[bool], Expression | int]]]
            The external edges, consisting of a tuple of the node data and a tuple of the edge direction and edge data.
            If the node data is the same, flip symmetries will be recognized.
        vertex_signatures: Sequence[Sequence[Tuple[Optional[bool], Expression | int]]]
            The allowed connections for each vertex.
        max_vertices: int, optional
            The maximum number of vertices in the graph.
        max_loops: int, optional
            The maximum number of loops in the graph.
        max_bridges: int, optional
            The maximum number of bridges in the graph.
        allow_self_loops: bool, optional
            Whether self-edges are allowed.
        """

    def to_dot(self) -> str:
        """Convert the graph to a graphviz dot string."""

    def to_mermaid(self) -> str:
        """Convert the graph to a mermaid string."""

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""

    def num_edges(self) -> int:
        """Get the number of edges in the graph."""

    def num_loops(self) -> int:
        """Get the number of loops in the graph."""

    def node(self, idx: int) -> tuple[Sequence[int], Expression]:
        """Get the `idx`th node, consisting of the edge indices and the data."""

    def edge(self, idx: int) -> tuple[int, int, bool, Expression]:
        """Get the `idx`th edge, consisting of the the source vertex, target vertex, whether the edge is directed, and the data."""

    def add_node(self, data: Optional[Expression | int] = None) -> int:
        """Add a node with data `data` to the graph, returning the index of the node.
        The default data is the number 0.
        """

    def add_edge(self, source: int, target: int, directed: bool = False, data: Optional[Expression | int] = None) -> int:
        """Add an edge between the `source` and `target` nodes, returning the index of the edge.

        Optionally, the edge can be set as directed. The default data is the number 0.
        """

    def set_node_data(self, index: int, data: Expression | int) -> Expression:
        """Set the data of the node at index `index`, returning the old data."""

    def set_edge_data(self, index: int, data: Expression | int) -> Expression:
        """Set the data of the edge at index `index`, returning the old data."""

    def set_directed(self, index: int, directed: bool) -> bool:
        """Set the directed status of the edge at index `index`, returning the old value."""

    def canonize(self) -> Tuple[Graph, Sequence[int], Expression, Sequence[int]]:
        """Write the graph in a canonical form. Returns the canonized graph, the vertex map, the automorphism group size, and the orbit."""

    def canonize_edges(self) -> None:
        """Sort and relabel the edges of the graph, keeping the vertices fixed."""

    def is_isomorphic(self, other: Graph) -> bool:
        """Check if the graph is isomorphic to another graph."""


class Integer:
    @classmethod
    def prime_iter(_cls, start: int = 1) -> Iterator[int]:
        """Create an iterator over all 64-bit prime numbers starting from `start`."""

    @classmethod
    def is_prime(_cls, n: int) -> bool:
        """Check if the 64-bit number `n` is a prime number."""

    @classmethod
    def solve_integer_relation(_cls, x: Sequence[int | float | Decimal], tolerance: float | Decimal, max_coeff: Optional[int] = None, gamma: Optional[float | Decimal] = None) -> Sequence[int]:
        """Use the PSLQ algorithm to find a vector of integers `a` that satisfies `a.x = 0`,
        where every element of `a` is less than `max_coeff`, using a specified tolerance and number
        of iterations. The parameter `gamma` must be more than or equal to `2/sqrt(3)`.

        Examples
        --------
        Solve a `32.0177=b*pi+c*e` where `b` and `c` are integers:

        >>> r = Integer.solve_integer_relation([-32.0177, 3.1416, 2.7183], 1e-5, 100)
        >>> print(r)

        yields `[1,5,6]`.
        """
