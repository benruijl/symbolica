"""
Symbolica Python API.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, overload, Iterator, Optional, Sequence, Tuple, List


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


def get_offline_license_key() -> str:
    """Get a license key for offline use, generated from a licensed Symbolica session. The key will remain valid for 24 hours."""


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
    >>> x = Expression.var('x')
    >>> e = x**2 + 2 - x + 1 / x**4
    >>> print(e)
    """

    E: Expression
    """Euler's number `e`."""

    PI: Expression
    """The mathematical constant `π`."""

    I: Expression
    """The mathematical constant `i`, where `i^2 = -1`."""

    @classmethod
    def var(_cls, name: str) -> Expression:
        """
        Create a Symbolica expression that is a single variable.

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
    def fun(_cls, name: str, is_symmetric: bool = False, is_antisymmetric: bool = False, is_linear: bool = False) -> Function:
        """
        Create a new Symbolica function with a given name.

        Examples
        --------
        >>> f = Expression.fun('f')
        >>> e = f(1,2)
        >>> print(e)
        f(1,2)


        Define a symmetric function:
        >>> f = Expression.fun('f', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)
        f(1,2)

        Define a linear and symmetric function:
        >>> p1, p2, p3, p4 = Expression.vars('p1', 'p2', 'p3', 'p4')
        >>> dot = Expression.fun('dot', is_symmetric=True, is_linear=True)
        >>> e = dot(p2+2*p3,p1+3*p2-p3)
        dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)
        """

    @classmethod
    def funs(_cls, *names: str) -> Sequence[Function]:
        """
        Create a Symbolica function for every name in `*names`.
        """

    @classmethod
    def num(_cls, num: int | float, max_denom: Optional[int] = None) -> Expression:
        """Create a new Symbolica number from an int or a float.
        A floating point number is converted to its rational number equivalent,
        but it can also be truncated by specifying the maximal denominator value.

        Examples
        --------
        >>> e = Expression.num(1) / 2
        >>> print(e)
        1/2

        >>> print(Expression.num(0.33))
        >>> print(Expression.num(0.33, 5))
        5944751508129055/18014398509481984
        1/3
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

    def __copy__(self) -> Expression:
        """
        Copy the expression.
        """

    def __str__(self) -> str:
        """
        Convert the expression into a human-readable string.
        """

    def get_byte_size(self) -> int:
        """ Get the number of bytes that this expression takes up in memory."""

    def pretty_str(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
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
        Multiply this expression with `other`, returning the result.
        """

    def __rmul__(self, other: Expression | int) -> Expression:
        """
        Multiply this expression with `other`, returning the result.
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

    def req(
        self,
        filter_fn: Callable[[Expression], bool],
    ) -> PatternRestriction:
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

    def req_lt(self, num: Expression | int, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun("f")
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_lt(2))
        """

    def req_gt(self, num: Expression | int, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun("f")
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_gt(2))
        """

    def req_le(self, num: Expression | int, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun("f")
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace_all(f(x_), 1, x_.req_le(2))
        """

    def req_ge(self, num: Expression | int, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun("f")
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
        >>> x_, y_ = Expression.var('x_', 'y_')
        >>> f = Expression.fun("f")
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
        >>> x_, y_ = Expression.var('x_', 'y_')
        >>> f = Expression.fun("f")
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
        >>> x_, y_ = Expression.var('x_', 'y_')
        >>> f = Expression.fun("f")
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
        >>> x_, y_ = Expression.var('x_', 'y_')
        >>> f = Expression.fun("f")
        >>> e = f(1,2)
        >>> e = e.replace_all(f(x_,y_), 1, x_.req_cmp_ge(y_))
        """

    def __eq__(self, other: Expression | int) -> bool:
        """
        Compare two expressions.
        """

    def __neq__(self, other: Expression | int) -> bool:
        """
        Compare two expressions.
        """

    def __lt__(self, other: Expression | int) -> bool:
        """
        Compare two expressions. Both expressions must be a number.
        """

    def __le__(self, other: Expression | int) -> bool:
        """
        Compare two expressions. Both expressions must be a number.
        """

    def __gt__(self, other: Expression | int) -> bool:
        """
        Compare two expressions. Both expressions must be a number.
        """

    def __ge__(self, other: Expression | int) -> bool:
        """
        Compare two expressions. Both expressions must be a number.
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
    ) -> Expression:
        """
        Map the transformations to every term in the expression.
        The execution happen in parallel.


        No new functions or variables can be defined and no new
        expressions can be parsed inside the map. Doing so will
        result in a deadlock.

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
        vars : Sequence[Expression]
                A list of variables
        """

    def expand(self) -> Expression:
        """
        Expand the expression.
        """

    def collect(
        self,
        x: Expression,
        key_map: Optional[Callable[[Expression], Expression]] = None,
        coeff_map: Optional[Callable[[Expression], Expression]] = None,
    ) -> Expression:
        """
        Collect terms involving the same power of `x`, where `x` is a variable or function name.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Both the key (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` respectively.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y = Expression.vars('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x))

        yields `x^2+x*(y+5)+5`.

        >>> from symbolica import Expression
        >>> x, y = Expression.vars('x', 'y')
        >>> var, coeff = Expression.funs('var', 'coeff')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x, key_map=lambda x: var(x), coeff_map=lambda x: coeff(x)))

        yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.

        Parameters
        ----------
        key_map: A function to be applied to the quantity collected in
        coeff_map: A function to be applied to the coefficient
        """

    def coefficient_list(
        self, x: Expression
    ) -> Sequence[Tuple[Expression, Expression]]:
        """Collect terms involving the same power of `x`, where `x` is a variable or function name.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = Expression.vars('x', 'y')
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
        >>> x, y = Expression.vars('x', 'y')
        >>> e = 5*x + x * y + x**2 + y*x**2
        >>> print(e.coefficient(x**2))

        yields

        ```
        y + 1
        ```
        """

    def derivative(self, x: Expression) -> Expression:
        """Derive the expression w.r.t the variable `x`."""

    def taylor_series(
        self,
        x: Expression,
        expansion_point: Expression | int,
        depth: int,
    ) -> Expression:
        """Taylor expand in `x` around `expansion_point` to depth `depth`."""

    def to_polynomial(self, vars: Optional[Sequence[Expression]] = None) -> Polynomial:
        """Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
        All non-polynomial parts will be converted to new, independent variables.
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

    def to_rational_polynomial_small_exponent(
        self,
        vars: Optional[Sequence[Expression]] = None,
    ) -> RationalPolynomial:
        """Similar to `to_rational_polynomial()`, but the power of each variable is limited to 255."""

    def match(
        self,
        lhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
    ) -> MatchIterator:
        """
        Return an iterator over the pattern `self` matching to `lhs`.
        Restrictions on pattern can be supplied through `cond`.

        The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
        The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
        depending on `level_is_tree_depth`.

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
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
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

        Parameters
        ----------
        lhs: The pattern to match.
        rhs: The right-hand side to replace the matched subexpression with.
        cond: Conditions on the pattern.
        level_range: Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth: If set to `True`, the level is increased when going one level deeper in the expression tree.
        """

    def replace_all(
        self,
        pattern: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        repeat: Optional[bool] = False,
    ) -> Expression:
        """
        Replace all subexpressions matching the pattern `pattern` by the right-hand side `rhs`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
        >>> print(r)

        Parameters
        ----------
        self: The expression to match and replace on.
        pattern: The pattern to match.
        rhs: The right-hand side to replace the matched subexpression with.
        cond: Conditions on the pattern.
        non_greedy_wildcards: Wildcards that try to match as little as possible.
        level_range: Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth: If set to `True`, the level is increased when going one level deeper in the expression tree.
        repeat: If set to `True`, the entire operation will be repeated until there are no more matches.
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
        >>> x, y, c = Expression.vars('x', 'y', 'c')
        >>> f = Expression.fun('f')
        >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
        >>> print('x =', x_r, ', y =', y_r)
        """

    def evaluate(
        self, constants: dict[Expression, float], funs: dict[Expression | Function, Callable[[Sequence[float]], float]]
    ) -> float:
        """Evaluate the expression, using a map of all the variables and
        user functions to a float.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> f = Expression.fun('f')
        >>> e = Expression.parse('cos(x)')*3 + f(x,2)
        >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
        """

    def evaluate_complex(
        self, constants: dict[Expression, float | complex], funs: dict[Expression | Function, Callable[[Sequence[complex]], float | complex]]
    ) -> complex:
        """Evaluate the expression, using a map of all the variables and
        user functions to a complex number.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x, y = Expression.vars('x', 'y')
        >>> e = Expression.parse('sqrt(x)')*y
        >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
        """


class PatternRestriction:
    """A restriction on wildcards."""

    def __and__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold)."""

    def __or__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold)."""

    def __invert__(self) -> PatternRestriction:
        """Create a new pattern restriction that takes the logical 'not' of the current restriction."""


class CompareOp:
    """One of the following comparison operators: `<`,`>`,`<=`,`>=`,`==`,`!=`."""


class Function:
    """A Symbolica function. Will turn into an expression or a transformer when called with arguments."""

    COEFF: Function
    """The built-in function that convert a rational polynomials to a coefficient."""

    COS: Function
    """The built-in cosine function."""

    SIN: Function
    """The built-in sine function."""

    EXP: Function
    """The built-in exponential function."""

    LOG: Function
    """The built-in logarithm function."""

    def __new__(_cls, name: str, is_symmetric: Optional[bool], is_antisymmetric: Optional[bool], is_linear: Optional[bool]) -> Function:
        """
        Create a new function from a `name`. Can be turned into a symmetric function
        using `is_symmetric=True` or into an antisymmetric function using `is_antisymmetric=True`.
        The function can be made multilinear using `is_linear=True`.

        Once attributes are defined on a function, they cannot be redefined later.
        """

    def is_symmetric(self) -> bool:
        """Returns `True` iff this function is symmetric."""

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
    def __call__(self, *args: Transformer | Expression | int) -> Transformer:
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
    """Operations that transform an expression."""

    def __new__(_cls) -> Transformer:
        """Create a new transformer for a term provided by `Expression.map`."""

    def expand(self) -> Transformer:
        """Create a transformer that expands products and powers.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x, x_ = Expression.vars('x', 'x_')
        >>> f = Expression.fun('f')
        >>> e = f((x+1)**2).replace_all(f(x_), x_.transform().expand())
        >>> print(e)
        """

    def prod(self) -> Transformer:
        """Create a transformer that computes the product of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.var('x__')
        >>> f = Expression.fun('f')
        >>> e = f(2,3).replace_all(f(x__), x__.transform().prod())
        >>> print(e)
        """

    def sum(self) -> Transformer:
        """Create a transformer that computes the sum of a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.var('x__')
        >>> f = Expression.fun('f')
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
        >>> x__ = Expression.var('x__')
        >>> f = Expression.fun('f')
        >>> e = f(2,3,4).replace_all(f(x__), x__.transform().nargs())
        >>> print(e)
        """

    def sort(self) -> Transformer:
        """Create a transformer that sorts a list of arguments.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.var('x__')
        >>> f = Expression.fun('f')
        >>> e = f(3,2,1).replace_all(f(x__), x__.transform().sort())
        >>> print(e)
        """

    def deduplicate(self) -> Transformer:
        """Create a transformer that removes elements from a list if they occur
        earlier in the list as well.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x__ = Expression.var('x__')
        >>> f = Expression.fun('f')
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
        >>> x, x__ = Expression.vars('x', 'x__')
        >>> f = Expression.fun('f')
        >>> e = (x + 1).replace_all(x__, f(x_.transform().split()))
        >>> print(e)
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
        >>> x_, f_id, g_id = Expression.vars('x_', 'f', 'g')
        >>> f = Expression.fun('f')
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
        >>> x_, f_id = Expression.vars('x_', 'f')
        >>> f = Expression.fun('f')
        >>> e = f(1,2,1,2).replace_all(f(x_), x_.transform().permutations(f_id)
        >>> print(e)

        yields:
        ```
        4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)
        ```
        """

    def map(self, f: Callable[[Expression], Expression | int]) -> Transformer:
        """Create a transformer that applies a Python function.

        Examples
        --------
        >>> from symbolica import Expression, Transformer
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = f(2).replace_all(f(x_), x_.transform().map(lambda r: r**2))
        >>> print(e)
        """

    def check_interrupt(self) -> Transformer:
        """Create a transformer that checks for a Python interrupt,
        such as ctrl-c and aborts the current transformer.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
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
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
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
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = Expression.parse("f(5)")
        >>> e = e.transform().chain(
        >>>     Transformer().expand(),
        >>>     Transformer().replace_all(f(x_), f(5))
        >>> ).execute()
        """

    def execute(self) -> Expression:
        """Execute the transformer.

        Examples
        --------
        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> e = (x+1)**5
        >>> e = e.transform().expand().execute()
        >>> print(e)
        """

    def derivative(self, x: Transformer | Expression) -> Transformer:
        """Create a transformer that derives `self` w.r.t the variable `x`."""

    def taylor_series(
        self,
        x: Expression,
        expansion_point: Expression,
        depth: int,
    ) -> Expression:
        """Create a transformer that Taylor expands in `x` around `expansion_point` to depth `depth`.

        Example
        -------
        >>> from symbolica import Expression
        >>> x, y = Expression.vars('x', 'y')
        >>> f = Expression.fun('f')
        >>>
        >>> e = 2* x**2 * y + f(x)
        >>> e = e.taylor_series(x, 0, 2)
        >>>
        >>> print(e)

        yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.
        """

    def replace_all(
        self,
        pat: Transformer | Expression | int,
        rhs: Transformer | Expression | int,
        cond: Optional[PatternRestriction] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
    ) -> Transformer:
        """
        Create a transformer that replaces all subexpressions matching the pattern `pat` by the right-hand side `rhs`.

        Examples
        --------

        >>> x, w1_, w2_ = Expression.vars('x','w1_','w2_')
        >>> f = Expression.fun('f')
        >>> e = f(3,x)
        >>> r = e.transform().replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
        >>> print(r)

        Parameters
        ----------
        pat: The pattern to match.
        rhs: The right-hand side to replace the matched subexpression with.
        cond: Conditions on the pattern.
        non_greedy_wildcards: Wildcards that try to match as little as possible.
        level_range: Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth: If set to `True`, the level is increased when going one level deeper in the expression tree.
        repeat: If set to `True`, the entire operation will be repeated until there are no more matches.
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
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
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
        >>> x_ = Expression.var('x_')
        >>> f = Expression.fun('f')
        >>> e = Expression.parse("f(5)")
        >>> e = e.transform().stats('replace', Transformer().replace_all(f(x_), 1)).execute()

        yields
        ```log
        Stats for replace:
            In  │ 1 │  10.00 B │
            Out │ 1 │   3.00 B │ ⧗ 40.15µs
        ```
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
    """An iterator over matches."""

    def __iter__(self) -> MatchIterator:
        """Create the iterator."""

    def __next__(self) -> dict[Expression, Expression]:
        """Return the next match."""


class ReplaceIterator:
    """An iterator over single replacments."""

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

    def pretty_str(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.pretty_str(symmetric_representation_for_finite_field=True))

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

    def quot_rem(self, rhs: Polynomial) -> Polynomial:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

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

    def optimize(self, iterations: int = 1000, to_file: str | None = None) -> Evaluator:
        """
        Optimize the polynomial for evaluation using `iterations` number of iterations.
        The optimized output can be exported in a C++ format using `to_file`.

        Returns an evaluator for the polynomial.
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
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def content(self) -> Polynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, x: Expression) -> list[Tuple[int, Polynomial]]:
        """Get the coefficient list in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
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
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
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

    def pretty_str(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.pretty_str(symmetric_representation_for_finite_field=True))

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

    def quot_rem(self, rhs: IntegerPolynomial) -> IntegerPolynomial:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

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
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def content(self) -> IntegerPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, x: Expression) -> list[Tuple[int, IntegerPolynomial]]:
        """Get the coefficient list in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
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
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """


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

    def pretty_str(
        self,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_finite_field: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        latex: bool = False,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.pretty_str(symmetric_representation_for_finite_field=True))

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

    def quot_rem(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __neg__(self) -> FiniteFieldPolynomial:
        """Negate the polynomial."""

    def gcd(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Compute the greatest common divisor (GCD) of two polynomials."""

    def resultant(self, rhs: FiniteFieldPolynomial, var: Expression) -> FiniteFieldPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def optimize(self, iterations: int = 1000, to_file: str | None = None) -> Evaluator:
        """
        Optimize the polynomial for evaluation using `iterations` number of iterations.
        The optimized output can be exported in a C++ format using `to_file`.

        Returns an evaluator for the polynomial.
        """

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
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def content(self) -> FiniteFieldPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import Expression
        >>> p = Expression.parse('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def coefficient_list(self, x: Expression) -> list[Tuple[int, FiniteFieldPolynomial]]:
        """Get the coefficient list in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
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
        grevlex: if `True`, reverse graded lexicographical ordering is used, otherwise the ordering is lexicographical.
        print_stats: if `True`, intermediate statistics will be printed.
        """

    def replace(self, x: Expression, v: Polynomial) -> Polynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> p = Expression.parse('x*y+2*x+x^2').to_polynomial()
        >>> r = Expression.parse('y+1').to_polynomial())
        >>> p.replace(x, r)
        """


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

    def apart(self, x: Expression) -> List[RationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class RationalPolynomialSmallExponent:
    """A Symbolica rational polynomial with variable powers limited to 255."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str]) -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = RationalPolynomialSmallExponent.parse('(3/4*x^2+y+y*4)/(1+x)', ['x', 'y'])

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

    def get_var_list(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(
        self, rhs: RationalPolynomialSmallExponent
    ) -> RationalPolynomialSmallExponent:
        """Add two rational polynomials `self` and `rhs`, returning the result."""

    def __sub__(
        self, rhs: RationalPolynomialSmallExponent
    ) -> RationalPolynomialSmallExponent:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(
        self, rhs: RationalPolynomialSmallExponent
    ) -> RationalPolynomialSmallExponent:
        """Multiply two rational polynomials `self` and `rhs`, returning the result."""

    def __truediv__(
        self, rhs: RationalPolynomialSmallExponent
    ) -> RationalPolynomialSmallExponent:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> RationalPolynomialSmallExponent:
        """Negate the rational polynomial."""

    def gcd(
        self, rhs: RationalPolynomialSmallExponent
    ) -> RationalPolynomialSmallExponent:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import Expression
        >>> e = Expression.parse('(x*y+2*x+x^2)/(x^7+y+1)')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def apart(self, x: Expression) -> List[RationalPolynomialSmallExponent]:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
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

    def apart(self, x: Expression) -> List[FiniteFieldRationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        Examples
        --------

        >>> from symbolica import Expression
        >>> x = Expression.var('x')
        >>> p = Expression.parse('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class Matrix:
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
        """Create a new row vector from a list of scalars."""

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

    def content(self) -> RationalPolynomial:
        """Get the content, i.e., the GCD of the coefficients."""

    def primitive_part(self) -> Matrix:
        """Construct the same matrix, but with the content removed."""

    def map(self, f: Callable[[RationalPolynomial], RationalPolynomial]) -> Matrix:
        """Apply a function `f` to every entry of the matrix."""

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
        """Multiply two matrices `self` and `rhs`, returning the result."""

    def __rmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Multiply two matrices `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Divide this matrix by scalar `rhs` and return the result."""

    def __neg__(self) -> Matrix:
        """Negate the matrix, returning the result."""


class Evaluator:
    def evaluate(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        """Evaluate the polynomial for multiple inputs and return the result."""


class NumericalIntegrator:
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
    accessible with `d` (empty when there are not discrete layers), and the final continous layer `c` if it is present."""

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
