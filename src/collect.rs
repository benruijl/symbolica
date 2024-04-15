use ahash::HashMap;

use crate::{
    atom::{Add, AsAtomView, Atom, AtomView, Symbol},
    domains::{integer::Z, rational::Q},
    state::Workspace,
};

impl Atom {
    /// Collect terms involving the same power of `x`, where `x` is a variable or function name, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub fn collect(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_view().collect(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub fn collect_into(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        out: &mut Atom,
    ) {
        self.as_view().collect_into(x, key_map, coeff_map, out)
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list(&self, x: Symbol) -> (Vec<(AtomView<'_>, Atom)>, Atom) {
        Workspace::get_local().with(|ws| self.as_view().coefficient_list_with_ws(x, ws))
    }

    /// Collect terms involving the literal occurrence of `x`.
    pub fn coefficient<'a, T: AsAtomView<'a>>(&self, x: T) -> Atom {
        Workspace::get_local().with(|ws| self.as_view().coefficient_with_ws(x.as_atom_view(), ws))
    }

    /// Write the expression over a common denominator.
    pub fn together(&self) -> Atom {
        self.as_view().together()
    }

    /// Write the expression as a sum of terms with minimal denominators.
    pub fn apart(&self, x: Symbol) -> Atom {
        self.as_view().apart(x)
    }
}

impl<'a> AtomView<'a> {
    /// Collect terms involving the same power of `x`, where `x` is a variable or function name, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub fn collect(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.collect_with_ws_into(x, ws, key_map, coeff_map, &mut out);
            out.into_inner()
        })
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub fn collect_into(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        out: &mut Atom,
    ) {
        Workspace::get_local().with(|ws| self.collect_with_ws_into(x, ws, key_map, coeff_map, out))
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub fn collect_with_ws_into(
        &self,
        x: Symbol,
        workspace: &Workspace,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        out: &mut Atom,
    ) {
        let (h, rest) = self.coefficient_list_with_ws(x, workspace);

        let mut add_h = workspace.new_atom();
        let add = add_h.to_add();

        fn map_key_coeff(
            key: AtomView,
            coeff: Atom,
            workspace: &Workspace,
            key_map: &Option<Box<dyn Fn(AtomView, &mut Atom)>>,
            coeff_map: &Option<Box<dyn Fn(AtomView, &mut Atom)>>,
            add: &mut Add,
        ) {
            let mut mul_h = workspace.new_atom();
            let mul = mul_h.to_mul();

            if let Some(key_map) = &key_map {
                let mut handle = workspace.new_atom();
                key_map(key, &mut handle);
                mul.extend(handle.as_view());
            } else {
                mul.extend(key);
            }

            if let Some(coeff_map) = &coeff_map {
                let mut handle = workspace.new_atom();
                coeff_map(coeff.as_view(), &mut handle);
                mul.extend(handle.as_view());
            } else {
                mul.extend(coeff.as_view());
            }

            add.extend(mul_h.as_view());
        }

        for (key, coeff) in h {
            map_key_coeff(key, coeff, workspace, &key_map, &coeff_map, add);
        }

        if key_map.is_some() {
            let key = workspace.new_num(1);
            map_key_coeff(key.as_view(), rest, workspace, &key_map, &coeff_map, add);
        } else if let Some(coeff_map) = coeff_map {
            let mut handle = workspace.new_atom();
            coeff_map(rest.as_view(), &mut handle);
            add.extend(handle.as_view());
        } else {
            add.extend(rest.as_view());
        }

        add_h.as_view().normalize(workspace, out);
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list(&self, x: Symbol) -> (Vec<(AtomView<'a>, Atom)>, Atom) {
        Workspace::get_local().with(|ws| self.coefficient_list_with_ws(x, ws))
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list_with_ws(
        &self,
        x: Symbol,
        workspace: &Workspace,
    ) -> (Vec<(AtomView<'a>, Atom)>, Atom) {
        let mut h = HashMap::default();
        let mut rest = workspace.new_atom();
        let mut rest_add = rest.to_add();

        match self {
            AtomView::Add(a) => {
                for arg in a.iter() {
                    arg.collect_factor_list(x, workspace, &mut h, &mut rest_add)
                }
            }
            _ => self.collect_factor_list(x, workspace, &mut h, &mut rest_add),
        }

        let mut rest_norm = Atom::new();
        rest.as_view().normalize(workspace, &mut rest_norm);

        let mut r: Vec<_> = h
            .into_iter()
            .map(|(k, v)| {
                (k, {
                    let mut a = Atom::new();
                    v.as_view().normalize(workspace, &mut a);
                    a
                })
            })
            .collect();
        r.sort_unstable_by_key(|(a, _)| *a);

        (r, rest_norm)
    }

    /// Check if a factor contains `x` at the ground level.
    #[inline]
    fn has_key(&self, x: Symbol) -> bool {
        match self {
            AtomView::Var(v) => v.get_symbol() == x,
            AtomView::Fun(f) => f.get_symbol() == x,
            AtomView::Pow(p) => {
                let (base, _) = p.get_base_exp();
                match base {
                    AtomView::Var(v) => v.get_symbol() == x,
                    AtomView::Fun(f) => f.get_symbol() == x,
                    _ => false,
                }
            }
            AtomView::Mul(_) => unreachable!("Mul is not a factor"),
            _ => false,
        }
    }

    fn collect_factor_list(
        &self,
        x: Symbol,
        workspace: &Workspace,
        h: &mut HashMap<AtomView<'a>, Add>,
        rest: &mut Add,
    ) {
        match self {
            AtomView::Add(_) => {}
            AtomView::Mul(m) => {
                if m.iter().any(|a| a.has_key(x)) {
                    let mut collected = workspace.new_atom();
                    let mul = collected.to_mul();

                    // we could have a double match if x*x(..)
                    // we then only collect on the first hit
                    let mut bracket = None;

                    for a in m.iter() {
                        if bracket.is_none() && a.has_key(x) {
                            bracket = Some(a);
                        } else {
                            mul.extend(a);
                        }
                    }

                    h.entry(bracket.unwrap())
                        .and_modify(|e| {
                            e.extend(collected.as_view());
                        })
                        .or_insert({
                            let mut a = Add::new();
                            a.extend(collected.as_view());
                            a
                        });

                    return;
                }
            }
            _ => {
                if self.has_key(x) {
                    // add the coefficient 1
                    let collected = workspace.new_num(1);
                    h.entry(*self)
                        .and_modify(|e| {
                            e.extend(collected.as_view());
                        })
                        .or_insert({
                            let mut a = Add::new();
                            a.extend(collected.as_view());
                            a
                        });
                    return;
                }
            }
        }

        rest.extend(*self);
    }

    /// Collect terms involving the literal occurrence of `x`.
    pub fn coefficient(&self, x: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| self.coefficient_with_ws(x, ws))
    }

    /// Collect terms involving the literal occurrence of `x`.
    pub fn coefficient_with_ws(&self, x: AtomView<'_>, workspace: &Workspace) -> Atom {
        let mut coeffs = workspace.new_atom();
        let mut coeff_add = coeffs.to_add();

        match self {
            AtomView::Add(a) => {
                for arg in a.iter() {
                    arg.collect_factor(x, workspace, &mut coeff_add)
                }
            }
            _ => self.collect_factor(x, workspace, &mut coeff_add),
        }

        let mut rest_norm = Atom::new();
        coeffs.as_view().normalize(workspace, &mut rest_norm);
        rest_norm
    }

    fn collect_factor(&self, x: AtomView<'_>, workspace: &Workspace, coeff: &mut Add) {
        match self {
            AtomView::Add(_) => {}
            AtomView::Mul(m) => {
                if m.iter().any(|a| a == x) {
                    let mut collected = workspace.new_atom();
                    let mul = collected.to_mul();

                    // we could have a double match if x*x(..)
                    // we then only collect on the first hit
                    let mut bracket = None;

                    for a in m.iter() {
                        if bracket.is_none() && a == x {
                            bracket = Some(a);
                        } else {
                            mul.extend(a);
                        }
                    }

                    coeff.extend(collected.as_view());
                }
            }
            _ => {
                if *self == x {
                    // add the coefficient 1
                    let collected = workspace.new_num(1);
                    coeff.extend(collected.as_view());
                }
            }
        }
    }

    /// Write the expression over a common denominator.
    pub fn together(&self) -> Atom {
        let mut out = Atom::new();
        self.together_into(&mut out);
        out
    }

    /// Write the expression over a common denominator.
    pub fn together_into(&self, out: &mut Atom) {
        self.to_rational_polynomial::<_, _, u32>(&Q, &Z, None)
            .to_expression_into(out);
    }

    /// Write the expression as a sum of terms with minimal denominators.
    pub fn apart(&self, x: Symbol) -> Atom {
        let mut out = Atom::new();

        Workspace::get_local().with(|ws| {
            self.apart_with_ws_into(x, ws, &mut out);
        });

        out
    }

    /// Write the expression as a sum of terms with minimal denominators.
    pub fn apart_with_ws_into(&self, x: Symbol, ws: &Workspace, out: &mut Atom) {
        let poly = self.to_rational_polynomial::<_, _, u32>(&Q, &Z, None);
        if let Some(v) = poly.get_variables().iter().position(|v| v == &x.into()) {
            let mut a = ws.new_atom();
            let add = a.to_add();

            let mut a = ws.new_atom();
            for x in poly.apart(v) {
                x.to_expression_into(&mut a);
                add.extend(a.as_view());
            }

            add.as_view().normalize(ws, out);
        } else {
            out.set_from_view(self);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, FunctionBuilder},
        fun,
        state::State,
    };

    #[test]
    fn coefficient_list() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = State::get_symbol("v1");

        let (r, rest) = input.coefficient_list(x);

        let res = vec![
            (
                Atom::parse("v1").unwrap(),
                Atom::parse("v3+5*v2+1").unwrap(),
            ),
            (Atom::parse("v1^2").unwrap(), Atom::parse("1").unwrap()),
            (Atom::parse("v1^3").unwrap(), Atom::parse("1").unwrap()),
        ];
        let res_rest = Atom::parse("v2^2+f1(5,v1)+2").unwrap();

        let res_ref = res
            .iter()
            .map(|(a, b)| (a.as_view(), b.clone()))
            .collect::<Vec<_>>();

        assert_eq!(r, res_ref);
        assert_eq!(rest, res_rest);
    }

    #[test]
    fn collect() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = State::get_symbol("v1");

        let out = input.collect(x, None, None);

        let ref_out = Atom::parse("v1^2+v1^3+v2^2+f1(5,v1)+v1*(5*v2+v3+1)+2").unwrap();
        assert_eq!(out, ref_out)
    }

    #[test]
    fn collect_wrap() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = State::get_symbol("v1");
        let key = State::get_symbol("f3");
        let coeff = State::get_symbol("f4");
        println!("> Collect in x with wrapping:");
        let out = input.collect(
            x,
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = fun!(key, out);
            })),
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = fun!(coeff, out);
            })),
        );

        let ref_out = Atom::parse(
            "f3(1)*f4(v2^2+f1(5,v1)+2)+f3(v1)*f4(5*v2+v3+1)+f3(v1^2)*f4(1)+f3(v1^3)*f4(1)",
        )
        .unwrap();

        assert_eq!(out, ref_out);
    }

    #[test]
    fn together() {
        let input = Atom::parse("v1^2/2+v1^3/v4*v2+v3/(1+v4)").unwrap();
        let out = input.together();

        let ref_out =
            Atom::parse("(2*v4+2*v4^2)^-1*(2*v3*v4+v1^2*v4+v1^2*v4^2+2*v1^3*v2+2*v1^3*v2*v4)")
                .unwrap();

        assert_eq!(out, ref_out);
    }

    #[test]
    fn apart() {
        let input =
            Atom::parse("(2*v4+2*v4^2)^-1*(2*v3*v4+v1^2*v4+v1^2*v4^2+2*v1^3*v2+2*v1^3*v2*v4)")
                .unwrap();
        let out = input.apart(State::get_symbol("v4"));

        let ref_out = Atom::parse("1/2*v1^2+v3*(v4+1)^-1+v1^3*v2*v4^-1").unwrap();

        assert_eq!(out, ref_out);
    }
}
