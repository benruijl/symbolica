use ahash::HashMap;

use crate::{
    atom::{Add, AsAtomView, Atom, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::{integer::Z, rational::Q},
    poly::{factor::Factorize, polynomial::MultivariatePolynomial},
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
    pub fn coefficient_list(&self, x: Symbol) -> (Vec<(Atom, Atom)>, Atom) {
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

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub fn cancel(&self) -> Atom {
        self.as_view().cancel()
    }

    /// Factor the expression over the rationals.
    pub fn factor(&self) -> Atom {
        self.as_view().factor()
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
            map_key_coeff(key.as_view(), coeff, workspace, &key_map, &coeff_map, add);
        }

        if !rest.is_zero() {
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
        }

        add_h.as_view().normalize(workspace, out);
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list(&self, x: Symbol) -> (Vec<(Atom, Atom)>, Atom) {
        Workspace::get_local().with(|ws| self.coefficient_list_with_ws(x, ws))
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list_with_ws(
        &self,
        x: Symbol,
        workspace: &Workspace,
    ) -> (Vec<(Atom, Atom)>, Atom) {
        let mut h = HashMap::default();
        let mut rest = workspace.new_atom();
        let mut rest_add = rest.to_add();

        let mut expanded = workspace.new_atom();
        self.expand_with_ws_into(workspace, Some(x), &mut expanded);

        match expanded.as_view() {
            AtomView::Add(a) => {
                for arg in a.iter() {
                    arg.collect_factor_list(x, workspace, &mut h, &mut rest_add)
                }
            }
            _ => expanded
                .as_view()
                .collect_factor_list(x, workspace, &mut h, &mut rest_add),
        }

        let mut rest_norm = Atom::new();
        rest.as_view().normalize(workspace, &mut rest_norm);

        let mut r: Vec<_> = h
            .into_iter()
            .map(|(k, v)| {
                (
                    {
                        let mut a = Atom::new();
                        a.set_from_view(&k);
                        a
                    },
                    {
                        let mut a = Atom::new();
                        v.as_view().normalize(workspace, &mut a);
                        a
                    },
                )
            })
            .collect();
        r.sort_unstable_by(|(a, _), (b, _)| a.as_view().cmp(&b.as_view()));

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

                if let AtomView::Mul(y) = x {
                    // check if all factors occur
                    for xx in y.iter() {
                        if !m.iter().any(|a| a == xx) {
                            return;
                        }
                    }

                    let mut collected = workspace.new_atom();
                    let mul = collected.to_mul();

                    for xx in m.iter() {
                        if !y.iter().any(|a| a == xx) {
                            mul.extend(xx);
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

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub fn cancel(&self) -> Atom {
        let mut out = Atom::new();
        self.cancel_into(&mut out);
        out
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub fn cancel_into(&self, out: &mut Atom) {
        Workspace::get_local().with(|ws| {
            self.cancel_with_ws_into(ws, out);
        });
    }

    fn cancel_with_ws_into(&self, ws: &Workspace, out: &mut Atom) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) | AtomView::Pow(_) => {
                out.set_from_view(self);
                false
            }
            AtomView::Mul(m) => {
                // split between numerator, denominator and rest
                // any numerator or denominator part that does not cancel will be kept as is
                let mut numerators = vec![];
                let mut denominators = vec![];
                let mut num_changed = vec![];
                let mut den_changed = vec![];
                let mut rest = vec![];

                for a in m.iter() {
                    if let AtomView::Pow(p) = a {
                        let (b, e) = p.get_base_exp();
                        if let AtomView::Num(n) = e {
                            if let CoefficientView::Natural(n, d) = n.get_coeff_view() {
                                if n < 0 && d == 1 {
                                    denominators.push(
                                        b.to_polynomial::<_, u16>(&Q, None).pow(n.abs() as usize),
                                    );
                                    den_changed.push((a, false));
                                    continue;
                                } else if n > 0 && d == 1 {
                                    numerators.push(a.to_polynomial::<_, u16>(&Q, None));
                                    num_changed.push((a, false));
                                    continue;
                                }
                            }
                        }

                        rest.push(a);
                    } else {
                        numerators.push(a.to_polynomial(&Q, None));
                        num_changed.push((a, false));
                    }
                }

                if numerators.is_empty() || denominators.is_empty() {
                    out.set_from_view(self);
                    return false;
                }

                MultivariatePolynomial::unify_variables_list(&mut numerators);
                MultivariatePolynomial::unify_variables_list(&mut denominators);
                numerators[0].unify_variables(&mut denominators[0]);
                MultivariatePolynomial::unify_variables_list(&mut numerators);
                MultivariatePolynomial::unify_variables_list(&mut denominators);

                let mut changed = false;
                for (d, ds) in denominators.iter_mut().zip(&mut den_changed) {
                    for (n, ns) in numerators.iter_mut().zip(&mut num_changed) {
                        let g = n.gcd(d);
                        if !g.is_one() {
                            changed = true;
                            ds.1 = true;
                            ns.1 = true;
                            *n = &*n / &g;
                            *d = &*d / &g;
                        }
                    }
                }

                if !changed {
                    out.set_from_view(self);
                    return false;
                }

                let mut mul = ws.new_atom();
                let mul_view = mul.to_mul();

                let mut tmp = ws.new_atom();
                for (n, (orig, changed)) in numerators.iter().zip(num_changed) {
                    if changed {
                        n.to_expression_into(&mut tmp);
                        mul_view.extend(tmp.as_view());
                    } else {
                        mul_view.extend(orig);
                    }
                }

                for (d, (orig, changed)) in denominators.iter().zip(den_changed) {
                    if changed {
                        d.to_expression_into(&mut tmp);

                        let mut pow = ws.new_atom();
                        let exp = ws.new_num(-1);
                        pow.to_pow(tmp.as_view(), exp.as_view());

                        mul_view.extend(pow.as_view());
                    } else {
                        mul_view.extend(orig);
                    }
                }

                for r in rest {
                    mul_view.extend(r);
                }

                mul_view.as_view().normalize(ws, out);
                true
            }
            AtomView::Add(a) => {
                let mut add = ws.new_atom();
                let add_view = add.to_add();

                let mut changed = false;
                let mut tmp = ws.new_atom();
                for arg in a.iter() {
                    if arg.cancel_with_ws_into(ws, &mut tmp) {
                        changed = true;
                        add_view.extend(tmp.as_view());
                    } else {
                        add_view.extend(arg);
                    }
                }

                if changed {
                    add_view.as_view().normalize(ws, out);
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
        }
    }

    /// Factor the expression over the rationals.
    pub fn factor(&self) -> Atom {
        let r = self.to_rational_polynomial::<_, _, u16>(&Q, &Z, None);
        let f_n = r.numerator.factor();
        let f_d = r.denominator.factor();

        let mut out = Atom::new();
        let mul = out.to_mul();

        let mut pow = Atom::new();
        for (k, v) in f_n {
            if v > 1 {
                let exp = Atom::new_num(v as i64);
                pow.to_pow(k.to_expression().as_view(), exp.as_view());
                mul.extend(pow.as_view());
            } else {
                mul.extend(k.to_expression().as_view());
            }
        }

        for (k, v) in f_d {
            let exp = Atom::new_num(-(v as i64));
            pow.to_pow(k.to_expression().as_view(), exp.as_view());
            mul.extend(pow.as_view());
        }

        Workspace::get_local().with(|ws| {
            out.as_view().normalize(ws, &mut pow);
        });

        pow
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
            .map(|(a, b)| (a.clone(), b.clone()))
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
    fn collect_nested() {
        let input = Atom::parse("(1+v1)^2*v1+(1+v2)^100").unwrap();
        let x = State::get_symbol("v1");

        let out = input.collect(x, None, None);

        let ref_out = Atom::parse("v1+2*v1^2+v1^3+(v2+1)^100").unwrap();
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

    #[test]
    fn cancel() {
        let input =
            Atom::parse("1/(v1+1)^2 + (v1^2 - 1)*(v2+1)^10/(v1 - 1)+ 5 + (v1+1)/(v1^2+2v1+1)")
                .unwrap();
        let out = input.cancel();

        let ref_out = Atom::parse("(v1+1)^-2+(v1+1)^-1+(v1+1)*(v2+1)^10+5").unwrap();

        assert_eq!(out, ref_out);
    }

    #[test]
    fn factor() {
        let input =
            Atom::parse("(6 + v1)/(7776 + 6480*v1 + 2160*v1^2 + 360*v1^3 + 30*v1^4 + v1^5)")
                .unwrap();
        let out = input.factor();

        let ref_out = Atom::parse("(v1+6)^-4").unwrap();

        assert_eq!(out, ref_out);
    }
}
