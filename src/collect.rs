use crate::{
    atom::{Add, Atom, AtomCore, AtomView, Symbol},
    coefficient::{Coefficient, CoefficientView},
    domains::{integer::Z, rational::Q},
    poly::{factor::Factorize, polynomial::MultivariatePolynomial, Exponent},
    state::Workspace,
};
use std::sync::Arc;

impl<'a> AtomView<'a> {
    /// Collect terms involving the same power of `x`, where `x` is an indeterminate, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub(crate) fn collect<E: Exponent, T: AtomCore>(
        &self,
        x: T,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.collect_multiple::<E, T>(std::slice::from_ref(&x), key_map, coeff_map)
    }

    pub(crate) fn collect_multiple<E: Exponent, T: AtomCore>(
        &self,
        xs: &[T],
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local()
            .with(|ws| self.collect_multiple_impl::<E, T>(xs, ws, key_map, coeff_map, &mut out));
        out
    }

    pub(crate) fn collect_multiple_impl<E: Exponent, T: AtomCore>(
        &self,
        xs: &[T],
        ws: &Workspace,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        out: &mut Atom,
    ) {
        let r = self.coefficient_list::<E, T>(xs);

        let mut add_h = Atom::new();
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

        for (key, coeff) in r {
            map_key_coeff(key.as_view(), coeff, ws, &key_map, &coeff_map, add);
        }

        add_h.as_view().normalize(ws, out);
    }

    /// Collect terms involving the same powers of `x` in `xs`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs.
    pub(crate) fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        let vars = xs
            .iter()
            .map(|x| x.as_atom_view().to_owned().into())
            .collect::<Vec<_>>();

        let p = self.to_polynomial_in_vars::<E>(&Arc::new(vars));

        let mut coeffs = vec![];
        for t in p.into_iter() {
            let mut key = Atom::new_num(1);

            for (p, v) in t.exponents.iter().zip(xs) {
                let mut pow = Atom::new();
                pow.to_pow(v.as_atom_view(), Atom::new_num(p.to_i32() as i64).as_view());
                key = key * pow;
            }

            coeffs.push((key, t.coefficient.clone()));
        }

        coeffs
    }

    /// Collect terms involving the literal occurrence of `x`.
    pub fn coefficient_with_ws(&self, x: AtomView<'_>, workspace: &Workspace) -> Atom {
        let mut coeffs = workspace.new_atom();
        let coeff_add = coeffs.to_add();

        match self {
            AtomView::Add(a) => {
                for arg in a {
                    arg.collect_factor(x, workspace, coeff_add)
                }
            }
            _ => self.collect_factor(x, workspace, coeff_add),
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

                    for a in m {
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

                    for xx in m {
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
    pub(crate) fn cancel(&self) -> Atom {
        let mut out = Atom::new();
        self.cancel_into(&mut out);
        out
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub(crate) fn cancel_into(&self, out: &mut Atom) {
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

                for a in m {
                    if let AtomView::Pow(p) = a {
                        let (b, e) = p.get_base_exp();
                        if let AtomView::Num(n) = e {
                            if let CoefficientView::Natural(n, d) = n.get_coeff_view() {
                                if n < 0 && d == 1 {
                                    denominators.push(
                                        b.to_polynomial::<_, u16>(&Q, None)
                                            .pow(n.unsigned_abs() as usize),
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
                for arg in a {
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

        if f_n.is_empty() {
            return Atom::new_num(0);
        }

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

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    pub fn collect_num(&self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut coeff = Atom::new();
            self.collect_num_impl(ws, &mut coeff);
            coeff
        })
    }

    fn collect_num_impl(&self, ws: &Workspace, out: &mut Atom) -> bool {
        fn get_num(a: AtomView) -> Option<Coefficient> {
            match a {
                AtomView::Num(n) => Some(n.get_coeff_view().to_owned()),
                AtomView::Add(add) => {
                    // perform GCD of all arguments
                    // make sure the first argument is positive
                    let mut is_negative = false;
                    let mut gcd: Option<Coefficient> = None;
                    for arg in add.iter() {
                        if let Some(num) = get_num(arg) {
                            if let Some(g) = gcd {
                                gcd = Some(g.gcd(&num));
                            } else {
                                is_negative = num.is_negative();
                                gcd = Some(num);
                            }
                        }
                    }

                    if let Some(g) = gcd {
                        if is_negative && !g.is_negative() {
                            Some(-g)
                        } else {
                            Some(g)
                        }
                    } else {
                        None
                    }
                }
                AtomView::Mul(mul) => {
                    if mul.has_coefficient() {
                        for aa in mul.iter() {
                            if let AtomView::Num(n) = aa {
                                return Some(n.get_coeff_view().to_owned());
                            }
                        }

                        unreachable!()
                    } else {
                        None
                    }
                }
                AtomView::Pow(p) => {
                    let (b, e) = p.get_base_exp();
                    if let Ok(e) = i64::try_from(e) {
                        if let Some(n) = get_num(b) {
                            if let Coefficient::Rational(r) = n {
                                if e < 0 {
                                    return Some(r.pow((-e) as u64).inv().into());
                                } else {
                                    return Some(r.pow(e as u64).into());
                                }
                            }
                        }
                    }

                    None
                }
                AtomView::Var(_) | AtomView::Fun(_) => None,
            }
        }

        match self {
            AtomView::Add(a) => {
                let mut r = ws.new_atom();
                let ra = r.to_add();
                let mut na = ws.new_atom();
                let mut changed = false;
                for arg in a {
                    changed |= arg.collect_num_impl(ws, &mut na);
                    ra.extend(na.as_view());
                }

                if !changed {
                    out.set_from_view(self);
                } else {
                    r.as_view().normalize(ws, out);
                }

                if let AtomView::Add(aa) = out.as_view() {
                    if let Some(n) = get_num(out.as_view()) {
                        let v = ws.new_num(n);
                        // divide every term by n
                        let ra = r.to_add();
                        let mut div = ws.new_atom();
                        for arg in aa.iter() {
                            arg.div_with_ws_into(ws, v.as_view(), &mut div);
                            ra.extend(div.as_view());
                        }

                        let m = div.to_mul();
                        m.extend(r.as_view());
                        m.extend(v.as_view());
                        m.as_view().normalize(ws, out);
                        changed = true;
                    }
                }

                changed
            }
            AtomView::Mul(m) => {
                let mut r = ws.new_atom();
                let ra = r.to_mul();
                let mut na = ws.new_atom();
                let mut changed = false;
                for arg in m {
                    changed |= arg.collect_num_impl(ws, &mut na);
                    ra.extend(na.as_view());
                }

                if !changed {
                    out.set_from_view(self);
                } else {
                    r.as_view().normalize(ws, out);
                }

                changed
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                let mut changed = false;
                let mut nb = ws.new_atom();
                changed |= b.collect_num_impl(ws, &mut nb);
                let mut ne = ws.new_atom();
                changed |= e.collect_num_impl(ws, &mut ne);

                if !changed {
                    out.set_from_view(self);
                } else {
                    let mut np = ws.new_atom();
                    np.to_pow(nb.as_view(), ne.as_view());
                    np.as_view().normalize(ws, out);
                }

                changed
            }
            _ => {
                out.set_from_view(self);
                false
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{representation::InlineVar, Atom, AtomCore, Symbol},
        function,
    };

    #[test]
    fn collect_num() {
        let input = Atom::parse("2*v1+4*v1^2+6*v1^3").unwrap();
        let out = input.collect_num();
        let ref_out = Atom::parse("2*(v1+2v1^2+3v1^3)").unwrap();
        assert_eq!(out, ref_out);

        let input = Atom::parse("(-3*v1+3*v2)(2*v3+2*v4)").unwrap();
        let out = input.collect_num();
        let ref_out = Atom::parse("-6*(v4+v3)*(v1-v2)").unwrap();
        assert_eq!(out, ref_out);

        let input = Atom::parse("v1+v2+2*(v1+v2)").unwrap();
        let out = input.expand_num().collect_num();
        let ref_out = Atom::parse("3*(v1+v2)").unwrap();
        assert_eq!(out, ref_out);
    }

    #[test]
    fn coefficient_list() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = Symbol::new("v1");

        let r = input.coefficient_list::<i8, InlineVar>(&[x.into()]);

        let res = vec![
            (
                Atom::parse("1").unwrap(),
                Atom::parse("v2^2+f1(5,v1)+2").unwrap(),
            ),
            (
                Atom::parse("v1").unwrap(),
                Atom::parse("v3+5*v2+1").unwrap(),
            ),
            (Atom::parse("v1^2").unwrap(), Atom::parse("1").unwrap()),
            (Atom::parse("v1^3").unwrap(), Atom::parse("1").unwrap()),
        ];

        assert_eq!(r, res);
    }

    #[test]
    fn collect() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = Symbol::new("v1");

        let out = input.collect::<i8, InlineVar>(x.into(), None, None);

        let ref_out = Atom::parse("v1^2+v1^3+v2^2+f1(5,v1)+v1*(5*v2+v3+1)+2").unwrap();
        assert_eq!(out, ref_out)
    }

    #[test]
    fn collect_nested() {
        let input = Atom::parse("(1+v1)^2*v1+(1+v2)^100").unwrap();
        let x = Symbol::new("v1");

        let out = input.collect::<i8, InlineVar>(x.into(), None, None);

        let ref_out = Atom::parse("v1+2*v1^2+v1^3+(v2+1)^100").unwrap();
        assert_eq!(out, ref_out)
    }

    #[test]
    fn collect_wrap() {
        let input = Atom::parse("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3").unwrap();
        let x = Symbol::new("v1");
        let key = Symbol::new("f3");
        let coeff = Symbol::new("f4");
        println!("> Collect in x with wrapping:");
        let out = input.collect::<i8, InlineVar>(
            x.into(),
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = function!(key, out);
            })),
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = function!(coeff, out);
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
        let out = input.apart(Symbol::new("v4"));

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

    #[test]
    fn coefficient_list_multiple() {
        let input = Atom::parse(
            "(v1+v2+v3)^2+v1+v1^2+ v2 + 5*v1*v2^2 + v3 + v2*(v4+1)^10 + v1*v5(1,2,3)^2 + v5(1,2)",
        )
        .unwrap();

        let out = input.as_view().coefficient_list::<i16, _>(&[
            Atom::new_var(Symbol::new("v1")),
            Atom::new_var(Symbol::new("v2")),
            Atom::parse("v5(1,2,3)").unwrap(),
        ]);

        assert_eq!(out.len(), 8);
    }
}
