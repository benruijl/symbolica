use ahash::HashMap;

use crate::{
    representations::{
        Add, AsAtomView, Atom, AtomSet, AtomView, Fun, Identifier, Mul, OwnedAdd, OwnedMul, Pow,
        Var,
    },
    state::{State, Workspace},
};

impl<'a, P: AtomSet> AtomView<'a, P> {
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
        x: Identifier,
        workspace: &Workspace<P>,
        state: &State,
        key_map: Option<Box<dyn Fn(AtomView<P>, &mut Atom<P>)>>,
        coeff_map: Option<Box<dyn Fn(AtomView<P>, &mut Atom<P>)>>,
        out: &mut Atom<P>,
    ) {
        let (h, rest) = self.coefficient_list(x, workspace, state);

        let mut add_h = workspace.new_atom();
        let add = add_h.to_add();

        fn map_key_coeff<P: AtomSet>(
            key: AtomView<P>,
            coeff: Atom<P>,
            workspace: &Workspace<P>,
            key_map: &Option<Box<dyn Fn(AtomView<P>, &mut Atom<P>)>>,
            coeff_map: &Option<Box<dyn Fn(AtomView<P>, &mut Atom<P>)>>,
            add: &mut P::OA,
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

            mul.set_dirty(true);

            add.extend(mul_h.as_view());
        }

        for (key, coeff) in h {
            map_key_coeff(key, coeff, workspace, &key_map, &coeff_map, add);
        }

        if key_map.is_some() {
            let key = workspace.new_num(1);
            map_key_coeff(key.as_view(), rest, workspace, &key_map, &coeff_map, add);
        } else {
            if let Some(coeff_map) = coeff_map {
                let mut handle = workspace.new_atom();
                coeff_map(rest.as_view(), &mut handle);
                add.extend(handle.as_view());
            } else {
                add.extend(rest.as_view());
            }
        }

        add.set_dirty(true);
        add_h.as_view().normalize(workspace, state, out);
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function name.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    pub fn coefficient_list(
        &self,
        x: Identifier,
        workspace: &Workspace<P>,
        state: &State,
    ) -> (Vec<(AtomView<'a, P>, Atom<P>)>, Atom<P>) {
        let mut h = HashMap::default();
        let mut rest = workspace.new_num(0);

        match self {
            AtomView::Add(a) => {
                for arg in a.iter() {
                    arg.collect_factor(x, workspace, state, &mut h, &mut rest)
                }
            }
            _ => self.collect_factor(x, workspace, state, &mut h, &mut rest),
        }

        (
            h.into_iter().collect(),
            Atom::new_from_view(&rest.as_view()),
        )
    }

    /// Check if a factor contains `x` at the ground level.
    fn has_key(&self, x: Identifier) -> bool {
        match self {
            AtomView::Var(v) => v.get_name() == x,
            AtomView::Fun(f) => f.get_name() == x,
            AtomView::Pow(p) => {
                let (base, _) = p.get_base_exp();
                match base {
                    AtomView::Var(v) => v.get_name() == x,
                    AtomView::Fun(f) => f.get_name() == x,
                    _ => false,
                }
            }
            AtomView::Mul(_) => unreachable!("Mul is not a factor"),
            _ => false,
        }
    }

    fn collect_factor(
        &self,
        x: Identifier,
        workspace: &Workspace<P>,
        state: &State,
        h: &mut HashMap<AtomView<'a, P>, Atom<P>>,
        rest: &mut Atom<P>,
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

                    mul.set_dirty(true);

                    let mut col_n = workspace.new_atom();
                    collected.as_view().normalize(workspace, state, &mut col_n);

                    h.entry(bracket.unwrap())
                        .and_modify(|e| {
                            let mut res = workspace.new_atom();
                            e.add(state, workspace, col_n.as_view(), &mut res);
                            std::mem::swap(e, &mut res);
                        })
                        .or_insert(Atom::new_from_view(&col_n.as_view()));

                    return;
                }
            }
            _ => {
                if self.has_key(x) {
                    // add the coefficient 1
                    let col_n = workspace.new_num(1);
                    h.entry(*self)
                        .and_modify(|e| {
                            let mut res = workspace.new_atom();
                            e.add(state, workspace, col_n.as_view(), &mut res);
                            std::mem::swap(e, &mut res);
                        })
                        .or_insert(Atom::new_from_view(&col_n.as_view()));

                    return;
                }
            }
        }

        let mut new_atom = workspace.new_atom();
        rest.add(state, workspace, *self, &mut new_atom);
        std::mem::swap(rest, new_atom.get_mut());
    }
}
