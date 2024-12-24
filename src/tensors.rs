//! Methods for tensor manipulation and linear algebra.

use crate::{
    atom::{representation::InlineNum, Atom, AtomOrView, AtomView, Mul},
    graph::{Graph, HiddenData},
    state::{RecycledAtom, Workspace},
};

pub mod matrix;

impl<'a> AtomView<'a> {
    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices are the arguments.
    /// The repeated indices should be provided in `contracted_indices`.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide an optional group marker for each index using `index_group`.
    /// This makes sure that an index will not be renamed to an index from a different group.
    pub(crate) fn canonize_tensors(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView]>,
    ) -> Result<Atom, String> {
        if self.is_zero() {
            return Ok(self.to_owned());
        }

        if let Some(c) = index_group {
            if c.len() != contracted_indices.len() {
                return Err(
                    "Index group must have the same length as contracted indices".to_owned(),
                );
            }
        }

        // sort all contracted indices, this is required to canonize antisymmetric tensors
        if !contracted_indices.windows(2).all(|w| w[0] < w[1]) {
            if let Some(groups) = index_group {
                let mut index = (0..contracted_indices.len()).collect::<Vec<_>>();
                index.sort_by_key(|&i| contracted_indices[i]);
                let c = index
                    .iter()
                    .map(|&i| contracted_indices[i])
                    .collect::<Vec<_>>();
                let g = index.iter().map(|&i| groups[i]).collect::<Vec<_>>();
                return self.canonize_tensors(&c, Some(&g));
            } else {
                let mut c = contracted_indices.to_vec();
                c.sort();
                return self.canonize_tensors(&c, None);
            }
        }

        Workspace::get_local().with(|ws| {
            if let AtomView::Add(a) = self {
                let mut aa = ws.new_atom();
                let add = aa.to_add();

                for a in a.iter() {
                    add.extend(
                        a.canonize_tensor_product(contracted_indices, index_group, ws)?
                            .as_view(),
                    );
                }

                let mut out = Atom::new();
                aa.as_view().normalize(ws, &mut out);
                Ok(out)
            } else {
                Ok(self
                    .canonize_tensor_product(contracted_indices, index_group, ws)?
                    .into_inner())
            }
        })
    }

    /// Canonize a tensor product by relabeling repeated indices.
    fn canonize_tensor_product(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView]>,
        ws: &Workspace,
    ) -> Result<RecycledAtom, String> {
        let mut g = Graph::new();
        let mut connections = vec![None; contracted_indices.len()];

        let mut t = ws.new_atom();
        let mul = t.to_mul();

        self.tensor_to_graph_impl(
            contracted_indices,
            index_group,
            &mut connections,
            &mut g,
            mul,
        )?;

        for (i, f) in contracted_indices.iter().zip(&connections) {
            if f.is_some() {
                return Err(format!("Index {} is not contracted", i));
            }
        }

        let gc = g.canonize().graph;

        let mut funcs = vec![];
        for n in gc.nodes() {
            funcs.push((true, n.data.clone().as_mut().clone(), vec![]));
        }

        // connect dummy indices
        let mut used_indices = vec![false; contracted_indices.len()];
        for e in gc.edges() {
            if e.directed {
                continue;
            }

            // find first free index that belongs to the same group
            let (index, used) = used_indices
                .iter_mut()
                .enumerate()
                .find(|(p, x)| {
                    !**x && index_group
                        .map(|c| c[*p] == e.data.1.as_view())
                        .unwrap_or(true)
                })
                .unwrap();
            *used = true;

            if let Atom::Fun(f) = &mut funcs[e.vertices.0].1 {
                f.add_arg(contracted_indices[index]);

                if f.to_fun_view().is_antisymmetric() {
                    funcs[e.vertices.0].2.push(e.data.0.hidden);
                }
            } else {
                unreachable!("Only functions should be left");
            }

            if let Atom::Fun(f) = &mut funcs[e.vertices.1].1 {
                f.add_arg(contracted_indices[index]);

                if f.to_fun_view().is_antisymmetric() {
                    funcs[e.vertices.1].2.push(e.data.0.hidden);
                }
            } else {
                unreachable!("Only functions should be left");
            }
        }

        // now join all regular and cyclesymmetric functions
        // the start of the cyclesymmetric function is determined by its
        // first encountered node
        for fi in 0..funcs.len() {
            if !funcs[fi].0 {
                continue;
            }

            if let AtomView::Fun(ff) = funcs[fi].1.as_view() {
                if ff.get_symbol().is_symmetric() {
                    funcs[fi].0 = false;
                    mul.extend(funcs[fi].1.as_view());
                    continue;
                } else if ff.get_symbol().is_antisymmetric() {
                    funcs[fi].0 = false;
                    mul.extend(funcs[fi].1.as_view());

                    // for antisymmetric functions, the original arguments were sorted
                    // according to the contracted indices order (since that is sorted)
                    // the original position is stored in the hidden data of the edge
                    let mut order: Vec<_> = funcs[fi].2.iter().enumerate().collect();
                    order.sort_by_key(|(_, x)| *x);

                    // find the number of swaps needed to sort the arguments
                    let mut order: Vec<_> = (0..order.len())
                        .map(|i| order.iter().position(|(j, _)| *j == i).unwrap())
                        .collect();
                    let mut swaps = 0;
                    for i in 0..order.len() {
                        let pos = order[i..].iter().position(|&x| x == i).unwrap();
                        order.copy_within(i..i + pos, i + 1);
                        swaps += pos;
                    }

                    if swaps % 2 == 1 {
                        mul.extend(InlineNum::new(-1, 1).as_view());
                    }

                    continue;
                } else if !ff.get_symbol().is_cyclesymmetric() {
                    // check if the current index is the start of a regular function
                    if gc.node(fi).edges.iter().any(|ei| {
                        let e = gc.edge(*ei);
                        e.directed && (e.vertices.0 != fi || e.data.0.data != 1)
                    }) {
                        continue;
                    }
                }
            }

            let mut ff = funcs[fi].1.clone();
            let mut cur_pos = fi;
            'next: loop {
                funcs[cur_pos].0 = false;

                for ei in &gc.node(cur_pos).edges {
                    let e = gc.edge(*ei);

                    if e.directed && e.vertices.0 == cur_pos {
                        debug_assert!(e.vertices.0 != e.vertices.1);

                        if e.vertices.1 == fi {
                            // cycle completed
                            break 'next;
                        }

                        debug_assert!(funcs[e.vertices.1].0);

                        if let Atom::Fun(ff) = &mut ff {
                            if let AtomView::Fun(f) = funcs[e.vertices.1].1.as_view() {
                                for a in f.iter() {
                                    ff.add_arg(a);
                                }
                            }
                        }

                        cur_pos = e.vertices.1;

                        continue 'next;
                    }
                }

                break;
            }

            mul.extend(ff.as_view());
        }

        debug_assert!(funcs.iter().all(|f| !f.0));

        let mut out = ws.new_atom();
        t.as_view().normalize(ws, &mut out);

        Ok(out)
    }

    fn tensor_to_graph_impl(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView<'a>]>,
        connections: &mut [Option<usize>],
        g: &mut Graph<AtomOrView<'a>, (HiddenData<usize, usize>, AtomOrView<'a>)>,
        remainder: &mut Mul,
    ) -> Result<(), String> {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                remainder.extend(*self);
                Ok(())
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if !contracted_indices.iter().any(|a| b.contains(*a)) {
                    remainder.extend(*self);
                    Ok(())
                } else {
                    if let Ok(n) = e.try_into() {
                        if n > 0 {
                            for _ in 0..n {
                                b.tensor_to_graph_impl(
                                    contracted_indices,
                                    index_group,
                                    connections,
                                    g,
                                    remainder,
                                )?;
                            }
                            Ok(())
                        } else {
                            Err("Only tensors raised to positive powers are supported".to_owned())
                        }
                    } else {
                        Err("Only tensors raised to positive powers are supported".to_owned())
                    }
                }
            }
            AtomView::Fun(f) => {
                if !f.iter().any(|a| contracted_indices.contains(&a)) {
                    remainder.extend(*self);
                    return Ok(());
                }

                let nargs = f.get_nargs();
                if f.is_symmetric() || f.is_antisymmetric() || nargs == 1 {
                    // collect all non-dummy arguments
                    let mut ff = Atom::new();
                    let fff = ff.to_fun(f.get_symbol());

                    for a in f.iter() {
                        if !contracted_indices.contains(&a) {
                            fff.add_arg(a);
                        }
                    }
                    fff.set_normalized(true);

                    let n = g.add_node(ff.into());
                    for a in f.iter() {
                        if let Some(p) = contracted_indices.iter().position(|x| x == &a) {
                            if let Some(n2) = connections[p] {
                                g.add_edge(
                                    n,
                                    n2,
                                    false,
                                    (
                                        HiddenData::new(0, p),
                                        index_group
                                            .map(|c| c[p].into())
                                            .unwrap_or(Atom::Zero.into()),
                                    ),
                                )
                                .unwrap();
                                connections[p] = None;
                            } else {
                                connections[p] = Some(n);
                            }
                        }
                    }

                    Ok(())
                } else {
                    let is_cyclesymmetric = f.is_cyclesymmetric();

                    // add a node for every slot
                    let start = g.nodes().len();
                    for (i, a) in f.iter().enumerate() {
                        let mut ff = Atom::new();
                        let fff = ff.to_fun(f.get_symbol());

                        if let Some(p) = contracted_indices.iter().position(|x| x == &a) {
                            ff.set_normalized(true);
                            g.add_node(ff.into());

                            if let Some(n2) = connections[p] {
                                g.add_edge(
                                    start + i,
                                    n2,
                                    false,
                                    (
                                        HiddenData::new(0, p),
                                        index_group
                                            .map(|c| c[p].into())
                                            .unwrap_or(Atom::Zero.into()),
                                    ),
                                )
                                .unwrap();
                                connections[p] = None;
                            } else {
                                connections[p] = Some(start + i);
                            }
                        } else {
                            fff.add_arg(a);
                            fff.set_normalized(true);
                            g.add_node(ff.into());
                        }

                        if i != 0 {
                            g.add_edge(
                                start + i - 1,
                                start + i,
                                true,
                                (
                                    if is_cyclesymmetric {
                                        HiddenData::new(0, 0)
                                    } else {
                                        HiddenData::new(i, 0)
                                    },
                                    Atom::Zero.into(),
                                ),
                            )
                            .unwrap();
                        }
                    }

                    if is_cyclesymmetric {
                        g.add_edge(
                            start + nargs - 1,
                            start,
                            true,
                            (HiddenData::new(0, 0), Atom::Zero.into()),
                        )
                        .unwrap();
                    }

                    Ok(())
                }
            }
            AtomView::Mul(m) => {
                for a in m.iter() {
                    a.tensor_to_graph_impl(
                        contracted_indices,
                        index_group,
                        connections,
                        g,
                        remainder,
                    )?;
                }
                Ok(())
            }
            AtomView::Add(_) => {
                if !contracted_indices.iter().any(|a| self.contains(*a)) {
                    remainder.extend(*self);
                    Ok(())
                } else {
                    Err(
                        "Nested additions containing contracted indices is not supported"
                            .to_owned(),
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::atom::{representation::InlineVar, Atom, AtomCore, Symbol};

    #[test]
    fn index_group() {
        let mus: Vec<_> = (0..4)
            .map(|i| InlineVar::new(Symbol::new(format!("mu{}", i + 1))))
            .collect();
        let mu_ref = mus.iter().map(|x| x.as_view()).collect::<Vec<_>>();

        let a1 = Atom::parse("fc1(mu1,mu2,mu1,mu3,mu4)*fs1(mu2,mu3,mu4)").unwrap();

        let colors = vec![
            Atom::new_num(1),
            Atom::new_num(2),
            Atom::new_num(1),
            Atom::new_num(2),
        ];
        let col_ref = colors.iter().map(|x| x.as_view()).collect::<Vec<_>>();

        let r1 = a1.canonize_tensors(&mu_ref, Some(&col_ref)).unwrap();

        let a2 = Atom::parse("fc1(mu4,mu3,mu1,mu2,mu3)*fs1(mu2,mu1,mu4)").unwrap();

        let r2 = a2.canonize_tensors(&mu_ref, Some(&col_ref)).unwrap();

        assert_eq!(r1, r2);

        let r3 = a1.canonize_tensors(&mu_ref, None).unwrap();

        assert_ne!(r1, r3);
    }

    #[test]
    fn canonize_tensors() {
        let mus: Vec<_> = (0..10)
            .map(|i| InlineVar::new(Symbol::new(format!("mu{}", i + 1))))
            .collect();
        let mu_ref = mus.iter().map(|x| x.as_view()).collect::<Vec<_>>();

        // fs1 is symmetric and fc1 is cyclesymmetric
        let a1 = Atom::parse(
                "fs1(k2,mu1,mu2)*fs1(mu1,mu3)*fc1(mu4,mu2,k1,mu4,k1,mu3)*(1+x)*f(k)*fs1(mu5,mu6)^2*f(mu7,mu9,k3,mu9,mu7)*h(mu8)*i(mu8)+fc1(mu4,mu5,mu6)*fc1(mu5,mu4,mu6)",
            )
            .unwrap();

        let r1 = a1.canonize_tensors(&mu_ref, None).unwrap();

        let a2 = Atom::parse(
                "fs1(k2,mu2,mu9)*fs1(mu2,mu5)*fc1(k1,mu8,k1,mu5,mu8,mu9)*(1+x)*f(k)*fs1(mu3,mu6)^2*f(mu7,mu1,k3,mu1,mu7)*h(mu4)*i(mu4)+fc1(mu1,mu4,mu6)*fc1(mu4,mu1,mu6)",
            )
            .unwrap();

        let r2 = a2.canonize_tensors(&mu_ref, None).unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn canonize_antisymmetric() {
        let mus: Vec<_> = (0..4)
            .map(|i| InlineVar::new(Symbol::new(format!("mu{}", i + 1))))
            .collect();
        let mu_ref = mus.iter().map(|x| x.as_view()).collect::<Vec<_>>();

        let a1 = Atom::parse("f1(mu3,mu2,mu3,mu1)*fa1(mu1,mu2)").unwrap();

        let r1 = a1.canonize_tensors(&mu_ref, None).unwrap();

        let a2 = Atom::parse("-f1(mu1,mu2,mu1,mu3)*fa1(mu2,mu3)").unwrap();

        let r2 = a2.canonize_tensors(&mu_ref, None).unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn canonize_constant() {
        let a1 = Atom::parse("x+5").unwrap();
        let r1 = a1.canonize_tensors(&[], None).unwrap();
        assert_eq!(a1, r1);
    }
}
