//! Methods for tensor manipulation and linear algebra.

use crate::{
    atom::{Atom, AtomOrView, AtomView, FunctionAttribute, Symbol},
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

        Workspace::get_local().with(|ws| {
            if let AtomView::Add(a) = self {
                let mut aa = ws.new_atom();
                let add = aa.to_add();

                for a in a.iter() {
                    add.extend(
                        a.canonize_tensor_product(contracted_indices, index_group)?
                            .as_view(),
                    );
                }

                let mut out = Atom::new();
                aa.as_view().normalize(ws, &mut out);
                Ok(out)
            } else {
                Ok(self
                    .canonize_tensor_product(contracted_indices, index_group)?
                    .into_inner())
            }
        })
    }

    /// Canonize a tensor product by relabeling repeated indices.
    fn canonize_tensor_product(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView]>,
    ) -> Result<RecycledAtom, String> {
        let mut g = Graph::new();
        let mut connections = vec![vec![]; contracted_indices.len()];

        // TODO: strip all top-level products that do not have any contracted indices
        // this ensures that graphs that are the same up to multiplication of constants
        // map to the same graph

        self.tensor_to_graph_impl(contracted_indices, index_group, &mut connections, &mut g)?;

        for (i, f) in contracted_indices.iter().zip(&connections) {
            if !f.is_empty() {
                return Err(format!("Index {} is not contracted", i));
            }
        }

        let gc = g.canonize().graph;

        // connect dummy indices
        // TODO: recycle dummy indices that are contracted on a deeper level?
        let mut used_indices = vec![false; contracted_indices.len()];
        let mut map = vec![None; contracted_indices.len()];
        for e in gc.edges() {
            if e.directed {
                continue;
            }

            if !e.data.0.data {
                continue; // not a dummy index
            }

            let dummy_index = e.data.0.hidden;
            if map[dummy_index].is_none() {
                // find first free index that belongs to the same group
                let (index, used) = used_indices
                    .iter_mut()
                    .enumerate()
                    .find(|(p, x)| {
                        !**x && index_group
                            .map(|c| c[*p] == e.data.1.as_ref().unwrap().as_view())
                            .unwrap_or(true)
                    })
                    .unwrap();
                *used = true;
                map[dummy_index] = Some(index);
            }
        }

        // map the contracted indices
        Ok(self
            .replace_map(&|a, _ctx, out| {
                if let Some(p) = contracted_indices.iter().position(|x| *x == a) {
                    if let Some(q) = map[p] {
                        out.set_from_view(&contracted_indices[q]);
                        true
                    } else {
                        unreachable!()
                    }
                } else {
                    false
                }
            })
            .into())
    }

    fn tensor_to_graph_impl(
        &self,
        contracted_indices: &[AtomView],
        index_group: Option<&[AtomView<'a>]>,
        connections: &mut [Vec<usize>],
        g: &mut Graph<AtomOrView<'a>, (HiddenData<bool, usize>, Option<AtomOrView<'a>>)>,
    ) -> Result<usize, String> {
        if !contracted_indices.iter().any(|a| self.contains(*a)) {
            let node = g.add_node(self.into());
            return Ok(node);
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                Err("Dummy index appears as variable instead of as a function argument".to_owned())
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if let Ok(n) = e.try_into() {
                    if n > 0 {
                        let mut nodes = vec![];
                        for _ in 0..n {
                            nodes.push(b.tensor_to_graph_impl(
                                contracted_indices,
                                index_group,
                                connections,
                                g,
                            )?);
                        }

                        let node = g.add_node(
                            Symbol::new_with_attributes("PROD", &[FunctionAttribute::Symmetric])
                                .unwrap()
                                .into(),
                        );

                        for n in nodes {
                            g.add_edge(n, node, false, (HiddenData::new(false, 0), None))
                                .unwrap();
                        }

                        Ok(node)
                    } else {
                        Err("Only tensors raised to positive powers are supported".to_owned())
                    }
                } else {
                    Err("Only tensors raised to positive powers are supported".to_owned())
                }
            }
            AtomView::Fun(f) => {
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
                            if connections[p].is_empty() {
                                connections[p].push(n);
                            } else {
                                for n2 in connections[p].drain(..) {
                                    g.add_edge(
                                        n,
                                        n2,
                                        false,
                                        (
                                            HiddenData::new(true, p),
                                            index_group.map(|c| c[p].into()),
                                        ),
                                    )
                                    .unwrap();
                                }
                            }
                        }
                    }

                    Ok(n)
                } else {
                    let is_cyclesymmetric = f.is_cyclesymmetric();

                    // create function header node
                    let header = g.add_node(f.get_symbol().into());

                    // add a node for every slot
                    let start = g.nodes().len();
                    for (i, a) in f.iter().enumerate() {
                        let mut ff = Atom::new();
                        let fff = ff.to_fun(f.get_symbol());

                        if let Some(p) = contracted_indices.iter().position(|x| x == &a) {
                            ff.set_normalized(true);
                            g.add_node(Atom::Zero.into());

                            if connections[p].is_empty() {
                                connections[p].push(start + i);
                            } else {
                                for n2 in connections[p].drain(..) {
                                    g.add_edge(
                                        start + i,
                                        n2,
                                        false,
                                        (
                                            HiddenData::new(true, p),
                                            index_group.map(|c| c[p].into()),
                                        ),
                                    )
                                    .unwrap();
                                }
                            }
                        } else {
                            fff.add_arg(a);
                            fff.set_normalized(true);
                            g.add_node(Atom::Zero.into());
                        }

                        if is_cyclesymmetric || i == 0 {
                            g.add_edge(header, start + i, true, (HiddenData::new(false, 0), None))
                                .unwrap();
                        }

                        if i != 0 {
                            g.add_edge(
                                start + i - 1,
                                start + i,
                                true,
                                (HiddenData::new(false, 0), None),
                            )
                            .unwrap();
                        }
                    }

                    if is_cyclesymmetric {
                        g.add_edge(
                            start + nargs - 1,
                            start,
                            true,
                            (HiddenData::new(false, 0), None),
                        )
                        .unwrap();
                    }

                    Ok(header)
                }
            }
            AtomView::Mul(m) => {
                let mut nodes = vec![];
                for a in m.iter() {
                    nodes.push(a.tensor_to_graph_impl(
                        contracted_indices,
                        index_group,
                        connections,
                        g,
                    )?);
                }
                let node = g.add_node(
                    Symbol::new_with_attributes("PROD", &[FunctionAttribute::Symmetric])
                        .unwrap()
                        .into(),
                );

                for n in nodes {
                    g.add_edge(n, node, false, (HiddenData::new(false, 0), None))
                        .unwrap();
                }

                Ok(node)
            }
            AtomView::Add(a) => {
                let mut subgraphs = vec![];

                for arg in a {
                    let mut sub_connections = connections.to_vec();

                    let node = arg.tensor_to_graph_impl(
                        contracted_indices,
                        index_group,
                        &mut sub_connections,
                        g,
                    )?;

                    subgraphs.push((node, sub_connections));
                }

                if subgraphs.iter().any(|x| {
                    x.1.iter()
                        .zip(&subgraphs[0].1)
                        .any(|(a, b)| a.len() != b.len())
                }) {
                    return Err(
                        "All components of nested sums must have the same open indices".to_owned(),
                    );
                }

                let node = g.add_node(
                    Symbol::new_with_attributes("PLUS", &[FunctionAttribute::Symmetric])
                        .unwrap()
                        .into(),
                );

                connections.clone_from_slice(&subgraphs[0].1);

                for (n, cons) in subgraphs {
                    g.add_edge(n, node, false, (HiddenData::new(false, 0), None))
                        .unwrap();

                    for c in connections.iter_mut().zip(cons) {
                        // add new open indices from this subgraph
                        if *c.0 != c.1 {
                            c.0.extend(c.1);
                        }
                    }
                }

                Ok(node)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::atom::{representation::InlineVar, Atom, AtomCore, Symbol};

    #[test]
    fn nested_sum() {
        let mus: Vec<_> = (0..4)
            .map(|i| InlineVar::new(Symbol::new(format!("mu{}", i + 1))))
            .collect();
        let mu_ref = mus.iter().map(|x| x.as_view()).collect::<Vec<_>>();

        let a1 =
            Atom::parse("fs1(mu1,mu2)*(f1(mu2, mu1) + (m*fs2(mu1,mu2)+f3(mu2)*f4(mu1)))").unwrap();
        let r1 = a1.canonize_tensors(&mu_ref, None).unwrap();

        let a2 =
            Atom::parse("fs1(mu1,mu2)*(f1(mu1, mu2) + (m*fs2(mu1,mu2)+f3(mu1)*f4(mu2)))").unwrap();
        let r2 = a2.canonize_tensors(&mu_ref, None).unwrap();

        assert_eq!(r1, r2);
    }

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
