//! Methods for tensor manipulation and linear algebra.

use crate::{
    atom::{Atom, AtomCore, AtomView, FunctionBuilder, Symbol},
    graph::{Graph, HiddenData},
    printer::PrintOptions,
    state::{RecycledAtom, Workspace},
};

pub mod matrix;

/// A node in a graph representation of a tensor network.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TensorGraphNode<'a> {
    Mul,
    Add,
    Fun(Symbol),
    Slot(Option<AtomView<'a>>),
}

impl<'a> std::fmt::Display for TensorGraphNode<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorGraphNode::Mul => write!(f, "Mul"),
            TensorGraphNode::Add => write!(f, "Add"),
            TensorGraphNode::Fun(s) => write!(f, "Fun({})", s),
            TensorGraphNode::Slot(d) => {
                if let Some(d) = d {
                    write!(f, "{}", d)
                } else {
                    write!(f, "")
                }
            }
        }
    }
}

impl<'a> AtomView<'a> {
    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices as the arguments.
    /// Indices should be provided in `indices`.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide an optional group marker for each index using `index_group`.
    /// This makes sure that an index will not be renamed to an index from a different group.
    pub(crate) fn canonize_tensors<G: Ord + std::hash::Hash>(
        &self,
        indices: &[(AtomView<'a>, G)],
    ) -> Result<Atom, String> {
        if self.is_zero() {
            return Ok(self.to_owned());
        }

        Workspace::get_local().with(|ws| {
            if let AtomView::Add(a) = self {
                let mut aa = ws.new_atom();
                let add = aa.to_add();

                for a in a.iter() {
                    add.extend(a.canonize_tensor_product(indices)?.as_view());
                }

                let mut out = Atom::new();
                aa.as_view().normalize(ws, &mut out);
                Ok(out)
            } else {
                Ok(self.canonize_tensor_product(indices)?.into_inner())
            }
        })
    }

    /// Canonize a tensor product by relabeling repeated indices.
    fn canonize_tensor_product<G: Ord + std::hash::Hash>(
        &self,
        indices: &[(AtomView<'a>, G)],
    ) -> Result<RecycledAtom, String> {
        // strip all top-level factors that do not have any indices, so that
        // they do not influence the canonization
        if let AtomView::Mul(m) = self {
            if let Some(r) = Workspace::get_local().with(|ws| {
                let mut stripped = ws.new_atom();
                let mut constants = ws.new_atom();
                let mm = stripped.to_mul();
                let r = constants.to_mul();
                for a in m.iter() {
                    if indices.iter().any(|x| a.contains(x.0.as_atom_view())) {
                        mm.extend(a);
                    } else {
                        r.extend(a);
                    }
                }

                if r.get_nargs() != 0 {
                    let mut res = stripped.as_view().canonize_tensor_product(indices)?;
                    let mut p = Atom::new();
                    let m = p.to_mul();
                    m.extend(res.as_view());
                    m.extend(constants.as_view());
                    p.as_view().normalize(ws, &mut res);
                    Ok::<_, String>(Some(res))
                } else {
                    Ok(None)
                }
            })? {
                return Ok(r);
            }
        }

        let mut g = Graph::new();
        let mut connections = vec![(vec![], false, 0); indices.len()];
        let mut used_indices = vec![(false, 0); indices.len()];

        let root = self.tensor_to_graph_impl(indices, &mut connections, &mut g)?;

        if let TensorGraphNode::Slot(Some(s)) = g.node(root).data {
            return Ok(s.to_owned().into());
        }

        for (i, (ii, (f, used, _))) in indices.iter().zip(&connections).enumerate() {
            if !f.is_empty() && *used {
                return Err(format!(
                    "Index {} is contracted more than once",
                    ii.0.as_atom_view()
                ));
            } else if !f.is_empty() && !used {
                used_indices[i] = (true, 0);

                for ff in f {
                    let mut data = g.node(*ff).data.clone();

                    if let TensorGraphNode::Slot(d) = &mut data {
                        *d = Some(indices[i].0);
                    } else {
                        unreachable!()
                    }

                    // set the open index in the graph
                    g.set_node_data(*ff, data);
                }
            }
        }

        let canon = g.canonize();
        let root = canon.vertex_map[root];

        /*
        // write the graph for easy debugging
        let mut g2 = Graph::new();
        for n in canon.graph.nodes() {
            g2.add_node(n.data.clone());
        }
        for e in canon.graph.edges() {
            g2.add_edge(
                e.vertices.0,
                e.vertices.1,
                e.directed,
                if e.directed {
                    format!("{}|{}", e.data.data.0, e.data.hidden)
                } else {
                    String::new()
                },
            )
            .unwrap();
        }

        println!("{}", g2.to_mermaid());
        */

        let mut res = Atom::new();
        Workspace::get_local().with(|ws| {
            Self::reconstruct(
                indices,
                &canon.graph,
                root,
                &mut used_indices,
                &mut vec![None; g.edges().len()],
                ws,
                &mut res,
            )
        });

        return Ok(res.into());
    }

    fn reconstruct<G: Ord + std::hash::Hash>(
        indices: &[(AtomView<'a>, G)],
        graph: &Graph<TensorGraphNode, HiddenData<(usize, Option<&G>), usize>>,
        cur_node: usize,
        used_indices: &mut [(bool, usize)],
        processed_slots: &mut [Option<usize>],
        ws: &Workspace,
        out: &mut Atom,
    ) {
        let n = graph.node(cur_node);

        match n.data {
            TensorGraphNode::Mul => {
                let mut prod = ws.new_atom();
                let pp = prod.to_mul();

                let mut arg = ws.new_atom();

                for e in &n.edges {
                    let edge = graph.edge(*e);
                    if edge.vertices.0 != cur_node {
                        continue;
                    }

                    Self::reconstruct(
                        indices,
                        graph,
                        edge.vertices.1,
                        used_indices,
                        processed_slots,
                        ws,
                        &mut arg,
                    );

                    pp.extend(arg.as_view());
                }

                prod.as_view().normalize(ws, out);
            }
            TensorGraphNode::Add => {
                let mut cur_used = used_indices.to_vec();

                let mut add = ws.new_atom();
                let pp = add.to_add();

                let mut arg = ws.new_atom();

                for e in &n.edges {
                    let edge = graph.edge(*e);
                    if edge.vertices.0 != cur_node {
                        continue;
                    }

                    let mut available = cur_used.clone();
                    Self::reconstruct(
                        indices,
                        graph,
                        edge.vertices.1,
                        &mut available,
                        processed_slots,
                        ws,
                        &mut arg,
                    );

                    // block open indices for every branch of the sum
                    for (i, a) in cur_used.iter_mut().zip(&available) {
                        if a.1 == 1 {
                            i.0 = true;
                        }
                    }

                    // block all used indices at higher levels in the graph
                    for (i, a) in used_indices.iter_mut().zip(&available) {
                        i.0 |= a.0;

                        // set the counter for used open indices
                        if a.1 == 1 {
                            assert!(i.1 < 2);
                            i.1 = 1;
                        }
                    }

                    pp.extend(arg.as_view());
                }

                add.as_view().normalize(ws, out);
            }
            TensorGraphNode::Fun(s) => {
                let mut args = vec![];

                for c in &n.edges {
                    let par_edge = graph.edge(*c);
                    if par_edge.vertices.0 != cur_node {
                        continue;
                    }

                    let child = par_edge.vertices.1;
                    let child_node = graph.node(child);

                    // the index the slot had in the function
                    // this is hidden data for any function with symmetry, as it should
                    // not affect the canonization
                    let original_child_index =
                        if s.is_symmetric() || s.is_antisymmetric() || s.is_cyclesymmetric() {
                            par_edge.data.hidden
                        } else {
                            par_edge.data.data.0
                        };

                    // check if any of the connecting slots has an index assigned to it
                    let mut rep_edge = None;
                    let mut added = false;
                    for e in &child_node.edges {
                        let edge = graph.edge(*e);

                        if edge.directed {
                            continue; // no dummy index
                        }

                        rep_edge = Some(edge);

                        let slot = if edge.vertices.0 == child {
                            edge.vertices.1
                        } else {
                            edge.vertices.0
                        };

                        if let Some(p) = processed_slots[slot] {
                            if processed_slots[child].is_none() {
                                processed_slots[child] = Some(p);
                            }

                            assert_eq!(p, processed_slots[child].unwrap());
                            if !added {
                                args.push((original_child_index, indices[p].0.as_atom_view()));
                                used_indices[p].1 += 1;
                                added = true;
                            }
                        }
                    }

                    if rep_edge.is_none() {
                        if let TensorGraphNode::Slot(data) = child_node.data {
                            args.push((original_child_index, data.unwrap()));
                        } else {
                            unreachable!("Slot is missing open index");
                        }

                        continue;
                    }

                    if processed_slots[child].is_none() {
                        let rep_edge = rep_edge.unwrap();

                        // find first available in the group
                        let (index, used) = used_indices
                            .iter_mut()
                            .enumerate()
                            .find(|(p, (used, _))| {
                                !*used && indices[*p].1 == *rep_edge.data.data.1.unwrap()
                            })
                            .unwrap();

                        args.push((original_child_index, indices[index].0.as_atom_view()));

                        *used = (true, 1);
                        processed_slots[child] = Some(index);

                        // set the slot index for all neighbours
                        for e in &child_node.edges {
                            let edge = graph.edge(*e);

                            if edge.directed {
                                continue; // no dummy index
                            }

                            let slot = if edge.vertices.0 == child {
                                edge.vertices.1
                            } else {
                                edge.vertices.0
                            };

                            processed_slots[slot] = Some(index);
                        }
                    }
                }

                args.sort_by_key(|a| a.0);
                let mut f = FunctionBuilder::new(s);
                for a in args {
                    f = f.add_arg(a.1);
                }
                *out = f.finish();
            }
            TensorGraphNode::Slot(s) => {
                if let Some(s) = s {
                    *out = s.to_owned().into();
                } else {
                    unreachable!("Encountered empty slot during tree walk")
                }
            }
        }
    }

    fn tensor_to_graph_impl<'b, G: Ord + std::hash::Hash>(
        &self,
        indices: &'b [(AtomView<'a>, G)],
        connections: &mut [(Vec<usize>, bool, usize)],
        g: &mut Graph<TensorGraphNode<'a>, HiddenData<(usize, Option<&'b G>), usize>>,
    ) -> Result<usize, String> {
        if !indices.iter().any(|a| self.contains(a.0.as_atom_view())) {
            let node = g.add_node(TensorGraphNode::Slot(Some(*self)));
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
                            nodes.push(b.tensor_to_graph_impl(indices, connections, g)?);
                        }

                        let node = g.add_node(TensorGraphNode::Mul);

                        for n in nodes {
                            g.add_edge(node, n, true, HiddenData::new((0, None), 0))
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

                let is_symmetric = f.is_symmetric();
                let is_cyclesymmetric = f.is_cyclesymmetric();
                let is_antisymmetric = f.is_antisymmetric();

                // create function header node
                let header = g.add_node(TensorGraphNode::Fun(f.get_symbol().into()));

                // add a node for every slot
                let start = g.nodes().len();
                for (i, a) in f.iter().enumerate() {
                    if let Some(p) = indices.iter().position(|x| x.0.as_atom_view() == a) {
                        g.add_node(TensorGraphNode::Slot(None));

                        if connections[p].1 {
                            return Err(format!(
                                "Index {} is contracted more than once",
                                indices[p].0.as_atom_view()
                            ));
                        }

                        if connections[p].0.is_empty() {
                            connections[p].0.push(start + i);
                        } else {
                            for n2 in connections[p].0.drain(..) {
                                connections[p].1 = true;
                                g.add_edge(
                                    start + i,
                                    n2,
                                    false,
                                    HiddenData::new((0, Some(&indices[p].1)), 0),
                                )
                                .unwrap();
                            }
                        }
                    } else {
                        g.add_node(TensorGraphNode::Slot(Some(a)));
                    }

                    if is_symmetric || is_antisymmetric || is_cyclesymmetric {
                        g.add_edge(header, start + i, true, HiddenData::new((0, None), i))
                            .unwrap();
                    } else {
                        g.add_edge(header, start + i, true, HiddenData::new((i, None), 0))
                            .unwrap();
                    }

                    if is_cyclesymmetric && i > 0 {
                        g.add_edge(
                            start + i - 1,
                            start + i,
                            true,
                            HiddenData::new((0, None), 0),
                        )
                        .unwrap();
                    }
                }

                if is_cyclesymmetric {
                    g.add_edge(
                        start + nargs - 1,
                        start,
                        true,
                        HiddenData::new((0, None), 0),
                    )
                    .unwrap();
                }

                Ok(header)
            }
            AtomView::Mul(m) => {
                let mut nodes = vec![];

                for a in m.iter() {
                    nodes.push(a.tensor_to_graph_impl(indices, connections, g)?);
                }

                let node = g.add_node(TensorGraphNode::Mul);

                for n in nodes {
                    g.add_edge(node, n, true, HiddenData::new((0, None), 0))
                        .unwrap();
                }

                Ok(node)
            }
            AtomView::Add(a) => {
                let mut subgraphs = vec![];

                for arg in a {
                    let mut sub_connections = connections.to_vec();

                    let node = arg.tensor_to_graph_impl(indices, &mut sub_connections, g)?;

                    // increase the index counter for every new contraction in the subgraph
                    for (con, sub_con) in connections.iter_mut().zip(&sub_connections) {
                        if con.1 != sub_con.1 {
                            con.2 += 1;
                        }
                    }

                    subgraphs.push((node, sub_connections));
                }

                if subgraphs.iter().any(|x| {
                    x.1.iter()
                        .zip(&subgraphs[0].1)
                        .any(|(a, b)| a.0.is_empty() != b.0.is_empty())
                }) {
                    return Err(format!(
                        "All components of {} must have the same open indices",
                        self.printer(PrintOptions::file())
                    ));
                }

                let node = g.add_node(TensorGraphNode::Add);

                connections.clone_from_slice(&subgraphs[0].1);

                for (n, cons) in subgraphs {
                    g.add_edge(node, n, true, HiddenData::new((0, None), 0))
                        .unwrap();

                    for (con, sub_con) in connections.iter_mut().zip(cons) {
                        // add new open indices from this subgraph
                        if *con.0 != sub_con.0 {
                            con.0.extend(sub_con.0);
                        }
                        con.1 |= sub_con.1;
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
        let mus: Vec<_> = (0..5)
            .map(|i| (InlineVar::new(Symbol::new(format!("mu{}", i + 1))), 0))
            .collect();

        let a1 = Atom::parse(
            "(f(mu1,mu4)*(f(mu2,mu2,mu1) + f2(mu3,mu2,mu3,mu2,mu1)) + f4(mu3,mu3,mu4))*g(mu4)",
        )
        .unwrap();
        let r1 = a1.canonize_tensors(&mus).unwrap();

        let a2 = Atom::parse(
            "(f(mu1,mu4)*(f(mu2,mu2,mu1) + f2(mu2,mu3,mu2,mu3,mu1)) + f4(mu1,mu1,mu4))*g(mu4)",
        )
        .unwrap();
        let r2 = a2.canonize_tensors(&mus).unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn index_group() {
        let mus: Vec<_> = (0..4)
            .map(|i| (InlineVar::new(Symbol::new(format!("mu{}", i + 1))), i % 2))
            .collect();

        let a1 = Atom::parse("fc1(mu1,mu2,mu1,mu3,mu4)*fs1(mu2,mu3,mu4)").unwrap();

        let r1 = a1.canonize_tensors(&mus).unwrap();

        let a2 = Atom::parse("fc1(mu4,mu3,mu1,mu2,mu3)*fs1(mu2,mu1,mu4)").unwrap();

        let r2 = a2.canonize_tensors(&mus).unwrap();

        assert_eq!(r1, r2);

        let mus: Vec<_> = (0..4)
            .map(|i| (InlineVar::new(Symbol::new(format!("mu{}", i + 1))), 0))
            .collect();

        let r3 = a1.canonize_tensors(&mus).unwrap();

        assert_ne!(r1, r3);
    }

    #[test]
    fn canonize_tensors() {
        let mus: Vec<_> = (0..10)
            .map(|i| (InlineVar::new(Symbol::new(format!("mu{}", i + 1))), 0))
            .collect();

        // fs1 is symmetric and fc1 is cyclesymmetric
        let a1 = Atom::parse(
                "fs1(k2,mu1,mu2)*fs1(mu1,mu3)*fc1(mu4,mu2,k1,mu4,k1,mu3)*(1+x)*f(k)*fs1(mu5,mu6)^2*f(mu7,mu9,k3,mu9,mu7)*h(mu8)*i(mu8)+fc1(mu4,mu5,mu6)*fc1(mu5,mu4,mu6)",
            )
            .unwrap();

        let r1 = a1.canonize_tensors(&mus).unwrap();

        let a2 = Atom::parse(
                "fs1(k2,mu2,mu9)*fs1(mu2,mu5)*fc1(k1,mu8,k1,mu5,mu8,mu9)*(1+x)*f(k)*fs1(mu3,mu6)^2*f(mu7,mu1,k3,mu1,mu7)*h(mu4)*i(mu4)+fc1(mu1,mu4,mu6)*fc1(mu4,mu1,mu6)",
            )
            .unwrap();

        let r2 = a2.canonize_tensors(&mus).unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn canonize_antisymmetric() {
        let mus: Vec<_> = (0..4)
            .map(|i| (InlineVar::new(Symbol::new(format!("mu{}", i + 1))), 0))
            .collect();

        let a1 = Atom::parse("f1(mu3,mu2,mu3,mu1)*fa1(mu1,mu2)").unwrap();

        let r1 = a1.canonize_tensors(&mus).unwrap();

        let a2 = Atom::parse("-f1(mu1,mu2,mu1,mu3)*fa1(mu2,mu3)").unwrap();

        let r2 = a2.canonize_tensors(&mus).unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn canonize_constant() {
        let a1 = Atom::parse("x+5").unwrap();
        let r1 = a1.canonize_tensors::<Atom, usize>(&[]).unwrap();
        assert_eq!(a1, r1);
    }
}
