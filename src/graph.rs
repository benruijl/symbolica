use ahash::HashMap;
use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    hash::Hash,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<NodeData = Empty> {
    data: NodeData,
    edges: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge<EdgeData = Empty> {
    vertices: (usize, usize),
    data: EdgeData,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Empty;

impl Display for Empty {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "_")
    }
}

/// A multigraph with support for arbitrary node and edge data.
#[derive(Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct Graph<NodeData = Empty, EdgeData = Empty> {
    nodes: Vec<Node<NodeData>>,
    edges: Vec<Edge<EdgeData>>,
}

impl<N: Display, E: Display> std::fmt::Display for Graph<N, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (i, x) in self.nodes.iter().enumerate() {
            writeln!(f, "N{}: {}", i, x.data)?;
        }

        for x in &self.edges {
            writeln!(f, "{} {}--{}", x.data, x.vertices.0, x.vertices.1)?;
        }
        Ok(())
    }
}

impl<N, E> Graph<N, E> {
    /// Create an empty graph.
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph, with arbitrary data, and return its index.
    pub fn add_node(&mut self, data: N) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            edges: Vec::new(),
            data,
        });
        index
    }

    /// Add an edge between vertex indices `source` and `target` to the graph, with arbitrary data.
    pub fn add_edge(&mut self, source: usize, target: usize, data: E) {
        let index = self.edges.len();
        self.edges.push(Edge {
            vertices: (source, target),
            data,
        });
        self.nodes[source].edges.push(index);
        self.nodes[target].edges.push(index);
    }

    /// Get the node with index `index`.
    pub fn node(&self, index: usize) -> &Node<N> {
        &self.nodes[index]
    }

    /// Get the edge with index `index`.
    pub fn edge(&self, index: usize) -> &Edge<E> {
        &self.edges[index]
    }

    /// Get all nodes of the graph.
    pub fn nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    /// Get all edges of the graph.
    pub fn edges(&self) -> &[Edge<E>] {
        &self.edges
    }
}

impl<N: Clone + PartialOrd + Ord + Eq + Hash, E: Clone + PartialOrd + Ord + Eq + Hash> Graph<N, E> {
    /// Canonize the graph using McKay's canonical graph labeling algorithm.
    pub fn canonize(&self) -> Self {
        if self.nodes.len() <= u16::MAX as usize {
            self.canonize_impl::<u16>()
        } else if self.nodes.len() <= u32::MAX as usize {
            self.canonize_impl::<u32>()
        } else {
            self.canonize_impl::<usize>()
        }
    }

    fn canonize_impl<I: NodeIndex>(&self) -> Self {
        // split on vertex color
        let mut h = HashMap::default();
        for (i, x) in self.nodes.iter().enumerate() {
            h.entry(&x.data)
                .or_insert_with(Vec::new)
                .push(I::from_usize(i));
        }
        let mut partition: Vec<_> = h.into_iter().collect();
        partition.sort_by_key(|x| x.0);

        let mut stack = vec![SearchTreeNode {
            partition: partition.into_iter().map(|x| x.1).collect(),
            selected_part: None,
            selected_vertex: None,
            children_to_visit: vec![],
            invariant: Invariant::default(),
        }];

        let mut automorphisms = vec![];
        let mut leaf_nodes: HashMap<_, (Vec<_>, Vec<_>)> = HashMap::default(); // TODO: limit growth
        let mut current_best: Option<(Graph<&N, &E>, Vec<Invariant<I>>)> = None;

        let mut node_buffer = vec![];

        while let Some(mut node) = stack.pop() {
            if node.selected_vertex.is_none() {
                node.refine(self);
            }

            if let Some((_best, best_invariant)) = &current_best {
                // the canonical form is defined as the maximal isomorphism, prepended with the node invariants of the path
                // at each tree level, the node invariant must therefore be at least as good as the best
                // to be a potential canonical form
                match node
                    .invariant
                    .cmp(&best_invariant[stack.len().min(best_invariant.len() - 1)])
                {
                    Ordering::Less => {
                        node_buffer.push(node);
                        continue;
                    }
                    Ordering::Greater => {
                        // we will find a better isomorphism on this path
                        current_best = None;
                    }
                    Ordering::Equal => {}
                }
            }

            if node.partition.iter().all(|x| x.len() == 1) {
                let partition: Vec<_> = node.partition.iter().map(|x| x[0]).collect();

                let mut g = Graph::new();
                for i in &partition {
                    g.add_node(&self.node(i.to_usize()).data);
                }
                let mut edges = vec![];
                for e in &self.edges {
                    let a = partition
                        .iter()
                        .position(|&x| x.to_usize() == e.vertices.0)
                        .unwrap();
                    let b = partition
                        .iter()
                        .position(|&x| x.to_usize() == e.vertices.1)
                        .unwrap();

                    if a < b {
                        edges.push((a, b, &e.data));
                    } else {
                        edges.push((b, a, &e.data));
                    }
                }
                edges.sort();
                for x in edges {
                    g.add_edge(x.0, x.1, x.2);
                }

                let path: Vec<_> = stack.iter().map(|x| x.selected_vertex.unwrap()).collect();

                if let Some((old_partition, old_path)) = leaf_nodes.get(&g) {
                    // construct the automorphism transformation
                    let mut seen = vec![false; partition.len()];
                    let mut fixed = vec![];
                    let mut orbits = vec![];
                    for x in &partition {
                        let mut cur = *x;
                        if seen[cur.to_usize()] {
                            continue;
                        }

                        let mut orbit = vec![cur];

                        let parts = [old_partition, &partition];
                        let mut part_i = 0;
                        loop {
                            cur = parts[(part_i + 1) % 2]
                                [parts[part_i].iter().position(|y| y == &cur).unwrap()];

                            if cur == *x {
                                break;
                            } else {
                                seen[cur.to_usize()] = true;
                                orbit.push(cur);
                                part_i = (part_i + 1) % 2;
                            }
                        }

                        orbit.sort();

                        if orbit.len() == 1 {
                            fixed.push(orbit[0]);
                        } else {
                            // only store the minimal representative per orbit
                            orbits.push(orbit[0]);
                        }
                    }

                    automorphisms.push((fixed, orbits));

                    // fall back to common ancestor in the search tree
                    let mut i = 0;
                    for (p1, p2) in old_path.iter().rev().zip(path.iter().rev()) {
                        if p1 != p2 {
                            i += 1;
                        } else {
                            break;
                        }
                    }

                    node_buffer.extend(stack.drain(stack.len() + 1 - i..));
                    continue;
                }

                if let Some((best, _)) = &current_best {
                    debug_assert!(g != *best);
                    if g < *best {
                        // isomorph does not improve the best
                        // add it to the list of terminal nodes anyway to discover new
                        // automorphisms
                        leaf_nodes.insert(g, (partition, path));
                        node_buffer.push(node);
                        continue;
                    }
                }

                let mut best_invariant: Vec<_> =
                    stack.iter().map(|x| x.invariant.clone()).collect();
                best_invariant.push(node.invariant.clone());
                current_best = Some((g.clone(), best_invariant));
                leaf_nodes.insert(g, (partition, path));
                node_buffer.push(node);
            } else {
                let p = if let Some(p) = node.selected_part {
                    if node.children_to_visit.is_empty() {
                        node_buffer.push(node);
                        continue;
                    }

                    // upon a repeat visit, filter the list of possible children with
                    // the automorphism group, taking the smallest out of every orbit
                    for (fixed, orbits) in &automorphisms {
                        // only use automorphisms that fix the vertices that are fixed by the partition
                        if !node
                            .partition
                            .iter()
                            .filter(|x| x.len() == 1)
                            .all(|x| fixed.contains(&x[0]))
                        {
                            continue;
                        }

                        node.children_to_visit.retain(|x| orbits.contains(&x));
                    }

                    if node.children_to_visit.is_empty() {
                        node_buffer.push(node);
                        continue;
                    }

                    p.to_usize()
                } else {
                    // find the first minimal length non-trivial part to individualize
                    let smallest = node
                        .partition
                        .iter()
                        .map(|x| x.len())
                        .filter(|x| *x > 1)
                        .min()
                        .unwrap();

                    let (p, part) = node
                        .partition
                        .iter()
                        .enumerate()
                        .find(|x| x.1.len() == smallest)
                        .unwrap();

                    node.children_to_visit = part.clone();
                    node.selected_part = Some(I::from_usize(p));
                    p
                };

                // individualize x
                let x = node.children_to_visit.remove(0);
                node.selected_vertex = Some(x);

                let mut new_node = match node_buffer.pop() {
                    Some(mut n) => {
                        n.partition.clear();
                        n.children_to_visit.clear();
                        n.selected_part = None;
                        n.selected_vertex = None;
                        n
                    }
                    None => SearchTreeNode::default(),
                };

                new_node
                    .partition
                    .extend(node.partition.iter().take(p).cloned());
                let part = &node.partition[p.to_usize()];
                new_node.partition.push(vec![x]);
                new_node
                    .partition
                    .push(part.iter().filter(|&y| *y != x).cloned().collect());
                new_node
                    .partition
                    .extend(node.partition.iter().skip(p + 1).cloned());

                stack.push(node);
                stack.push(new_node);
            }
        }

        let (best, _) = current_best.unwrap();
        let mut g = Graph::new();
        for i in 0..best.nodes.len() {
            g.add_node(best.node(i).data.clone());
        }
        for e in &best.edges {
            g.add_edge(e.vertices.0, e.vertices.1, e.data.clone());
        }
        g
    }
}

/// An index type that can be used to index nodes in a graph.
/// Used to save memory for small graphs.
trait NodeIndex: Default + Copy + PartialOrd + Ord + Eq + Hash + Display + Debug {
    fn to_usize(&self) -> usize;
    fn from_usize(x: usize) -> Self;
}

impl NodeIndex for usize {
    fn to_usize(&self) -> usize {
        *self
    }

    fn from_usize(x: usize) -> Self {
        x
    }
}

impl NodeIndex for u32 {
    fn to_usize(&self) -> usize {
        *self as usize
    }

    fn from_usize(x: usize) -> Self {
        assert!(x <= u32::MAX as usize);
        x as u32
    }
}

impl NodeIndex for u16 {
    fn to_usize(&self) -> usize {
        *self as usize
    }

    fn from_usize(x: usize) -> Self {
        assert!(x <= u16::MAX as usize);
        x as u16
    }
}

/// A node invariant.
#[derive(Default, Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
struct Invariant<N: NodeIndex> {
    partition_lengths: Vec<N>,
}

/// A search tree node used for canonization.
#[derive(Default)]
struct SearchTreeNode<N: NodeIndex> {
    partition: Vec<Vec<N>>,
    selected_part: Option<N>,
    selected_vertex: Option<N>,
    children_to_visit: Vec<N>,
    invariant: Invariant<N>,
}

impl<I: NodeIndex> SearchTreeNode<I> {
    /// Compute a node invariant.
    fn update_invariant(&mut self) {
        self.invariant.partition_lengths.clear();
        // TODO: improve invariant
        self.invariant
            .partition_lengths
            .extend(self.partition.iter().map(|x| I::from_usize(x.len())));
    }

    /// Apply equitable coloring rules and set the node invariant.
    fn refine<N, E: PartialOrd + Ord>(&mut self, g: &Graph<N, E>) {
        'next: loop {
            for (ii, i) in self.partition.iter().enumerate() {
                if i.len() == 1 {
                    continue;
                }

                for j in &self.partition {
                    // sorted edge colors of edges in i that connect to vertices in j
                    // the length of this vector is the degree of the vertex i in j
                    let mut degrees: Vec<_> = i
                        .iter()
                        .map(|v| {
                            (
                                {
                                    let mut edge_data: Vec<_> = g
                                        .node(v.to_usize())
                                        .edges
                                        .iter()
                                        .filter_map(|&e| {
                                            let e = g.edge(e);
                                            let k = if e.vertices.0 == v.to_usize() {
                                                NodeIndex::from_usize(e.vertices.1)
                                            } else {
                                                NodeIndex::from_usize(e.vertices.0)
                                            };
                                            if j.contains(&k) {
                                                Some(&e.data)
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    edge_data.sort();
                                    edge_data
                                },
                                *v,
                            )
                        })
                        .collect();

                    if degrees.windows(2).all(|w| w[0].0 == w[1].0) {
                        continue;
                    }

                    degrees.sort();

                    let mut degs = vec![];
                    let mut cur = vec![degrees[0].1];
                    for x in degrees.windows(2) {
                        if x[0].0 != x[1].0 {
                            degs.push(cur);
                            cur = vec![x[1].1];
                        } else {
                            cur.push(x[1].1);
                        }
                    }
                    degs.push(cur);

                    self.partition.splice(ii..=ii, degs);
                    continue 'next;
                }
            }

            break;
        }

        self.update_invariant();
    }
}

#[test]
fn canonize() {
    let mut g = Graph::new();
    let n0 = g.add_node(1);
    let n1 = g.add_node(0);
    let n2 = g.add_node(1);
    let n3 = g.add_node(0);
    let n4 = g.add_node(2);
    let n5 = g.add_node(0);
    let n6 = g.add_node(1);
    let n7 = g.add_node(0);
    let n8 = g.add_node(1);

    g.add_edge(n0, n1, 0);
    g.add_edge(n0, n3, 0);
    g.add_edge(n1, n2, 0);
    g.add_edge(n1, n3, 0);
    g.add_edge(n1, n4, 0);
    g.add_edge(n1, n5, 0);
    g.add_edge(n2, n5, 0);
    g.add_edge(n3, n4, 0);
    g.add_edge(n3, n6, 0);
    g.add_edge(n3, n7, 0);
    g.add_edge(n4, n5, 0);
    g.add_edge(n4, n7, 0);
    g.add_edge(n5, n7, 0);
    g.add_edge(n5, n8, 0);
    g.add_edge(n6, n7, 0);
    g.add_edge(n7, n8, 0);

    let c = g.canonize();

    assert_eq!(c.edge(0).vertices, (0, 2));
}
