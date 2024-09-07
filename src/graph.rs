use ahash::HashMap;
use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    hash::Hash,
};

/// A node in a graph, with arbitrary data.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<NodeData = Empty> {
    /// Arbitrary data associated with the node.
    pub data: NodeData,
    /// Indices of the edges connected to the node.
    pub edges: Vec<usize>,
}

/// An edge in a graph, with arbitrary data.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge<EdgeData = Empty> {
    /// Indices of the vertices connected by the edge.
    pub vertices: (usize, usize),
    /// If `true`, the edge is directed from `vertices.0` to `vertices.1`.
    pub directed: bool,
    /// Arbitrary data associated with the edge.
    pub data: EdgeData,
}

/// Empty data type.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Empty;

impl Display for Empty {
    fn fmt(&self, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Ok(())
    }
}

/// Data that has a public part and a private part. The private
/// part is not used for equality or hashing.
#[derive(Clone)]
pub struct HiddenData<T, U> {
    pub data: T,
    pub hidden: U,
}

impl<T, U> HiddenData<T, U> {
    pub fn new(data: T, hidden: U) -> Self {
        HiddenData { data, hidden }
    }
}

impl<T: PartialEq, U> PartialEq for HiddenData<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq, U> Eq for HiddenData<T, U> {}

impl<T: Hash, U> Hash for HiddenData<T, U> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T: Display, U: Display> Display for HiddenData<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} ({})", self.data, self.hidden)
    }
}

impl<T: PartialOrd, U> PartialOrd for HiddenData<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl<T: Ord, U> Ord for HiddenData<T, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

/// A multigraph with support for arbitrary node and edge data.
///
/// Use [HiddenData] to hide parts of the data from all equality and hashing.
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
            if x.directed {
                writeln!(f, "{} {}->{}", x.data, x.vertices.0, x.vertices.1)?;
            } else {
                writeln!(f, "{} {}--{}", x.data, x.vertices.0, x.vertices.1)?;
            }
        }
        Ok(())
    }
}

impl<N: Display, E: Display> Graph<N, E> {
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph G {\n");

        for (i, x) in self.nodes.iter().enumerate() {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, x.data));
        }

        for x in &self.edges {
            if x.directed {
                out.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    x.vertices.0, x.vertices.1, x.data
                ));
            } else {
                out.push_str(&format!(
                    "  {} -> {} [dir=none,label=\"{}\"];\n",
                    x.vertices.0, x.vertices.1, x.data
                ));
            }
        }

        out.push_str("}\n");
        out
    }
}

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub position: Option<usize>,
    pub parent: usize,
    pub chain_id: Option<usize>,
    pub external: bool,
    pub back_edges: Vec<usize>, // back edges starting from this node
}

/// A spanning tree representation of a graph.
/// Parts of the graph may not be in the tree.
#[derive(Clone, Debug)]
pub struct SpanningTree {
    pub nodes: Vec<NodeInfo>,
    pub order: Vec<usize>,
}

impl SpanningTree {
    pub fn is_connected(&self) -> bool {
        self.nodes.iter().all(|x| x.position.is_some())
    }

    pub fn chain_decomposition(&mut self) {
        // now build the chains, starting from the DFS root
        for &n in &self.order {
            let mut back_edge_index = 0;

            while back_edge_index < self.nodes[n].back_edges.len() {
                let node = self.nodes[n].back_edges[back_edge_index];
                back_edge_index += 1;

                if self.nodes[node].parent == n {
                    continue;
                }

                // set blocker
                if self.nodes[n].chain_id.is_none() {
                    self.nodes[n].chain_id = Some(n);
                }

                let mut target = node;
                while self.nodes[target].chain_id.is_none() {
                    let nn = &mut self.nodes[target];
                    nn.chain_id = Some(n);
                    target = nn.parent;
                }

                // the start node is always excluded from the chain,
                // as we define the chain to contain the edges
                // that connect the node to its parent
                if self.nodes[n].chain_id == Some(n) {
                    self.nodes[n].chain_id = None;
                }
            }
        }
    }

    /// Count non-external bridge nodes. Make sure to call [Self::chain_decomposition] first.
    pub fn count_bridges(&self) -> usize {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(n, x)| {
                x.chain_id.is_none()
                    && !self.nodes[x.parent].external
                    && !x.external
                    && !self.nodes[x.parent].back_edges.iter().any(|end| n == end)
            })
            .count()
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
    /// If `directed` is true, the edge is directed from `source` to `target`.
    pub fn add_edge(&mut self, source: usize, target: usize, directed: bool, data: E) {
        let index = self.edges.len();
        self.edges.push(Edge {
            vertices: if !directed && source > target {
                (target, source)
            } else {
                (source, target)
            },
            directed,
            data,
        });
        self.nodes[source].edges.push(index);
        self.nodes[target].edges.push(index);
    }

    /// Delete the last added edge. This operation is O(1).
    pub fn delete_last_edge(&mut self) -> Option<Edge<E>> {
        if let Some(edge) = self.edges.pop() {
            self.nodes[edge.vertices.0].edges.pop();
            self.nodes[edge.vertices.1].edges.pop();
            Some(edge)
        } else {
            None
        }
    }

    /// Remove the last added empty node. This operation is O(1).
    pub fn delete_last_empty_node(&mut self) -> Option<Node<N>> {
        if let Some(node) = self.nodes.last() {
            if node.edges.is_empty() {
                return self.nodes.pop();
            }
        }
        None
    }

    /// Get the node with index `index`.
    #[inline(always)]
    pub fn node(&self, index: usize) -> &Node<N> {
        &self.nodes[index]
    }

    /// Get the edge with index `index`.
    #[inline(always)]
    pub fn edge(&self, index: usize) -> &Edge<E> {
        &self.edges[index]
    }

    /// Get all nodes of the graph.
    #[inline(always)]
    pub fn nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    /// Get all edges of the graph.
    #[inline(always)]
    pub fn edges(&self) -> &[Edge<E>] {
        &self.edges
    }

    // Get the number of loop in the graph, using E - V + 1
    pub fn num_loops(&self) -> usize {
        self.edges.len() - self.nodes.len() + 1
    }

    /// Generate a spanning tree of the graph, starting at `start_vertex`.
    pub fn get_spanning_tree(&self, start_vertex: usize) -> SpanningTree {
        let mut nodes_to_visit = vec![(start_vertex, start_vertex)];
        let mut tree_nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|n| NodeInfo {
                position: None,
                parent: 0,
                chain_id: None,
                external: n.edges.len() == 1,
                back_edges: vec![],
            })
            .collect();
        let mut order = vec![0; self.nodes.len()];

        let mut index = 0;
        while let Some((n, parent)) = nodes_to_visit.pop() {
            if let Some(p) = tree_nodes[n].position {
                let par = &mut tree_nodes[parent];
                if par.position.unwrap() < p {
                    par.back_edges.push(n);
                }
                continue;
            }

            tree_nodes[n].position = Some(index);
            tree_nodes[n].parent = parent;
            order[index] = n;
            index += 1;

            for e in &self.node(n).edges {
                let edge = self.edge(*e);
                let target = if edge.vertices.0 == n {
                    edge.vertices.1
                } else {
                    edge.vertices.0
                };

                if tree_nodes[target].position.is_none() {
                    nodes_to_visit.push((target, n));
                }
            }
        }

        SpanningTree {
            nodes: tree_nodes,
            order,
        }
    }

    /// Check if the graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        self.get_spanning_tree(0)
            .nodes
            .iter()
            .all(|x| x.position.is_some())
    }
}

struct GenerationSettings<'a, E> {
    vertex_signatures: &'a [Vec<E>],
    max_vertices: Option<usize>,
    max_loops: Option<usize>,
    max_bridges: Option<usize>,
    allow_self_loops: bool,
    min_degree: usize,
    max_degree: usize,
}

impl<E: Clone + Ord> Graph<Empty, E> {
    /// Generate all connected graphs with `external_edges` as external half-edges and the given allowed list
    /// of vertex connections.
    pub fn generate(
        external_edges: &[E],
        vertex_signatures: &[Vec<E>],
        max_vertices: Option<usize>,
        max_loops: Option<usize>,
        max_bridges: Option<usize>,
        allow_self_loops: bool,
    ) -> Vec<Self> {
        let vertex_sorted: Vec<_> = vertex_signatures
            .iter()
            .map(|x| {
                let mut x = x.clone();
                x.sort();
                x
            })
            .collect();

        let mut g = Self::new();
        for _ in 0..external_edges.len() {
            g.add_node(Empty);
            g.add_node(Empty);
        }

        for (i, e) in external_edges.iter().enumerate() {
            g.add_edge(i, external_edges.len() + i, false, e.clone());
        }

        if external_edges.len() == 0 {
            g.add_node(Empty);
        }

        let settings = GenerationSettings {
            vertex_signatures: &vertex_sorted,
            max_vertices,
            max_loops,
            max_bridges,
            allow_self_loops,
            min_degree: vertex_sorted.iter().map(|x| x.len()).min().unwrap_or(0),
            max_degree: vertex_sorted.iter().map(|x| x.len()).max().unwrap_or(0),
        };

        let mut out = vec![];
        g.generate_impl(external_edges.len(), &settings, &mut out);
        out
    }

    fn generate_impl(
        &mut self,
        cur_vertex: usize,
        settings: &GenerationSettings<E>,
        out: &mut Vec<Self>,
    ) {
        if let Some(max_vertices) = settings.max_vertices {
            if self.nodes.len() > max_vertices {
                return;
            }
        }

        if let Some(max_loops) = settings.max_loops {
            // filter based on an underestimate of the loop count
            // fuse every open vertex into one vertex
            // use the minimal vertex degree to see how many minimal
            // loops we need to fuse all open vertices into one
            // connected component
            //
            // TODO: use the maximum vertex degree as well
            let open_vertices = self
                .nodes
                .iter()
                .skip(cur_vertex)
                .filter(|x| x.edges.len() < settings.min_degree)
                .count();

            let mut extra_edges = self
                .nodes
                .iter()
                .skip(cur_vertex)
                .map(|x| {
                    if x.edges.len() < settings.min_degree {
                        settings.min_degree - x.edges.len()
                    } else {
                        0
                    }
                })
                .sum::<usize>();
            if extra_edges % 2 == 1 {
                extra_edges = extra_edges / 2 + 1;
            } else {
                extra_edges /= 2;
            }

            let extra_loops = if extra_edges + 1 < open_vertices {
                // cannot form one connected component
                0
            } else {
                extra_edges + 1 - open_vertices
            };

            let loops = if open_vertices > 0 {
                self.edges.len() + extra_loops + open_vertices - self.nodes.len()
            } else {
                self.edges.len() + 1 - self.nodes.len()
            };

            if loops > max_loops {
                return;
            }
        }

        if cur_vertex == self.nodes.len() {
            let mut spanning_tree = self.get_spanning_tree(0);

            if !spanning_tree.is_connected() {
                return;
            }

            if let Some(max_bridges) = settings.max_bridges {
                spanning_tree.chain_decomposition();
                if spanning_tree.count_bridges() > max_bridges {
                    return;
                }
            }

            out.push(self.clone());
            return;
        }

        // find completions for the current vertex
        let mut cur_edges: Vec<_> = self
            .node(cur_vertex)
            .edges
            .iter()
            .map(|e| self.edges[*e].data.clone())
            .collect();
        cur_edges.sort();

        let mut edges_left: Vec<(E, usize)> = vec![];
        'next_signature: for d in settings.vertex_signatures {
            // check if the current state is compatible
            if d.len() < cur_edges.len() {
                continue;
            }

            if *d == cur_edges {
                self.generate_impl(cur_vertex + 1, settings, out);
                continue;
            }

            edges_left.clear();
            let mut edge_pos = 0;
            for e in d {
                if edge_pos < cur_edges.len() {
                    if cur_edges[edge_pos] == *e {
                        edge_pos += 1;
                        continue;
                    } else if cur_edges[edge_pos] < *e {
                        // incompatible
                        continue 'next_signature;
                    }
                }

                if let Some(last) = edges_left.last_mut() {
                    if last.0 == *e {
                        last.1 += 1;
                        continue;
                    }
                }

                edges_left.push((e.clone(), 1));
            }

            if edge_pos < cur_edges.len() {
                // incompatible
                continue;
            }

            self.distribute_edges(cur_vertex, cur_vertex, &mut edges_left, 0, settings, out);
        }
    }

    fn distribute_edges(
        &mut self,
        source: usize,
        cur_target: usize,
        edge_count: &mut [(E, usize)],
        cur_edge_count_group_index: usize,
        settings: &GenerationSettings<E>,
        out: &mut Vec<Self>,
    ) {
        if edge_count.iter().all(|x| x.1 == 0) {
            return self.generate_impl(source + 1, settings, out);
        }

        let mut grown = false;
        if cur_target == self.nodes.len() {
            grown = true;
            self.add_node(Empty);
        } else {
            self.distribute_edges(source, cur_target + 1, edge_count, 0, settings, out);
        }

        let consume_count = if source == cur_target {
            if !settings.allow_self_loops {
                return;
            }

            2
        } else {
            1
        };

        // TODO: do more extensive compatibility checks
        if self.node(cur_target).edges.len() + consume_count > settings.max_degree {
            return;
        }

        if let Some((p, (e, count))) = edge_count[cur_edge_count_group_index..]
            .iter_mut()
            .enumerate()
            .find(|x| x.1 .1 >= consume_count)
        {
            *count -= consume_count;
            self.add_edge(source, cur_target, false, e.clone());
            self.distribute_edges(
                source,
                cur_target,
                edge_count,
                cur_edge_count_group_index + p,
                settings,
                out,
            );

            self.delete_last_edge(); // TODO: cache edge data

            edge_count[cur_edge_count_group_index + p].1 += consume_count;
        }

        if grown {
            self.delete_last_empty_node().unwrap();
        }
    }
}

impl<N: Clone + PartialOrd + Ord + Eq + Hash, E: Clone + PartialOrd + Ord + Eq + Hash> Graph<N, E> {
    /// Canonize the graph using McKay's canonical graph labeling algorithm,
    /// returning the vertex mapping and the canonical form.
    pub fn canonize(&self) -> (Vec<usize>, Self) {
        if self.nodes.is_empty() {
            return (vec![], self.clone());
        }

        if self.nodes.len() <= u16::MAX as usize {
            let r = self.canonize_impl::<u16>();
            (r.0.into_iter().map(|x| x as usize).collect(), r.1)
        } else if self.nodes.len() <= u32::MAX as usize {
            let r = self.canonize_impl::<u32>();
            (r.0.into_iter().map(|x| x as usize).collect(), r.1)
        } else {
            self.canonize_impl::<usize>()
        }
    }

    fn canonize_impl<I: NodeIndex>(&self) -> (Vec<I>, Self) {
        let mut stack = vec![SearchTreeNode::new(self)];
        let mut automorphisms = vec![];
        let mut leaf_nodes: HashMap<_, (Vec<_>, Vec<_>)> = HashMap::default(); // TODO: limit growth
        let mut current_best: Option<(Graph<&N, &E>, Vec<I>, Vec<Invariant<I>>)> = None;

        let mut node_buffer = vec![];

        while let Some(mut node) = stack.pop() {
            if node.selected_vertex.is_none() {
                node.refine(self);
            }

            if let Some((_, _, best_invariant)) = &current_best {
                // the canonical form is defined as the maximal isomorph, prepended with the node invariants of the path
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
                        // we will find a better isomorph on this path
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

                    if e.directed || a <= b {
                        edges.push((a, b, e.directed, &e.data));
                    } else {
                        edges.push((b, a, e.directed, &e.data));
                    }
                }
                edges.sort();
                for (v1, v2, dir, d) in edges {
                    g.add_edge(v1, v2, dir, d);
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

                if let Some((best, _, _)) = &current_best {
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
                current_best = Some((g.clone(), partition.clone(), best_invariant));
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

        let (best, map, _) = current_best.unwrap();
        let mut g = Graph::new();
        for i in 0..best.nodes.len() {
            g.add_node(best.node(i).data.clone());
        }
        for e in &best.edges {
            g.add_edge(e.vertices.0, e.vertices.1, e.directed, e.data.clone());
        }

        (map, g)
    }

    /// Returns `true` iff the graph is isomorphic to `other`.
    pub fn is_isomorphic(&self, other: &Self) -> bool {
        if self.nodes.len() != other.nodes.len() || self.edges.len() != other.edges.len() {
            return false;
        }

        if self == other {
            return true;
        }

        let mut node = SearchTreeNode::<usize>::new(self);
        let mut other_node = SearchTreeNode::new(other);

        if node.partition.len() != other_node.partition.len() {
            return false;
        }

        // check if the vertex colors are the same
        for (i, j) in node.partition.iter().zip(&other_node.partition) {
            if i.len() != j.len() || self.node(i[0]).data != other.node(j[0]).data {
                return false;
            }
        }

        // check if the first refinement has the same node invariant
        node.refine(self);
        other_node.refine(other);

        if node.invariant != other_node.invariant {
            return false;
        }

        self.canonize().1 == other.canonize().1
    }
}

/// An index type that can be used to index nodes in a graph.
/// Used to save memory for small graphs.
trait NodeIndex: Default + Copy + PartialOrd + Ord + Eq + Hash + Display + Debug {
    fn to_usize(&self) -> usize;
    fn from_usize(x: usize) -> Self;
}

impl NodeIndex for usize {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        x
    }
}

impl NodeIndex for u32 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self as usize
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u32::MAX as usize);
        x as u32
    }
}

impl NodeIndex for u16 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self as usize
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u16::MAX as usize);
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
    /// Create a new search tree node with a partition refined on vertex color.
    fn new<N: PartialOrd + Ord + Hash, E>(g: &Graph<N, E>) -> Self {
        let mut h = HashMap::default();
        for (i, x) in g.nodes.iter().enumerate() {
            h.entry(&x.data)
                .or_insert_with(Vec::new)
                .push(I::from_usize(i));
        }
        let mut partition: Vec<_> = h.into_iter().collect();
        partition.sort_by_key(|x| x.0);

        SearchTreeNode {
            partition: partition.into_iter().map(|x| x.1).collect(),
            selected_part: None,
            selected_vertex: None,
            children_to_visit: vec![],
            invariant: Invariant::default(),
        }
    }

    /// Compute a node invariant.
    fn update_invariant(&mut self) {
        self.invariant.partition_lengths.clear();
        // TODO: improve invariant
        self.invariant
            .partition_lengths
            .extend(self.partition.iter().map(|x| I::from_usize(x.len())));
    }

    /// Apply a label-invariant refinement to the partition.
    /// The current refinement is based on the signature of the neighbors of a vertex.
    /// The signature takes directional edges and edge colors into account.
    fn refine<N, E: PartialOrd + Ord>(&mut self, g: &Graph<N, E>) {
        let largest_partition = self.partition.iter().map(|x| x.len()).max().unwrap();
        let mut degrees = vec![(vec![], I::from_usize(0)); largest_partition];

        'next: loop {
            for (ii, i) in self.partition.iter().enumerate() {
                if i.len() == 1 {
                    continue;
                }

                for j in &self.partition {
                    // sorted edge colors of edges in i that connect to vertices in j
                    // the length of this vector is the degree of the vertex i in j
                    for ((edge_data, vert), v) in degrees.iter_mut().zip(i) {
                        edge_data.clear();

                        for e in &g.node(v.to_usize()).edges {
                            let e = g.edge(*e);
                            let (k, is_source) = if e.vertices.0 == v.to_usize() {
                                (NodeIndex::from_usize(e.vertices.1), true)
                            } else {
                                (NodeIndex::from_usize(e.vertices.0), false)
                            };
                            if j.contains(&k) {
                                if e.directed {
                                    edge_data.push((&e.data, e.directed, is_source));
                                } else {
                                    edge_data.push((&e.data, false, false));
                                }
                            }
                        }
                        edge_data.sort();
                        *vert = *v;
                    }

                    if degrees[..i.len()].windows(2).all(|w| w[0].0 == w[1].0) {
                        continue;
                    }

                    degrees[..i.len()].sort();

                    let mut degs = vec![];
                    let mut cur = vec![degrees[0].1];
                    for x in degrees[..i.len()].windows(2) {
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

#[cfg(test)]
mod test {
    use crate::graph::{Empty, Graph, SearchTreeNode};

    #[test]
    fn directed() {
        let mut g = Graph::new();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        g.add_edge(n0, n1, false, 0);
        g.add_edge(n0, n1, false, 1);

        let mut node = SearchTreeNode::<usize>::default();
        node.partition = vec![vec![0, 1]];
        node.refine(&g);
        assert_eq!(node.partition.len(), 1); // cannot refine

        let mut g = Graph::new();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        g.add_edge(n0, n1, true, 0);
        g.add_edge(n0, n1, false, 1);

        let mut node = SearchTreeNode::<usize>::default();
        node.partition = vec![vec![0, 1]];
        node.refine(&g);
        assert_eq!(node.partition.len(), 2); // distinguish two nodes based on directed edge
    }

    #[test]
    fn isomorphic() {
        let mut g = Graph::new();
        let n0 = g.add_node(0);
        g.add_edge(n0, n0, false, 0);
        g.add_edge(n0, n0, false, 0);

        let mut g1 = Graph::new();
        let n0 = g1.add_node(0);
        g1.add_edge(n0, n0, false, 0);

        assert!(!g.is_isomorphic(&g1));

        g1.add_edge(n0, n0, true, 0);
        assert!(!g.is_isomorphic(&g1));

        g.add_edge(n0, n0, true, 0);
        g1.add_edge(n0, n0, false, 0);
        assert!(g.is_isomorphic(&g1));

        let _ = g.add_node(1);
        let _ = g1.add_node(0);
        assert!(!g.is_isomorphic(&g1));

        let _ = g1.add_node(1);
        let _ = g.add_node(0);
        assert!(g.is_isomorphic(&g1));
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

        g.add_edge(n0, n1, false, 0);
        g.add_edge(n0, n3, false, 0);
        g.add_edge(n1, n2, false, 0);
        g.add_edge(n1, n3, false, 0);
        g.add_edge(n1, n4, false, 0);
        g.add_edge(n1, n5, false, 0);
        g.add_edge(n2, n5, false, 0);
        g.add_edge(n3, n4, false, 0);
        g.add_edge(n3, n6, false, 0);
        g.add_edge(n3, n7, false, 0);
        g.add_edge(n4, n5, false, 0);
        g.add_edge(n4, n7, false, 0);
        g.add_edge(n5, n7, false, 0);
        g.add_edge(n5, n8, false, 0);
        g.add_edge(n6, n7, false, 0);
        g.add_edge(n7, n8, false, 0);

        let c = g.canonize();

        assert_eq!(c.1.edge(0).vertices, (0, 2));
    }

    #[test]
    fn generate() {
        let gs = Graph::<Empty, &str>::generate(
            &["g", "g"],
            &[
                vec!["g", "g", "g"],
                vec!["q", "qb", "g"],
                vec!["g", "g", "g", "g"],
            ],
            None,
            Some(3),
            Some(0),
            false,
        );

        assert_eq!(gs.len(), 129);
    }
}
