use ahash::HashMap;
use smallvec::SmallVec;
use std::{cmp::Ordering, hash::Hash};

use crate::domains::integer::Integer;

/// An iterator for combinations without replacement.
pub struct CombinationIterator {
    n: usize,
    indices: Vec<usize>,
    init: bool,
}

impl CombinationIterator {
    pub fn new(n: usize, k: usize) -> CombinationIterator {
        CombinationIterator {
            indices: (0..k).collect(),
            n,
            init: false,
        }
    }

    pub fn next(&mut self) -> Option<&[usize]> {
        if self.indices.is_empty() || self.indices.len() > self.n {
            return None;
        }

        if !self.init {
            self.init = true;

            return Some(&self.indices);
        }

        if self.indices.is_empty() {
            return None;
        }

        let mut done = true;
        for (i, v) in self.indices.iter().enumerate().rev() {
            if *v < self.n - self.indices.len() + i {
                let a = *v + 1;
                for (p, vv) in &mut self.indices[i..].iter_mut().enumerate() {
                    *vv = a + p;
                }

                done = false;
                break;
            }
        }

        if done {
            None
        } else {
            Some(&self.indices)
        }
    }
}

#[test]
fn test() {
    let mut c = CombinationIterator::new(10, 5);
    while let Some(a) = c.next() {
        println!("{:?}", a);
    }
}

/// An iterator for combinations with replacement.
pub struct CombinationWithReplacementIterator {
    indices: SmallVec<[u32; 10]>,
    k: u32,
    init: bool,
}

impl CombinationWithReplacementIterator {
    pub fn new(n: usize, k: u32) -> CombinationWithReplacementIterator {
        CombinationWithReplacementIterator {
            indices: (0..n).map(|_| 0).collect(),
            k,
            init: false,
        }
    }

    pub fn next(&mut self) -> Option<&[u32]> {
        if self.indices.is_empty() {
            return None;
        }

        if !self.init {
            self.init = true;
            self.indices[0] = self.k;
            return Some(&self.indices);
        }

        if self.k == 0 {
            return None;
        }

        // find the last non-zero index that is not at the end
        let mut i = self.indices.len() - 1;
        while self.indices[i] == 0 {
            i -= 1;
        }

        // cannot move to the right more
        // find the next index
        let mut last_val = 0;
        if i == self.indices.len() - 1 {
            last_val = self.indices[i];
            self.indices[i] = 0;

            if self.indices.len() == 1 {
                return None;
            }

            i = self.indices.len() - 2;
            while self.indices[i] == 0 {
                if i == 0 {
                    return None;
                }

                i -= 1;
            }
        }

        self.indices[i] -= 1;
        self.indices[i + 1] = last_val + 1;

        Some(&self.indices)
    }
}

/// Generate all unique permutations of the `list` entries.
///
/// The combinatorial prefactor of each element is `list.len()! / out.len()` where
/// `out` is the returned list.
pub fn unique_permutations<T: Clone + Hash + Ord>(list: &[T]) -> (Integer, Vec<Vec<T>>) {
    let mut unique: HashMap<&T, usize> = HashMap::default();
    for e in list {
        *unique.entry(e).or_insert(0) += 1;
    }
    let mut unique: Vec<_> = unique.into_iter().collect();

    // determine pre-factor
    let mut prefactor = Integer::one();
    for (_, count) in &unique {
        prefactor *= &Integer::factorial(*count as u32);
    }

    let mut out = vec![];
    unique_permutations_impl(
        &mut unique,
        &mut Vec::with_capacity(list.len()),
        list.len(),
        &mut out,
    );
    (prefactor, out)
}

fn unique_permutations_impl<T: Clone>(
    unique: &mut Vec<(&T, usize)>,
    accum: &mut Vec<T>,
    len: usize,
    out: &mut Vec<Vec<T>>,
) {
    if accum.len() == len {
        out.push(accum.to_vec());
    }

    for i in 0..unique.len() {
        let (entry, count) = &mut unique[i];
        if *count > 0 {
            *count -= 1;
            accum.push(entry.clone());
            unique_permutations_impl(unique, accum, len, out);
            accum.pop();
            unique[i].1 += 1;
        }
    }
}

/// Partition the unordered list `elements` into named bins of unordered lists with a given length,
/// returning all partitions and their multiplicity.
///
/// For example:
/// ```
/// partitions(&[1, 1, 1, 2, 2],
///     &[('f', 2), ('g', 2), ('f', 1)],
///     false,
///     false
/// )
/// ```
/// generates all possible ways to partition the elements of three sets
/// and yields:
/// ```plain
/// [(3, [('g', [1]), ('f', [1, 1]), ('f', [2, 2])]), (6, [('g', [1]), ('f', [1, 2]),
/// ('f', [1, 2])]), (6, [('g', [2]), ('f', [1, 1]), ('f', [1, 2])])]
/// ```
///
/// If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
/// will add all remaining elements to the last set.
///
/// Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
/// if possible.
pub fn partitions<T: Ord + Hash + Copy, B: Ord + Hash + Copy>(
    elements: &[T],
    bins: &[(B, usize)],
    fill_last: bool,
    repeat: bool,
) -> Vec<(Integer, Vec<(B, Vec<T>)>)> {
    if bins.is_empty() {
        return vec![];
    }

    let bin_sum = bins.iter().map(|b| b.1).sum::<usize>();
    match elements.len().cmp(&bin_sum) {
        Ordering::Less => {
            return vec![];
        }
        Ordering::Equal => {}
        Ordering::Greater => {
            if !fill_last && (!repeat || elements.len() % bin_sum != 0) {
                return vec![];
            }
        }
    }

    // create groups of equal elements
    let mut element_groups: HashMap<T, usize> = HashMap::default();
    for e in elements {
        *element_groups.entry(*e).or_insert(0) += 1;
    }

    let mut element_sorted: Vec<(T, usize)> = element_groups.into_iter().collect();
    element_sorted.sort();

    let mut sorted_bins = bins.to_vec();

    // extend the bins if needed
    if fill_last {
        let last_bin = sorted_bins.last_mut().unwrap();
        last_bin.1 += elements.len() - bin_sum;
    }

    if repeat {
        for _ in 1..elements.len() / bin_sum {
            sorted_bins.extend_from_slice(bins);
        }
    }

    // sort the bins from largest to smallest and based on the bin id
    sorted_bins.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    fn fill_bin<T: Copy>(
        len: usize,
        elems: &mut [(T, usize)],
        accum: &mut Vec<T>,
        result: &mut Vec<Vec<T>>,
    ) {
        if len == 0 {
            result.push(accum.clone());
            return;
        }

        for i in 0..elems.len() {
            let (name, count) = &mut elems[i];
            if *count > 0 {
                *count -= 1;
                accum.push(*name);
                fill_bin(len - 1, &mut elems[i..], accum, result);
                accum.pop();
                elems[i].1 += 1;
            }
        }
    }

    fn fill_rec<T: Ord + Copy, B: Copy + Eq>(
        bins: &[(B, usize)],
        elems: &mut [(T, usize)],
        single_bin_accum: &mut Vec<T>,
        single_bin_fill: &mut Vec<Vec<T>>,
        accum: &mut Vec<(B, Vec<T>)>,
        result: &mut Vec<(Integer, Vec<(B, Vec<T>)>)>,
    ) {
        if bins.is_empty() {
            if elems.iter().all(|x| x.1 == 0) {
                result.push((Integer::one(), accum.clone()));
            }
            return;
        }
        debug_assert!(elems.iter().any(|x| x.1 > 0));

        let (bin_id, bin_len) = &bins[0];

        // find all possible ways to fill fun_len
        fill_bin(*bin_len, elems, single_bin_accum, single_bin_fill);

        let mut new_bin_fill = vec![];
        for a in single_bin_fill.drain(..) {
            // make sure we generate a descending list
            if let Some(l) = accum.last() {
                if l.0 == *bin_id && a.len() == l.1.len() && a < l.1 {
                    continue;
                }
            }

            // remove uses from the counters
            for x in &a {
                elems.iter_mut().find(|e| e.0 == *x).unwrap().1 -= 1;
            }

            accum.push((*bin_id, a.clone()));
            fill_rec(
                &bins[1..],
                elems,
                single_bin_accum,
                &mut new_bin_fill,
                accum,
                result,
            );
            accum.pop();

            for x in &a {
                elems.iter_mut().find(|e| e.0 == *x).unwrap().1 += 1;
            }
        }
    }

    let mut res = vec![];
    fill_rec(
        &mut sorted_bins,
        &mut element_sorted,
        &mut vec![],
        &mut vec![],
        &mut vec![],
        &mut res,
    );

    // compute the prefactor
    let mut counter = vec![];
    let mut bin_goups: HashMap<&(B, Vec<T>), usize> = HashMap::default();
    for (pref, sol) in &mut res {
        for (e, _) in &element_sorted {
            counter.clear();
            for (_, bin) in &*sol {
                let c = bin.iter().filter(|be| *be == e).count();
                if c > 0 {
                    counter.push(c as u32);
                }
            }
            *pref *= &Integer::multinom(&counter);
        }

        // count the number of unique bins
        for named_bin in &*sol {
            *bin_goups.entry(named_bin).or_insert(0) += 1;
        }

        for (_, p) in bin_goups.drain() {
            *pref /= &Integer::new(p as i64);
        }
    }

    res
}
