use smallvec::{smallvec, SmallVec};

use crate::{
    representations::{
        Add, Atom, AtomView, Fun, Identifier, ListIterator, ListSlice, Mul, OwnedAtom, Pow,
        SliceType,
    },
    state::State,
};

pub enum Pattern<P: Atom> {
    Wildcard(Identifier, usize, usize),    // min range, max range
    Fn(Identifier, bool, Vec<Pattern<P>>), // bool signifies that the identifier is a wildcard
    Pow(Box<[Pattern<P>; 2]>),
    Mul(Vec<Pattern<P>>),
    Add(Vec<Pattern<P>>),
    Literal(OwnedAtom<P>), // a literal
}

impl<P: Atom> std::fmt::Debug for Pattern<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wildcard(arg0, arg1, arg2) => f
                .debug_tuple("Wildcard")
                .field(arg0)
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::Fn(arg0, arg1, arg2) => f
                .debug_tuple("Fn")
                .field(arg0)
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::Pow(arg0) => f.debug_tuple("Pow").field(arg0).finish(),
            Self::Mul(arg0) => f.debug_tuple("Mul").field(arg0).finish(),
            Self::Add(arg0) => f.debug_tuple("Add").field(arg0).finish(),
            Self::Literal(arg0) => f.debug_tuple("Literal").field(arg0).finish(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Match<'a, P: Atom> {
    Single(AtomView<'a, P>),
    Multiple(SliceType, SmallVec<[AtomView<'a, P>; 10]>),
    FunctionName(Identifier),
}

impl<'a, P: Atom> std::fmt::Debug for Match<'a, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(arg0) => f.debug_tuple("").field(arg0).finish(),
            Self::Multiple(arg0, arg1) => f.debug_tuple("").field(arg0).field(arg1).finish(),
            Self::FunctionName(arg0) => f.debug_tuple("Fn").field(arg0).finish(),
        }
    }
}

/// An insertion-ordered map of wildcard identifiers to a subexpressions.
pub struct MatchStack<'a, P: Atom>(Vec<(Identifier, Match<'a, P>)>);

impl<'a, P: Atom> std::fmt::Debug for MatchStack<'a, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("MatchStack").field(&self.0).finish()
    }
}

impl<'a, P: Atom> MatchStack<'a, P> {
    /// Create a new match stack.
    pub fn new() -> MatchStack<'a, P> {
        MatchStack(Vec::new())
    }

    /// Add a new map of identifier `key` to value `value` to the stack and return the size the stack had inserting this new entry.
    /// If the entry `(key, value)` already exists, it is not inserted again and therefore the returned size is the actual size.
    /// If the `key` exists in the map, but the `value` is different, the insertion is ignored and `None` is returned.
    pub fn insert(&mut self, key: Identifier, value: Match<'a, P>) -> Option<usize> {
        for (rk, rv) in self.0.iter() {
            if rk == &key {
                if rv == &value {
                    return Some(self.0.len());
                } else {
                    return None;
                }
            }
        }

        self.0.push((key, value));
        Some(self.0.len() - 1)
    }

    /// Return the length of the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Truncate the stack to `len`.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }
}

impl<'a, 'b, P: Atom> IntoIterator for &'b MatchStack<'a, P> {
    type Item = &'b (Identifier, Match<'a, P>);
    type IntoIter = std::slice::Iter<'b, (Identifier, Match<'a, P>)>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

struct WildcardIter {
    name: Identifier,
    indices: SmallVec<[u32; 10]>,
    size_target: u32,
    max_size: u32,
}

enum PatternIter<'a, 'b, P: Atom> {
    Literal(Option<usize>, AtomView<'a, P>),
    Wildcard(WildcardIter),
    Fn(
        Option<usize>,
        Identifier,
        bool,
        &'a [Pattern<P>],
        Box<Option<SubSliceIterator<'a, 'b, P>>>,
    ), // index first
    Sequence(
        Option<usize>,
        SliceType,
        &'a [Pattern<P>],
        Box<Option<SubSliceIterator<'a, 'b, P>>>,
    ),
}

/// An iterator that matches a slice of patterns to a slice of atoms.
/// Use the [`SubSliceIterator::next`] to get the next match, if any.
///  
/// The flag `complete` determines whether the pattern should match the entire
/// slice `target. The flag `ordered_gapless` determines whether the the patterns
/// may match the slice of atoms in any order. For a non-symmetric function, this
/// flag should likely be set.
pub struct SubSliceIterator<'a, 'b, P: Atom> {
    pattern: &'a [Pattern<P>], // input term
    target: P::S<'b>,
    iterators: SmallVec<[PatternIter<'a, 'b, P>; 10]>,
    used_flag: SmallVec<[bool; 10]>,
    initialized: bool,
    matches: SmallVec<[usize; 10]>, // track match stack length
    state: &'b State,
    complete: bool,        // match needs to consume entire target
    ordered_gapless: bool, // pattern should appear ordered and have no gaps
}

impl<'a, 'b, P: Atom> SubSliceIterator<'a, 'b, P> {
    /// Create a new sub-slice iterator.
    pub fn new(
        pattern: &'a [Pattern<P>],
        target: P::S<'b>,
        state: &'b State,
        complete: bool,
        ordered: bool,
    ) -> SubSliceIterator<'a, 'b, P> {
        let mut shortcut_done = false;

        // shortcut if the number of arguments is wrong
        let min_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(_, min_size, _) => *min_size,
                _ => 1,
            })
            .sum();

        if pattern.len() > target.len() + min_length {
            shortcut_done = true;
        };

        let max_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(_, _, max_size) => *max_size,
                _ => 1,
            })
            .sum();

        if complete && max_length < target.len() {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern,
            iterators: SmallVec::new(),
            matches: SmallVec::new(),
            used_flag: smallvec![false; target.len()],
            target,
            state,
            initialized: shortcut_done,
            complete,
            ordered_gapless: ordered,
        }
    }

    /// Get the next matches, where the map of matches is written into `match_stack`.
    /// The function returns the length of the match stack before the last subiterator
    /// matched. This value can be ignored by the end-user. If `None` is returned,
    /// all potential matches will have been generated and the iterator will generate
    /// `None` if called again.
    pub fn next(&mut self, match_stack: &mut MatchStack<'b, P>) -> Option<usize> {
        let mut forward_pass = !self.initialized;

        'next_match: loop {
            self.initialized = true;

            if !forward_pass && self.iterators.len() == 0 {
                return None; // done as all options have been exhausted
            }

            if forward_pass && self.iterators.len() == self.pattern.len() {
                // check the proposed solution for extra conditions
                if self.complete && self.used_flag.iter().any(|x| !*x) {
                    // not done as the entire target is not used
                    // TODO: optimize bounds on iterators to make this case rare
                    forward_pass = false;
                } else {
                    // yield the current match
                    return Some(*self.matches.last().unwrap());
                }
            }

            if forward_pass {
                // add new iterator
                let it = match &self.pattern[self.iterators.len()] {
                    Pattern::Wildcard(name, min, max) => PatternIter::Wildcard(WildcardIter {
                        name: *name,
                        indices: SmallVec::new(),
                        size_target: *min as u32,
                        max_size: *max as u32,
                    }),
                    Pattern::Fn(name, is_wildcard, args) => {
                        PatternIter::Fn(None, *name, *is_wildcard, args, Box::new(None))
                    }
                    Pattern::Pow(base_exp) => PatternIter::Sequence(
                        None,
                        SliceType::Pow,
                        base_exp.as_slice(),
                        Box::new(None),
                    ),
                    Pattern::Mul(pat) => {
                        PatternIter::Sequence(None, SliceType::Mul, pat, Box::new(None))
                    }
                    Pattern::Add(pat) => {
                        PatternIter::Sequence(None, SliceType::Add, pat, Box::new(None))
                    }
                    Pattern::Literal(atom) => PatternIter::Literal(None, atom.to_view()),
                };

                self.iterators.push(it);
            } else {
                // update an existing iterator, so pop the latest matches (this implies every iter pushes to the match)
                match_stack.truncate(self.matches.pop().unwrap());
            }

            // assume we are in forward pass mode
            // if the iterator does not match this variable is set to false
            forward_pass = true;

            match self.iterators.last_mut().unwrap() {
                PatternIter::Wildcard(w) => {
                    // using empty list as toggle for initialized state of iterator
                    // TODO: does not work when empty list is valid solution
                    let mut wildcard_forward_pass = w.indices.is_empty();

                    'next_wildcard_match: while !wildcard_forward_pass
                        || w.indices.len() < w.max_size as usize
                    {
                        // a wildcard collects indices in increasing order
                        let start_index = w.indices.last().map(|x| *x as usize + 1).unwrap_or(0);

                        if !wildcard_forward_pass {
                            let last_iterator_empty = w.indices.is_empty();
                            if let Some(last_index) = w.indices.pop() {
                                self.used_flag[last_index as usize] = false;
                            }

                            if last_iterator_empty {
                                // the current iterator is exhausted
                                // increase the size of the wildcard range if possible, otherwise we are done
                                let max_space = self.used_flag.iter().filter(|x| !*x).count();

                                if w.size_target < w.max_size.min(max_space as u32) {
                                    w.size_target += 1;

                                    wildcard_forward_pass = true;
                                    continue 'next_wildcard_match;
                                } else {
                                    break; // done
                                }
                            } else if self.ordered_gapless {
                                // after the first match, fall back to an empty iterator
                                // then a size increase will be performed
                                continue 'next_wildcard_match;
                            }
                        }

                        let mut tried_first_option = false;
                        for k in start_index..self.target.len() {
                            if self.ordered_gapless && tried_first_option {
                                break;
                            }

                            if self.used_flag[k] {
                                continue;
                            }

                            tried_first_option = true;
                            self.used_flag[k] = true;
                            w.indices.push(k as u32);

                            if w.indices.len() == w.size_target as usize {
                                // simplify case of 1 argument, this is important for matching to work, since mul(x) = add(x) = arg(x) for any x
                                let matched = if w.indices.len() == 1 {
                                    match self.target.get(w.indices[0] as usize) {
                                        AtomView::Mul(m) => Match::Multiple(SliceType::Mul, {
                                            let mut iter = m.into_iter();
                                            let mut v = SmallVec::new();
                                            while let Some(x) = iter.next() {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        AtomView::Add(a) => Match::Multiple(SliceType::Add, {
                                            let mut iter = a.into_iter();
                                            let mut v = SmallVec::new();
                                            while let Some(x) = iter.next() {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        x => Match::Single(x),
                                    }
                                } else {
                                    Match::Multiple(
                                        self.target.get_type(),
                                        w.indices
                                            .iter()
                                            .map(|ii| self.target.get(*ii as usize))
                                            .collect(),
                                    )
                                };

                                // add the match to the stack if it is compatible
                                if let Some(new_stack_len) = match_stack.insert(w.name, matched) {
                                    self.matches.push(new_stack_len);
                                    continue 'next_match;
                                } else {
                                    // no match,
                                    w.indices.pop();
                                    self.used_flag[k] = false;
                                }
                            } else {
                                // go to next iteration?
                                wildcard_forward_pass = true;
                                continue 'next_wildcard_match;
                            }
                        }

                        // no match found and last element popped,
                        // try to increase the index of the current last element
                        wildcard_forward_pass = false;
                    }
                }
                PatternIter::Fn(index, name, is_wildcard, args, s) => {
                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some(x) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                if *is_wildcard {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }

                                self.used_flag[*jj] = false;
                                **s = None;
                                *jj + 1
                            }
                        }
                        None => 0,
                    };

                    // find a new match and create a new iterator
                    let mut tried_first_option = false;
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        if let AtomView::Fun(f) = self.target.get(ii) {
                            let target_name = f.get_name();
                            let name_match = if *is_wildcard {
                                if let Some(new_stack_len) =
                                    match_stack.insert(*name, Match::FunctionName(target_name))
                                {
                                    self.matches.push(new_stack_len);
                                    true
                                } else {
                                    ii += 1;
                                    continue;
                                }
                            } else {
                                f.get_name() == *name
                            };

                            if name_match {
                                let mut it = SubSliceIterator::new(
                                    args,
                                    f.to_slice(),
                                    &self.state,
                                    true,
                                    true,
                                );

                                if let Some(x) = it.next(match_stack) {
                                    *index = Some(ii);
                                    **s = Some(it);
                                    self.matches.push(x);
                                    self.used_flag[ii] = true;

                                    continue 'next_match;
                                }

                                if *is_wildcard {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }
                            }
                        }

                        ii += 1;
                    }
                }
                PatternIter::Literal(index, atom) => {
                    let mut ii = match index {
                        Some(jj) => {
                            self.used_flag[*jj] = false;
                            *jj + 1
                        }
                        None => 0,
                    };

                    let mut tried_first_option = false;
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        if self.target.get(ii) == *atom {
                            *index = Some(ii);
                            self.matches.push(match_stack.len());
                            self.used_flag[ii] = true;
                            continue 'next_match;
                        }
                        ii += 1;
                    }
                }
                PatternIter::Sequence(index, slice_type, pattern, s) => {
                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some(x) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                self.used_flag[*jj] = false;
                                *jj + 1
                            }
                        }
                        None => 0,
                    };

                    // find a new match and create a new iterator
                    let mut tried_first_option = false;
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        let slice = match (self.target.get(ii), &slice_type) {
                            (AtomView::Mul(m), SliceType::Mul) => m.to_slice(),
                            (AtomView::Add(a), SliceType::Add) => a.to_slice(),
                            (AtomView::Pow(a), SliceType::Pow) => a.to_slice(),
                            _ => {
                                ii += 1;
                                continue;
                            }
                        };

                        let ordered = match slice_type {
                            SliceType::Add | SliceType::Mul => false,
                            SliceType::Pow => true, // make sure pattern (base,exp) is not exchanged
                            _ => unreachable!(),
                        };

                        let mut it =
                            SubSliceIterator::new(pattern, slice, &self.state, true, ordered);

                        if let Some(x) = it.next(match_stack) {
                            *index = Some(ii);
                            **s = Some(it);
                            self.matches.push(x);
                            self.used_flag[ii] = true;

                            continue 'next_match;
                        }

                        ii += 1;
                    }
                }
            }

            // no match, so fall back one level
            forward_pass = false;
            self.iterators.pop();
        }
    }
}
