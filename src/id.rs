use ahash::HashMap;
use smallvec::{smallvec, SmallVec};

use crate::{
    representations::{
        Add, Atom, AtomView, Fun, Identifier, ListIterator, ListSlice, Mul, OwnedAtom, Pow,
        SliceType, Var,
    },
    state::{ResettableBuffer, State},
};

pub enum Pattern<P: Atom> {
    Wildcard(Identifier),
    Fn(Identifier, bool, Vec<Pattern<P>>), // bool signifies that the identifier is a wildcard
    Pow(Box<[Pattern<P>; 2]>),
    Mul(Vec<Pattern<P>>),
    Add(Vec<Pattern<P>>),
    Literal(OwnedAtom<P>), // a literal
}

impl<P: Atom> Pattern<P> {
    /// Check if the expression `atom` contains a wildcard.
    fn has_wildcard(atom: AtomView<'_, P>, state: &State) -> bool {
        match atom {
            AtomView::Num(_) => false,
            AtomView::Var(v) => state.is_wildcard(v.get_name()).unwrap(),
            AtomView::Fun(f) => {
                if state.is_wildcard(f.get_name()).unwrap() {
                    return true;
                }

                let mut it = f.into_iter();
                while let Some(arg) = it.next() {
                    if Self::has_wildcard(arg, state) {
                        return true;
                    }
                }
                false
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                Self::has_wildcard(base, state) || Self::has_wildcard(exp, state)
            }
            AtomView::Mul(m) => {
                let mut it = m.into_iter();
                while let Some(arg) = it.next() {
                    if Self::has_wildcard(arg, state) {
                        return true;
                    }
                }
                false
            }
            AtomView::Add(a) => {
                let mut it = a.into_iter();
                while let Some(arg) = it.next() {
                    if Self::has_wildcard(arg, state) {
                        return true;
                    }
                }
                false
            }
        }
    }

    pub fn from_view(atom: AtomView<'_, P>, state: &State) -> Pattern<P> {
        if Self::has_wildcard(atom, state) {
            match atom {
                AtomView::Var(v) => Pattern::Wildcard(v.get_name()),
                AtomView::Fun(f) => {
                    let name = f.get_name();

                    let mut args = Vec::with_capacity(f.get_nargs());
                    let mut it = f.into_iter();

                    while let Some(arg) = it.next() {
                        args.push(Self::from_view(arg, state));
                    }

                    Pattern::Fn(name, state.is_wildcard(name).unwrap(), args)
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    Pattern::Pow(Box::new([
                        Self::from_view(base, state),
                        Self::from_view(exp, state),
                    ]))
                }
                AtomView::Mul(m) => {
                    let mut args = Vec::with_capacity(m.get_nargs());
                    let mut it = m.into_iter();

                    while let Some(arg) = it.next() {
                        args.push(Self::from_view(arg, state));
                    }

                    Pattern::Mul(args)
                }
                AtomView::Add(a) => {
                    let mut args = Vec::with_capacity(a.get_nargs());
                    let mut it = a.into_iter();

                    while let Some(arg) = it.next() {
                        args.push(Self::from_view(arg, state));
                    }

                    Pattern::Add(args)
                }
                AtomView::Num(_) => unreachable!("Number cannot have wildcard"),
            }
        } else {
            let mut oa = OwnedAtom::new();
            oa.from_view(&atom);
            Pattern::Literal(oa)
        }
    }
}

impl<P: Atom> std::fmt::Debug for Pattern<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wildcard(arg0) => f.debug_tuple("Wildcard").field(arg0).finish(),
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

/// Restrictions for a wildcard. Note that a length restriction
/// applies at any level and therefore
/// `x_*f(x_) : length(x) == 2`
/// does not match to `x*y*f(x*y)`, since the pattern `x_` has length
/// 1 inside the function argument.
pub enum PatternRestriction<P>
where
    P: Atom,
{
    Length(usize, Option<usize>), // min-max range
    Filter(Box<dyn Fn(&Match<'_, P>) -> bool>),
    Cmp(
        Identifier,
        Box<dyn Fn(&Match<'_, P>, &Match<'_, P>) -> bool>,
    ),
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
/// It keeps track of all restrictions on wilcards and will check them
/// before inserting.
pub struct MatchStack<'a, P: Atom> {
    stack: Vec<(Identifier, Match<'a, P>)>,
    restrictions: &'a HashMap<Identifier, Vec<PatternRestriction<P>>>,
}

impl<'a, P: Atom> std::fmt::Debug for MatchStack<'a, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchStack")
            .field("stack", &self.stack)
            .finish()
    }
}

impl<'a, P: Atom> MatchStack<'a, P> {
    /// Create a new match stack.
    pub fn new(
        restrictions: &'a HashMap<Identifier, Vec<PatternRestriction<P>>>,
    ) -> MatchStack<'a, P> {
        MatchStack {
            stack: Vec::new(),
            restrictions,
        }
    }

    /// Add a new map of identifier `key` to value `value` to the stack and return the size the stack had inserting this new entry.
    /// If the entry `(key, value)` already exists, it is not inserted again and therefore the returned size is the actual size.
    /// If the `key` exists in the map, but the `value` is different, the insertion is ignored and `None` is returned.
    pub fn insert(&mut self, key: Identifier, value: Match<'a, P>) -> Option<usize> {
        for (rk, rv) in self.stack.iter() {
            if rk == &key {
                if rv == &value {
                    return Some(self.stack.len());
                } else {
                    return None;
                }
            }
        }

        // test whether the current value passes all restrictions
        if let Some(res) = self.restrictions.get(&key) {
            for r in res {
                match r {
                    PatternRestriction::Length(min, max) => match &value {
                        Match::Single(_) | Match::FunctionName(_) => {
                            if *min <= 1 && max.map(|m| m >= 1).unwrap_or(true) {
                                continue;
                            }
                        }
                        Match::Multiple(_, slice) => {
                            if *min <= slice.len() && max.map(|m| m >= slice.len()).unwrap_or(true)
                            {
                                continue;
                            }
                        }
                    },
                    PatternRestriction::Filter(f) => {
                        if f(&value) {
                            continue;
                        }
                    }
                    PatternRestriction::Cmp(other_id, f) => {
                        // get match stack to get the value of other_id
                        if let Some((_, value2)) = self.stack.iter().find(|(k, _)| k == other_id) {
                            if f(&value, value2) {
                                continue;
                            }
                        } else {
                            // TODO: if the value does not exist, add this check to a list of TODOs
                            continue;
                        }
                    }
                }

                return None;
            }
        }

        self.stack.push((key, value));
        Some(self.stack.len() - 1)
    }

    /// Return the length of the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Truncate the stack to `len`.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.stack.truncate(len)
    }

    /// Get the range of an identifier based on previous matches and based
    /// on restrictions.
    pub fn get_range(&self, identifier: Identifier) -> (usize, Option<usize>) {
        for (rk, rv) in self.stack.iter() {
            if rk == &identifier {
                return match rv {
                    Match::Single(_) => (1, Some(1)),
                    Match::Multiple(slice_type, slice) => {
                        match slice_type {
                            SliceType::Empty => (0, Some(0)),
                            SliceType::Arg => (slice.len(), Some(slice.len())),
                            _ => {
                                // the length needs to include 1 since for example x*y is only
                                // length one in f(x*y)
                                // TODO: the length can only be 1 or slice.len() and no values in between
                                // so we could optimize this
                                (1, Some(slice.len()))
                            }
                        }
                    }
                    Match::FunctionName(_) => (1, Some(1)),
                };
            }
        }

        let mut minimal = None;
        let mut maximal = None;

        if let Some(res) = self.restrictions.get(&identifier) {
            for r in res {
                if let PatternRestriction::Length(min, max) = r {
                    minimal = Some(minimal.map_or(*min, |v: usize| v.max(*min)));
                    maximal = max.map_or(maximal, |v| Some(maximal.map_or(v, |v1| v.min(v1))));
                }
            }
        }

        // defaulft the minimum to 1
        (minimal.unwrap_or(1), maximal)
    }
}

impl<'a, 'b, P: Atom> IntoIterator for &'b MatchStack<'a, P> {
    type Item = &'b (Identifier, Match<'a, P>);
    type IntoIter = std::slice::Iter<'b, (Identifier, Match<'a, P>)>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.stack).into_iter()
    }
}

struct WildcardIter {
    initialized: bool,
    name: Identifier,
    indices: SmallVec<[u32; 10]>,
    size_target: u32,
    max_size: u32,
}

enum PatternIter<'a, 'b, P: Atom> {
    Literal(Option<usize>, AtomView<'b, P>),
    Wildcard(WildcardIter),
    Fn(
        Option<usize>,
        Identifier,
        bool,
        &'b [Pattern<P>],
        Box<Option<SubSliceIterator<'a, 'b, P>>>,
    ), // index first
    Sequence(
        Option<usize>,
        SliceType,
        &'b [Pattern<P>],
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
    pattern: &'b [Pattern<P>], // input term
    target: P::S<'a>,
    iterators: SmallVec<[PatternIter<'a, 'b, P>; 10]>,
    used_flag: SmallVec<[bool; 10]>,
    initialized: bool,
    matches: SmallVec<[usize; 10]>, // track match stack length
    state: &'a State,
    complete: bool,        // match needs to consume entire target
    ordered_gapless: bool, // pattern should appear ordered and have no gaps
}

impl<'a, 'b, P: Atom> SubSliceIterator<'a, 'b, P> {
    /// Create an iterator over a pattern applied to a target.
    pub fn new(
        pattern: &'b Pattern<P>,
        target: AtomView<'a, P>,
        state: &'a State,
        match_stack: &MatchStack<'a, P>,
    ) -> SubSliceIterator<'a, 'b, P> {
        let mut shortcut_done = false;

        // a pattern and target can either be a single atom or a list
        // for (list, list)  create a subslice iterator on the lists that is not complete
        // for (single, list), upgrade single to a slice with one element

        let (pat_list, target_list) = match (pattern, target) {
            (Pattern::Mul(m1), AtomView::Mul(m2)) => (m1.as_slice(), m2.to_slice()),
            (Pattern::Add(a1), AtomView::Add(a2)) => (a1.as_slice(), a2.to_slice()),
            (Pattern::Mul(arg) | Pattern::Add(arg), _) => {
                shortcut_done = true; // cannot match
                (arg.as_slice(), ListSlice::from_one(target))
            }
            (_, AtomView::Mul(m2)) => (std::slice::from_ref(pattern), m2.to_slice()),
            (_, AtomView::Add(a2)) => (std::slice::from_ref(pattern), a2.to_slice()),
            (_, _) => (std::slice::from_ref(pattern), ListSlice::from_one(target)),
        };

        // shortcut if the number of arguments is wrong
        let min_length: usize = pat_list
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        if min_length > target_list.len() {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern: pat_list,
            iterators: SmallVec::new(),
            matches: SmallVec::new(),
            used_flag: smallvec![false; target_list.len()],
            target: target_list,
            state,
            initialized: shortcut_done,
            complete: false,
            ordered_gapless: false,
        }
    }

    /// Create a new sub-slice iterator.
    pub fn from_list(
        pattern: &'b [Pattern<P>],
        target: P::S<'a>,
        state: &'a State,
        match_stack: &MatchStack<'a, P>,
        complete: bool,
        ordered: bool,
    ) -> SubSliceIterator<'a, 'b, P> {
        let mut shortcut_done = false;

        // shortcut if the number of arguments is wrong
        let min_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        if min_length > target.len() {
            shortcut_done = true;
        };

        let max_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).1.unwrap_or(target.len()),
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
    pub fn next(&mut self, match_stack: &mut MatchStack<'a, P>) -> Option<usize> {
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
                    Pattern::Wildcard(name) => {
                        let range = match_stack.get_range(*name);

                        PatternIter::Wildcard(WildcardIter {
                            initialized: false,
                            name: *name,
                            indices: SmallVec::new(),
                            size_target: range.0 as u32,
                            max_size: range.1.unwrap_or(self.target.len()) as u32,
                        })
                    }
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
                    let mut wildcard_forward_pass = !w.initialized;

                    'next_wildcard_match: while !wildcard_forward_pass
                        || w.indices.len() <= w.max_size as usize
                    {
                        w.initialized = true;
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

                        // check for an empty slice match
                        if w.size_target == 0 && w.indices.is_empty() {
                            if let Some(new_stack_len) = match_stack
                                .insert(w.name, Match::Multiple(SliceType::Empty, SmallVec::new()))
                            {
                                self.matches.push(new_stack_len);
                                continue 'next_match;
                            } else {
                                wildcard_forward_pass = false;
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
                                let mut it = SubSliceIterator::from_list(
                                    args,
                                    f.to_slice(),
                                    &self.state,
                                    &match_stack,
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

                        let mut it = SubSliceIterator::from_list(
                            pattern,
                            slice,
                            &self.state,
                            &match_stack,
                            true,
                            ordered,
                        );

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

pub struct AtomTreeIterator<'a, P: Atom> {
    stack: SmallVec<[(Option<usize>, AtomView<'a, P>); 10]>,
}

impl<'a, P: Atom> AtomTreeIterator<'a, P> {
    pub fn new(target: AtomView<'a, P>) -> AtomTreeIterator<'a, P> {
        AtomTreeIterator {
            stack: smallvec![(None, target)],
        }
    }

    pub fn next(&mut self) -> Option<(usize, AtomView<'a, P>)> {
        while let Some((ind, atom)) = self.stack.pop() {
            if let Some(ind) = ind {
                let slice = match atom {
                    AtomView::Fun(f) => f.to_slice(),
                    AtomView::Pow(p) => p.to_slice(),
                    AtomView::Mul(m) => m.to_slice(),
                    AtomView::Add(a) => a.to_slice(),
                    _ => {
                        continue; // not iterable
                    }
                };

                if ind < slice.len() {
                    let new_atom = slice.get(ind);

                    self.stack.push((Some(ind + 1), atom));
                    self.stack.push((None, new_atom)); // push the new element on the stack
                }
            } else {
                // return full match and set the position to the first sub element
                self.stack.push((Some(0), atom));
                return Some((self.stack.len(), atom));
            }
        }

        None
    }
}

pub struct PatternAtomTreeIterator<'a, 'b, P: Atom> {
    pattern: &'b Pattern<P>,
    atom_tree_iterator: AtomTreeIterator<'a, P>,
    current_target: Option<AtomView<'a, P>>,
    pattern_iter: Option<SubSliceIterator<'a, 'b, P>>,
    state: &'a State,
    match_stack: MatchStack<'a, P>,
    level: usize,
}

impl<'a, 'b, P: Atom> PatternAtomTreeIterator<'a, 'b, P> {
    pub fn new(
        pattern: &'b Pattern<P>,
        target: AtomView<'a, P>,
        state: &'a State,
        restrictions: &'a HashMap<Identifier, Vec<PatternRestriction<P>>>,
    ) -> PatternAtomTreeIterator<'a, 'b, P> {
        PatternAtomTreeIterator {
            pattern,
            atom_tree_iterator: AtomTreeIterator::new(target),
            current_target: None,
            pattern_iter: None,
            state,
            match_stack: MatchStack::new(restrictions),
            level: 0,
        }
    }

    pub fn next(&mut self) -> Option<(usize, AtomView<'a, P>, &MatchStack<'a, P>)> {
        loop {
            if let Some(ct) = self.current_target {
                if let Some(it) = self.pattern_iter.as_mut() {
                    if let Some(_) = it.next(&mut self.match_stack) {
                        return Some((self.level, ct, &self.match_stack));
                    } else {
                        // no match: bail
                        self.current_target = None;
                        self.pattern_iter = None;
                        continue;
                    }
                } else {
                    self.pattern_iter = Some(SubSliceIterator::new(
                        self.pattern,
                        ct,
                        self.state,
                        &self.match_stack,
                    ));
                }
            } else {
                let tree_pos = self.atom_tree_iterator.next();

                if let Some(t) = tree_pos {
                    self.level = t.0;
                    self.current_target = Some(t.1);
                } else {
                    return None;
                }
            }
        }
    }
}
