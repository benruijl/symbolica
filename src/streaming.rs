use std::{sync::Mutex, thread::LocalKey};

use rayon::prelude::*;

use crate::{
    coefficient::Coefficient,
    representations::{default::Linear, Add, Atom, AtomSet, AtomView, Num, OwnedAdd, OwnedNum},
    state::{ResettableBuffer, State, Workspace},
};

thread_local!(static WORKSPACE: Workspace<Linear> = Workspace::new());

pub trait GetLocalWorkspace: AtomSet + Sized {
    /// Get a reference to a thread-local workspace.
    fn get_local_workspace<'a>() -> &'a LocalKey<Workspace<Self>>;
}

impl GetLocalWorkspace for Linear {
    fn get_local_workspace<'a>() -> &'a LocalKey<Workspace<Self>> {
        &WORKSPACE
    }
}

struct TermInputStream<P: AtomSet> {
    mem_buf: Vec<Atom<P>>,
}

impl<P: AtomSet> Iterator for TermInputStream<P> {
    type Item = Atom<P>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.mem_buf.pop() {
            return Some(v);
        }

        None
    }
}

struct TermOutputStream<P: AtomSet> {
    mem_buf: Vec<Atom<P>>,
}

impl<P: AtomSet> TermOutputStream<P> {
    /// Add terms to the buffer.
    fn push(&mut self, a: Atom<P>) {
        match a {
            Atom::Add(aa) => {
                for arg in aa.to_add_view().iter() {
                    self.mem_buf.push(Atom::new_from_view(&arg));
                }
            }
            _ => {
                self.mem_buf.push(a);
            }
        }
    }

    /// Sort all the terms.
    fn sort(&mut self, workspace: &Workspace<P>, state: &State) {
        self.mem_buf
            .par_sort_by(|a, b| a.as_view().cmp_terms(&b.as_view()));

        let mut out = Vec::with_capacity(self.mem_buf.len());

        if !self.mem_buf.is_empty() {
            let mut last_buf = self.mem_buf.remove(0);

            let mut handle = workspace.new_atom();
            let helper = handle.get_mut();
            let mut cur_len = 0;

            for mut cur_buf in self.mem_buf.drain(..) {
                if !last_buf.merge_terms(&mut cur_buf, helper, state) {
                    // we are done merging
                    {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                out.push(last_buf);
                                cur_len += 1;
                            }
                        } else {
                            out.push(last_buf);
                            cur_len += 1;
                        }
                    }
                    last_buf = cur_buf;
                }
            }

            if cur_len == 0 {
                out.push(last_buf);
            } else {
                out.push(last_buf);
            }
        }

        self.mem_buf = out;
    }

    fn to_expression(&mut self, workspace: &Workspace<P>, state: &State) -> Atom<P> {
        self.sort(workspace, state);

        if self.mem_buf.is_empty() {
            let mut out = Atom::<P>::new();
            out.to_num().set_from_number(Coefficient::Natural(0, 1));
            out
        } else if self.mem_buf.len() == 1 {
            self.mem_buf.pop().unwrap()
        } else {
            let mut out = Atom::<P>::new();
            let add = out.to_add();
            for x in self.mem_buf.drain(..) {
                add.extend(x.as_view());
            }
            out
        }
    }
}

/// A term streamer that allows for mapping
pub struct TermStreamer<P: AtomSet> {
    exp_in: TermInputStream<P>,
    exp_out: TermOutputStream<P>,
}

impl<P: AtomSet + GetLocalWorkspace + Send + 'static> Default for TermStreamer<P>
where
    for<'a> AtomView<'a, P>: Send,
    Atom<P>: Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<P: AtomSet + GetLocalWorkspace + Send + 'static> TermStreamer<P>
where
    for<'a> AtomView<'a, P>: Send,
    Atom<P>: Send,
{
    /// Create a new term streamer.
    pub fn new() -> TermStreamer<P> {
        TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: TermOutputStream { mem_buf: vec![] },
        }
    }

    /// Create a new term streamer that contains the
    /// terms in atom `a`. More terms can be added using `self.push`.
    pub fn new_from(a: Atom<P>) -> TermStreamer<P> {
        let mut s = TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: TermOutputStream { mem_buf: vec![] },
        };

        s.push(a);
        s
    }

    /// Add terms to the streamer.
    pub fn push(&mut self, a: Atom<P>) {
        self.exp_out.push(a);
    }

    fn move_out_to_in(&mut self) {
        std::mem::swap(&mut self.exp_in.mem_buf, &mut self.exp_out.mem_buf);
    }

    /// Map every term in the stream using the function `f`. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map(
        mut self,
        f: impl Fn(&Workspace<P>, Atom<P>) -> Atom<P> + Send + Sync,
    ) -> TermStreamer<P> {
        self.move_out_to_in();

        let out_wrap = Mutex::new(self.exp_out);

        self.exp_in.par_bridge().for_each(|x| {
            P::get_local_workspace().with(|workspace| {
                out_wrap.lock().unwrap().push(f(workspace, x));
            })
        });

        TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: out_wrap.into_inner().unwrap(),
        }
    }

    /// Convert the term stream into an expression. This may exceed the available memory.
    pub fn to_expression(&mut self, workspace: &Workspace<P>, state: &State) -> Atom<P> {
        self.exp_out.to_expression(workspace, state)
    }
}
