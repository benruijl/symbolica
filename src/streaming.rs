use std::{sync::Mutex, thread::LocalKey};

use rayon::prelude::*;

use crate::{
    representations::{
        default::DefaultRepresentation, number::Number, Add, Atom, AtomView, Num, OwnedAdd,
        OwnedAtom, OwnedNum,
    },
    state::{ResettableBuffer, State, Workspace},
};

thread_local!(static WORKSPACE: Workspace<DefaultRepresentation> = Workspace::new());

pub trait GetLocalWorkspace: Atom + Sized {
    /// Get a reference to a thread-local workspace.
    fn get_local_workspace<'a>() -> &'a LocalKey<Workspace<Self>>;
}

impl GetLocalWorkspace for DefaultRepresentation {
    fn get_local_workspace<'a>() -> &'a LocalKey<Workspace<Self>> {
        &WORKSPACE
    }
}

struct TermInputStream<P: Atom> {
    mem_buf: Vec<OwnedAtom<P>>,
}

impl<P: Atom> Iterator for TermInputStream<P> {
    type Item = OwnedAtom<P>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mem_buf.pop()
    }
}

struct TermOutputStream<P: Atom> {
    mem_buf: Vec<OwnedAtom<P>>,
}

impl<P: Atom> TermOutputStream<P> {
    /// Add terms to the buffer.
    fn push(&mut self, a: OwnedAtom<P>) {
        match a {
            OwnedAtom::Add(aa) => {
                for arg in aa.to_add_view().iter() {
                    self.mem_buf.push(OwnedAtom::new_from_view(&arg));
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
            .sort_by(|a, b| a.to_view().cmp_terms(&b.to_view()));

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
                        let v = last_buf.to_view();
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

    fn to_expression(&mut self, workspace: &Workspace<P>, state: &State) -> OwnedAtom<P> {
        self.sort(workspace, state);

        if self.mem_buf.is_empty() {
            let mut out = OwnedAtom::<P>::new();
            out.transform_to_num()
                .set_from_number(Number::Natural(0, 1));
            out
        } else if self.mem_buf.len() == 1 {
            self.mem_buf.pop().unwrap()
        } else {
            let mut out = OwnedAtom::<P>::new();
            let add = out.transform_to_add();
            for x in self.mem_buf.drain(..) {
                add.extend(x.to_view());
            }
            out
        }
    }
}

/// A term streamer that allows for mapping
pub struct TermStreamer<P: Atom> {
    exp_in: TermInputStream<P>,
    exp_out: TermOutputStream<P>,
}

impl<P: Atom + GetLocalWorkspace + Send + 'static> TermStreamer<P>
where
    for<'a> AtomView<'a, P>: Send,
    OwnedAtom<P>: Send,
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
    pub fn new_from(a: OwnedAtom<P>) -> TermStreamer<P> {
        let mut s = TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: TermOutputStream { mem_buf: vec![] },
        };

        s.push(a);
        s
    }

    /// Add terms to the streamer.
    pub fn push(&mut self, a: OwnedAtom<P>) {
        self.exp_out.push(a);
    }

    fn move_out_to_in(&mut self) {
        std::mem::swap(&mut self.exp_in.mem_buf, &mut self.exp_out.mem_buf);
    }

    /// Map every term in the stream using the function `f`. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map(
        mut self,
        f: impl Fn(&Workspace<P>, OwnedAtom<P>) -> OwnedAtom<P> + Send + Sync,
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
    pub fn to_expression(&mut self, workspace: &Workspace<P>, state: &State) -> OwnedAtom<P> {
        self.exp_out.to_expression(workspace, state)
    }
}
