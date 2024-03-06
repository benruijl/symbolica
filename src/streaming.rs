use std::sync::Mutex;

use rayon::prelude::*;

use crate::{
    representations::{Atom, AtomView},
    state::{RecycledAtom, Workspace},
};

struct TermInputStream {
    mem_buf: Vec<Atom>,
}

impl Iterator for TermInputStream {
    type Item = Atom;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.mem_buf.pop() {
            return Some(v);
        }

        None
    }
}

struct TermOutputStream {
    mem_buf: Vec<Atom>,
}

impl TermOutputStream {
    /// Add terms to the buffer.
    fn push(&mut self, a: Atom) {
        if let AtomView::Add(aa) = a.as_view() {
            for arg in aa.iter() {
                self.mem_buf.push(arg.to_owned());
            }
        } else {
            self.mem_buf.push(a);
        }
    }

    /// Sort all the terms.
    fn sort(&mut self) {
        self.mem_buf
            .par_sort_by(|a, b| a.as_view().cmp_terms(&b.as_view()));

        let mut out = Vec::with_capacity(self.mem_buf.len());

        if !self.mem_buf.is_empty() {
            let mut last_buf = self.mem_buf.remove(0);

            let mut handle: RecycledAtom = Atom::new().into();
            let mut cur_len = 0;

            for cur_buf in self.mem_buf.drain(..) {
                if !last_buf.merge_terms(cur_buf.as_view(), &mut handle) {
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

    fn to_expression(&mut self) -> Atom {
        self.sort();

        if self.mem_buf.is_empty() {
            Atom::new_num(0)
        } else if self.mem_buf.len() == 1 {
            self.mem_buf.pop().unwrap()
        } else {
            let mut out = Atom::default();
            let add = out.to_add();
            for x in self.mem_buf.drain(..) {
                add.extend(x.as_view());
            }
            out
        }
    }
}

/// A term streamer that allows for mapping
pub struct TermStreamer {
    exp_in: TermInputStream,
    exp_out: TermOutputStream,
}

impl Default for TermStreamer
where
    for<'a> AtomView<'a>: Send,
    Atom: Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl TermStreamer
where
    for<'a> AtomView<'a>: Send,
    Atom: Send,
{
    /// Create a new term streamer.
    pub fn new() -> TermStreamer {
        TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: TermOutputStream { mem_buf: vec![] },
        }
    }

    /// Create a new term streamer that contains the
    /// terms in atom `a`. More terms can be added using `self.push`.
    pub fn new_from(a: Atom) -> TermStreamer {
        let mut s = TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: TermOutputStream { mem_buf: vec![] },
        };

        s.push(a);
        s
    }

    /// Add terms to the streamer.
    pub fn push(&mut self, a: Atom) {
        self.exp_out.push(a);
    }

    fn move_out_to_in(&mut self) {
        std::mem::swap(&mut self.exp_in.mem_buf, &mut self.exp_out.mem_buf);
    }

    /// Map every term in the stream using the function `f`. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map(mut self, f: impl Fn(&Workspace, Atom) -> Atom + Send + Sync) -> TermStreamer {
        self.move_out_to_in();

        let out_wrap = Mutex::new(self.exp_out);

        self.exp_in.par_bridge().for_each(|x| {
            Workspace::get_local().with(|workspace| {
                out_wrap.lock().unwrap().push(f(workspace, x));
            })
        });

        TermStreamer {
            exp_in: TermInputStream { mem_buf: vec![] },
            exp_out: out_wrap.into_inner().unwrap(),
        }
    }

    /// Convert the term stream into an expression. This may exceed the available memory.
    pub fn to_expression(&mut self) -> Atom {
        self.exp_out.to_expression()
    }
}
