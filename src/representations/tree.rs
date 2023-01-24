use std::mem::size_of;

use super::{Identifier, number::Number};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomTree {
    Var(Identifier),
    Fn(Identifier, Vec<AtomTree>), // name and args
    Num(Number),
    Pow(Box<(AtomTree, AtomTree)>),
    Mul(Vec<AtomTree>),
    Add(Vec<AtomTree>),
}

impl AtomTree {
    pub fn len(&self) -> usize {
        size_of::<AtomTree>()
            + match self {
                AtomTree::Fn(_, args) | AtomTree::Mul(args) | AtomTree::Add(args) => {
                    args.iter().map(|a| a.len()).sum()
                }
                AtomTree::Pow(p) => p.0.len() + p.1.len(),
                _ => 0,
            }
    }
}
