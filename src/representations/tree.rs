use std::mem::size_of;

use super::{number::Number, Identifier};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomTree {
    Var(Identifier),
    Fn(Identifier, Vec<Self>), // name and args
    Num(Number),
    Pow(Box<(Self, Self)>),
    Mul(Vec<Self>),
    Add(Vec<Self>),
}

impl AtomTree {
    pub fn len(&self) -> usize {
        size_of::<Self>()
            + match self {
                Self::Fn(_, args) | Self::Mul(args) | Self::Add(args) => {
                    args.iter().map(|a| a.len()).sum()
                }
                Self::Pow(p) => p.0.len() + p.1.len(),
                _ => 0,
            }
    }
}
