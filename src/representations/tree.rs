use std::mem::size_of;

use crate::utils;

use super::Identifier;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number {
    pub num: i64,
    pub den: i64,
}

impl Number {
    pub fn new(num: i64, den: i64) -> Number {
        Number { num, den }
    }

    pub fn mul(&self, other: &Number) -> Number {
        let c = (self.num * other.num, self.den * other.den);
        let gcd = utils::gcd_signed(c.0 as i64, c.1 as i64);

        Number {
            num: c.0 / gcd,
            den: c.1 / gcd,
        }
    }
}

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
