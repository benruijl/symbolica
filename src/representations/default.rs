use byteorder::{LittleEndian, WriteBytesExt};
use bytes::{Buf, BufMut};
use std::{cmp::Ordering, io::Cursor};

use crate::{representations::tree::Number, state::ResettableBuffer, utils};

use super::{
    number::{RationalNumberReader, RationalNumberWriter},
    tree::AtomTree,
    Add, Atom, AtomView, Fn, Identifier, ListIteratorT, Mul, Num, OwnedAdd, OwnedAtom,
    OwnedFn, OwnedMul, OwnedNum, OwnedPow, OwnedVar, Pow, Var,
};

const NUM_ID: u8 = 1;
const VAR_ID: u8 = 2;
const FN_ID: u8 = 3;
const MUL_ID: u8 = 4;
const POW_ID: u8 = 5;
const ADD_ID: u8 = 6;
const TYPE_MASK: u8 = 0b00000111;
const DIRTY_FLAG: u8 = 0b10000000;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DefaultRepresentation {}

#[derive(Debug, Clone)]
pub struct OwnedAtomD {
    data: Vec<u8>,
}

impl OwnedAtom for OwnedAtomD {
    type P = DefaultRepresentation;

    fn from_num(source: <Self::P as Atom>::ON) -> Self {
        OwnedAtomD { data: source.data }
    }

    fn write<'a>(&mut self, source: &AtomView<'a, Self::P>) {
        // TODO: does not work yet, as an upgrade to Term is needed and sizes need to be changed
        self.data.extend(source.get_data());
    }

    fn write_tree(&mut self, source: &AtomTree) {
        // TODO: does not work yet, as an upgrade to Term is needed and sizes need to be changed
        self.linearize(source);
    }

    fn from_tree(a: &AtomTree) -> Self {
        let mut owned_data = OwnedAtomD { data: vec![] };
        owned_data.linearize(a);
        owned_data
    }

    fn to_tree(&self) -> AtomTree {
        OwnedAtomD::write_to_tree(self.data.as_slice()).0
    }

    fn to_view<'a>(&'a self) -> AtomView<'a, Self::P> {
        match self.data[0] & TYPE_MASK {
            VAR_ID => AtomView::Var(VarViewD { data: &self.data }),
            FN_ID => AtomView::Fn(FunctionViewD { data: &self.data }),
            NUM_ID => AtomView::Num(NumberViewD { data: &self.data }),
            MUL_ID => AtomView::Mul(MulViewD { data: &self.data }),
            POW_ID => AtomView::Pow(PowViewD { data: &self.data }),
            ADD_ID => AtomView::Add(AddViewD { data: &self.data }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl ResettableBuffer for OwnedAtomD {
    fn new() -> Self {
        OwnedAtomD { data: vec![] }
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone)]
pub struct OwnedNumD {
    data: Vec<u8>,
}

impl OwnedNum for OwnedNumD {
    type P = DefaultRepresentation;

    fn from_u64_frac(&mut self, num: u64, den: u64) {
        self.data.clear();
        self.data.put_u8(NUM_ID);
        num.write_frac(den, &mut self.data);
    }

    fn from_view<'a>(&mut self, a: NumberViewD<'a>) {
        self.data.clear();
        self.data.extend(a.data);
    }

    fn add<'a>(&mut self, other: &NumberViewD<'a>) {
        let a = self.to_num_view().get_numden();
        let b = other.get_numden();

        let c = (a.0 * b.0, a.1 * b.1);
        let gcd = utils::gcd_unsigned(c.0 as u64, c.1 as u64);

        self.data.truncate(1);
        (c.0 as u64 / gcd).write_frac(c.1 as u64 / gcd, &mut self.data);
    }

    fn to_num_view(&self) -> NumberViewD {
        assert!(self.data[0] & TYPE_MASK == NUM_ID);
        NumberViewD { data: &self.data }
    }
}

impl ResettableBuffer for OwnedNumD {
    fn new() -> Self {
        let mut data = Vec::new();
        data.put_u8(NUM_ID);
        0u64.write_num(&mut data); // TODO: should this be written?

        OwnedNumD { data }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.data.put_u8(NUM_ID);
        0u64.write_num(&mut self.data);
    }
}

#[derive(Debug, Clone)]
pub struct OwnedVarD {
    data: Vec<u8>,
}

impl OwnedVar for OwnedVarD {
    type P = DefaultRepresentation;

    fn from_id(&mut self, id: Identifier) {
        self.data.put_u8(VAR_ID);
        (id.to_u32() as u64).write_num(&mut self.data);
    }

    fn to_var_view<'a>(&'a self) -> <Self::P as Atom>::V<'a> {
        VarViewD { data: &self.data }
    }

    fn to_atom(&mut self, out: &mut OwnedAtomD) {
        out.data.clone_from(&self.data);
    }
}

impl ResettableBuffer for OwnedVarD {
    fn new() -> Self {
        OwnedVarD { data: vec![] }
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone)]
pub struct OwnedFnD {
    data: Vec<u8>,
}

impl OwnedFn for OwnedFnD {
    type P = DefaultRepresentation;

    fn from_name(&mut self, id: Identifier) {
        self.data.put_u8(VAR_ID);
        (id.to_u32() as u64).write_num(&mut self.data);
    }

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] &= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    fn add_arg(&mut self, other: AtomView<Self::P>) {
        // TODO: update the size
        self.data.extend(other.get_data());
        todo!()
    }

    fn to_fn_view<'a>(&'a self) -> <Self::P as Atom>::F<'a> {
        FunctionViewD { data: &self.data }
    }

    fn to_atom(&mut self, out: &mut OwnedAtomD) {
        out.data.clone_from(&self.data);
    }
}

impl ResettableBuffer for OwnedFnD {
    fn new() -> Self {
        OwnedFnD { data: vec![] }
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone)]
pub struct OwnedPowD {
    data: Vec<u8>,
}

impl OwnedPow for OwnedPowD {
    type P = DefaultRepresentation;

    fn from_base_and_exp(&mut self, base: AtomView<Self::P>, exp: AtomView<Self::P>) {
        self.data.put_u8(POW_ID);
        self.data.extend(base.get_data());
        self.data.extend(exp.get_data());
    }

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] &= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    fn to_pow_view<'a>(&'a self) -> <Self::P as Atom>::P<'a> {
        PowViewD { data: &self.data }
    }

    fn to_atom(&mut self, out: &mut OwnedAtomD) {
        out.data.clone_from(&self.data);
    }
}

impl ResettableBuffer for OwnedPowD {
    fn new() -> Self {
        OwnedPowD { data: vec![] }
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

pub struct OwnedMulD {
    data: Vec<u8>,
}

impl OwnedMul for OwnedMulD {
    type P = DefaultRepresentation;

    fn extend<'a>(&mut self, other: AtomView<'a, DefaultRepresentation>) {
        // may increase size of the num of args
        let c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, _) = c.get_frac_u64();

        match other {
            AtomView::Mul(_t) => {
                todo!();
            }
            _ => {
                n_args += 1;
                self.data.extend(other.get_data());
            }
        }

        // FIXME: this may overwrite the rest of the term
        // assume for now it does not
        n_args.write_frac_fixed(1, &mut self.data[1 + 4..]);

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    fn to_mul_view<'a>(&'a self) -> <Self::P as Atom>::M<'a> {
        MulViewD { data: &self.data }
    }

    fn to_atom(&mut self, out: &mut OwnedAtomD) {
        out.data.clone_from(&self.data);
    }
}

impl ResettableBuffer for OwnedMulD {
    fn new() -> Self {
        let mut data = Vec::new();
        data.put_u8(MUL_ID);
        data.put_u32_le(0 as u32);
        0u64.write_num(&mut data);

        OwnedMulD { data }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.data.put_u8(MUL_ID);
        self.data.put_u32_le(0 as u32);
        0u64.write_num(&mut self.data);
    }
}

pub struct OwnedAddD {
    data: Vec<u8>,
}

impl OwnedAdd for OwnedAddD {
    type P = DefaultRepresentation;

    fn extend<'a>(&mut self, other: AtomView<'a, DefaultRepresentation>) {
        // may increase size of the num of args
        let c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, _) = c.get_frac_u64();

        match other {
            AtomView::Add(_t) => {
                todo!();
            }
            _ => {
                n_args += 1;
                self.data.extend(other.get_data());
            }
        }

        // FIXME: this may overwrite the rest of the term
        // assume for now it does not
        n_args.write_frac_fixed(1, &mut self.data[1 + 4..]);

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    fn to_add_view<'a>(&'a self) -> <Self::P as Atom>::A<'a> {
        AddViewD { data: &self.data }
    }

    fn to_atom(&mut self, out: &mut OwnedAtomD) {
        out.data.clone_from(&self.data);
    }
}

impl ResettableBuffer for OwnedAddD {
    fn new() -> Self {
        let mut data = Vec::new();
        data.put_u8(ADD_ID);
        data.put_u32_le(0 as u32);
        0u64.write_num(&mut data);

        OwnedAddD { data }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.data.put_u8(ADD_ID);
        self.data.put_u32_le(0 as u32);
        0u64.write_num(&mut self.data);
    }
}

impl Atom for DefaultRepresentation {
    type N<'a> = NumberViewD<'a>;
    type V<'a> = VarViewD<'a>;
    type F<'a> = FunctionViewD<'a>;
    type P<'a> = PowViewD<'a>;
    type M<'a> = MulViewD<'a>;
    type A<'a> = AddViewD<'a>;
    type O = OwnedAtomD;
    type ON = OwnedNumD;
    type OV = OwnedVarD;
    type OF = OwnedFnD;
    type OP = OwnedPowD;
    type OM = OwnedMulD;
    type OA = OwnedAddD;
}

impl<'a> Var<'a> for VarViewD<'a> {
    type P = DefaultRepresentation;

    #[inline]
    fn get_name(&self) -> Identifier {
        Identifier::from((&self.data[1..]).get_frac_u64().0 as u32)
    }
}

impl OwnedAtomD {
    pub fn linearize(&mut self, atom: &AtomTree) {
        match atom {
            AtomTree::Var(name) => {
                self.data.put_u8(VAR_ID);
                (name.to_u32() as u64).write_num(&mut self.data);
            }
            AtomTree::Fn(name, args) => {
                self.data.put_u8(FN_ID);
                let size_pos = self.data.len();
                self.data.put_u32_le(0 as u32); // length of entire fn without flag
                let buf_pos = self.data.len();

                // pack name and args
                (name.to_u32() as u64).write_frac(args.len() as u64, &mut self.data);

                for a in args {
                    self.linearize(a);
                }
                let new_buf_pos = self.data.len();

                let mut cursor: Cursor<&mut [u8]> = Cursor::new(&mut self.data[size_pos..]);

                cursor
                    .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
                    .unwrap();
            }
            AtomTree::Num(n) => {
                self.data.put_u8(NUM_ID);
                n.num.write_frac(n.den, &mut self.data);
            }
            AtomTree::Pow(p) => {
                self.data.put_u8(POW_ID);
                self.linearize(&p.0);
                self.linearize(&p.1);
            }
            AtomTree::Mul(args) | AtomTree::Add(args) => {
                if let AtomTree::Mul(_) = atom {
                    self.data.put_u8(MUL_ID);
                } else {
                    self.data.put_u8(ADD_ID);
                }

                let size_pos = self.data.len();
                self.data.put_u32_le(0 as u32); // length of entire fn without flag
                let buf_pos = self.data.len();

                (args.len() as u64).write_num(&mut self.data);

                for a in args {
                    self.linearize(a);
                }
                let new_buf_pos = self.data.len();

                let mut cursor: Cursor<&mut [u8]> = Cursor::new(&mut self.data[size_pos..]);

                cursor
                    .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
                    .unwrap();
            }
        }
    }

    fn write_to_tree(mut source: &[u8]) -> (AtomTree, &[u8]) {
        let d = source.get_u8() & TYPE_MASK;
        match d {
            VAR_ID => {
                let name;
                (name, _, source) = source.get_frac_u64();
                (AtomTree::Var(Identifier::from(name as u32)), source)
            }
            FN_ID => {
                source.get_u32_le(); // size

                let (name, n_args);
                (name, n_args, source) = source.get_frac_u64();

                let mut args = Vec::with_capacity(n_args as usize);
                for _ in 0..n_args {
                    let (a, s) = OwnedAtomD::write_to_tree(source);
                    source = s;
                    args.push(a);
                }

                (AtomTree::Fn(Identifier::from(name as u32), args), source)
            }
            NUM_ID => {
                let (num, den);
                (num, den, source) = source.get_frac_u64();
                (AtomTree::Num(Number::new(num, den)), source)
            }
            POW_ID => {
                let (base, exp);
                (base, source) = OwnedAtomD::write_to_tree(source);
                (exp, source) = OwnedAtomD::write_to_tree(source);
                (AtomTree::Pow(Box::new((base, exp))), source)
            }
            MUL_ID | ADD_ID => {
                source.get_u32_le(); // size

                let n_args;
                (n_args, _, source) = source.get_frac_u64();

                let mut args = Vec::with_capacity(n_args as usize);
                for _ in 0..n_args {
                    let (a, s) = OwnedAtomD::write_to_tree(source);
                    source = s;
                    args.push(a);
                }

                if d == MUL_ID {
                    (AtomTree::Mul(args), source)
                } else {
                    (AtomTree::Add(args), source)
                }
            }
            x => unreachable!("Bad id: {}", x),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VarViewD<'a> {
    pub data: &'a [u8],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionViewD<'a> {
    pub data: &'a [u8],
}

impl<'a> Fn<'a> for FunctionViewD<'a> {
    type P = DefaultRepresentation;
    type I = ListIterator<'a>;

    fn get_name(&self) -> Identifier {
        Identifier::from((&self.data[1 + 4..]).get_frac_u64().0 as u32)
    }

    fn get_nargs(&self) -> usize {
        (&self.data[1 + 4..]).get_frac_u64().1 as usize
    }

    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    fn cmp(&self, other: &Self) -> Ordering {
        self.get_name().cmp(&other.get_name())
    }

    #[inline]
    fn into_iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_u64(); // name

        ListIterator {
            data: c,
            length: n_args as u32,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NumberViewD<'a> {
    pub data: &'a [u8],
}

impl<'a> Num<'a> for NumberViewD<'a> {
    type P = DefaultRepresentation;

    fn is_one(&self) -> bool {
        self.data.is_one()
    }

    fn add<'b>(&self, other: &Self, out: &mut OwnedAtomD) {
        let a = self.get_numden();
        let b = other.get_numden();

        let c = (a.0 * b.0, a.1 * b.1);
        let gcd = utils::gcd_unsigned(c.0 as u64, c.1 as u64);

        out.data.put_u8(NUM_ID);

        (c.0 as u64 / gcd).write_frac(c.1 as u64 / gcd, &mut out.data);
    }

    #[inline]
    fn get_numden(&self) -> (u64, u64) {
        let mut c = self.data;
        c.get_u8();

        let num;
        let den;
        (num, den, _) = c.get_frac_u64();

        (num, den)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PowViewD<'a> {
    pub data: &'a [u8],
}

impl<'a> Pow<'a> for PowViewD<'a> {
    type P = DefaultRepresentation;

    #[inline]
    fn get_base(&self) -> AtomView<Self::P> {
        let (b, _) = self.get_base_exp();
        b
    }

    #[inline]
    fn get_exp(&self) -> AtomView<Self::P> {
        let (_, e) = self.get_base_exp();
        e
    }

    #[inline]
    fn get_base_exp(&self) -> (AtomView<Self::P>, AtomView<Self::P>) {
        let mut it = ListIterator {
            data: &self.data[1..],
            length: 2,
        };

        (it.next().unwrap(), it.next().unwrap())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MulViewD<'a> {
    pub data: &'a [u8],
}

impl<'a> Mul<'a> for MulViewD<'a> {
    type P = DefaultRepresentation;
    type I = ListIterator<'a>;

    #[inline]
    fn into_iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListIterator {
            data: c,
            length: n_args as u32,
        }
    }

    fn get_nargs(&self) -> usize {
        (&self.data[1 + 4..]).get_frac_u64().0 as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddViewD<'a> {
    pub data: &'a [u8],
}

impl<'a> Add<'a> for AddViewD<'a> {
    type P = DefaultRepresentation;
    type I = ListIterator<'a>;

    #[inline]
    fn into_iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListIterator {
            data: c,
            length: n_args as u32,
        }
    }

    fn get_nargs(&self) -> usize {
        (&self.data[1 + 4..]).get_frac_u64().0 as usize
    }
}

impl<'a> AtomView<'a, DefaultRepresentation> {
    pub fn from(source: &'a [u8]) -> AtomView<'a, DefaultRepresentation> {
        match source[0] {
            VAR_ID => AtomView::Var(VarViewD { data: source }),
            FN_ID => AtomView::Fn(FunctionViewD { data: source }),
            NUM_ID => AtomView::Num(NumberViewD { data: source }),
            POW_ID => AtomView::Pow(PowViewD { data: source }),
            MUL_ID => AtomView::Mul(MulViewD { data: source }),
            ADD_ID => AtomView::Add(AddViewD { data: source }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    pub fn to_atom(&self) -> AtomTree {
        OwnedAtomD::write_to_tree(self.get_data()).0
    }

    pub fn get_data(&self) -> &[u8] {
        match self {
            AtomView::Num(n) => n.data,
            AtomView::Var(v) => v.data,
            AtomView::Fn(f) => f.data,
            AtomView::Pow(p) => p.data,
            AtomView::Mul(t) => t.data,
            AtomView::Add(e) => e.data,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ListIterator<'a> {
    data: &'a [u8],
    length: u32,
}

impl<'a> ListIteratorT<'a> for ListIterator<'a> {
    type P = DefaultRepresentation;

    #[inline(always)]
    fn next(&mut self) -> Option<AtomView<'a, Self::P>> {
        if self.length == 0 {
            return None;
        }

        self.length -= 1;

        let start = self.data;

        let start_id = self.data.get_u8() & TYPE_MASK;
        let mut cur_id = start_id;

        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = 1;
        loop {
            match cur_id {
                VAR_ID => {
                    self.data = self.data.skip_rational();
                }
                NUM_ID => {
                    self.data = self.data.skip_rational();
                }
                FN_ID => {
                    let n_size = self.data.get_u32_le();
                    self.data.advance(n_size as usize);
                }
                POW_ID => {
                    skip_count += 2;
                }
                MUL_ID | ADD_ID => {
                    let n_size = self.data.get_u32_le();
                    self.data.advance(n_size as usize);
                }
                x => unreachable!("Bad id {}", x),
            }

            skip_count -= 1;

            if skip_count == 0 {
                break;
            } else {
                cur_id = self.data.get_u8() & TYPE_MASK;
            }
        }

        let len = unsafe { self.data.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        match start_id {
            VAR_ID => {
                return Some(AtomView::Var(VarViewD { data }));
            }
            NUM_ID => {
                return Some(AtomView::Num(NumberViewD { data }));
            }
            FN_ID => {
                return Some(AtomView::Fn(FunctionViewD { data }));
            }
            POW_ID => {
                return Some(AtomView::Pow(PowViewD { data }));
            }
            MUL_ID => {
                return Some(AtomView::Mul(MulViewD { data }));
            }
            ADD_ID => {
                return Some(AtomView::Add(AddViewD { data }));
            }
            x => unreachable!("Bad id {}", x),
        }
    }
}

#[test]
pub fn representation_size() {
    let a = AtomTree::Fn(
        Identifier::from(1),
        vec![
            AtomTree::Var(Identifier::from(2)),
            AtomTree::Fn(
                Identifier::from(3),
                vec![
                    AtomTree::Mul(vec![
                        AtomTree::Num(Number::new(3, 1)),
                        AtomTree::Num(Number::new(13, 1)),
                    ]),
                    AtomTree::Add(vec![
                        AtomTree::Num(Number::new(3, 1)),
                        AtomTree::Num(Number::new(13, 1)),
                    ]),
                    AtomTree::Mul(vec![
                        AtomTree::Num(Number::new(3, 1)),
                        AtomTree::Num(Number::new(13, 1)),
                    ]),
                    AtomTree::Mul(vec![
                        AtomTree::Num(Number::new(3, 1)),
                        AtomTree::Num(Number::new(13, 1)),
                    ]),
                    AtomTree::Num(Number::new(4, 2)),
                    AtomTree::Num(Number::new(4, 2)),
                    AtomTree::Num(Number::new(4, 2)),
                    AtomTree::Num(Number::new(4, 2)),
                ],
            ),
            AtomTree::Var(Identifier::from(6)),
            AtomTree::Num(Number::new(2, 1)),
            AtomTree::Pow(Box::new((
                AtomTree::Add(vec![
                    AtomTree::Num(Number::new(3, 1)),
                    AtomTree::Num(Number::new(13, 1)),
                ]),
                AtomTree::Var(Identifier::from(2)),
            ))),
        ],
    );
    println!("expr={:?}", a);

    let b = OwnedAtomD::from_tree(&a);

    println!("lin size: {:?} bytes", b.data.len());

    let c = b.to_tree();

    if a != c {
        panic!("in and out is different: {:?} vs {:?}", a, c);
    }

    b.to_view().dbg_print_tree(0);
}
