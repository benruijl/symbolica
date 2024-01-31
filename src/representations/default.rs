use byteorder::{LittleEndian, WriteBytesExt};
use bytes::{Buf, BufMut};
use std::cmp::Ordering;

use crate::state::{ResettableBuffer, State};

use super::{
    number::{BorrowedNumber, Number, PackedRationalNumberReader, PackedRationalNumberWriter},
    Add, Atom, AtomSet, AtomView, Convert, Fun, Identifier, ListSlice, Mul, Num, OwnedAdd,
    OwnedFun, OwnedMul, OwnedNum, OwnedPow, OwnedVar, Pow, SliceType, Var,
};

const NUM_ID: u8 = 1;
const VAR_ID: u8 = 2;
const FUN_ID: u8 = 3;
const MUL_ID: u8 = 4;
const POW_ID: u8 = 5;
const ADD_ID: u8 = 6;
const TYPE_MASK: u8 = 0b00000111;
const DIRTY_FLAG: u8 = 0b10000000;
const HAS_COEFF_FLAG: u8 = 0b01000000;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Linear {}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct OwnedNumD {
    data: Vec<u8>,
}

impl OwnedNum for OwnedNumD {
    type P = Linear;

    fn set_from_number(&mut self, num: Number) {
        self.data.clear();
        self.data.put_u8(NUM_ID);
        num.write_packed(&mut self.data);
    }

    fn set_from_view(&mut self, a: &NumViewD<'_>) {
        self.data.clear();
        self.data.extend(a.data);
    }

    fn add(&mut self, other: &NumViewD<'_>, state: &State) {
        let nv = self.to_num_view();
        let a = nv.get_number_view();
        let b = other.get_number_view();
        let n = a.add(&b, state);

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    fn mul(&mut self, other: &NumViewD<'_>, state: &State) {
        let nv = self.to_num_view();
        let a = nv.get_number_view();
        let b = other.get_number_view();
        let n = a.mul(&b, state);

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    fn to_num_view(&self) -> NumViewD {
        assert!(self.data[0] & TYPE_MASK == NUM_ID);
        NumViewD { data: &self.data }
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Num(self.to_num_view())
    }
}

impl Convert<Linear> for OwnedNumD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedNumD {
    #[inline(always)]
    fn new() -> Self {
        OwnedNumD { data: vec![] }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct OwnedVarD {
    data: Vec<u8>,
}

impl OwnedVar for OwnedVarD {
    type P = Linear;

    fn set_from_id(&mut self, id: Identifier) {
        self.data.clear();
        self.data.put_u8(VAR_ID);
        (id.to_u32() as u64, 1).write_packed(&mut self.data);
    }

    fn to_var_view(&self) -> <Self::P as AtomSet>::V<'_> {
        VarViewD { data: &self.data }
    }

    fn set_from_view<'a>(&mut self, view: &VarViewD) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Var(self.to_var_view())
    }
}

impl Convert<Linear> for OwnedVarD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedVarD {
    #[inline(always)]
    fn new() -> Self {
        OwnedVarD { data: vec![] }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct OwnedFunD {
    data: Vec<u8>,
}

impl OwnedFun for OwnedFunD {
    type P = Linear;

    fn set_from_name(&mut self, id: Identifier) {
        self.data.clear();
        self.data.put_u8(FUN_ID);
        self.data.put_u32_le(0_u32);

        let buf_pos = self.data.len();

        (id.to_u32() as u64, 0).write_packed(&mut self.data);

        let new_buf_pos = self.data.len();
        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] |= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    fn add_arg(&mut self, other: AtomView<Self::P>) {
        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let name;
        let mut n_args;
        (name, n_args, c) = c.get_frac_u64();

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        n_args += 1;

        let new_size = (name, n_args).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                self.data.resize(self.data.len() - old_size + new_size, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (name, n_args).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        self.data.extend(other.get_data());

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    #[inline(always)]
    fn to_fun_view(&self) -> <Self::P as AtomSet>::F<'_> {
        FnViewD { data: &self.data }
    }

    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::F<'_>) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Fun(self.to_fun_view())
    }
}

impl Convert<Linear> for OwnedFunD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedFunD {
    #[inline(always)]
    fn new() -> Self {
        OwnedFunD { data: vec![] }
    }
    #[inline(always)]

    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct OwnedPowD {
    data: Vec<u8>,
}

impl OwnedPow for OwnedPowD {
    type P = Linear;

    fn set_from_base_and_exp(&mut self, base: AtomView<Self::P>, exp: AtomView<Self::P>) {
        self.data.clear();
        self.data.put_u8(POW_ID);
        self.data.extend(base.get_data());
        self.data.extend(exp.get_data());
    }

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] |= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    #[inline(always)]
    fn to_pow_view(&self) -> <Self::P as AtomSet>::P<'_> {
        PowViewD { data: &self.data }
    }

    #[inline(always)]
    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::P<'_>) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Pow(self.to_pow_view())
    }
}

impl Convert<Linear> for OwnedPowD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedPowD {
    #[inline(always)]
    fn new() -> Self {
        OwnedPowD { data: vec![] }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Clone, PartialEq, Hash)]
pub struct OwnedMulD {
    data: Vec<u8>,
}

impl OwnedMul for OwnedMulD {
    type P = Linear;

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] |= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::M<'_>) {
        self.data.clear();
        self.data.extend(view.data);
    }

    fn extend(&mut self, other: AtomView<'_, Linear>) {
        if self.data.is_empty() {
            self.data.put_u8(MUL_ID);
            self.data.put_u32_le(0_u32);
            (0u64, 1).write_packed(&mut self.data);
        }

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64(); // TODO: pack size and n_args

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        let new_slice = match other {
            AtomView::Mul(m) => m.to_slice(),
            _ => ListSliceD::from_one(other),
        };

        n_args += new_slice.len() as u64;

        let new_size = (n_args, 1).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                self.data.resize(self.data.len() - old_size + new_size, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (n_args, 1).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        for child in new_slice.iter() {
            self.data.extend_from_slice(child.get_data());
        }

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    fn replace_last(&mut self, other: AtomView<Self::P>) {
        if self.data.is_empty() {
            panic!("Cannot pop empty mul");
        }

        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let n_args;
        (n_args, _, c) = c.get_frac_u64(); // TODO: pack size and n_args

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        let new_size = (n_args, 1).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                self.data.resize(self.data.len() - old_size + new_size, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (n_args, 1).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        // remove the last entry
        let s = self.to_mul_view().to_slice();
        let last_index = s.get(s.len() - 1);
        let start_pointer_of_last = last_index.get_data().as_ptr();
        let dist = unsafe { start_pointer_of_last.offset_from(self.data.as_ptr()) } as usize;
        self.data.drain(dist..);
        self.data.extend_from_slice(other.get_data());

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    fn to_mul_view(&self) -> <Self::P as AtomSet>::M<'_> {
        MulViewD { data: &self.data }
    }

    fn set_has_coefficient(&mut self, has_coeff: bool) {
        if has_coeff {
            self.data[0] |= HAS_COEFF_FLAG;
        } else {
            self.data[0] &= !HAS_COEFF_FLAG;
        }
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Mul(self.to_mul_view())
    }
}

impl Convert<Linear> for OwnedMulD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedMulD {
    #[inline(always)]
    fn new() -> Self {
        OwnedMulD { data: vec![] }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.data.clear();
    }
}

#[derive(Clone, PartialEq, Hash)]
pub struct OwnedAddD {
    data: Vec<u8>,
}

impl OwnedAdd for OwnedAddD {
    type P = Linear;

    fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.data[0] |= DIRTY_FLAG;
        } else {
            self.data[0] &= !DIRTY_FLAG;
        }
    }

    fn extend(&mut self, other: AtomView<'_, Linear>) {
        if self.data.is_empty() {
            self.data.put_u8(ADD_ID);
            self.data.put_u32_le(0_u32);
            (0u64, 1).write_packed(&mut self.data);
        }

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64();

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        let new_slice = match other {
            AtomView::Add(m) => m.to_slice(),
            _ => ListSliceD::from_one(other),
        };

        n_args += new_slice.len() as u64;

        let new_size = (n_args, 1).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                self.data.resize(self.data.len() - old_size + new_size, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (n_args, 1).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        for child in new_slice.iter() {
            self.data.extend_from_slice(child.get_data());
        }

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];

        assert!(new_buf_pos - buf_pos < u32::MAX as usize, "Term too large");

        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    #[inline(always)]
    fn to_add_view(&self) -> <Self::P as AtomSet>::A<'_> {
        AddViewD { data: &self.data }
    }

    #[inline(always)]
    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::A<'_>) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    fn as_view(&self) -> AtomView<Self::P> {
        AtomView::Add(self.to_add_view())
    }
}

impl Convert<Linear> for OwnedAddD {
    #[inline(always)]
    fn to_owned_var(mut self) -> OwnedVarD {
        self.data.clear();
        OwnedVarD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_pow(mut self) -> OwnedPowD {
        self.data.clear();
        OwnedPowD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_num(mut self) -> OwnedNumD {
        self.data.clear();
        OwnedNumD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_fun(mut self) -> OwnedFunD {
        self.data.clear();
        OwnedFunD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_add(mut self) -> OwnedAddD {
        self.data.clear();
        OwnedAddD { data: self.data }
    }

    #[inline(always)]
    fn to_owned_mul(mut self) -> OwnedMulD {
        self.data.clear();
        OwnedMulD { data: self.data }
    }
}

impl ResettableBuffer for OwnedAddD {
    fn new() -> Self {
        let mut data = Vec::new();
        data.put_u8(ADD_ID);
        data.put_u32_le(0_u32);
        (0u64, 1).write_packed(&mut data);

        OwnedAddD { data }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.data.put_u8(ADD_ID);
        self.data.put_u32_le(0_u32);
        (0u64, 1).write_packed(&mut self.data);
    }
}

impl AtomSet for Linear {
    type N<'a> = NumViewD<'a>;
    type V<'a> = VarViewD<'a>;
    type F<'a> = FnViewD<'a>;
    type P<'a> = PowViewD<'a>;
    type M<'a> = MulViewD<'a>;
    type A<'a> = AddViewD<'a>;
    type ON = OwnedNumD;
    type OV = OwnedVarD;
    type OF = OwnedFunD;
    type OP = OwnedPowD;
    type OM = OwnedMulD;
    type OA = OwnedAddD;
    type S<'a> = ListSliceD<'a>;
}

impl<'a> Var<'a> for VarViewD<'a> {
    type P = Linear;

    #[inline(always)]
    fn get_name(&self) -> Identifier {
        Identifier::from(self.data[1..].get_frac_i64().0 as u32)
    }

    #[inline]
    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Var(*self)
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

impl Atom<Linear> {
    pub fn get_data(&self) -> &[u8] {
        match self {
            Atom::Num(n) => &n.data,
            Atom::Var(v) => &v.data,
            Atom::Fun(f) => &f.data,
            Atom::Pow(p) => &p.data,
            Atom::Mul(m) => &m.data,
            Atom::Add(a) => &a.data,
            Atom::Empty => unreachable!(),
        }
    }

    /// Get the total byte length of the atom and its children.
    pub fn len(&self) -> usize {
        self.get_data().len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct VarViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<VarViewD<'b>> for VarViewD<'a> {
    fn eq(&self, other: &VarViewD<'b>) -> bool {
        self.data == other.data
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct FnViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<FnViewD<'b>> for FnViewD<'a> {
    fn eq(&self, other: &FnViewD<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> Fun<'a> for FnViewD<'a> {
    type P = Linear;
    type I = ListIteratorD<'a>;

    #[inline(always)]
    fn get_name(&self) -> Identifier {
        Identifier::from(self.data[1 + 4..].get_frac_i64().0 as u32)
    }

    #[inline(always)]
    fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_i64().1 as usize
    }

    #[inline(always)]
    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    #[inline]
    fn iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_i64(); // name

        ListIteratorD {
            data: c,
            length: n_args as u32,
        }
    }

    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Fun(*self)
    }

    fn to_slice(&self) -> ListSliceD<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_i64(); // name

        ListSliceD {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Arg,
        }
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }

    fn fast_cmp(&self, other: <Self::P as AtomSet>::F<'_>) -> Ordering {
        self.data.cmp(other.data)
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct NumViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<NumViewD<'b>> for NumViewD<'a> {
    #[inline]
    fn eq(&self, other: &NumViewD<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> Num<'a> for NumViewD<'a> {
    type P = Linear;

    #[inline]
    fn is_zero(&self) -> bool {
        self.data.is_zero_rat()
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.data.is_one_rat()
    }

    #[inline]
    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    #[inline]
    fn get_number_view(&self) -> BorrowedNumber<'_> {
        self.data[1..].get_number_view().0
    }

    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Num(*self)
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct PowViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<PowViewD<'b>> for PowViewD<'a> {
    #[inline]
    fn eq(&self, other: &PowViewD<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> Pow<'a> for PowViewD<'a> {
    type P = Linear;

    #[inline]
    fn get_base(&self) -> AtomView<'a, Self::P> {
        let (b, _) = self.get_base_exp();
        b
    }

    #[inline]
    fn get_exp(&self) -> AtomView<'a, Self::P> {
        let (_, e) = self.get_base_exp();
        e
    }

    #[inline]
    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    #[inline]
    fn get_base_exp(&self) -> (AtomView<'a, Self::P>, AtomView<'a, Self::P>) {
        let mut it = ListIteratorD {
            data: &self.data[1..],
            length: 2,
        };

        (it.next().unwrap(), it.next().unwrap())
    }

    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Pow(*self)
    }

    fn to_slice(&self) -> ListSliceD<'a> {
        ListSliceD {
            data: &self.data[1..],
            length: 2,
            slice_type: SliceType::Pow,
        }
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct MulViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<MulViewD<'b>> for MulViewD<'a> {
    #[inline]
    fn eq(&self, other: &MulViewD<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> Mul<'a> for MulViewD<'a> {
    type P = Linear;
    type I = ListIteratorD<'a>;

    #[inline]
    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_i64().0 as usize
    }

    #[inline]
    fn iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_i64();

        ListIteratorD {
            data: c,
            length: n_args as u32,
        }
    }

    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Mul(*self)
    }

    fn to_slice(&self) -> ListSliceD<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_i64();

        ListSliceD {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Mul,
        }
    }

    #[inline]
    fn has_coefficient(&self) -> bool {
        (self.data[0] & HAS_COEFF_FLAG) != 0
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct AddViewD<'a> {
    pub data: &'a [u8],
}

impl<'a, 'b> PartialEq<AddViewD<'b>> for AddViewD<'a> {
    #[inline]
    fn eq(&self, other: &AddViewD<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> Add<'a> for AddViewD<'a> {
    type P = Linear;
    type I = ListIteratorD<'a>;

    #[inline(always)]
    fn is_dirty(&self) -> bool {
        (self.data[0] & DIRTY_FLAG) != 0
    }

    #[inline(always)]
    fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_i64().0 as usize
    }

    #[inline]
    fn iter(&self) -> Self::I {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_i64();

        ListIteratorD {
            data: c,
            length: n_args as u32,
        }
    }

    #[inline]
    fn as_view(&self) -> AtomView<'a, Self::P> {
        AtomView::Add(*self)
    }

    fn to_slice(&self) -> ListSliceD<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_i64();

        ListSliceD {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Add,
        }
    }

    fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AtomView<'a, Linear> {
    pub fn from(source: &'a [u8]) -> AtomView<'a, Linear> {
        match source[0] {
            VAR_ID => AtomView::Var(VarViewD { data: source }),
            FUN_ID => AtomView::Fun(FnViewD { data: source }),
            NUM_ID => AtomView::Num(NumViewD { data: source }),
            POW_ID => AtomView::Pow(PowViewD { data: source }),
            MUL_ID => AtomView::Mul(MulViewD { data: source }),
            ADD_ID => AtomView::Add(AddViewD { data: source }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    pub fn get_data(&self) -> &'a [u8] {
        match self {
            AtomView::Num(n) => n.data,
            AtomView::Var(v) => v.data,
            AtomView::Fun(f) => f.data,
            AtomView::Pow(p) => p.data,
            AtomView::Mul(t) => t.data,
            AtomView::Add(e) => e.data,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ListIteratorD<'a> {
    data: &'a [u8],
    length: u32,
}

impl<'a> Iterator for ListIteratorD<'a> {
    type Item = AtomView<'a, Linear>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
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
                FUN_ID => {
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
                return Some(AtomView::Num(NumViewD { data }));
            }
            FUN_ID => {
                return Some(AtomView::Fun(FnViewD { data }));
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListSliceD<'a> {
    data: &'a [u8],
    length: usize,
    slice_type: SliceType,
}

impl<'a> ListSliceD<'a> {
    #[inline(always)]
    fn skip_one(mut pos: &[u8]) -> &[u8] {
        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = 1u32;
        while skip_count > 0 {
            skip_count -= 1;

            match pos.get_u8() & TYPE_MASK {
                VAR_ID => {
                    pos = pos.skip_rational();
                }
                NUM_ID => {
                    pos = pos.skip_rational();
                }
                FUN_ID => {
                    let n_size = pos.get_u32_le();
                    pos.advance(n_size as usize);
                }
                POW_ID => {
                    skip_count += 2;
                }
                MUL_ID | ADD_ID => {
                    let n_size = pos.get_u32_le();
                    pos.advance(n_size as usize);
                }
                x => unreachable!("Bad id {}", x),
            }
        }
        pos
    }

    fn fast_forward(&self, index: usize) -> ListSliceD<'a> {
        let mut pos = self.data;

        for _ in 0..index {
            pos = Self::skip_one(pos);
        }

        ListSliceD {
            data: pos,
            length: self.length - index,
            slice_type: self.slice_type,
        }
    }

    fn get_entry(start: &'a [u8]) -> (AtomView<'a, Linear>, &[u8]) {
        let start_id = start[0] & TYPE_MASK;
        let end = Self::skip_one(start);
        let len = unsafe { end.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        (
            match start_id {
                VAR_ID => AtomView::Var(VarViewD { data }),
                NUM_ID => AtomView::Num(NumViewD { data }),
                FUN_ID => AtomView::Fun(FnViewD { data }),
                POW_ID => AtomView::Pow(PowViewD { data }),
                MUL_ID => AtomView::Mul(MulViewD { data }),
                ADD_ID => AtomView::Add(AddViewD { data }),
                x => unreachable!("Bad id {}", x),
            },
            end,
        )
    }
}

impl<'a> ListSlice<'a> for ListSliceD<'a> {
    type P = Linear;
    type ListSliceIterator = ListSliceIteratorD<'a>;

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    #[inline]
    fn get(&self, index: usize) -> AtomView<'a, Self::P> {
        let start = self.fast_forward(index);
        Self::get_entry(start.data).0
    }

    fn get_subslice(&self, range: std::ops::Range<usize>) -> Self {
        let start = self.fast_forward(range.start);

        let mut s = start.data;
        for _ in 0..range.len() {
            s = Self::skip_one(s);
        }

        let len = unsafe { s.as_ptr().offset_from(start.data.as_ptr()) } as usize;
        ListSliceD {
            data: &start.data[..len],
            length: range.len(),
            slice_type: self.slice_type,
        }
    }

    #[inline]
    fn eq(&self, other: &ListSliceD<'_>) -> bool {
        self.data == other.data
    }

    #[inline]
    fn get_type(&self) -> SliceType {
        self.slice_type
    }

    #[inline]
    fn from_one(view: AtomView<'a, Self::P>) -> Self {
        ListSliceD {
            data: view.get_data(),
            length: 1,
            slice_type: SliceType::One,
        }
    }

    #[inline]
    fn iter(&self) -> Self::ListSliceIterator {
        ListSliceIteratorD { data: *self }
    }
}

pub struct ListSliceIteratorD<'a> {
    data: ListSliceD<'a>,
}

impl<'a> Iterator for ListSliceIteratorD<'a> {
    type Item = AtomView<'a, Linear>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.length > 0 {
            let (res, end) = ListSliceD::get_entry(self.data.data);
            self.data = ListSliceD {
                data: end,
                length: self.data.length - 1,
                slice_type: self.data.slice_type,
            };

            Some(res)
        } else {
            None
        }
    }
}
