use byteorder::{LittleEndian, WriteBytesExt};
use bytes::{Buf, BufMut};
use std::{
    cmp::Ordering,
    io::{Read, Write},
};

use crate::{
    coefficient::{Coefficient, CoefficientView},
    state::{StateMap, Workspace},
};

use super::{
    coefficient::{PackedRationalNumberReader, PackedRationalNumberWriter},
    Atom, AtomView, SliceType, Symbol,
};

const NUM_ID: u8 = 1;
const VAR_ID: u8 = 2;
const FUN_ID: u8 = 3;
const MUL_ID: u8 = 4;
const ADD_ID: u8 = 5;
const POW_ID: u8 = 6;
const TYPE_MASK: u8 = 0b00000111;
const NOT_NORMALIZED: u8 = 0b10000000;
const VAR_WILDCARD_LEVEL_MASK: u8 = 0b00011000;
const VAR_WILDCARD_LEVEL_1: u8 = 0b00001000;
const VAR_WILDCARD_LEVEL_2: u8 = 0b00010000;
const VAR_WILDCARD_LEVEL_3: u8 = 0b00011000;
const FUN_SYMMETRIC_FLAG: u8 = 0b00100000;
const FUN_LINEAR_FLAG: u8 = 0b01000000;
const VAR_ANTISYMMETRIC_FLAG: u8 = 0b10000000;
const FUN_ANTISYMMETRIC_FLAG: u64 = 1 << 32; // stored in the function id
const MUL_HAS_COEFF_FLAG: u8 = 0b01000000;

const ZERO_DATA: [u8; 3] = [NUM_ID, 1, 0];

pub type RawAtom = Vec<u8>;

impl Atom {
    /// Read from a binary stream. The format is the byte-length first
    /// followed by the data.
    pub(crate) fn read<R: Read>(&mut self, mut source: R) -> Result<(), std::io::Error> {
        let mut dest = std::mem::replace(self, Atom::Zero).into_raw();

        // should also set whether rat poly coefficient needs to be converted
        let mut flags_buf = [0; 1];
        let mut size_buf = [0; 8];

        source.read_exact(&mut flags_buf)?;
        source.read_exact(&mut size_buf)?;

        let n_size = u64::from_le_bytes(size_buf);

        dest.extend(size_buf);
        dest.resize(n_size as usize, 0);
        source.read_exact(&mut dest)?;

        unsafe {
            match dest[0] & TYPE_MASK {
                NUM_ID => *self = Atom::Num(Num::from_raw(dest)),
                VAR_ID => *self = Atom::Var(Var::from_raw(dest)),
                FUN_ID => *self = Atom::Fun(Fun::from_raw(dest)),
                MUL_ID => *self = Atom::Mul(Mul::from_raw(dest)),
                ADD_ID => *self = Atom::Add(Add::from_raw(dest)),
                POW_ID => *self = Atom::Pow(Pow::from_raw(dest)),
                _ => unreachable!("Unknown type {}", dest[0]),
            }
        }

        Ok(())
    }

    pub fn import<R: Read>(source: R, state_map: &StateMap) -> Result<Atom, std::io::Error> {
        let mut a = Atom::new();
        a.read(source)?;
        Ok(a.as_view().rename(state_map))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Num {
    data: RawAtom,
}

impl Num {
    #[inline(always)]
    pub fn zero(mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.put_u8(NUM_ID);
        buffer.put_u8(1);
        buffer.put_u8(0);
        Num { data: buffer }
    }

    #[inline]
    pub fn new(num: Coefficient) -> Num {
        let mut buffer = Vec::new();
        buffer.put_u8(NUM_ID);
        num.write_packed(&mut buffer);
        Num { data: buffer }
    }

    #[inline(always)]
    pub fn new_into(num: Coefficient, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.put_u8(NUM_ID);
        num.write_packed(&mut buffer);
        Num { data: buffer }
    }

    #[inline]
    pub fn from_view_into(a: &NumView<'_>, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(a.data);
        Num { data: buffer }
    }

    #[inline]
    pub fn set_from_coeff(&mut self, num: Coefficient) {
        self.data.clear();
        self.data.put_u8(NUM_ID);
        num.write_packed(&mut self.data);
    }

    #[inline]
    pub fn set_from_view(&mut self, a: &NumView<'_>) {
        self.data.clear();
        self.data.extend(a.data);
    }

    pub fn add(&mut self, other: &NumView<'_>) {
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a + b;

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    pub fn mul(&mut self, other: &NumView<'_>) {
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a * b;

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    #[inline]
    pub fn to_num_view(&self) -> NumView {
        NumView { data: &self.data }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Num(self.to_num_view())
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Num {
        Num { data: raw }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Var {
    data: RawAtom,
}

impl Var {
    #[inline]
    pub fn new(symbol: Symbol) -> Var {
        Self::new_into(symbol, RawAtom::new())
    }

    #[inline]
    pub fn new_into(symbol: Symbol, buffer: RawAtom) -> Var {
        let mut f = Var { data: buffer };
        f.set_from_symbol(symbol);
        f
    }

    #[inline]
    pub fn from_view_into(a: &VarView<'_>, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(a.data);
        Var { data: buffer }
    }

    #[inline]
    pub fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();

        let mut flags = VAR_ID;
        match symbol.wildcard_level {
            0 => {}
            1 => flags |= VAR_WILDCARD_LEVEL_1,
            2 => flags |= VAR_WILDCARD_LEVEL_2,
            _ => flags |= VAR_WILDCARD_LEVEL_3,
        }

        if symbol.is_symmetric {
            flags |= FUN_SYMMETRIC_FLAG;
        }
        if symbol.is_linear {
            flags |= FUN_LINEAR_FLAG;
        }
        if symbol.is_antisymmetric {
            flags |= VAR_ANTISYMMETRIC_FLAG;
        }

        self.data.put_u8(flags);

        (symbol.id as u64, 1).write_packed(&mut self.data);
    }

    #[inline]
    pub fn to_var_view(&self) -> VarView {
        VarView { data: &self.data }
    }

    #[inline]
    pub fn set_from_view(&mut self, view: &VarView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Var(self.to_var_view())
    }

    #[inline]
    pub fn get_symbol(&self) -> Symbol {
        self.to_var_view().get_symbol()
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Var {
        Var { data: raw }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fun {
    data: RawAtom,
}

impl Fun {
    #[inline]
    pub(crate) fn new_into(id: Symbol, buffer: RawAtom) -> Fun {
        let mut f = Fun { data: buffer };
        f.set_from_symbol(id);
        f
    }

    #[inline]
    pub fn from_view_into(a: &FunView<'_>, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(a.data);
        Fun { data: buffer }
    }

    #[inline]
    pub(crate) fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();

        let mut flags = FUN_ID | NOT_NORMALIZED;
        match symbol.wildcard_level {
            0 => {}
            1 => flags |= VAR_WILDCARD_LEVEL_1,
            2 => flags |= VAR_WILDCARD_LEVEL_2,
            _ => flags |= VAR_WILDCARD_LEVEL_3,
        }

        if symbol.is_symmetric {
            flags |= FUN_SYMMETRIC_FLAG;
        }
        if symbol.is_linear {
            flags |= FUN_LINEAR_FLAG;
        }

        self.data.put_u8(flags);

        self.data.put_u32_le(0_u32);

        let buf_pos = self.data.len();

        let id = if symbol.is_antisymmetric {
            symbol.id as u64 | FUN_ANTISYMMETRIC_FLAG
        } else {
            symbol.id as u64
        };

        (id, 0).write_packed(&mut self.data);

        let new_buf_pos = self.data.len();
        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    pub(crate) fn add_arg(&mut self, other: AtomView) {
        self.data[0] |= NOT_NORMALIZED;

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
    pub fn to_fun_view(&self) -> FunView {
        FunView { data: &self.data }
    }

    pub fn set_from_view(&mut self, view: &FunView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Fun(self.to_fun_view())
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        self.to_fun_view().get_symbol()
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_fun_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Fun {
        Fun { data: raw }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pow {
    data: RawAtom,
}

impl Pow {
    #[inline]
    pub(crate) fn new_into(base: AtomView, exp: AtomView, buffer: RawAtom) -> Pow {
        let mut f = Pow { data: buffer };
        f.set_from_base_and_exp(base, exp);
        f
    }

    #[inline]
    pub fn from_view_into(a: &PowView<'_>, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(a.data);
        Pow { data: buffer }
    }

    #[inline]
    pub(crate) fn set_from_base_and_exp(&mut self, base: AtomView, exp: AtomView) {
        self.data.clear();
        self.data.put_u8(POW_ID | NOT_NORMALIZED);
        self.data.extend(base.get_data());
        self.data.extend(exp.get_data());
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline(always)]
    pub fn to_pow_view(&self) -> PowView {
        PowView { data: &self.data }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &PowView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Pow(self.to_pow_view())
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Pow {
        Pow { data: raw }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Mul {
    data: RawAtom,
}

impl Default for Mul {
    fn default() -> Self {
        Self::new()
    }
}

impl Mul {
    #[inline]
    pub(crate) fn new() -> Mul {
        Self::new_into(RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_into(mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.put_u8(MUL_ID | NOT_NORMALIZED);
        buffer.put_u32_le(0_u32);
        (0u64, 1).write_packed(&mut buffer);
        let len = buffer.len() as u32 - 1 - 4;
        (&mut buffer[1..]).put_u32_le(len);

        Mul { data: buffer }
    }

    #[inline]
    pub fn from_view_into(a: &MulView<'_>, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(a.data);
        Mul { data: buffer }
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline]
    pub fn set_from_view(&mut self, view: &MulView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: AtomView<'_>) {
        self.data[0] |= NOT_NORMALIZED;

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64(); // TODO: pack size and n_args

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        let data_start = match other {
            AtomView::Mul(m) => {
                let mut sd = &m.data[1 + 4..];
                let sub_n_args;
                (sub_n_args, _, sd) = sd.get_frac_u64();

                n_args += sub_n_args;
                sd
            }
            _ => {
                n_args += 1;
                &other.get_data()
            }
        };

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

        self.data.extend_from_slice(data_start);

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor
            .write_u32::<LittleEndian>((new_buf_pos - buf_pos) as u32)
            .unwrap();
    }

    pub(crate) fn replace_last(&mut self, other: AtomView) {
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

    #[inline]
    pub fn to_mul_view(&self) -> MulView {
        MulView { data: &self.data }
    }

    pub(crate) fn set_has_coefficient(&mut self, has_coeff: bool) {
        if has_coeff {
            self.data[0] |= MUL_HAS_COEFF_FLAG;
        } else {
            self.data[0] &= !MUL_HAS_COEFF_FLAG;
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Mul(self.to_mul_view())
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_mul_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Mul {
        Mul { data: raw }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Add {
    data: RawAtom,
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Add {
    #[inline]
    pub(crate) fn new() -> Add {
        Self::new_into(RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_into(mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.put_u8(ADD_ID | NOT_NORMALIZED);
        (0u64, 0).write_packed(&mut buffer);
        Add { data: buffer }
    }

    #[inline]
    pub fn from_view_into(a: &AddView<'_>, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(a.data);
        Add { data: buffer }
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: AtomView<'_>) {
        self.data[0] |= NOT_NORMALIZED;

        let mut c = &self.data[1..];

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64();

        let old_header_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize;

        match other {
            AtomView::Add(m) => {
                let mut sd = &m.data[1..];
                let sub_n_args;
                (sub_n_args, _, sd) = sd.get_frac_u64();

                n_args += sub_n_args;
                self.data.extend_from_slice(&sd);
            }
            _ => {
                n_args += 1;
                self.data.extend_from_slice(&other.get_data());
            }
        };

        let new_len = self.data.len() - old_header_size;
        let new_header_size = (n_args, new_len as u64).get_packed_size() as usize + 1;

        match new_header_size.cmp(&old_header_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(old_header_size.., new_header_size);
                self.data
                    .resize(self.data.len() - old_header_size + new_header_size, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data
                    .resize(old_len + new_header_size - old_header_size, 0);
                self.data
                    .copy_within(old_header_size..old_len, new_header_size);
            }
        }

        (n_args, new_len as u64).write_packed_fixed(&mut self.data[1..new_header_size]);
    }

    #[inline(always)]
    pub fn to_add_view(&self) -> AddView {
        AddView { data: &self.data }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: AddView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView {
        AtomView::Add(self.to_add_view())
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_add_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Add {
        Add { data: raw }
    }
}

impl<'a> VarView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Var {
        Var::from_view_into(self, Vec::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Var) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(self.data);
        Var { data: buffer }
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        Symbol::init_fn(
            self.data[1..].get_frac_u64().0 as u32,
            self.get_wildcard_level(),
            self.data[0] & FUN_SYMMETRIC_FLAG != 0,
            self.data[0] & VAR_ANTISYMMETRIC_FLAG != 0,
            self.data[0] & FUN_LINEAR_FLAG != 0,
        )
    }

    #[inline(always)]
    pub fn get_wildcard_level(&self) -> u8 {
        match self.data[0] & VAR_WILDCARD_LEVEL_MASK {
            0 => 0,
            VAR_WILDCARD_LEVEL_1 => 1,
            VAR_WILDCARD_LEVEL_2 => 2,
            VAR_WILDCARD_LEVEL_3 => 3,
            _ => 0,
        }
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Var(*self)
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct VarView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<VarView<'b>> for VarView<'a> {
    fn eq(&self, other: &VarView<'b>) -> bool {
        self.data == other.data
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct FunView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<FunView<'b>> for FunView<'a> {
    fn eq(&self, other: &FunView<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> FunView<'a> {
    pub fn to_owned(&self) -> Fun {
        Fun::from_view_into(self, Vec::new())
    }

    pub fn clone_into(&self, target: &mut Fun) {
        target.set_from_view(self);
    }

    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(self.data);
        Fun { data: buffer }
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        let id = self.data[1 + 4..].get_frac_u64().0;

        Symbol::init_fn(
            id as u32,
            self.get_wildcard_level(),
            self.is_symmetric(),
            id & FUN_ANTISYMMETRIC_FLAG != 0,
            self.is_linear(),
        )
    }

    #[inline(always)]
    pub fn is_symmetric(&self) -> bool {
        self.data[0] & FUN_SYMMETRIC_FLAG != 0
    }

    #[inline(always)]
    pub fn is_antisymmetric(&self) -> bool {
        let id = self.data[1 + 4..].get_frac_u64().0;
        id & FUN_ANTISYMMETRIC_FLAG != 0
    }

    #[inline(always)]
    pub fn is_linear(&self) -> bool {
        self.data[0] & FUN_LINEAR_FLAG != 0
    }

    #[inline(always)]
    pub fn get_wildcard_level(&self) -> u8 {
        match self.data[0] & VAR_WILDCARD_LEVEL_MASK {
            0 => 0,
            VAR_WILDCARD_LEVEL_1 => 1,
            VAR_WILDCARD_LEVEL_2 => 2,
            VAR_WILDCARD_LEVEL_3 => 3,
            _ => 0,
        }
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_u64().1 as usize
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
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

    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Fun(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_u64(); // name

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Arg,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn fast_cmp(&self, other: FunView) -> Ordering {
        self.data.cmp(other.data)
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct NumView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<NumView<'b>> for NumView<'a> {
    #[inline]
    fn eq(&self, other: &NumView<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> NumView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Num {
        Num::from_view_into(self, Vec::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Num) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(self.data);
        Num { data: buffer }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data.is_zero_rat()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.data.is_one_rat()
    }

    #[inline]
    pub fn get_coeff_view(&self) -> CoefficientView<'a> {
        self.data[1..].get_coeff_view().0
    }

    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Num(*self)
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct PowView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<PowView<'b>> for PowView<'a> {
    #[inline]
    fn eq(&self, other: &PowView<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> PowView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Pow {
        Pow::from_view_into(self, Vec::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Pow) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(self.data);
        Pow { data: buffer }
    }

    #[inline]
    pub fn get_base(&self) -> AtomView<'a> {
        let (b, _) = self.get_base_exp();
        b
    }

    #[inline]
    pub fn get_exp(&self) -> AtomView<'a> {
        let (_, e) = self.get_base_exp();
        e
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline]
    pub fn get_base_exp(&self) -> (AtomView<'a>, AtomView<'a>) {
        let mut it = ListIterator {
            data: &self.data[1..],
            length: 2,
        };

        (it.next().unwrap(), it.next().unwrap())
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Pow(*self)
    }

    #[inline]
    pub fn to_slice(&self) -> ListSlice<'a> {
        ListSlice {
            data: &self.data[1..],
            length: 2,
            slice_type: SliceType::Pow,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct MulView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<MulView<'b>> for MulView<'a> {
    #[inline]
    fn eq(&self, other: &MulView<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> MulView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Mul {
        Mul::from_view_into(self, Vec::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Mul) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(self.data);
        Mul { data: buffer }
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    pub fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_u64().0 as usize
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
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

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Mul(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Mul,
        }
    }

    #[inline]
    pub fn has_coefficient(&self) -> bool {
        (self.data[0] & MUL_HAS_COEFF_FLAG) != 0
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct AddView<'a> {
    data: &'a [u8],
}

impl<'a, 'b> PartialEq<AddView<'b>> for AddView<'a> {
    #[inline]
    fn eq(&self, other: &AddView<'b>) -> bool {
        self.data == other.data
    }
}

impl<'a> AddView<'a> {
    pub fn to_owned(&self) -> Add {
        Add::from_view_into(self, Vec::new())
    }

    pub fn clone_into(&self, target: &mut Add) {
        target.set_from_view(*self);
    }

    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(self.data);
        Add { data: buffer }
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.data[1..].get_frac_u64().0 as usize
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
        let mut c = self.data;
        c.get_u8();

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListIterator {
            data: c,
            length: n_args as u32,
        }
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Add(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Add,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AtomView<'a> {
    pub const ZERO: Self = Self::Num(NumView { data: &ZERO_DATA });

    pub fn from(source: &'a [u8]) -> AtomView<'a> {
        match source[0] & TYPE_MASK {
            VAR_ID => AtomView::Var(VarView { data: source }),
            FUN_ID => AtomView::Fun(FunView { data: source }),
            NUM_ID => AtomView::Num(NumView { data: source }),
            POW_ID => AtomView::Pow(PowView { data: source }),
            MUL_ID => AtomView::Mul(MulView { data: source }),
            ADD_ID => AtomView::Add(AddView { data: source }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    #[inline(always)]
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

    /// Write the expression to a binary stream. The format is the byte-length first
    /// followed by the data. To import the expression in new session, also export the [`State`].
    #[inline(always)]
    pub fn write<W: Write>(&self, mut dest: W) -> Result<(), std::io::Error> {
        let d = self.get_data();
        dest.write_u8(0)?;
        dest.write_u64::<LittleEndian>(d.len() as u64)?;
        dest.write_all(d)
    }

    pub(crate) fn rename(&self, state_map: &StateMap) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.rename_no_norm(state_map, ws, &mut a);
            let mut r = Atom::new();
            a.as_view().normalize(ws, &mut r);
            r
        })
    }

    fn rename_no_norm(&self, state_map: &StateMap, ws: &Workspace, out: &mut Atom) {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::FiniteField(e, i) => {
                    if let Some(s) = state_map.finite_fields.get(&i) {
                        out.to_num(Coefficient::FiniteField(e, *s));
                    } else {
                        out.set_from_view(self);
                    }
                }
                CoefficientView::RationalPolynomial(r) => {
                    let (old_id, _, _) = r.0.get_frac_u64();

                    if let Some(nv) = state_map.variables_lists.get(&old_id) {
                        let mut rr = r.deserialize();
                        rr.numerator.variables = nv.clone();
                        rr.denominator.variables = nv.clone();
                        out.to_num(Coefficient::RationalPolynomial(rr));
                    } else {
                        out.set_from_view(self);
                    }
                }
                _ => out.set_from_view(self),
            },
            AtomView::Var(v) => {
                if let Some(s) = state_map.symbols.get(&v.get_symbol().get_id()) {
                    out.to_var(*s);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Fun(f) => {
                if let Some(s) = state_map.symbols.get(&f.get_symbol().get_id()) {
                    let nf = out.to_fun(*s);

                    let mut na = ws.new_atom();
                    for a in f.iter() {
                        a.rename_no_norm(state_map, ws, &mut na);
                        nf.add_arg(na.as_view());
                    }
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                let mut nb = ws.new_atom();
                b.rename_no_norm(state_map, ws, &mut nb);
                let mut ne = ws.new_atom();
                e.rename_no_norm(state_map, ws, &mut ne);

                out.to_pow(nb.as_view(), ne.as_view());
            }
            AtomView::Mul(m) => {
                let nm = out.to_mul();

                let mut na = ws.new_atom();
                for a in m.iter() {
                    a.rename_no_norm(state_map, ws, &mut na);
                    nm.extend(na.as_view());
                }
            }
            AtomView::Add(add) => {
                let nm = out.to_add();

                let mut na = ws.new_atom();
                for a in add.iter() {
                    a.rename_no_norm(state_map, ws, &mut na);
                    nm.extend(na.as_view());
                }
            }
        }
    }
}

impl PartialEq<AtomView<'_>> for AtomView<'_> {
    #[inline(always)]
    fn eq(&self, other: &AtomView) -> bool {
        self.get_data() == other.get_data()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ListIterator<'a> {
    data: &'a [u8],
    length: u32,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = AtomView<'a>;

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
                NUM_ID | VAR_ID => {
                    self.data = self.data.skip_rational();
                }
                FUN_ID | MUL_ID => {
                    let n_size = self.data.get_u32_le();
                    self.data.advance(n_size as usize);
                }
                ADD_ID => {
                    let (_, size, np) = self.data.get_frac_u64();
                    self.data = np;
                    self.data.advance(size as usize);
                }
                POW_ID => {
                    skip_count += 2;
                }
                _ => unreachable!("Bad id"),
            }

            skip_count -= 1;

            if skip_count == 0 {
                break;
            }

            cur_id = self.data.get_u8() & TYPE_MASK;
        }

        let len = unsafe { self.data.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        match start_id {
            NUM_ID => Some(AtomView::Num(NumView { data })),
            VAR_ID => Some(AtomView::Var(VarView { data })),
            FUN_ID => Some(AtomView::Fun(FunView { data })),
            MUL_ID => Some(AtomView::Mul(MulView { data })),
            ADD_ID => Some(AtomView::Add(AddView { data })),
            POW_ID => Some(AtomView::Pow(PowView { data })),
            x => unreachable!("Bad id {}", x),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListSlice<'a> {
    data: &'a [u8],
    length: usize,
    slice_type: SliceType,
}

impl<'a> ListSlice<'a> {
    #[inline(always)]
    fn skip(mut pos: &[u8], n: u32) -> &[u8] {
        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = n;
        while skip_count > 0 {
            skip_count -= 1;

            match pos.get_u8() & TYPE_MASK {
                NUM_ID | VAR_ID => {
                    pos = pos.skip_rational();
                }
                FUN_ID | MUL_ID => {
                    let n_size = pos.get_u32_le();
                    pos.advance(n_size as usize);
                }
                ADD_ID => {
                    let (_, size, np) = pos.get_frac_u64();
                    pos = np;
                    pos.advance(size as usize);
                }
                POW_ID => {
                    skip_count += 2;
                }
                _ => unreachable!("Bad id"),
            }
        }
        pos
    }

    fn fast_forward(&self, index: usize) -> ListSlice<'a> {
        let mut pos = self.data;

        pos = Self::skip(pos, index as u32);

        ListSlice {
            data: pos,
            length: self.length - index,
            slice_type: self.slice_type,
        }
    }

    fn get_entry(start: &'a [u8]) -> (AtomView<'a>, &[u8]) {
        let start_id = start[0] & TYPE_MASK;
        let end = Self::skip(start, 1);
        let len = unsafe { end.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        (
            match start_id {
                NUM_ID => AtomView::Num(NumView { data }),
                VAR_ID => AtomView::Var(VarView { data }),
                FUN_ID => AtomView::Fun(FunView { data }),
                MUL_ID => AtomView::Mul(MulView { data }),
                ADD_ID => AtomView::Add(AddView { data }),
                POW_ID => AtomView::Pow(PowView { data }),
                x => unreachable!("Bad id {}", x),
            },
            end,
        )
    }
}

impl<'a> ListSlice<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn get(&self, index: usize) -> AtomView<'a> {
        let start = self.fast_forward(index);
        Self::get_entry(start.data).0
    }

    pub fn get_subslice(&self, range: std::ops::Range<usize>) -> Self {
        let start = self.fast_forward(range.start);

        let mut s = start.data;
        s = Self::skip(s, range.len() as u32);

        let len = unsafe { s.as_ptr().offset_from(start.data.as_ptr()) } as usize;
        ListSlice {
            data: &start.data[..len],
            length: range.len(),
            slice_type: self.slice_type,
        }
    }

    #[inline]
    pub fn get_type(&self) -> SliceType {
        self.slice_type
    }

    #[inline]
    pub fn from_one(view: AtomView<'a>) -> Self {
        ListSlice {
            data: view.get_data(),
            length: 1,
            slice_type: SliceType::One,
        }
    }

    #[inline]
    pub fn iter(&self) -> ListSliceIterator<'a> {
        ListSliceIterator { data: *self }
    }
}

pub struct ListSliceIterator<'a> {
    data: ListSlice<'a>,
}

impl<'a> Iterator for ListSliceIterator<'a> {
    type Item = AtomView<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.length > 0 {
            let (res, end) = ListSlice::get_entry(self.data.data);
            self.data = ListSlice {
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
