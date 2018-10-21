use std::ops::{BitAnd, BitOr};
use std::fmt;
use std::iter::FromIterator;
use std::cmp::Ordering;

pub trait AndNot<RHS = Self> {
    type Output;
    fn andnot(self, rhs: RHS) -> Self::Output;
}

pub trait IDL {
    fn push_id(&mut self, value: u64);
    fn len(&self) -> usize;
}

pub mod idl_simple;
pub mod idl_range;

