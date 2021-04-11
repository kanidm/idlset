//! IDLSet - Fast u64 integer set operations
//!
//! IDLSet is a specialised library for fast logical set operations on
//! u64. For example, this means union (or), intersection (and) and not
//! operations on sets. In the best case, speed ups of 15x have been observed
//! with the general case performing approximately 4x faster that a Vec<u64>
//! based implementation.
//!
//! These operations are heavily used in low-level implementations of databases
//! for their indexing logic, but has applications with statistical analysis and
//! other domains that require logical set operations.
//!

#![warn(missing_docs)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate smallvec;

pub mod v1;
pub mod v2;

/// Bit trait representing the equivalent of a & (!b). This allows set operations
/// such as "The set A does not contain any element of set B".
pub trait AndNot<RHS = Self> {
    /// The type of set implementation to return.
    type Output;

    /// Perform an AndNot (exclude) operation between two sets. This returns
    /// a new set containing the results. The set on the right is the candidate
    /// set to exclude from the set of the left. As an example this would
    /// behave as `[1,2,3].andnot([2]) == [1, 3]`.
    fn andnot(self, rhs: RHS) -> Self::Output;
}
