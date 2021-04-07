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
//! This seems very specific to only use u64, but has been chosen for a good reason. On
//! 64bit cpus, native 64bit operations are faster than 32/16. Additionally,
//! due to the design of the library, unsigned types are simpler to operate
//! on for the set operations.
//!

#![warn(missing_docs)]

#[macro_use]
extern crate serde_derive;

#[cfg(feature = "use_smallvec")]
extern crate smallvec;

pub mod v2;

#[cfg(feature = "use_smallvec")]
use smallvec::SmallVec;

use std::cmp::Ordering;
use std::iter::FromIterator;
use std::ops::{BitAnd, BitOr};
use std::{fmt, slice};

/// Default number of IDL ranges to keep in stack before we spill into heap. As many
/// operations in a system like kanidm are either single item indexes (think equality)
/// or very large indexes (think pres, class), we can keep this small.
#[cfg(feature = "use_smallvec")]
const DEFAULT_STACK_ALLOC: usize = 1;

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

/// The core representation of sets of integers in compressed format.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct IDLRange {
    range: u64,
    mask: u64,
}

// To make binary search, Ord only applies to range.

impl Ord for IDLRange {
    fn cmp(&self, other: &Self) -> Ordering {
        self.range.cmp(&other.range)
    }
}

impl PartialOrd for IDLRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for IDLRange {
    fn eq(&self, other: &Self) -> bool {
        self.range == other.range
    }
}

impl Eq for IDLRange {}

impl IDLRange {
    fn new(range: u64, mask: u64) -> Self {
        IDLRange {
            range: range,
            mask: mask,
        }
    }

    fn push_id(&mut self, value: u64) {
        let nmask = 1 << value;
        self.mask ^= nmask;
    }
}

/// An ID List of `u64` values, that uses a compressed representation of `u64` to
/// speed up set operations, improve cpu cache behaviour and consume less memory.
///
/// This is essentially a `Vec<u64>`, but requires less storage with large values
/// and natively supports logical operations for set manipulation. Today this
/// supports And, Or, AndNot. Future types may be added such as xor.
///
/// # How does it work?
///
/// The `IDLBitRange` stores a series of tuples (IDRange) that represents a
/// range prefix `u64` and a `u64` mask of bits representing the presence of that
/// integer in the set. For example, the number `1` when inserted would create
/// an idl range of: `IDRange { range: 0, mask: 2 }`. The mask has the "second"
/// bit set, so we add range and recieve `1`. (if mask was 1, this means the value
/// 0 is present!)
///
/// Other examples would be `IDRange { range: 0, mask: 3 }`. Because 3 means
/// "the first and second bit is set" this would extract to `[0, 1]`
/// `IDRange { range: 0, mask: 38}` represents the set `[1, 2, 5]` as the.
/// second, third and sixth bits are set. Finally, a value of `IDRange { range: 64, mask: 4096 }`
/// represents the set `[76, ]`.
///
/// Using this, we can store up to 64 integers in an IDRange. Once there are
/// at least 3 bits set in mask, the compression is now saving memory space compared
/// to raw unpacked `Vec<u64>`.
///
/// The set operations can now be performed by applying `u64` bitwise operations
/// on the mask components for a given matching range prefix. If the range
/// prefix is not present in the partner set, we choose a correct course of
/// action (Or copies the range to the result, And skips the range entirely)
///
/// As an example, if we had the values `IDRange { range: 0, mask: 38 }` (`[1, 2, 5]`) and
/// `IDRange { range: 0, mask: 7 }` (`[0, 1, 2]`), and we were to perform an `&` operation
/// on these sets, the result would be `7 & 38 == 6`. The result now is
/// `IDRange { range: 0, mask: 6 }`, which decompresses to `[1, 2]` - the correct
/// result of our set And operation.
///
/// The important note here is that with a single cpu `&` operation, we were
/// able to intersect up to 64 values at once. Contrast to a `Vec<u64>` where we
/// would need to perform cpu equality on each value. For our above example
/// this would have taken at most 4 cpu operations with the `Vec<u64>`, where
/// as the `IDLBitRange` performed 2 (range eq and mask `&`).
///
/// Worst case behaviour is sparse u64 sets where each IDRange only has a single
/// populated value. This yields a slow down of approx 20% compared to the `Vec<u64>`.
/// However, as soon as the IDRange contains at least 2 values they are equal
/// in performance, and three values begins to exceed. This applies to all
/// operation types and data sizes.
///
/// # Examples
/// ```
/// use idlset::IDLBitRange;
/// use std::iter::FromIterator;
///
/// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
/// let idl_b = IDLBitRange::from_iter(vec![2]);
///
/// // Conduct an and (intersection) of the two lists to find commont members.
/// let idl_result = idl_a & idl_b;
///
/// let idl_expect = IDLBitRange::from_iter(vec![2]);
/// assert_eq!(idl_result, idl_expect);
/// ```
#[derive(Serialize, Deserialize, PartialEq, Clone)]
pub struct IDLBitRange {
    #[cfg(not(feature = "use_smallvec"))]
    list: Vec<IDLRange>,
    #[cfg(feature = "use_smallvec")]
    list: SmallVec<[IDLRange; DEFAULT_STACK_ALLOC]>,
}

impl IDLBitRange {
    /// Construct a new, empty set.
    pub fn new() -> Self {
        IDLBitRange {
            #[cfg(not(feature = "use_smallvec"))]
            list: Vec::new(),
            #[cfg(feature = "use_smallvec")]
            list: SmallVec::new(),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        IDLBitRange {
            #[cfg(not(feature = "use_smallvec"))]
            list: Vec::with_capacity(cap),
            #[cfg(feature = "use_smallvec")]
            list: SmallVec::with_capacity(cap),
        }
    }

    /// Construct a set containing a single initial value. This is a special
    /// use case for database indexing where single value equality indexes are
    /// store uncompressed on disk.
    pub fn from_u64(id: u64) -> Self {
        let mut new = IDLBitRange::new();
        unsafe {
            new.push_id(id);
        }
        new
    }

    /// This does an optimised single and operation in the case
    /// one of the candidates has a single item. See bitand for more.
    fn bstbitand(&self, candidate: &IDLRange) -> Self {
        let mut result = IDLBitRange::new();
        if let Ok(idx) = self.list.binary_search(candidate) {
            let existing = self.list.get(idx).unwrap();
            let mask = existing.mask & candidate.mask;
            if mask > 0 {
                let newrange = IDLRange::new(candidate.range, mask);
                result.list.push(newrange);
            };
        };
        result
    }

    /// Returns `true` if the id `u64` value exists within the set.
    pub fn contains(&self, id: u64) -> bool {
        let bvalue: u64 = id % 64;
        let range: u64 = id - bvalue;
        // New takes a starting mask, not a raw bval, so shift it!
        let candidate = IDLRange::new(range, 1 << bvalue);

        if let Ok(idx) = self.list.binary_search(&candidate) {
            let existing = self.list.get(idx).unwrap();
            let mask = existing.mask & candidate.mask;
            if mask > 0 {
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Insert an id into the set, correctly sorted.
    pub fn insert_id(&mut self, value: u64) {
        // Determine our range
        let bvalue: u64 = value % 64;
        let range: u64 = value - bvalue;

        // We make a dummy range and mask to find our range
        let candidate = IDLRange::new(range, 1 << bvalue);

        let r = self.list.binary_search(&candidate);
        match r {
            Ok(idx) => {
                let mut existing = self.list.get_mut(idx).unwrap();
                existing.mask |= candidate.mask;
            }
            Err(idx) => {
                self.list.insert(idx, candidate);
            }
        }
    }

    /// Remove an id from the set, leaving it correctly sorted.
    ///
    /// If the value is not present, no action is taken.
    pub fn remove_id(&mut self, value: u64) {
        // Determine our range
        let bvalue: u64 = value % 64;
        let range: u64 = value - bvalue;

        // We make a dummy range and mask to find our range
        let candidate = IDLRange::new(range, 1 << bvalue);

        match self.list.binary_search(&candidate) {
            Ok(idx) => {
                // The listed range would contain our bit.
                // So we need to remove this, leaving all other bits in place.
                //
                // To do this, we not the candidate, so all other bits remain,
                // then we perform and &= so that the existing bits survive.
                let mut existing = self.list.get_mut(idx).unwrap();

                existing.mask &= !candidate.mask;

                if existing.mask == 0 {
                    // No more items in this range, remove it.
                    self.list.remove(idx);
                }
            }
            Err(_) => {
                // No action required, the value is not in any range.
            }
        }
    }

    /// Push an id into the set. The value is inserted onto the tail of the set
    /// which may cause you to break the structure if your input isn't sorted.
    /// You probably want `insert_id` instead.
    pub unsafe fn push_id(&mut self, value: u64) {
        // Get what range this should be
        let bvalue: u64 = value % 64;
        let range: u64 = value - bvalue;

        // Get the highest IDLRange out:
        if let Some(last) = self.list.last_mut() {
            if (*last).range == range {
                // Insert the bit.
                (*last).push_id(bvalue);
                return;
            }
        }

        // New takes a starting mask, not a raw bval, so shift it!
        let newrange = IDLRange::new(range, 1 << bvalue);
        self.list.push(newrange);
    }

    /// Returns the number of ids in the set. This operation iterates over
    /// the set, decompressing it to count the ids, which MAY be slow. If
    /// you want to see if the set is empty, us `is_empty()`
    #[inline(always)]
    pub fn len(&self) -> usize {
        // Today this is really inefficient using an iter to collect
        // and reduce the set. We could store a .count in the struct
        // if demand was required ...
        // Right now, this would require a complete walk of the bitmask.
        self.list
            .iter()
            .fold(0, |acc, i| (i.mask.count_ones() as usize) + acc)
    }

    /// Returns if the number of ids in this set exceed this threshold. While
    /// this must iterate to determine if this is true, since we shortcut return
    /// in the check, on long sets we will not iterate over the complete content
    /// making it faster than `len() < thresh`.
    ///
    /// Returns true if the set is smaller than threshold.
    #[inline(always)]
    pub fn below_threshold(&self, threshold: usize) -> bool {
        let mut ic: usize = 0;
        for i in self.list.iter() {
            ic += i.mask.count_ones() as usize;
            if ic >= threshold {
                return false;
            }
        }
        true
    }

    /// Show if this IDL set contains no elements
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.list.len() == 0
    }

    /// Show how many ranges we hold in this idlset.
    #[inline(always)]
    pub fn len_range(&self) -> usize {
        self.list.len()
    }

    /// Sum all the values contained into this set to yield a single result.
    #[inline(always)]
    pub fn sum(&self) -> u64 {
        let mut result: u64 = 0;
        for id in self {
            result += id;
        }
        return result;
    }

    #[inline(always)]
    fn bitand_inner(&self, rhs: &Self) -> Self {
        let llen = self.len_range();
        let rlen = rhs.len_range();

        // If any list only has a single range element, then it's only possible
        // for that single element to match in the other. In this case
        // rather than doing a full walk of the vecs, binary search for
        // the single item and compare it directly.
        if llen == 1 {
            return rhs.bstbitand(self.list.first().unwrap());
        } else if rlen == 1 {
            return self.bstbitand(rhs.list.first().unwrap());
        }

        // We only allocate the size of the smaller set since that's the
        // theoretical max alloc needed.
        let mut result = if llen < rlen {
            IDLBitRange::with_capacity(llen)
        } else {
            IDLBitRange::with_capacity(rlen)
        };

        let mut liter = self.list.iter();
        let mut riter = rhs.list.iter();

        let mut lnextrange = liter.next();
        let mut rnextrange = riter.next();

        while lnextrange.is_some() && rnextrange.is_some() {
            let l = lnextrange.unwrap();
            let r = rnextrange.unwrap();

            if l.range == r.range {
                let mask = l.mask & r.mask;
                if mask > 0 {
                    let newrange = IDLRange::new(l.range, mask);
                    result.list.push(newrange);
                }
                lnextrange = liter.next();
                rnextrange = riter.next();
            } else if l.range < r.range {
                lnextrange = liter.next();
            } else {
                rnextrange = riter.next();
            }
        }
        result
    }

    #[inline(always)]
    fn bitor_inner(&self, rhs: &Self) -> Self {
        let llen = self.len_range();
        let rlen = rhs.len_range();

        // TODO: This could actually be llen + rlen for worst
        // case situations.
        let mut result = if llen > rlen {
            IDLBitRange::with_capacity(llen)
        } else {
            IDLBitRange::with_capacity(rlen)
        };

        let mut liter = self.list.iter();
        let mut riter = rhs.list.iter();

        let mut lnextrange = liter.next();
        let mut rnextrange = riter.next();

        while lnextrange.is_some() && rnextrange.is_some() {
            let l = lnextrange.unwrap();
            let r = rnextrange.unwrap();

            let (range, mask) = if l.range == r.range {
                lnextrange = liter.next();
                rnextrange = riter.next();
                (l.range, l.mask | r.mask)
            } else if l.range < r.range {
                lnextrange = liter.next();
                (l.range, l.mask)
            } else {
                rnextrange = riter.next();
                (r.range, r.mask)
            };
            let newrange = IDLRange::new(range, mask);
            result.list.push(newrange);
        }

        while lnextrange.is_some() {
            let l = lnextrange.unwrap();

            let newrange = IDLRange::new(l.range, l.mask);
            result.list.push(newrange);
            lnextrange = liter.next();
        }

        while rnextrange.is_some() {
            let r = rnextrange.unwrap();

            let newrange = IDLRange::new(r.range, r.mask);
            result.list.push(newrange);
            rnextrange = riter.next();
        }
        result
    }

    #[inline(always)]
    fn bitandnot_inner(&self, rhs: &Self) -> Self {
        let llen = self.len_range();
        let rlen = rhs.len_range();

        // Must alloc size of the large, since all elements of r
        // could not be in l.
        let mut result = if llen > rlen {
            IDLBitRange::with_capacity(llen)
        } else {
            IDLBitRange::with_capacity(rlen)
        };

        let mut liter = self.list.iter();
        let mut riter = rhs.list.iter();

        let mut lnextrange = liter.next();
        let mut rnextrange = riter.next();

        while lnextrange.is_some() && rnextrange.is_some() {
            let l = lnextrange.unwrap();
            let r = rnextrange.unwrap();

            if l.range == r.range {
                let mask = l.mask & (!r.mask);
                if mask > 0 {
                    let newrange = IDLRange::new(l.range, mask);
                    result.list.push(newrange);
                }
                lnextrange = liter.next();
                rnextrange = riter.next();
            } else if l.range < r.range {
                // if the left range isn't in the right, just push it to the set and move
                // on.
                result.list.push(l.clone());
                lnextrange = liter.next();
            } else {
                rnextrange = riter.next();
            }
        }

        while lnextrange.is_some() {
            let l = lnextrange.unwrap();

            let newrange = IDLRange::new(l.range, l.mask);
            result.list.push(newrange);
            lnextrange = liter.next();
        }
        result
    }
}

impl FromIterator<u64> for IDLBitRange {
    /// Build an IDLBitRange from at iterator. If you provide a sorted input, a fast append
    /// mode is used. Unsorted inputs use a slower insertion sort method
    /// instead.
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower_bound, _) = iter.size_hint();
        let mut new = IDLBitRange {
            #[cfg(feature = "use_smallvec")]
            list: SmallVec::with_capacity(lower_bound),
            #[cfg(not(feature = "use_smallvec"))]
            list: Vec::with_capacity(lower_bound),
        };

        let mut max_seen = 0;
        iter.for_each(|i| {
            if i >= max_seen {
                // if we have a sorted list, we can take a fast append path.
                unsafe {
                    new.push_id(i);
                }
                max_seen = i;
            } else {
                // if not, we have to bst each time to get the right place.
                new.insert_id(i);
            }
        });
        new
    }
}

impl BitAnd for &IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an And (intersection) operation between two sets. This returns
    /// a new set containing the results.
    ///
    /// # Examples
    /// ```
    /// # use idlset::IDLBitRange;
    /// # use std::iter::FromIterator;
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a & idl_b;
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![2]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn bitand(self, rhs: &IDLBitRange) -> IDLBitRange {
        self.bitand_inner(rhs)
    }
}

impl BitAnd for IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an And (intersection) operation between two sets. This returns
    /// a new set containing the results.
    ///
    /// # Examples
    /// ```
    /// # use idlset::IDLBitRange;
    /// # use std::iter::FromIterator;
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a & idl_b;
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![2]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn bitand(self, rhs: IDLBitRange) -> IDLBitRange {
        self.bitand_inner(&rhs)
    }
}

impl BitOr for &IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an Or (union) operation between two sets. This returns
    /// a new set containing the results.
    ///
    /// # Examples
    /// ```
    /// # use idlset::IDLBitRange;
    /// # use std::iter::FromIterator;
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a | idl_b;
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn bitor(self, rhs: &IDLBitRange) -> IDLBitRange {
        self.bitor_inner(rhs)
    }
}

impl BitOr for IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an Or (union) operation between two sets. This returns
    /// a new set containing the results.
    ///
    /// # Examples
    /// ```
    /// # use idlset::IDLBitRange;
    /// # use std::iter::FromIterator;
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a | idl_b;
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn bitor(self, rhs: Self) -> IDLBitRange {
        self.bitor_inner(&rhs)
    }
}

impl AndNot for IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an AndNot (exclude) operation between two sets. This returns
    /// a new set containing the results. The set on the right is the candidate
    /// set to exclude from the set of the left.
    ///
    /// # Examples
    /// ```
    /// // Note the change to import the AndNot trait.
    /// use idlset::{IDLBitRange, AndNot};
    /// # use std::iter::FromIterator;
    ///
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a.andnot(idl_b);
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![1, 3]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    ///
    /// ```
    /// // Note the change to import the AndNot trait.
    /// use idlset::{IDLBitRange, AndNot};
    /// # use std::iter::FromIterator;
    ///
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// // Note how reversing a and b here will return an empty set.
    /// let idl_result = idl_b.andnot(idl_a);
    ///
    /// let idl_expect = IDLBitRange::new();
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn andnot(self, rhs: Self) -> IDLBitRange {
        self.bitandnot_inner(&rhs)
    }
}

impl AndNot for &IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an AndNot (exclude) operation between two sets. This returns
    /// a new set containing the results. The set on the right is the candidate
    /// set to exclude from the set of the left.
    ///
    /// # Examples
    /// ```
    /// // Note the change to import the AndNot trait.
    /// use idlset::{IDLBitRange, AndNot};
    /// # use std::iter::FromIterator;
    ///
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// let idl_result = idl_a.andnot(idl_b);
    ///
    /// let idl_expect = IDLBitRange::from_iter(vec![1, 3]);
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    ///
    /// ```
    /// // Note the change to import the AndNot trait.
    /// use idlset::{IDLBitRange, AndNot};
    /// # use std::iter::FromIterator;
    ///
    /// let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
    /// let idl_b = IDLBitRange::from_iter(vec![2]);
    ///
    /// // Note how reversing a and b here will return an empty set.
    /// let idl_result = idl_b.andnot(idl_a);
    ///
    /// let idl_expect = IDLBitRange::new();
    /// assert_eq!(idl_result, idl_expect);
    /// ```
    fn andnot(self, rhs: &IDLBitRange) -> IDLBitRange {
        self.bitandnot_inner(rhs)
    }
}

/// An iterator over the set of values that exists in an `IDLBitRange`. This
/// can be used to extract the decompressed values into another form of
/// datastructure, perform map functions or simply iteration with a for
/// loop.
///
/// # Examples
/// ```
/// # use idlset::IDLBitRange;
/// # use std::iter::FromIterator;
/// # let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
/// let ids: Vec<u64> = idl_a.into_iter().collect();
/// ```
///
/// ```
/// # use idlset::IDLBitRange;
/// # use std::iter::FromIterator;
/// # let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
/// # let mut total: u64 = 0;
/// for id in &idl_a {
///    total += id;
/// }
/// ```
#[derive(Debug)]
pub struct IDLBitRangeIter<'a> {
    // rangeiter: std::vec::IntoIter<IDLRange>,
    rangeiter: slice::Iter<'a, IDLRange>,
    currange: Option<&'a IDLRange>,
    curbit: u64,
}

impl<'a> Iterator for IDLBitRangeIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        while self.currange.is_some() {
            let range = self.currange.unwrap();
            while self.curbit < 64 {
                let m: u64 = 1 << self.curbit;
                let candidate: u64 = range.mask & m;
                if candidate > 0 {
                    let result = Some(self.curbit + range.range);
                    self.curbit += 1;
                    return result;
                }
                self.curbit += 1;
            }
            self.currange = self.rangeiter.next();
            self.curbit = 0;
        }
        None
    }
}

impl<'a> IntoIterator for &'a IDLBitRange {
    type Item = u64;
    type IntoIter = IDLBitRangeIter<'a>;

    fn into_iter(self) -> IDLBitRangeIter<'a> {
        let mut liter = (&self.list).into_iter();
        let nrange = liter.next();
        IDLBitRangeIter {
            rangeiter: liter,
            currange: nrange,
            curbit: 0,
        }
    }
}

impl fmt::Display for IDLBitRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IDLBitRange (compressed ranges) {:?} (decompressed) <optimised out>",
            self.list.len()
        )
    }
}

impl fmt::Debug for IDLBitRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IDLBitRange (compressed) {:?} (decompressed) [ ",
            self.list
        )?;
        for id in self {
            write!(f, "{}, ", id)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::{AndNot, IDLBitRange};
    use std::iter::FromIterator;

    #[test]
    fn test_store_zero() {
        let idl_a = IDLBitRange::from_iter(vec![0]);
        assert!(idl_a.contains(0));
    }

    #[test]
    fn test_contains() {
        let idl_a = IDLBitRange::from_iter(vec![0, 1, 2]);
        assert!(idl_a.contains(2));
        assert!(!idl_a.contains(3));
        assert!(!idl_a.contains(65));
    }

    #[test]
    fn test_remove_id() {
        let mut idl_a = IDLBitRange::from_iter(vec![0, 1, 2, 3, 4]);
        let idl_ex = IDLBitRange::from_iter(vec![0, 1, 3, 4]);
        idl_a.remove_id(2);
        assert!(idl_ex == idl_a);
        // Removing twice does nothing
        idl_a.remove_id(2);
        assert!(idl_ex == idl_a);
    }

    #[test]
    fn test_len() {
        let idl_a = IDLBitRange::new();
        assert_eq!(idl_a.len(), 0);
        let idl_b = IDLBitRange::from_iter(vec![0, 1, 2]);
        assert_eq!(idl_b.len(), 3);
        let idl_c = IDLBitRange::from_iter(vec![0, 64, 128]);
        assert_eq!(idl_c.len(), 3);
    }

    #[test]
    fn test_from_iter() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let idl_b = IDLBitRange::from_iter(vec![64, 68, 2, 1]);
        let idl_c = IDLBitRange::from_iter(vec![68, 64, 1, 2]);
        let idl_d = IDLBitRange::from_iter(vec![2, 1, 68, 64]);

        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        assert_eq!(idl_a, idl_expect);
        assert_eq!(idl_b, idl_expect);
        assert_eq!(idl_c, idl_expect);
        assert_eq!(idl_d, idl_expect);
    }

    #[test]
    fn test_range_intersection_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::from_iter(vec![2]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_2() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![4, 67]);
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_3() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 35, 64, 65, 128, 150]);
        let idl_b = IDLBitRange::from_iter(vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 3, 35, 64, 128, 150]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_4() {
        let idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let idl_b = IDLBitRange::from_iter(1..1024);
        let idl_expect = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_5() {
        let idl_a = IDLBitRange::from_iter(1..204800);
        let idl_b = IDLBitRange::from_iter(102400..307200);
        let idl_expect = IDLBitRange::from_iter(102400..204800);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_6() {
        let idl_a = IDLBitRange::from_iter(vec![307199]);
        let idl_b = IDLBitRange::from_iter(102400..307200);
        let idl_expect = IDLBitRange::from_iter(vec![307199]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_union_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3]);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_union_2() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![4, 67]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 67]);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_union_3() {
        let idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let idl_b = IDLBitRange::from_iter(1..1024);
        let idl_expect = IDLBitRange::from_iter(1..1024);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_not_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![3, 4]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 5, 6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_not_2() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![10]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_not_3() {
        let idl_a = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![1]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_not_4() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let idl_b = IDLBitRange::from_iter(vec![65]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 64, 66]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_threshold() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let idl_b = IDLBitRange::from_iter(vec![65]);

        assert!(idl_a.below_threshold(1) == false);
        assert!(idl_a.below_threshold(6) == false);
        assert!(idl_a.below_threshold(8) == true);

        assert!(idl_b.below_threshold(1) == false);
        assert!(idl_b.below_threshold(8) == true);
    }
}
