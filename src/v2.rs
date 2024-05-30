//! Version 2 of IDLSet - This is a self-adaptive version of the compressed integer
//! set library, that compresses dynamicly based on heuristics of your data. In the
//! case that your data is sparse, or a very small set, the data will remain uncompressed.
//! If your data is dense, then it will be compressed. Depending on the nature of
//! the operations you use, this means that when intersecting or unioning these
//! an optimised version for these behaviours can be chosen, significantly improving
//! performance in general cases over [`v1`] (which is always compressed).

use crate::AndNot;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::iter::FromIterator;
use std::ops::{BitAnd, BitOr};
use std::{fmt, slice};

/// Default number of IDL ranges to keep in stack before we spill into heap. As many
/// operations in a system like kanidm are either single item indexes (think equality)
/// or very large indexes (think pres, class), we can keep this small.
///
/// A sparse alloc of 2 keeps the comp vs sparse variants equal size in the non-overflow
/// case. Larger means we are losing space in the comp case.
const DEFAULT_SPARSE_ALLOC: usize = 2;

// After a lot of benchmarking, the cross over point is when there is an average
// of 12 bits set in a compressed range for general case to be faster.
#[cfg(target_arch = "x86_64")]
const AVG_RANGE_COMP_REQ: usize = 12;
// We improve intersection over union performance.
#[cfg(target_arch = "aarch64")]
const AVG_RANGE_COMP_REQ: usize = 5;

// Untested, but covers other build options.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const AVG_RANGE_COMP_REQ: usize = 5;

const FAST_PATH_BST_RATIO: usize = 8;

/// The core representation of sets of integers in compressed format.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename = "R")]
struct IDLRange {
    #[serde(rename = "r")]
    pub range: u64,
    #[serde(rename = "m")]
    pub mask: u64,
}

impl fmt::Debug for IDLRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IDLRange {{ range: {}, mask: {:x} }}",
            self.range, self.mask
        )
    }
}

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

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename = "S")]
enum IDLState {
    #[serde(rename = "s")]
    Sparse(SmallVec<[u64; DEFAULT_SPARSE_ALLOC]>),
    #[serde(rename = "c")]
    Compressed(Vec<IDLRange>),
}

impl IDLState {
    fn shrink_to_fit(&mut self) {
        match self {
            IDLState::Sparse(svec) => svec.shrink_to_fit(),
            IDLState::Compressed(vec) => vec.shrink_to_fit(),
        }
    }

    fn sparse_bitand_fast_path(smol: &[u64], lrg: &[u64]) -> Self {
        let mut nlist = SmallVec::with_capacity(smol.len());
        // We cache the idx inbetween to narrow the bst sizes.
        let mut idx_min = 0;
        smol.iter().for_each(|id| {
            let (_, partition) = lrg.split_at(idx_min);
            if let Ok(idx) = partition.binary_search(id) {
                debug_assert!(Ok(idx + idx_min) == lrg.binary_search(id));
                let idx = idx + idx_min;
                nlist.push(*id);
                debug_assert!(idx >= idx_min);
                idx_min = idx;
            }
        });

        nlist.shrink_to_fit();
        IDLState::Sparse(nlist)
    }

    fn sparse_bitor_fast_path(smol: &[u64], lrg: &[u64]) -> Self {
        let mut nlist = SmallVec::with_capacity(lrg.len() + smol.len());
        nlist.extend_from_slice(lrg);

        let mut idx_min = 0;
        smol.iter().for_each(|id| {
            let (_, partition) = nlist.split_at(idx_min);
            if let Err(idx) = partition.binary_search(id) {
                debug_assert!(Err(idx + idx_min) == nlist.binary_search(id));
                let idx = idx + idx_min;
                nlist.insert(idx, *id);
                debug_assert!(idx >= idx_min);
                if idx != 0 {
                    idx_min = idx - 1;
                }
            }
        });

        nlist.shrink_to_fit();
        IDLState::Sparse(nlist)
    }
}

impl fmt::Display for IDLBitRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.state {
            IDLState::Sparse(list) => write!(
                f,
                "IDLBitRange (sparse values) {:?} (data) <optimised out>",
                list.len()
            ),
            IDLState::Compressed(list) => write!(
                f,
                "IDLBitRange (compressed ranges) {:?} (decompressed) <optimised out>",
                list.len()
            ),
        }
    }
}

impl fmt::Debug for IDLBitRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.state {
            IDLState::Sparse(list) => {
                write!(f, "IDLBitRange (sparse values) {:?} (data) [ ", list.len())?;
            }
            IDLState::Compressed(list) => {
                write!(f, "IDLBitRange (compressed) {:?} (decompressed) [ ", list)?;
            }
        }
        for id in self {
            write!(f, "{}, ", id)?;
        }
        write!(f, "]")
    }
}

/// An ID List of `u64` values, that uses a compressed representation of `u64` to
/// speed up set operations, improve cpu cache behaviour and consume less memory.
///
/// This is essentially a `Vec<u64>`, but requires less storage with large values
/// and natively supports logical operations for set manipulation. Today this
/// supports And, Or, AndNot. Future types may be added such as xor.
///
/// # Examples
/// ```
/// use idlset::v2::IDLBitRange;
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
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename = "IDLV2")]
pub struct IDLBitRange {
    #[serde(rename = "t")]
    state: IDLState,
}

impl IDLRange {
    fn new(range: u64, mask: u64) -> Self {
        IDLRange { range, mask }
    }

    #[inline(always)]
    fn push_id(&mut self, value: u64) {
        self.mask ^= 1 << value;
    }
}

impl Default for IDLBitRange {
    /// Construct a new, empty set.
    fn default() -> Self {
        IDLBitRange {
            state: IDLState::Sparse(SmallVec::with_capacity(0)),
        }
    }
}

impl PartialEq for IDLBitRange {
    fn eq(&self, other: &Self) -> bool {
        let x = self & other;
        debug_assert!(other.len() == self.len() && other.len() == x.len());
        x.len() == other.len()
    }
}

impl IDLBitRange {
    /// Construct a new, empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a set containing a single initial value. This is a special
    /// use case for database indexing where single value equality indexes are
    /// store uncompressed on disk.
    pub fn from_u64(id: u64) -> Self {
        IDLBitRange {
            state: IDLState::Sparse(smallvec![id]),
        }
    }

    /// Show if this IDL set contains no elements
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Show if this data set is sparsely packed.
    pub fn is_sparse(&self) -> bool {
        matches!(self.state, IDLState::Sparse(_))
    }

    /// Show if this data set is compressed.
    pub fn is_compressed(&self) -> bool {
        matches!(self.state, IDLState::Compressed(_))
    }

    /// Returns the number of ids in the set. This operation iterates over
    /// the set, decompressing it to count the ids, which MAY be slow. If
    /// you want to see if the set is empty, us `is_empty()`
    #[inline(always)]
    pub fn len(&self) -> usize {
        match &self.state {
            IDLState::Sparse(list) => list.len(),
            IDLState::Compressed(list) => list
                .iter()
                .fold(0, |acc, i| (i.mask.count_ones() as usize) + acc),
        }
    }

    fn len_ranges(&self) -> usize {
        match &self.state {
            IDLState::Sparse(_list) => 0,
            IDLState::Compressed(list) => list.len(),
        }
    }

    /// Returns if the number of ids in this set exceed this threshold. While
    /// this must iterate to determine if this is true, since we shortcut return
    /// in the check, on long sets we will not iterate over the complete content
    /// making it faster than `len() < thresh`.
    ///
    /// Returns true if the set is smaller than threshold.
    #[inline(always)]
    pub fn below_threshold(&self, threshold: usize) -> bool {
        match &self.state {
            IDLState::Sparse(list) => list.len() < threshold,
            IDLState::Compressed(list) => {
                let mut ic: usize = 0;
                for i in list.iter() {
                    ic += i.mask.count_ones() as usize;
                    if ic >= threshold {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Sum all the values contained into this set to yield a single result.
    #[inline(always)]
    pub fn sum(&self) -> u64 {
        match &self.state {
            IDLState::Sparse(list) => list.iter().fold(0, |acc, x| x + acc),
            IDLState::Compressed(list) => IDLBitRangeIterComp::new(list).fold(0, |acc, x| x + acc),
        }
    }

    /// Returns `true` if the id `u64` value exists within the set.
    pub fn contains(&self, id: u64) -> bool {
        match &self.state {
            IDLState::Sparse(list) => list.as_slice().binary_search(&id).is_ok(),
            IDLState::Compressed(list) => {
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;
                let mask = 1 << bvalue;

                if let Ok(idx) = list.binary_search_by(|v| v.range.cmp(&range)) {
                    // We know this is safe and exists due to binary search.
                    let existing = unsafe { list.get_unchecked(idx) };
                    (existing.mask & mask) > 0
                } else {
                    false
                }
            }
        }
    }

    /// Push an id into the set. The value is appended onto the tail of the set.
    /// You probably want `insert_id` instead.
    ///
    /// # Safety
    ///
    /// Failure to insert sorted data will corrupt the set, and cause subsequent
    /// set operations to yield incorrect and inconsistent results.
    pub unsafe fn push_id(&mut self, id: u64) {
        match &mut self.state {
            IDLState::Sparse(list) => {
                list.push(id);
            }
            IDLState::Compressed(list) => {
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;

                if let Some(last) = list.last_mut() {
                    debug_assert!(id >= last.range);
                    if last.range == range {
                        // Insert the bit.
                        (*last).push_id(bvalue);
                        return;
                    }
                }
                // Range is greater, or the set is empty.
                list.push(IDLRange::new(range, 1 << bvalue));
            }
        } // end match self.state.
    }

    /// Insert an id into the set, correctly sorted.
    pub fn insert_id(&mut self, id: u64) {
        match &mut self.state {
            IDLState::Sparse(list) => {
                let r = list.binary_search(&id);
                // In the ok case, it's already present.
                if let Err(idx) = r {
                    list.insert(idx, id);
                }
            }
            IDLState::Compressed(list) => {
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;

                let candidate = IDLRange::new(range, 1 << bvalue);

                let r = list.binary_search(&candidate);
                match r {
                    Ok(idx) => {
                        let existing = list.get_mut(idx).unwrap();
                        existing.mask |= candidate.mask;
                    }
                    Err(idx) => {
                        list.insert(idx, candidate);
                    }
                };
            }
        }
    }

    /// Remove an id from the set, leaving it correctly sorted.
    ///
    /// If the value is not present, no action is taken.
    pub fn remove_id(&mut self, id: u64) {
        match &mut self.state {
            IDLState::Sparse(list) => {
                let r = list.binary_search(&id);
                if let Ok(idx) = r {
                    list.remove(idx);
                };
            }
            IDLState::Compressed(list) => {
                // Determine our range
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;

                // We make a dummy range and mask to find our range
                let candidate = IDLRange::new(range, 1 << bvalue);

                if let Ok(idx) = list.binary_search(&candidate) {
                    // The listed range would contain our bit.
                    // So we need to remove this, leaving all other bits in place.
                    //
                    // To do this, we not the candidate, so all other bits remain,
                    // then we perform and &= so that the existing bits survive.
                    let existing = list.get_mut(idx).unwrap();

                    existing.mask &= !candidate.mask;

                    if existing.mask == 0 {
                        // No more items in this range, remove it.
                        list.remove(idx);
                    }
                }
            }
        } // end match
    }

    /// Compress this IDL set. This may be needed if you wish to force a set
    /// to be compressed, even if the adaptive behaviour has not compressed
    /// it for you.
    pub fn compress(&mut self) {
        if self.is_compressed() {
            return;
        }
        let mut prev_state = IDLState::Compressed(Vec::with_capacity(0));
        std::mem::swap(&mut prev_state, &mut self.state);
        match prev_state {
            IDLState::Sparse(list) => list.into_iter().for_each(|i| unsafe {
                self.push_id(i);
            }),
            IDLState::Compressed(_) => panic!("Unexpected state!"),
        }
    }

    /// If it is viable, attempt to compress this IDL. This operation will scan the
    /// full IDL, so it's not recommended to call this frequently. Generally the use of
    /// `from_iter` will already make the correct decision for you.
    pub fn maybe_compress(&mut self) -> bool {
        let maybe_state = if let IDLState::Sparse(list) = &self.state {
            if list.len() < AVG_RANGE_COMP_REQ {
                None
            } else {
                let mut maybe = IDLBitRange {
                    state: IDLState::Compressed(Vec::with_capacity(0)),
                };
                list.iter().for_each(|id| unsafe { maybe.push_id(*id) });

                if maybe.len_ranges() > 0
                    && (maybe.len() / maybe.len_ranges()) >= AVG_RANGE_COMP_REQ
                {
                    let IDLBitRange { mut state } = maybe;
                    state.shrink_to_fit();
                    Some(state)
                } else {
                    None
                }
            }
        } else {
            None
        };
        if let Some(mut new_state) = maybe_state {
            std::mem::swap(&mut self.state, &mut new_state);
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn bitand_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Sparse(lhs), IDLState::Sparse(rhs)) => {
                // Fast path if there is a really large difference in the sizes.
                let state = if !lhs.is_empty() && (rhs.len() / lhs.len()) >= FAST_PATH_BST_RATIO {
                    IDLState::sparse_bitand_fast_path(lhs.as_slice(), rhs.as_slice())
                } else if !rhs.is_empty() && (lhs.len() / rhs.len()) >= FAST_PATH_BST_RATIO {
                    IDLState::sparse_bitand_fast_path(rhs.as_slice(), lhs.as_slice())
                } else {
                    let x = if rhs.len() > lhs.len() {
                        rhs.len()
                    } else {
                        lhs.len()
                    };

                    let mut nlist = SmallVec::with_capacity(x);

                    let mut liter = lhs.iter();
                    let mut riter = rhs.iter();

                    let mut lnext = liter.next();
                    let mut rnext = riter.next();

                    while lnext.is_some() && rnext.is_some() {
                        let l = lnext.unwrap();
                        let r = rnext.unwrap();

                        match l.cmp(r) {
                            Ordering::Equal => {
                                nlist.push(*l);
                                lnext = liter.next();
                                rnext = riter.next();
                            }
                            Ordering::Less => {
                                lnext = liter.next();
                            }
                            Ordering::Greater => {
                                rnext = riter.next();
                            }
                        }
                    }

                    nlist.shrink_to_fit();

                    IDLState::Sparse(nlist)
                };

                IDLBitRange { state }
            }
            (IDLState::Sparse(sparselist), IDLState::Compressed(list))
            | (IDLState::Compressed(list), IDLState::Sparse(sparselist)) => {
                // Could be be better to decompress instead? This currently
                // assumes sparse is MUCH smaller than compressed ...

                let mut nlist = SmallVec::with_capacity(sparselist.len());
                let mut idx_min = 0;

                sparselist.iter().for_each(|id| {
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;
                    let mask = 1 << bvalue;
                    let (_, partition) = list.split_at(idx_min);
                    if let Ok(idx) = partition.binary_search_by(|v| v.range.cmp(&range)) {
                        debug_assert!(
                            Ok(idx + idx_min) == list.binary_search_by(|v| v.range.cmp(&range))
                        );
                        let idx = idx + idx_min;
                        // We know this is safe and exists due to binary search.
                        let existing = unsafe { list.get_unchecked(idx) };
                        if (existing.mask & mask) > 0 {
                            nlist.push(*id);
                        }
                        debug_assert!(idx >= idx_min);
                        idx_min = idx;
                    }
                });

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Sparse(nlist),
                }
            }
            (IDLState::Compressed(list1), IDLState::Compressed(list2)) => {
                let mut nlist = Vec::with_capacity(0);
                let mut liter = list1.iter();
                let mut riter = list2.iter();

                let mut lnextrange = liter.next();
                let mut rnextrange = riter.next();

                while lnextrange.is_some() && rnextrange.is_some() {
                    let l = lnextrange.unwrap();
                    let r = rnextrange.unwrap();

                    match l.range.cmp(&r.range) {
                        Ordering::Equal => {
                            let mask = l.mask & r.mask;
                            if mask > 0 {
                                let newrange = IDLRange::new(l.range, mask);
                                nlist.push(newrange);
                            }
                            lnextrange = liter.next();
                            rnextrange = riter.next();
                        }
                        Ordering::Less => {
                            lnextrange = liter.next();
                        }
                        Ordering::Greater => {
                            rnextrange = riter.next();
                        }
                    }
                }
                if nlist.is_empty() {
                    IDLBitRange::new()
                } else {
                    nlist.shrink_to_fit();
                    IDLBitRange {
                        state: IDLState::Compressed(nlist),
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn bitor_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Sparse(lhs), IDLState::Sparse(rhs)) => {
                // If one is much smaller, we can clone the larger and just insert.
                let state = if !lhs.is_empty() && (rhs.len() / lhs.len()) >= FAST_PATH_BST_RATIO {
                    IDLState::sparse_bitor_fast_path(lhs.as_slice(), rhs.as_slice())
                } else if !rhs.is_empty() && (lhs.len() / rhs.len()) >= FAST_PATH_BST_RATIO {
                    IDLState::sparse_bitor_fast_path(rhs.as_slice(), lhs.as_slice())
                } else {
                    let mut nlist = SmallVec::with_capacity(lhs.len() + rhs.len());
                    let mut liter = lhs.iter();
                    let mut riter = rhs.iter();

                    let mut lnext = liter.next();
                    let mut rnext = riter.next();

                    while lnext.is_some() && rnext.is_some() {
                        let l = lnext.unwrap();
                        let r = rnext.unwrap();

                        let n = match l.cmp(r) {
                            Ordering::Equal => {
                                lnext = liter.next();
                                rnext = riter.next();
                                l
                            }
                            Ordering::Less => {
                                lnext = liter.next();
                                l
                            }
                            Ordering::Greater => {
                                rnext = riter.next();
                                r
                            }
                        };
                        nlist.push(*n);
                    }

                    while lnext.is_some() {
                        let l = lnext.unwrap();
                        nlist.push(*l);
                        lnext = liter.next();
                    }

                    while rnext.is_some() {
                        let r = rnext.unwrap();
                        nlist.push(*r);
                        rnext = riter.next();
                    }

                    nlist.shrink_to_fit();

                    IDLState::Sparse(nlist)
                };

                /*
                // Failed experiment XD
                let mut nlist = SmallVec::with_capacity(lhs.len() + rhs.len());
                nlist.extend_from_slice(lhs.as_slice());
                nlist.extend_from_slice(rhs.as_slice());
                nlist.as_mut_slice().sort_unstable();
                nlist.dedup();
                nlist.shrink_to_fit();

                let state = IDLState::Sparse(nlist);
                */

                IDLBitRange { state }
            }
            (IDLState::Sparse(sparselist), IDLState::Compressed(list))
            | (IDLState::Compressed(list), IDLState::Sparse(sparselist)) => {
                // Duplicate the compressed set.
                let mut list = list.clone();
                let mut idx_min = 0;

                sparselist.iter().for_each(|id| {
                    // Same algo as insert id.
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;

                    let candidate = IDLRange::new(range, 1 << bvalue);

                    let (_, partition) = list.split_at(idx_min);
                    let r = partition.binary_search(&candidate);
                    match r {
                        Ok(idx) => {
                            debug_assert!(Ok(idx + idx_min) == list.binary_search(&candidate));
                            let idx = idx + idx_min;
                            let existing = list.get_mut(idx).unwrap();
                            existing.mask |= candidate.mask;
                            debug_assert!(idx >= idx_min);
                            idx_min = idx;
                        }
                        Err(idx) => {
                            debug_assert!(Err(idx + idx_min) == list.binary_search(&candidate));
                            let idx = idx + idx_min;
                            list.insert(idx, candidate);
                            debug_assert!(idx >= idx_min);
                            if idx != 0 {
                                idx_min = idx - 1;
                            }
                        }
                    };
                });

                list.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Compressed(list),
                }
            }
            (IDLState::Compressed(list1), IDLState::Compressed(list2)) => {
                let llen = list1.len();
                let rlen = list2.len();

                let mut nlist = Vec::with_capacity(llen + rlen);

                let mut liter = list1.iter();
                let mut riter = list2.iter();

                let mut lnextrange = liter.next();
                let mut rnextrange = riter.next();

                while lnextrange.is_some() && rnextrange.is_some() {
                    let l = lnextrange.unwrap();
                    let r = rnextrange.unwrap();

                    let (range, mask) = match l.range.cmp(&r.range) {
                        Ordering::Equal => {
                            lnextrange = liter.next();
                            rnextrange = riter.next();
                            (l.range, l.mask | r.mask)
                        }
                        Ordering::Less => {
                            lnextrange = liter.next();
                            (l.range, l.mask)
                        }
                        Ordering::Greater => {
                            rnextrange = riter.next();
                            (r.range, r.mask)
                        }
                    };
                    let newrange = IDLRange::new(range, mask);
                    nlist.push(newrange);
                }

                while lnextrange.is_some() {
                    let l = lnextrange.unwrap();

                    let newrange = IDLRange::new(l.range, l.mask);
                    nlist.push(newrange);
                    lnextrange = liter.next();
                }

                while rnextrange.is_some() {
                    let r = rnextrange.unwrap();

                    let newrange = IDLRange::new(r.range, r.mask);
                    nlist.push(newrange);
                    rnextrange = riter.next();
                }

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Compressed(nlist),
                }
            }
        } // end match
    }

    #[inline(always)]
    fn bitandnot_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Sparse(lhs), IDLState::Sparse(rhs)) => {
                let mut nlist = SmallVec::with_capacity(lhs.len());

                let mut liter = lhs.iter();
                let mut riter = rhs.iter();

                let mut lnext = liter.next();
                let mut rnext = riter.next();

                while lnext.is_some() && rnext.is_some() {
                    let l = lnext.unwrap();
                    let r = rnext.unwrap();

                    match l.cmp(r) {
                        Ordering::Equal => {
                            // It's in right, so exclude.
                            lnext = liter.next();
                            rnext = riter.next();
                        }
                        Ordering::Less => {
                            nlist.push(*l);
                            lnext = liter.next();
                        }
                        Ordering::Greater => {
                            rnext = riter.next();
                        }
                    }
                }

                // Drain remaining left elements.
                while lnext.is_some() {
                    nlist.push(*lnext.unwrap());
                    lnext = liter.next();
                }

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Sparse(nlist),
                }
            }
            (IDLState::Sparse(sparselist), IDLState::Compressed(list)) => {
                let mut nlist = SmallVec::with_capacity(sparselist.len());

                let mut idx_min = 0;
                sparselist.iter().for_each(|id| {
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;
                    let mask = 1 << bvalue;
                    let (_, partition) = list.split_at(idx_min);
                    match partition.binary_search_by(|v| v.range.cmp(&range)) {
                        Ok(idx) => {
                            debug_assert!(
                                Ok(idx + idx_min) == list.binary_search_by(|v| v.range.cmp(&range))
                            );
                            let idx = idx + idx_min;
                            // Okay the range is there ...
                            let existing = unsafe { list.get_unchecked(idx) };
                            if (existing.mask & mask) == 0 {
                                // It didn't match the mask, so it's not in right.
                                nlist.push(*id);
                            }
                            debug_assert!(idx >= idx_min);
                            // Avoid an edge case where idx_min >= list.len
                            idx_min = idx;
                        }
                        Err(idx) => {
                            debug_assert!(
                                Err(idx + idx_min)
                                    == list.binary_search_by(|v| v.range.cmp(&range))
                            );
                            let idx = idx + idx_min;
                            // Didn't find the range, push.
                            nlist.push(*id);
                            if idx != 0 {
                                idx_min = idx - 1;
                            }
                        }
                    }
                });

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Sparse(nlist),
                }
            }
            (IDLState::Compressed(list), IDLState::Sparse(sparselist)) => {
                let mut nlist = list.clone();
                // This assumes the sparse is much much smaller and fragmented.
                // Alternately, we could compress right, and use the lower algo.
                let mut idx_min = 0;
                sparselist.iter().for_each(|id| {
                    // same algo as remove.
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;
                    let candidate = IDLRange::new(range, 1 << bvalue);
                    let (_, partition) = nlist.split_at(idx_min);
                    if let Ok(idx) = partition.binary_search(&candidate) {
                        debug_assert!(Ok(idx + idx_min) == nlist.binary_search(&candidate));
                        let idx = idx + idx_min;
                        let existing = nlist.get_mut(idx).unwrap();
                        existing.mask &= !candidate.mask;
                        if existing.mask == 0 {
                            nlist.remove(idx);
                        }
                        debug_assert!(idx >= idx_min);
                        idx_min = idx;
                    }
                });

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Compressed(nlist),
                }
            }
            (IDLState::Compressed(list1), IDLState::Compressed(list2)) => {
                // Worst case - we exclude nothing from list1.
                let mut nlist = Vec::with_capacity(list1.len());

                let mut liter = list1.iter();
                let mut riter = list2.iter();

                let mut lnextrange = liter.next();
                let mut rnextrange = riter.next();

                while lnextrange.is_some() && rnextrange.is_some() {
                    let l = lnextrange.unwrap();
                    let r = rnextrange.unwrap();

                    match l.range.cmp(&r.range) {
                        Ordering::Equal => {
                            let mask = l.mask & (!r.mask);
                            if mask > 0 {
                                let newrange = IDLRange::new(l.range, mask);
                                nlist.push(newrange);
                            }
                            lnextrange = liter.next();
                            rnextrange = riter.next();
                        }
                        Ordering::Less => {
                            // if the left range isn't in the right, just push it to the set and move
                            // on.
                            nlist.push(l.clone());
                            lnextrange = liter.next();
                        }
                        Ordering::Greater => {
                            rnextrange = riter.next();
                        }
                    }
                }

                // Drain the remaining left ranges into the set.
                while lnextrange.is_some() {
                    let l = lnextrange.unwrap();

                    let newrange = IDLRange::new(l.range, l.mask);
                    nlist.push(newrange);
                    lnextrange = liter.next();
                }

                nlist.shrink_to_fit();

                IDLBitRange {
                    state: IDLState::Compressed(nlist),
                }
            }
        } // end match
    }
}

impl FromIterator<u64> for IDLBitRange {
    /// Build an IDLBitRange from at iterator. If you provide a sorted input, a fast append
    /// mode is used. Unsorted inputs use a slower insertion sort method
    /// instead. Based on the provided input, this will adaptively choose sparse or compressed
    /// storage of the dataset.
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        let mut new_sparse = IDLBitRange {
            state: IDLState::Sparse(SmallVec::with_capacity(lower_bound)),
        };

        let mut max_seen = 0;
        iter.for_each(|i| {
            if i >= max_seen {
                // if we have a sorted list, we can take a fast append path.
                unsafe {
                    new_sparse.push_id(i);
                }
                max_seen = i;
            } else {
                // if not, we have to bst each time to get the right place.
                new_sparse.insert_id(i);
            }
        });

        if !new_sparse.maybe_compress() {
            // If the compression didn't occur, trim the vec anyway.
            new_sparse.state.shrink_to_fit();
        }
        new_sparse
    }
}

impl BitAnd for &IDLBitRange {
    type Output = IDLBitRange;

    /// Perform an And (intersection) operation between two sets. This returns
    /// a new set containing the results.
    ///
    /// # Examples
    /// ```
    /// # use idlset::v2::IDLBitRange;
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
    /// # use idlset::v2::IDLBitRange;
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
    /// # use idlset::v2::IDLBitRange;
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
    /// # use idlset::v2::IDLBitRange;
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
    /// use idlset::{v2::IDLBitRange, AndNot};
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
    /// use idlset::{v2::IDLBitRange, AndNot};
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
    /// use idlset::{v2::IDLBitRange, AndNot};
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
    /// use idlset::{v2::IDLBitRange, AndNot};
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

/// An internal component for compressed idl set iteration.
#[derive(Debug)]
pub struct IDLBitRangeIterComp<'a> {
    // rangeiter: std::vec::IntoIter<IDLRange>,
    rangeiter: slice::Iter<'a, IDLRange>,
    currange: Option<&'a IDLRange>,
    curbit: u64,
}

impl<'a> Iterator for IDLBitRangeIterComp<'a> {
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

impl<'a> IDLBitRangeIterComp<'a> {
    fn new(data: &'a [IDLRange]) -> Self {
        let mut rangeiter = data.iter();
        let currange = rangeiter.next();
        IDLBitRangeIterComp {
            rangeiter,
            currange,
            curbit: 0,
        }
    }
}

/// An iterator over the content of an IDLBitRange.
#[derive(Debug)]
pub enum IDLBitRangeIter<'a> {
    /// The sparse
    Sparse(slice::Iter<'a, u64>),
    /// The compressed
    Compressed(IDLBitRangeIterComp<'a>),
}

impl<'a> Iterator for IDLBitRangeIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            IDLBitRangeIter::Sparse(i) => i.next().copied(),
            IDLBitRangeIter::Compressed(i) => i.next(),
        }
    }
}

impl<'a> IntoIterator for &'a IDLBitRange {
    type Item = u64;
    type IntoIter = IDLBitRangeIter<'a>;

    fn into_iter(self) -> IDLBitRangeIter<'a> {
        match &self.state {
            IDLState::Sparse(list) => IDLBitRangeIter::Sparse((list).into_iter()),
            IDLState::Compressed(list) => {
                let mut liter = (list).iter();
                let nrange = liter.next();
                IDLBitRangeIter::Compressed(IDLBitRangeIterComp {
                    rangeiter: liter,
                    currange: nrange,
                    curbit: 0,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IDLBitRange;
    use super::IDLState;
    use super::AVG_RANGE_COMP_REQ;
    use crate::AndNot;
    use std::iter::FromIterator;

    #[test]
    fn test_struct_size() {
        let ibrsize = std::mem::size_of::<IDLBitRange>();
        eprintln!("Struct size {:?}", ibrsize);
        assert!(ibrsize <= 64);
        // assert!(ibrsize <= 128);
    }

    #[test]
    fn test_empty() {
        let idl_a = IDLBitRange::new();
        assert!(idl_a.is_empty());
        assert!(idl_a.len() == 0);
    }

    #[test]
    fn test_push_id_contains() {
        let mut idl_a = IDLBitRange::new();
        unsafe { idl_a.push_id(0) };
        assert!(idl_a.contains(0));
        assert!(idl_a.len() == 1);

        unsafe { idl_a.push_id(1) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.is_sparse());
        assert!(idl_a.len() == 2);

        unsafe { idl_a.push_id(2) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.contains(2));
        assert!(idl_a.is_sparse());
        assert!(idl_a.len() == 3);

        unsafe { idl_a.push_id(128) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.contains(2));
        assert!(idl_a.contains(128));
        assert!(idl_a.is_sparse());
        assert!(idl_a.len() == 4);
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
    fn test_sparse_remove_id() {
        let mut idl_a = IDLBitRange::new();
        idl_a.remove_id(100);
        assert!(idl_a.len() == 0);

        let mut idl_a = IDLBitRange::from_iter(vec![100]);
        idl_a.remove_id(100);
        assert!(idl_a.len() == 0);

        let mut idl_a = IDLBitRange::from_iter(vec![100, 101]);
        idl_a.remove_id(101);
        assert!(idl_a.len() == 1);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 64, 68]);
        idl_a.remove_id(1);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 64, 68]);
        idl_a.remove_id(2);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 68]);
        idl_a.remove_id(64);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 64]);
        idl_a.remove_id(68);
        assert_eq!(idl_a, idl_expect);
    }

    #[test]
    fn test_compressed_remove_id() {
        let mut idl_a = IDLBitRange::new();
        idl_a.compress();
        assert!(idl_a.is_compressed());
        idl_a.remove_id(100);
        assert!(idl_a.len() == 0);

        let mut idl_a = IDLBitRange::from_iter(vec![100]);
        idl_a.compress();
        idl_a.remove_id(100);
        assert!(idl_a.len() == 0);

        let mut idl_a = IDLBitRange::from_iter(vec![100, 101]);
        idl_a.compress();
        idl_a.remove_id(101);
        assert!(idl_a.len() == 1);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let mut idl_expect = IDLBitRange::from_iter(vec![2, 64, 68]);
        idl_a.compress();
        idl_expect.compress();
        idl_a.remove_id(1);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 64, 68]);
        idl_a.compress();
        idl_expect.compress();
        idl_a.remove_id(2);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 68]);
        idl_a.compress();
        idl_expect.compress();
        idl_a.remove_id(64);
        assert_eq!(idl_a, idl_expect);

        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 64, 68]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 64]);
        idl_a.compress();
        idl_expect.compress();
        idl_a.remove_id(68);
        assert_eq!(idl_a, idl_expect);
    }

    #[test]
    fn test_range_intersection_1() {
        let idl_a = IDLBitRange::new();
        let idl_b = IDLBitRange::new();
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_2() {
        let idl_a = IDLBitRange::new();
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);

        let idl_a = IDLBitRange::from_iter(vec![2]);
        let idl_b = IDLBitRange::new();
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_3() {
        let idl_a = IDLBitRange::from_iter(vec![2]);
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::from_iter(vec![2]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);

        let idl_a = IDLBitRange::from_iter(vec![2]);
        let idl_b = IDLBitRange::from_iter(vec![128]);
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_4() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::from_iter(vec![2]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);

        let idl_a = IDLBitRange::from_iter(vec![2]);
        let idl_b = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_expect = IDLBitRange::from_iter(vec![2]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);

        let idl_a = IDLBitRange::from_iter(vec![128]);
        let idl_b = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);

        // Exercises the fast path.
        let idl_a = IDLBitRange::from_iter(vec![64, 66]);
        let idl_b = IDLBitRange::from_iter(vec![1, 2, 60, 62, 64, 69]);
        let idl_expect = IDLBitRange::from_iter(vec![64]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_5() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![4, 67]);
        let idl_expect = IDLBitRange::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_6() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 35, 64, 65, 128, 150]);
        let idl_b = IDLBitRange::from_iter(vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 3, 35, 64, 128, 150]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_7() {
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
    fn test_range_intersection_8() {
        let idl_a = IDLBitRange::from_iter(1..204800);
        let idl_b = IDLBitRange::from_iter(102400..307200);
        let idl_expect = IDLBitRange::from_iter(102400..204800);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_intersection_9() {
        let idl_a = IDLBitRange::from_iter(vec![307199]);
        let idl_b = IDLBitRange::from_iter(102400..307200);
        let idl_expect = IDLBitRange::from_iter(vec![307199]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_compressed_intersection() {
        let mut idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let mut idl_b = IDLBitRange::from_iter(1..1024);
        let mut idl_expect = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_compressed_intersection() {
        let mut idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let idl_b = IDLBitRange::from_iter(1..1024);
        let mut idl_expect = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);

        idl_a.compress();
        idl_expect.compress();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_union_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3]);
        let idl_b = IDLBitRange::from_iter(vec![2]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3]);

        let idl_result = idl_a | idl_b;
        eprintln!("{:?}, {:?}", idl_result, idl_expect);
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
    fn test_range_union_compressed() {
        let mut idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let mut idl_b = IDLBitRange::from_iter(1..1024);
        let mut idl_expect = IDLBitRange::from_iter(1..1024);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_union_compressed() {
        let mut idl_a = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);
        let idl_b = IDLBitRange::from_iter(1..1024);
        let idl_expect = IDLBitRange::from_iter(1..1024);

        idl_a.compress();

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_not_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![3, 4]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 5, 6]);

        if AVG_RANGE_COMP_REQ <= 5 {
            idl_expect.compress();
        };

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_not_2() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![10]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_not_3() {
        let idl_a = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![1]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_not_4() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let idl_b = IDLBitRange::from_iter(vec![65]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 64, 66]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_not_1() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![3, 4]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 5, 6]);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_not_2() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![10]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_not_3() {
        let mut idl_a = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![1]);
        let mut idl_expect = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_not_4() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let mut idl_b = IDLBitRange::from_iter(vec![65]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 64, 66]);

        idl_a.compress();
        idl_b.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_comp_not_1() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![3, 4]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 5, 6]);

        idl_b.compress();

        if AVG_RANGE_COMP_REQ <= 5 {
            idl_expect.compress();
        };

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_comp_not_2() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![10]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);

        idl_b.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_comp_not_3() {
        let idl_a = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);
        let mut idl_b = IDLBitRange::from_iter(vec![1]);
        let idl_expect = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);

        idl_b.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_sparse_comp_not_4() {
        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let mut idl_b = IDLBitRange::from_iter(vec![65]);
        let idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 64, 66]);

        idl_b.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);

        let idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let mut idl_b = IDLBitRange::from_iter(vec![65, 80]);
        idl_b.compress();

        let mut idl_expect = IDLBitRange::from_iter(vec![80]);
        idl_expect.compress();

        let idl_result = idl_b.andnot(idl_a);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_sparse_not_1() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![3, 4]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 5, 6]);

        idl_a.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_sparse_not_2() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![10]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 4, 5, 6]);

        idl_a.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_sparse_not_3() {
        let mut idl_a = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);
        let idl_b = IDLBitRange::from_iter(vec![1]);
        let mut idl_expect = IDLBitRange::from_iter(vec![2, 3, 4, 5, 6]);

        idl_a.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_range_comp_sparse_not_4() {
        let mut idl_a = IDLBitRange::from_iter(vec![1, 2, 3, 64, 65, 66]);
        let idl_b = IDLBitRange::from_iter(vec![65]);
        let mut idl_expect = IDLBitRange::from_iter(vec![1, 2, 3, 64, 66]);

        idl_a.compress();
        idl_expect.compress();

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_sizeof_idlstate() {
        let sz = std::mem::size_of::<IDLState>();
        eprintln!("{}", sz);
    }
}
