
use std::iter::FromIterator;
use std::cmp::Ordering;
use std::ops::{BitOr, BitAnd};
use smallvec::SmallVec;
use std::slice;

/// Default number of IDL ranges to keep in stack before we spill into heap. As many
/// operations in a system like kanidm are either single item indexes (think equality)
/// or very large indexes (think pres, class), we can keep this small.
const DEFAULT_SPARSE_ALLOC: usize = 5;
// const DEFAULT_COMP_ALLOC: usize = 2;
// const DEFAULT_SPARSE_ALLOC: usize = 5 + 8;
// const DEFAULT_COMP_ALLOC: usize = 2 + 4;

/// The core representation of sets of integers in compressed format.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct IDLRange {
    pub range: u64,
    pub mask: u64,
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
enum IDLState {
    Sparse(SmallVec<[u64; DEFAULT_SPARSE_ALLOC]>),
    Compressed(Vec<IDLRange>),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct IDLBitRange {
    state: IDLState
}

impl IDLRange {
    fn new(range: u64, mask: u64) -> Self {
        IDLRange {
            range: range,
            mask: mask,
        }
    }

    #[inline(always)]
    fn push_id(&mut self, value: u64) {
        self.mask ^= 1 << value;
    }
}

impl Default for IDLBitRange {
    fn default() -> Self  {
        IDLBitRange {
            state: IDLState::Sparse(SmallVec::new())
        }
    }
}

impl IDLBitRange {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_u64(id: u64) -> Self {
        IDLBitRange {
            state: IDLState::Sparse(smallvec![id])
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn is_sparse(&self) -> bool {
        match self.state {
            IDLState::Sparse(_) => true,
            _ => false,
        }
    }

    pub fn is_compressed(&self) -> bool {
        match self.state {
            IDLState::Compressed(_) => true,
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        match &self.state {
            IDLState::Sparse(list) => list.len(),
            IDLState::Compressed(list) =>
                list
                    .iter()
                    .fold(0, |acc, i| (i.mask.count_ones() as usize) + acc),
        }
    }

    pub fn sum(&self) -> u64 {
        match &self.state {
            IDLState::Sparse(list) => list.iter().fold(0, |acc, x| x + acc),
            IDLState::Compressed(list) =>
                IDLBitRangeIter::new(&list)
                    .fold(0, |acc, x| x + acc),
        }
    }

    pub fn contains(&self, id: u64) -> bool {
        match &self.state {
            IDLState::Sparse(list) => {
                list.as_slice()
                    .binary_search(&id)
                    .is_ok()
            }
            IDLState::Compressed(list) => {
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;
                let mask = 1 << bvalue;

                if let Ok(idx) = list.binary_search_by(|v| v.range.cmp(&range)) {
                    // We know this is safe and exists due to binary search.
                    let existing = unsafe { list.get_unchecked(idx) };
                    return (existing.mask & mask) > 0;
                } else {
                    false
                }

            }
        }
    }

    pub unsafe fn push_id(&mut self, id: u64) {
        if let IDLState::Sparse(list) = &self.state {
            if list.len() >= DEFAULT_SPARSE_ALLOC {
                self.compress()
            }
        };

        match &mut self.state {
            IDLState::Sparse(list) => {
                list.push(id);
            }
            IDLState::Compressed(list) => {
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;

                if let Some(last) = list.last_mut() {
                    debug_assert!(id >= (*last).range);
                    if (*last).range == range {
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

    pub fn insert_id(&mut self, id: u64) {
        if let IDLState::Sparse(list) = &self.state {
            if list.len() >= DEFAULT_SPARSE_ALLOC {
                self.compress()
            }
        };

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
                        let mut existing = list.get_mut(idx).unwrap();
                        existing.mask |= candidate.mask;
                    }
                    Err(idx) => {
                        list.insert(idx, candidate);
                    }
                };
            }
        }
    }

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

                match list.binary_search(&candidate) {
                    Ok(idx) => {
                        // The listed range would contain our bit.
                        // So we need to remove this, leaving all other bits in place.
                        //
                        // To do this, we not the candidate, so all other bits remain,
                        // then we perform and &= so that the existing bits survive.
                        let mut existing = list.get_mut(idx).unwrap();

                        existing.mask &= !candidate.mask;

                        if existing.mask == 0 {
                            // No more items in this range, remove it.
                            list.remove(idx);
                        }
                    }
                    Err(_) => {
                        // No action required, the value is not in any range.
                    }
                }
            }
        } // end match
    }

    pub fn compress(&mut self) {
        if self.is_compressed() {
            return;
        }
        let mut prev_state = IDLState::Compressed(Vec::new());
        std::mem::swap(&mut prev_state, &mut self.state);
        match prev_state {
            IDLState::Sparse(list) => {
                list.into_iter().for_each(|i|
                    unsafe { self.push_id(i); })
            }
            IDLState::Compressed(_) => panic!("Unexpected state!"),
        }
    }

    #[inline(always)]
    fn bitand_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Sparse(lhs), IDLState::Sparse(rhs)) => {
                // If one is significantly smaller, can we do a binary search instead?
                let mut nlist = SmallVec::new();

                let mut liter = lhs.iter();
                let mut riter = rhs.iter();

                let mut lnext = liter.next();
                let mut rnext = riter.next();

                while lnext.is_some() && rnext.is_some() {
                    let l = *lnext.unwrap();
                    let r = *rnext.unwrap();

                    if l == r {
                        nlist.push(l);
                        lnext = liter.next();
                        rnext = riter.next();
                    } else if l < r {
                        lnext = liter.next();
                    } else {
                        rnext = riter.next();
                    }
                }

                IDLBitRange {
                    state: IDLState::Sparse(nlist)
                }
            }
            (IDLState::Sparse(sparselist), IDLState::Compressed(list)) |
            (IDLState::Compressed(list), IDLState::Sparse(sparselist)) => {
                let mut nlist = SmallVec::new();

                sparselist.iter().for_each(|id| {
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;
                    let mask = 1 << bvalue;
                    if let Ok(idx) = list.binary_search_by(|v| v.range.cmp(&range)) {
                        // We know this is safe and exists due to binary search.
                        let existing = unsafe { list.get_unchecked(idx) };
                        if (existing.mask & mask) > 0 {
                            nlist.push(*id);
                        }
                    }
                });

                IDLBitRange {
                    state: IDLState::Sparse(nlist)
                }
            }
            (IDLState::Compressed(list1), IDLState::Compressed(list2)) => {
                let mut nlist = Vec::new();
                let mut liter = list1.iter();
                let mut riter = list2.iter();

                let mut lnextrange = liter.next();
                let mut rnextrange = riter.next();

                while lnextrange.is_some() && rnextrange.is_some() {
                    let l = lnextrange.unwrap();
                    let r = rnextrange.unwrap();

                    if l.range == r.range {
                        let mask = l.mask & r.mask;
                        if mask > 0 {
                            let newrange = IDLRange::new(l.range, mask);
                            nlist.push(newrange);
                        }
                        lnextrange = liter.next();
                        rnextrange = riter.next();
                    } else if l.range < r.range {
                        lnextrange = liter.next();
                    } else {
                        rnextrange = riter.next();
                    }
                }
                if nlist.len() == 0 {
                    IDLBitRange::new()
                } else {
                    IDLBitRange {
                        state: IDLState::Compressed(nlist)
                    }
                }
            }
        }
    }


    #[inline(always)]
    fn bitor_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Sparse(lhs), IDLState::Sparse(rhs)) => {
                let mut nlist = SmallVec::with_capacity(lhs.len() + rhs.len());
                let mut liter = lhs.iter();
                let mut riter = rhs.iter();

                let mut lnext = liter.next();
                let mut rnext = riter.next();

                while lnext.is_some() && rnext.is_some() {
                    let l = lnext.unwrap();
                    let r = rnext.unwrap();

                    let n = if l == r {
                        lnext = liter.next();
                        rnext = riter.next();
                        l
                    } else if l < r {
                        lnext = liter.next();
                        l
                    } else {
                        rnext = riter.next();
                        r
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

                IDLBitRange {
                    state: IDLState::Sparse(nlist)
                }
            }
            (IDLState::Sparse(sparselist), IDLState::Compressed(list)) |
            (IDLState::Compressed(list), IDLState::Sparse(sparselist)) => {
                // Duplicate the compressed set.
                let mut list = list.clone();

                sparselist.iter().for_each(|id| {
                    // Same algo as insert id.
                    let bvalue: u64 = id % 64;
                    let range: u64 = id - bvalue;

                    let candidate = IDLRange::new(range, 1 << bvalue);

                    let r = list.binary_search(&candidate);
                    match r {
                        Ok(idx) => {
                            let mut existing = list.get_mut(idx).unwrap();
                            existing.mask |= candidate.mask;
                        }
                        Err(idx) => {
                            list.insert(idx, candidate);
                        }
                    };
                });

                IDLBitRange {
                    state: IDLState::Compressed(list)
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
                IDLBitRange {
                    state: IDLState::Compressed(nlist)
                }
            }
        } // end match
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
            state: IDLState::Sparse(SmallVec::with_capacity(lower_bound))
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

impl<'a> IDLBitRangeIter<'a> {
    fn new(data: &'a [IDLRange]) -> Self {
        let mut rangeiter = data.into_iter();
        let currange = rangeiter.next();
        IDLBitRangeIter {
            rangeiter,
            currange,
            curbit: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IDLBitRange;
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
        let mut idl_a = IDLBitRange::new();
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
        let idl_expect = IDLBitRange::from_iter(vec![
            2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900,
        ]);

        idl_a.compress();

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
        let mut idl_b = IDLBitRange::from_iter(1..1024);
        let mut idl_expect = IDLBitRange::from_iter(1..1024);

        idl_a.compress();

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }
}
