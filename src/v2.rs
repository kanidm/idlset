use std::iter::FromIterator;
use std::cmp::Ordering;
use std::ops::BitAnd;

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
    Empty,
    Single(u64),
    // Sparse(Vec<u64>)
    Compressed(Vec<IDLRange>)
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
            state: IDLState::Empty
        }
    }
}

impl IDLBitRange {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        match &self.state {
            IDLState::Empty => true,
            IDLState::Single(_) => false,
            IDLState::Compressed(list) => self.len() == 0,
        }
    }

    pub(crate) fn is_single(&self) -> bool {
        match self.state {
            IDLState::Single(_) => true,
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
        IDLState::Empty => 0,
        IDLState::Single(value) => 1,
        IDLState::Compressed(list) =>
            list
                .iter()
                .fold(0, |acc, i| (i.mask.count_ones() as usize) + acc),
        }
    }

    pub fn contains(&self, id: u64) -> bool {
        match &self.state {
            IDLState::Empty => false,
            IDLState::Single(value) => *value == id,
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
        self.state = match &mut self.state {
            IDLState::Empty =>
                IDLState::Single(id),
            IDLState::Single(prev) => {
                let prev = *prev;
                debug_assert!(id > prev);
                // Given prev, append id.
                let bvalue: u64 = prev % 64;
                let range: u64 = prev - bvalue;
                let mut candidate = IDLRange::new(range, 1 << bvalue);

                let bvalue2: u64 = id % 64;
                let range2: u64 = id - bvalue2;
                if range == range2 {
                    candidate.push_id(bvalue2);
                    IDLState::Compressed(vec![candidate])
                } else {
                    let candidate2 = IDLRange::new(range2, 1 << bvalue2);
                    IDLState::Compressed(vec![candidate, candidate2])
                }
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
                    } else {
                        list.push(IDLRange::new(range, 1 << bvalue));
                        return;
                    }
                } else {
                    panic!("Inserted value would corrupt the set!");
                }
            }
        } // end match self.state.
    }

    pub fn insert_id(&mut self, id: u64) {
        self.state = match &mut self.state {
            IDLState::Empty =>
                IDLState::Single(id),
            IDLState::Single(prev) => {
                let prev = *prev;
                let bvalue: u64 = prev % 64;
                let range: u64 = prev - bvalue;
                let bvalue2: u64 = id % 64;
                let range2: u64 = id - bvalue2;

                let mut candidate = IDLRange::new(range, 1 << bvalue);

                if range == range2 {
                    candidate.push_id(bvalue2);
                    IDLState::Compressed(vec![candidate])
                } else {
                    let candidate2 = IDLRange::new(range2, 1 << bvalue2);
                    if range > range2 {
                        IDLState::Compressed(vec![candidate2, candidate])
                    } else {
                        IDLState::Compressed(vec![candidate, candidate2])
                    }
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
                return;
            }
        }
    }

    pub fn remove_id(&mut self, id: u64) {
        self.state = match &mut self.state {
            IDLState::Empty => return,
            IDLState::Single(prev) => IDLState::Empty,
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
                if list.len() == 0 {
                    IDLState::Empty
                } else {
                    return;
                }
            }
        }
    }

    #[inline(always)]
    fn bitand_inner(&self, rhs: &Self) -> Self {
        match (&self.state, &rhs.state) {
            (IDLState::Empty, _) |
            (_, IDLState::Empty) => IDLBitRange::new(),
            (IDLState::Single(v1), IDLState::Single(v2)) => {
                if v1 == v2 {
                    IDLBitRange {
                        state: IDLState::Single(*v1)
                    }
                } else {
                    IDLBitRange::new()
                }
            }
            (IDLState::Single(id), IDLState::Compressed(list)) |
            (IDLState::Compressed(list), IDLState::Single(id)) => {
                let id = *id;
                let bvalue: u64 = id % 64;
                let range: u64 = id - bvalue;
                let mask = 1 << bvalue;

                if let Ok(idx) = list.binary_search_by(|v| v.range.cmp(&range)) {
                    // We know this is safe and exists due to binary search.
                    let existing = unsafe { list.get_unchecked(idx) };
                    if (existing.mask & mask) > 0 {
                        return IDLBitRange {
                            state: IDLState::Single(id)
                        };
                    }
                }
                // Catch all, not found.
                IDLBitRange::new()
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
}

impl FromIterator<u64> for IDLBitRange {
    /// Build an IDLBitRange from at iterator. If you provide a sorted input, a fast append
    /// mode is used. Unsorted inputs use a slower insertion sort method
    /// instead.
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        // We could check size hint, but it doesn't really work well for us here ...
        let mut new = IDLBitRange::new();

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

#[cfg(test)]
mod tests {
    use super::IDLBitRange;
    use std::iter::FromIterator;

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
        assert!(idl_a.is_single());
        assert!(idl_a.len() == 1);

        unsafe { idl_a.push_id(1) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.is_compressed());
        assert!(idl_a.len() == 2);

        unsafe { idl_a.push_id(2) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.contains(2));
        assert!(idl_a.is_compressed());
        assert!(idl_a.len() == 3);

        unsafe { idl_a.push_id(128) };
        assert!(idl_a.contains(0));
        assert!(idl_a.contains(1));
        assert!(idl_a.contains(2));
        assert!(idl_a.contains(128));
        assert!(idl_a.is_compressed());
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
    fn test_remove_id() {
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
}
