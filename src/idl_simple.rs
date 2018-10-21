use idlset::IDL;
use idlset::AndNot;
use std::ops::{BitAnd, BitOr};
use std::slice;
use std::iter::FromIterator;

#[derive(Debug, PartialEq)]
pub struct IDLSimple(Vec<u64>);

impl IDLSimple {
    pub fn new() -> Self {
        IDLSimple(Vec::with_capacity(128))
    }

    pub fn from_u64(id: u64) -> Self {
        let mut new = IDLSimple::new();
        new.push_id(id);
        new
    }

    fn bstbitand(&self, candidate: &u64) -> Self {
        let mut result = IDLSimple::new();
        if let Ok(_idx) = self.0.binary_search(candidate) {
            result.0.push(*candidate);
        };
        result
    }
}

impl IDL for IDLSimple {
    fn push_id(&mut self, value: u64) {
        let &mut IDLSimple(ref mut list) = self;
        list.push(value)
    }

    fn len(&self) -> usize {
        let &IDLSimple(ref list) = self;
        list.len()
    }

}

impl FromIterator<u64> for IDLSimple {
    fn from_iter<I: IntoIterator<Item=u64>>(iter: I) -> Self {
        let mut list = Vec::with_capacity(8);
        for i in iter {
            list.push(i);
        }
        IDLSimple(list)
    }
}

#[derive(Debug)]
pub struct IDLSimpleIter<'b> {
    simpleiter: slice::Iter<'b, u64>,
}

impl<'b> Iterator for IDLSimpleIter<'b> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if let Some(id) = self.simpleiter.next() {
            Some(id.clone())
        } else {
            None
        }
    }
}

impl<'b> IntoIterator for &'b IDLSimple {
    type Item = u64;
    type IntoIter = IDLSimpleIter<'b>;

    fn into_iter(self) -> Self::IntoIter {
        IDLSimpleIter {
            simpleiter: (&self.0).into_iter(),
        }
    }
}

impl BitAnd for IDLSimple
{
    type Output = Self;

    fn bitand(self, other: Self) -> Self {

        if self.0.len() == 1 {
            return other.bstbitand(self.0.first().unwrap());
        } else if other.0.len() == 1 {
            return self.bstbitand(other.0.first().unwrap());
        }

        let IDLSimple(rhs) = other;
        let IDLSimple(lhs) = self;

        let mut result = IDLSimple::new();

        let mut liter = lhs.iter();
        let mut riter = rhs.iter();

        let mut lnext = liter.next();
        let mut rnext = riter.next();

        while lnext.is_some() && rnext.is_some() {
            let l = lnext.unwrap();
            let r = rnext.unwrap();

            if l == r {
                // result.push_id(l.clone());
                result.push_id(*l);
                lnext = liter.next();
                rnext = riter.next();
            } else if l < r {
                lnext = liter.next();
            } else {
                rnext = riter.next();
            }

        }
        result

    }
}

impl BitOr for IDLSimple
{
    type Output = Self;

    fn bitor(self, IDLSimple(rhs): Self) -> Self {
        let IDLSimple(lhs) = self;
        let mut result = IDLSimple::new();

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
            result.push_id(n.clone());

        };

        while lnext.is_some() {
            let l = lnext.unwrap();
            result.push_id(l.clone());
            lnext = liter.next();
        }

        while rnext.is_some() {
            let r = rnext.unwrap();
            result.push_id(r.clone());
            rnext = riter.next();
        }
        result
    }
}

impl AndNot for IDLSimple {
    type Output = Self;

    fn andnot(self, IDLSimple(rhs): Self) -> Self {
        let IDLSimple(lhs) = self;
        let mut result = IDLSimple::new();

        /*  LEFT is the a not b, IE a - b set wise. */
        let mut liter = lhs.iter();
        let mut riter = rhs.iter();

        let mut lnext = liter.next();
        let mut rnext = riter.next();

        while lnext.is_some() && rnext.is_some() {
            let l = lnext.unwrap();
            let r = rnext.unwrap();

            if l < r {
                result.push_id(l.clone());
                lnext = liter.next();
            } else if l == r {
                lnext = liter.next();
                rnext = riter.next();
            } else if l > r {
                rnext = riter.next();
            }

        };

        /* Push the remaining A set elements. */
        while lnext.is_some() {
            let l = lnext.unwrap();
            result.push_id(l.clone());
            lnext = liter.next();
        }

        result
    }
}


#[cfg(test)]
mod tests {
    // use test::Bencher;
    use super::{IDLSimple, AndNot};
    use std::iter::FromIterator;

    #[test]
    fn test_simple_intersection_1() {
        let idl_a = IDLSimple::from_iter(vec![1, 2, 3]);
        let idl_b = IDLSimple::from_iter(vec![2]);
        let idl_expect = IDLSimple::from_iter(vec![2]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_intersection_2() {
        let idl_a = IDLSimple::from_iter(vec![1, 2, 3]);
        let idl_b = IDLSimple::from_iter(vec![4, 67]);
        let idl_expect = IDLSimple::new();

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_intersection_4() {
        let idl_a = IDLSimple::from_iter(vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900]);
        let idl_b = IDLSimple::from_iter(1..1024);
        let idl_expect = IDLSimple::from_iter(vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_intersection_5() {
        let idl_a = IDLSimple::from_iter(1..204800);
        let idl_b = IDLSimple::from_iter(102400..307200);
        let idl_expect = IDLSimple::from_iter(102400..204800);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_intersection_6() {
        let idl_a = IDLSimple::from_iter(vec![307199]);
        let idl_b = IDLSimple::from_iter(102400..307200);
        let idl_expect = IDLSimple::from_iter(vec![307199]);

        let idl_result = idl_a & idl_b;
        assert_eq!(idl_result, idl_expect);
    }


    #[test]
    fn test_simple_union_1() {
        let idl_a = IDLSimple::from_iter(vec![1,2,3]);
        let idl_b = IDLSimple::from_iter(vec![2]);
        let idl_expect = IDLSimple::from_iter(vec![1,2,3]);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_union_2() {
        let idl_a = IDLSimple::from_iter(vec![1,2,3]);
        let idl_b = IDLSimple::from_iter(vec![4,67]);
        let idl_expect = IDLSimple::from_iter(vec![1,2,3,4,67]);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_union_3() {
        let idl_a = IDLSimple::from_iter(vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900]);
        let idl_b = IDLSimple::from_iter(1..1024);
        let idl_expect = IDLSimple::from_iter(1..1024);

        let idl_result = idl_a | idl_b;
        assert_eq!(idl_result, idl_expect);
    }



    #[test]
    fn test_simple_not_1() {
        let idl_a = IDLSimple::from_iter(vec![1,2,3,4,5,6]);
        let idl_b = IDLSimple::from_iter(vec![3,4]);
        let idl_expect = IDLSimple::from_iter(vec![1,2,5,6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_not_2() {
        let idl_a = IDLSimple::from_iter(vec![1,2,3,4,5,6]);
        let idl_b = IDLSimple::from_iter(vec![10]);
        let idl_expect = IDLSimple::from_iter(vec![1,2,3,4,5,6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

    #[test]
    fn test_simple_not_3() {
        let idl_a = IDLSimple::from_iter(vec![2,3,4,5,6]);
        let idl_b = IDLSimple::from_iter(vec![1]);
        let idl_expect = IDLSimple::from_iter(vec![2,3,4,5,6]);

        let idl_result = idl_a.andnot(idl_b);
        assert_eq!(idl_result, idl_expect);
    }

}

