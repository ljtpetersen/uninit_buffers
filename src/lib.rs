// src/lib.rs
//
// Copyright (C) 2023-2025 James Petersen <m@jamespetersen.ca>
// Licensed under Apache 2.0 OR MIT. See LICENSE-APACHE or LICENSE-MIT

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

//! This crate aims to fill a hole in the currently-unstable [`MaybeUninit`] slice-filling API:
//! there is a safe way to fill a slice, but there is no safe way to drop elements of the slice.
//! For this purpose, we introduce a wrapper type, [`Initialized`], which will drop the initialized
//! elements when it goes out of scope.
//!
//! Regarding safety, we treat the [`Initialized`] structure as if it owns the elements that are
//! filled. It is instantiated by using the [`SliceExt`] trait, which is implemented on all
//! [`MaybeUninit`] slices.
//!
//! # Usage
//! This crate is on [crates.io](https://crates.io/crates/uninit_buffers) and can be used
//! by executing `cargo add uninit_buffers` or by adding the following to the dependencies in your
//! `Cargo.toml` file.
//!
//! ```toml
//! [dependencies]
//! uninit_buffers = "0.1"
//! ```
//!
//! From here, you will want to
//! ```no_run
//! use uninit_buffers::SliceExt;
//! ```
//! This will allow you to write into any [`MaybeUninit`] slices.
//!
//! # Examples
//! ## Fill a buffer from an iterator.
//! ```
//! use std::mem::MaybeUninit;
//! use uninit_buffers::SliceExt;
//!
//! let mut buf = [const { MaybeUninit::uninit() }; 8];
//!
//! let original = ["fundamental", "theorem", "of", "calculus"].as_ref();
//! // strings are allocated in to_owned
//! let iter = original.iter().map(ToOwned::to_owned);
//! let (initialized, remainder) = buf.write_iter_owned(iter);
//!
//! assert_eq!(&*initialized, original);
//! assert_eq!(remainder.len(), 4);
//!
//! // this will drop the created strings. no need for unsafe!
//! drop(initialized);
//! ```
//! ## Fill a byte buffer from an input stream.
//! ```
//! # use std::io;
//! # fn main() -> io::Result<()> {
//! use std::io::Read;
//! use std::mem::MaybeUninit;
//!
//! use uninit_buffers::SliceExt;
//!
//! let input = /* <omitted> */
//! # io::empty();
//! let mut buf = [const { MaybeUninit::uninit() }; 2048];
//!
//! let (read_data, _) = buf.try_write_iter_owned(input.bytes())?;
//!
//! println!("read {} bytes.", read_data.len());
//!
//! # Ok(())
//! # }
//! ```
//!
//! For more examples, see the documentation of the methods of the [`SliceExt`] trait.

use core::{
    borrow::{Borrow, BorrowMut},
    iter::FusedIterator,
    marker::PhantomData,
    mem::{MaybeUninit, forget, take, transmute},
    ops::{Deref, DerefMut},
    slice::IterMut,
};

/// This structure owns initialized data contained in a slice of
/// [`MaybeUninit`]. To obtain one, use the [`SliceExt`] trait, which is automatically
/// implemented on slices of [`MaybeUninit`].
///
/// Regarding safety: this object "owns" the initialized data, and will drop them
/// automatically.
#[derive(Debug)]
pub struct Initialized<'a, T> {
    // for this object to exist, all elements of data must be initialized.
    data: &'a mut [MaybeUninit<T>],
    _marker: PhantomData<T>,
}

type InitializedWithRemainder<'a, T> = (Initialized<'a, T>, &'a mut [MaybeUninit<T>]);

/// This trait implements methods to construct initialized data on a [`MaybeUninit`] buffer.
/// It mimics the (currently unstable) API in the [standard library](slice::write_copy_of_slice).
/// The implementation and documentation are near-identical copies for all but the
/// `try_write_iter_owned` method.
pub trait SliceExt {
    /// This is the item that may be uninitialized.
    type Item;

    /// Copies the elements from `src` to `self`, returning the [`Initialized`] structure
    /// which owns the now-initialized contents of `self`.
    ///
    /// If `T` does not implement `Copy`, use
    /// [`write_clone_of_slice_owned`](SliceExt::write_clone_of_slice_owned) instead.
    ///
    /// This is similar to [`slice::copy_from_slice`].
    ///
    /// # Panics
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut dst = [MaybeUninit::uninit(); 32];
    /// let src = [0; 32];
    ///
    /// let init = dst.write_copy_of_slice_owned(&src);
    ///
    /// assert_eq!(*init, src);
    /// ```
    fn write_copy_of_slice_owned(&mut self, src: &[Self::Item]) -> Initialized<Self::Item>
    where
        Self::Item: Copy;

    /// Clones the elements from `src` to `self`, returning the [`Initialized`] structure
    /// which owns the now-initialized contents of `self`. Any already initialized items
    /// will not be dropped.
    ///
    /// If `T` implements `Copy`, use
    /// [`write_copy_of_slice_owned`](SliceExt::write_copy_of_slice_owned) instead.
    ///
    /// This is similar to [`slice::clone_from_slice`] but does not drop existing elements.
    ///
    /// # Panics
    /// This function will panic if the two slices have different lengths, or if the implementation
    /// of `Clone` panics.
    ///
    /// If there is a panic, the already cloned elements will be dropped.
    ///
    /// # Examples
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut dst = [const { MaybeUninit::uninit() }; 5];
    /// let src = ["wibbly", "wobbly", "timey", "wimey", "stuff"].map(ToOwned::to_owned);
    ///
    /// let init = dst.write_clone_of_slice_owned(&src);
    ///
    /// assert_eq!(*init, src);
    /// ```
    fn write_clone_of_slice_owned(&mut self, src: &[Self::Item]) -> Initialized<Self::Item>
    where
        Self::Item: Clone;

    /// Fills a slice with elements by cloning `value`, returning a mutable reference to the now
    /// initialized contents of the slice. Any previously initialized elements will not be dropped.
    ///
    /// This is similar to [`slice::fill`].
    ///
    /// # Panics
    /// This function will panic if any call to `Clone` panics.
    ///
    /// If such a panic occurs, any elements previously initialized during this operation will be
    /// dropped.
    ///
    /// # Examples
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::uninit() }; 10];
    /// let initialized = buf.write_filled_owned(1);
    /// assert_eq!(&*initialized, [1; 10].as_ref());
    /// ```
    fn write_filled_owned(&mut self, value: Self::Item) -> Initialized<Self::Item>
    where
        Self::Item: Clone;

    /// Fills a slice with elements returned by calling a closure for each index.
    ///
    /// This method uses a closure to create new values. If you'd rather `Clone` a given value, use
    /// [`MaybeUninit::fill`]. If you want to use the `Default` trait to generate values, you can
    /// pass [`|_| Default::default()`](Default::default) as the argument.
    ///
    /// # Panics
    /// This function will panic if any call to the provided closure panics.
    ///
    /// If such a panic occurs, any elements previously initialized during this operation will be
    /// dropped.
    ///
    /// # Examples
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::<usize>::uninit() }; 5];
    /// let initialized = buf.write_with_owned(|idx| idx + 1);
    /// assert_eq!(&*initialized, &[1, 2, 3, 4, 5]);
    /// ```
    fn write_with_owned<F>(&mut self, f: F) -> Initialized<Self::Item>
    where
        F: FnMut(usize) -> Self::Item;

    /// Fills a slice with elements yielded by an iterator until either all elements have been
    /// initialized or the iterator is empty.
    ///
    /// Returns the owned initialized data in [`Initialized`] and the remainder in the second
    /// slice.
    ///
    /// # Panics
    /// This function panics if the iterator's `next` function panics.
    ///
    /// If such a panic occurs, any elements previously initialized during this operation will be
    /// dropped.
    ///
    /// # Examples
    /// Completely filling the slice:
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::uninit() }; 5];
    ///
    /// let iter = [1, 2, 3].into_iter().cycle();
    /// let (initialized, remainder) = buf.write_iter_owned(iter);
    ///
    /// assert_eq!(&*initialized, &[1, 2, 3, 1, 2]);
    /// assert!(remainder.is_empty());
    /// ```
    /// Partially filling the slice:
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::uninit() }; 5];
    /// let iter = [1, 2];
    /// let (initialized, remainder) = buf.write_iter_owned(iter);
    ///
    /// assert_eq!(&*initialized, &[1, 2]);
    /// assert_eq!(remainder.len(), 3);
    /// ```
    /// Checking an iterator after filling a slice:
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::uninit() }; 3];
    /// let mut iter = [1, 2, 3, 4, 5].into_iter();
    /// let (initialized, remainder) = buf.write_iter_owned(iter.by_ref());
    ///
    /// assert_eq!(&*initialized, &[1, 2, 3]);
    /// assert!(remainder.is_empty());
    /// assert_eq!(iter.as_slice(), &[4, 5]);
    /// ```
    fn write_iter_owned<I>(&mut self, it: I) -> InitializedWithRemainder<Self::Item>
    where
        I: IntoIterator<Item = Self::Item>;

    /// Tries to fill a slice with elements yielded by an iterator until either all elements have
    /// been initialized or the iterator is empty.
    ///
    /// Returns the owned initialized data in [`Initialized`] and the remainder in the second
    /// slice.
    ///
    /// If the iterator at any point returns `Err`, the previously-consumed elements are dropped
    /// and the error is returned.
    ///
    /// # Panics
    /// This function panics if the iterator's `next` function panics.
    ///
    /// If such a panic occurs, any elements previously initialized during this operation will be
    /// dropped.
    ///
    /// # Examples
    // this is identical to the example in the crate documentation. do not run it.
    /// ```no_run
    /// # use std::io;
    /// # fn main() -> io::Result<()> {
    /// use std::io::Read;
    /// use std::mem::MaybeUninit;
    ///
    /// use uninit_buffers::SliceExt;
    ///
    /// let input = /* <omitted> */
    /// # io::empty();
    /// let mut buf = [const { MaybeUninit::uninit() }; 2048];
    ///
    /// let (read_data, _) = buf.try_write_iter_owned(input.bytes())?;
    ///
    /// println!("read {} bytes.", read_data.len());
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn try_write_iter_owned<I, E>(
        &mut self,
        it: I,
    ) -> Result<InitializedWithRemainder<Self::Item>, E>
    where
        I: IntoIterator<Item = Result<Self::Item, E>>;
}

/// An iterator that moves out of [`Initialized`].
///
/// This `struct` is created by the `into_iter` method on [`Initialized`] (provided by the
/// [`IntoIterator`] trait).
#[derive(Debug, Default)]
pub struct IntoIter<'a, T> {
    iter: IterMut<'a, MaybeUninit<T>>,
    _marker: PhantomData<T>,
}

struct Guard<'a, T> {
    slice: &'a mut [MaybeUninit<T>],
    initialized: usize,
}

impl<T> Deref for Initialized<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: all elements of self.data are already initialized, and this object owns them.
        unsafe { &*(self.data as *const [MaybeUninit<T>] as *const [T]) }
    }
}

impl<T> DerefMut for Initialized<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: all elements of self.data are already initialized, and this object owns them.
        unsafe { &mut *(self.data as *mut [MaybeUninit<T>] as *mut [T]) }
    }
}

impl<'a, T> IntoIterator for Initialized<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(mut self) -> Self::IntoIter {
        let im = take(&mut self.data).iter_mut();
        forget(self);
        IntoIter {
            iter: im,
            _marker: PhantomData,
        }
    }
}

impl<T> Iterator for IntoIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| unsafe { v.assume_init_read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator for IntoIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|v| unsafe { v.assume_init_read() })
    }
}

impl<T> ExactSizeIterator for IntoIter<'_, T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T> Drop for Initialized<'_, T> {
    fn drop(&mut self) {
        for v in self.data.iter_mut() {
            unsafe { v.assume_init_drop() }
        }
    }
}

impl<T> Drop for IntoIter<'_, T> {
    fn drop(&mut self) {
        for v in self.iter.by_ref() {
            unsafe { v.assume_init_drop() }
        }
    }
}

impl<T> SliceExt for [MaybeUninit<T>] {
    type Item = T;

    fn write_copy_of_slice_owned(&mut self, src: &[Self::Item]) -> Initialized<Self::Item>
    where
        Self::Item: Copy,
    {
        let uninit_src: &[MaybeUninit<T>] = unsafe { transmute(src) };

        self.copy_from_slice(uninit_src);

        Initialized {
            data: self,
            _marker: PhantomData,
        }
    }

    fn write_clone_of_slice_owned(&mut self, src: &[Self::Item]) -> Initialized<Self::Item>
    where
        Self::Item: Clone,
    {
        assert_eq!(
            self.len(),
            src.len(),
            "destination and source slices have different lengths"
        );

        let len = self.len();
        let src = &src[..len];

        let mut guard = Guard {
            slice: self,
            initialized: 0,
        };

        for (i, val) in src.iter().enumerate() {
            guard.slice[i].write(val.clone());
            guard.initialized += 1;
        }

        forget(guard);

        Initialized {
            data: self,
            _marker: PhantomData,
        }
    }

    #[doc(alias = "memset")]
    fn write_filled_owned(&mut self, value: Self::Item) -> Initialized<Self::Item>
    where
        Self::Item: Clone,
    {
        // oh to long for specialization
        let mut guard = Guard {
            slice: self,
            initialized: 0,
        };

        if let Some((last, elems)) = guard.slice.split_last_mut() {
            for el in elems {
                el.write(value.clone());
                guard.initialized += 1;
            }

            last.write(value);
        }

        forget(guard);

        Initialized {
            data: self,
            _marker: PhantomData,
        }
    }

    fn write_with_owned<F>(&mut self, mut f: F) -> Initialized<Self::Item>
    where
        F: FnMut(usize) -> Self::Item,
    {
        let mut guard = Guard {
            slice: self,
            initialized: 0,
        };

        for (idx, element) in guard.slice.iter_mut().enumerate() {
            element.write(f(idx));
            guard.initialized += 1;
        }

        forget(guard);

        Initialized {
            data: self,
            _marker: PhantomData,
        }
    }

    fn write_iter_owned<I>(
        &mut self,
        it: I,
    ) -> (Initialized<Self::Item>, &mut [MaybeUninit<Self::Item>])
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let iter = it.into_iter();
        let mut guard = Guard {
            slice: self,
            initialized: 0,
        };

        for (element, val) in guard.slice.iter_mut().zip(iter) {
            element.write(val);
            guard.initialized += 1;
        }

        let initialized_len = guard.initialized;
        forget(guard);

        let (initted, remainder) = unsafe { self.split_at_mut_unchecked(initialized_len) };

        (
            Initialized {
                data: initted,
                _marker: PhantomData,
            },
            remainder,
        )
    }

    fn try_write_iter_owned<I, E>(
        &mut self,
        it: I,
    ) -> Result<(Initialized<Self::Item>, &mut [MaybeUninit<Self::Item>]), E>
    where
        I: IntoIterator<Item = Result<Self::Item, E>>,
    {
        let iter = it.into_iter();
        let mut guard = Guard {
            slice: self,
            initialized: 0,
        };

        for (element, val) in guard.slice.iter_mut().zip(iter) {
            element.write(val?);
            guard.initialized += 1;
        }

        let initialized_len = guard.initialized;
        forget(guard);

        let (initted, remainder) = unsafe { self.split_at_mut_unchecked(initialized_len) };

        Ok((
            Initialized {
                data: initted,
                _marker: PhantomData,
            },
            remainder,
        ))
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        for v in self.slice[..self.initialized].iter_mut() {
            unsafe {
                v.assume_init_drop();
            }
        }
    }
}

impl<T> AsRef<[T]> for Initialized<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T> AsRef<[T]> for IntoIter<'_, T> {
    fn as_ref(&self) -> &[T] {
        unsafe { &*(self.iter.as_slice() as *const [MaybeUninit<T>] as *const [T]) }
    }
}

impl<T> AsMut<[T]> for Initialized<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<T> Borrow<[T]> for Initialized<'_, T> {
    fn borrow(&self) -> &[T] {
        self.deref()
    }
}

impl<T> BorrowMut<[T]> for Initialized<'_, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

unsafe impl<T: Send> Send for Initialized<'_, T> {}
unsafe impl<T: Send> Send for IntoIter<'_, T> {}

unsafe impl<T: Sync> Sync for Initialized<'_, T> {}
unsafe impl<T: Sync> Sync for IntoIter<'_, T> {}

impl<T> Default for Initialized<'_, T> {
    fn default() -> Self {
        Self {
            data: &mut [],
            _marker: PhantomData,
        }
    }
}

impl<T> FusedIterator for IntoIter<'_, T> {}

impl<T> IntoIter<'_, T> {
    /// Returns the remaining items of this iterator as a slice.
    /// # Example
    /// ```
    /// use std::mem::MaybeUninit;
    /// use uninit_buffers::SliceExt;
    ///
    /// let mut buf = [const { MaybeUninit::uninit() }; 3];
    /// let mut into_iter = buf.write_copy_of_slice_owned(&['a', 'b', 'c']).into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// let _ = into_iter.next().unwrap();
    /// assert_eq!(into_iter.as_slice(), &['b', 'c']);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use core::{cell::RefCell, sync::atomic::AtomicUsize};
    use std::{collections::HashSet, panic::{catch_unwind, set_hook, take_hook, PanicHookInfo}, sync::mpsc::{channel, Sender}};

    use super::*;

    #[derive(Debug)]
    struct DropKey<'a> {
        index: usize,
        set: &'a RefCell<HashSet<usize>>,
        inset: bool,
    }

    impl<'a> DropKey<'a> {
        fn new(index: usize, set: &'a RefCell<HashSet<usize>>) -> Self {
            assert!(
                set.borrow_mut().insert(index),
                "set already contains index {index}"
            );
            Self {
                index,
                set,
                inset: true,
            }
        }

        fn new_notinset(index: usize, set: &'a RefCell<HashSet<usize>>) -> Self {
            Self {
                index,
                set,
                inset: false,
            }
        }
    }

    impl<'a> Drop for DropKey<'a> {
        fn drop(&mut self) {
            if self.inset {
                assert!(
                    self.set.borrow_mut().remove(&self.index),
                    "set does not contain index {}",
                    self.index
                );
            }
        }
    }

    impl<'a> Clone for DropKey<'a> {
        fn clone(&self) -> Self {
            assert!(
                self.set.borrow_mut().insert(self.index),
                "set already contains index {}",
                self.index
            );
            Self {
                index: self.index,
                set: self.set,
                inset: true,
            }
        }
    }

    struct DropCount<'a> {
        count: &'a RefCell<usize>,
    }

    impl<'a> DropCount<'a> {
        fn new(count: &'a RefCell<usize>) -> Self {
            *count.borrow_mut() += 1;
            Self { count }
        }
    }

    impl<'a> Clone for DropCount<'a> {
        fn clone(&self) -> Self {
            Self::new(self.count)
        }
    }

    impl<'a> Drop for DropCount<'a> {
        fn drop(&mut self) {
            *self.count.borrow_mut() -= 1;
        }
    }

    impl<'a> PartialEq for DropKey<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.index == other.index
        }
    }

    #[test]
    fn write_iter() {
        let set = RefCell::new(HashSet::new());
        const CAP: usize = 10;
        const NUM: usize = 5;
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let (written, uninit) = buf.write_iter_owned((0..NUM).map(|v| DropKey::new(v, &set)));

        let setcmp = RefCell::new(HashSet::new());
        let expected = (0..NUM)
            .map(|v| DropKey::new(v, &setcmp))
            .collect::<Vec<_>>();

        assert_eq!(*written, *expected);
        assert_eq!(uninit.len(), CAP - NUM);

        for idx in 0..NUM {
            assert!(
                set.borrow().contains(&idx),
                "set does not contain index {idx}"
            );
        }

        drop(written);
        assert!(set.borrow().is_empty(), "set is empty after dropping");
    }

    #[test]
    fn write_with() {
        let set = RefCell::new(HashSet::new());
        const CAP: usize = 10;
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let written = buf.write_with_owned(|i| DropKey::new(i, &set));

        let setcmp = RefCell::new(HashSet::new());
        let expected = (0..CAP)
            .map(|v| DropKey::new(v, &setcmp))
            .collect::<Vec<_>>();

        assert_eq!(*written, *expected);

        for idx in 0..CAP {
            assert!(
                set.borrow().contains(&idx),
                "set does not contain index {idx}"
            );
        }

        drop(written);
        assert!(set.borrow().is_empty(), "set is empty after dropping");
    }

    #[test]
    fn write_clone_of_slice() {
        let set = RefCell::new(HashSet::new());
        const CAP: usize = 10;
        let slc = (0..CAP)
            .map(|v| DropKey::new_notinset(v, &set))
            .collect::<Vec<_>>();
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let written = buf.write_clone_of_slice_owned(&slc);

        assert_eq!(*written, *slc);

        for idx in 0..CAP {
            assert!(
                set.borrow().contains(&idx),
                "set does not contain index {idx}"
            );
        }

        drop(written);
        assert!(set.borrow().is_empty(), "set is empty after dropping");
    }

    #[test]
    fn write_copy_of_slice() {
        const CAP: usize = 10;
        let slc = (0..CAP).collect::<Vec<_>>();
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let written = buf.write_copy_of_slice_owned(&slc);

        assert_eq!(*written, *slc);
    }

    #[test]
    fn write_filled() {
        const CAP: usize = 10;
        let count = RefCell::new(0);
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let written = buf.write_filled_owned(DropCount::new(&count));
        assert_eq!(*count.borrow(), 10);
        drop(written);
        assert_eq!(*count.borrow(), 0);
    }

    #[test]
    fn test_into_iter() {
        let set = RefCell::new(HashSet::new());
        const CAP: usize = 10;
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        let mut iter = buf.write_with_owned(|i| DropKey::new(i, &set)).into_iter();
        assert_eq!(iter.len(), 10);

        const TAKE: usize = 5;
        iter.nth(TAKE - 1).unwrap();

        let borrow = set.borrow();
        for j in 0..TAKE {
            assert!(!borrow.contains(&j), "set contains index {j}");
        }
        for j in TAKE..CAP {
            assert!(borrow.contains(&j), "set does not contains index {j}");
        }
        drop(borrow);

        assert_eq!(iter.len(), 5);
        iter.next_back().unwrap();

        let borrow = set.borrow();
        for j in 0..TAKE {
            assert!(!borrow.contains(&j), "set contains index {j}");
        }
        for j in TAKE..(CAP - 1) {
            assert!(borrow.contains(&j), "set does not contains index {j}");
        }
        assert!(!borrow.contains(&CAP), "set contains index 9");
        drop(borrow);

        drop(iter);
        assert!(set.borrow().is_empty(), "set is not empty");
    }

    #[derive(Debug, PartialEq)]
    enum DropLogEntry {
        Init(usize),
        Drop(usize),
    }

    #[derive(Debug)]
    struct DropLog {
        id: Option<usize>,
        log: Sender<DropLogEntry>,
    }

    impl DropLog {
        fn new(id: Option<usize>, log: Sender<DropLogEntry>) -> Self {
            if let Some(id) = id {
                log.send(DropLogEntry::Init(id)).unwrap();
            }
            Self { id, log }
        }
    }

    impl Drop for DropLog {
        fn drop(&mut self) {
            if let Some(id) = self.id.as_ref() {
                self.log.send(DropLogEntry::Drop(*id)).unwrap();
            }
        }
    }

    #[test]
    fn try_write_iter() {
        const CAP: usize = 10;
        let mut buf = [const { MaybeUninit::uninit() }; CAP];

        let first_batch = [Ok::<_, ()>(1), Ok(2), Ok(3), Ok(4)];
        let (initialized, remainder) = buf
            .try_write_iter_owned(first_batch.iter().cloned())
            .unwrap();
        let expected = first_batch
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), &*initialized);
        assert_eq!(remainder.len(), 6);
        drop(initialized);

        let second_batch = [Ok(1), Ok(2), Err("err")];
        assert_eq!(buf.try_write_iter_owned(second_batch).unwrap_err(), "err");

        let third_batch = (0..CAP).map(Result::Ok).chain([Err("err")]);
        let (initialized, remainder) = buf.try_write_iter_owned(third_batch).unwrap();
        assert!(initialized.into_iter().eq(0..CAP));
        assert!(remainder.is_empty());

        const GOOD: usize = 3;
        let (tx, rx) = channel();
        let fourth_batch = (0..GOOD).map(|j| Ok(DropLog::new(Some(j), tx.clone())))
            .chain([Err("err")]);
        let mut buf = [const { MaybeUninit::uninit() }; CAP];
        assert_eq!(buf.try_write_iter_owned(fourth_batch).unwrap_err(), "err");
        let expected = (0..GOOD).map(DropLogEntry::Init).chain((0..GOOD).map(DropLogEntry::Drop));
        drop(tx);
        assert!(rx.iter().eq(expected));
    }

    #[test]
    fn write_iter_panic() {
        let prev_hook = take_hook();
        let new_hook = move |info: &PanicHookInfo<'_>| {
            // if we intentionally panic, do nothing.
            if let Some(v) = info.payload().downcast_ref::<&'static str>() && *v == "intentional panic" {
                return;
            }
            prev_hook(info);
        };
        set_hook(Box::new(new_hook));

        let (tx, rx) = channel();
        const CAP: usize = 10;
        const GOOD: usize = 3;
        assert_eq!(
            catch_unwind(move || {
                let mut buf = [const { MaybeUninit::uninit() }; CAP];
                let iter = (0..GOOD).map(|j| DropLog::new(Some(j), tx.clone()))
                    .chain([()].into_iter().map(|_| panic!("intentional panic")));
                let _ = buf.write_iter_owned(iter);
            }).unwrap_err().downcast_ref::<&'static str>(),
            Some(&"intentional panic"),
        );
        // when the iterator panics, the previously-returned values are correctly dropped.
        let expected_log = (0..GOOD).map(DropLogEntry::Init).chain((0..GOOD).map(DropLogEntry::Drop));
        assert!(rx.iter().eq(expected_log));
    }

    #[test]
    fn try_write_iter_panic() {
        let prev_hook = take_hook();
        let new_hook = move |info: &PanicHookInfo<'_>| {
            // if we intentionally panic, do nothing.
            if let Some(v) = info.payload().downcast_ref::<&'static str>() && *v == "intentional panic" {
                return;
            }
            prev_hook(info);
        };
        set_hook(Box::new(new_hook));

        let (tx, rx) = channel();
        const CAP: usize = 10;
        const GOOD: usize = 3;
        assert_eq!(
            catch_unwind(move || {
                let mut buf = [const { MaybeUninit::uninit() }; CAP];
                let iter = (0..GOOD).map(|j| Ok::<_, ()>(DropLog::new(Some(j), tx.clone())))
                    .chain([()].into_iter().map(|_| panic!("intentional panic")));
                let _ = buf.write_iter_owned(iter);
            }).unwrap_err().downcast_ref::<&'static str>(),
            Some(&"intentional panic"),
        );
        // when the iterator panics, the previously-returned values are correctly dropped.
        let expected_log = (0..GOOD).map(DropLogEntry::Init).chain((0..GOOD).map(DropLogEntry::Drop));
        assert!(rx.iter().eq(expected_log));
    }

    #[test]
    fn write_with_panic() {
        let prev_hook = take_hook();
        let new_hook = move |info: &PanicHookInfo<'_>| {
            // if we intentionally panic, do nothing.
            if let Some(v) = info.payload().downcast_ref::<&'static str>() && *v == "intentional panic" {
                return;
            }
            prev_hook(info);
        };
        set_hook(Box::new(new_hook));

        let (tx, rx) = channel();
        const CAP: usize = 10;
        const GOOD: usize = 3;
        assert_eq!(
            catch_unwind(move || {
                let mut buf = [const { MaybeUninit::uninit() }; CAP];
                let _ = buf.write_with_owned(|j| {
                    if j == GOOD { panic!("intentional panic") }
                    else {
                        DropLog::new(Some(j), tx.clone())
                    }
                });
            }).unwrap_err().downcast_ref::<&'static str>(),
            Some(&"intentional panic"),
        );
        // when the iterator panics, the previously-returned values are correctly dropped.
        let expected_log = (0..GOOD).map(DropLogEntry::Init).chain((0..GOOD).map(DropLogEntry::Drop));
        assert!(rx.iter().eq(expected_log));
    }

    #[derive(Debug)]
    struct BadCloneFill<'a> {
        drop_log: DropLog,
        shared_count: &'a AtomicUsize,
        panic_on: usize,
    }

    impl<'a> BadCloneFill<'a> {
        fn new(log: Sender<DropLogEntry>, shared_count: &'a AtomicUsize, panic_on: usize) -> Self {
            shared_count.store(0, core::sync::atomic::Ordering::Relaxed);
            Self {
                drop_log: DropLog::new(Some(0), log),
                shared_count,
                panic_on,
            }
        }
    }

    impl Clone for BadCloneFill<'_> {
        fn clone(&self) -> Self {
            let oldcount = self.shared_count.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            if oldcount + 1 == self.panic_on {
                panic!("intentional panic");
            }
            Self {
                drop_log: DropLog::new(Some(oldcount + 1), self.drop_log.log.clone()),
                shared_count: self.shared_count,
                panic_on: self.panic_on,
            }
        }
    }

    #[test]
    fn write_filled_panic() {
        let prev_hook = take_hook();
        let new_hook = move |info: &PanicHookInfo<'_>| {
            // if we intentionally panic, do nothing.
            if let Some(v) = info.payload().downcast_ref::<&'static str>() && *v == "intentional panic" {
                return;
            }
            prev_hook(info);
        };
        set_hook(Box::new(new_hook));

        let (tx, rx) = channel();
        const CAP: usize = 10;
        const GOOD: usize = 3;
        assert_eq!(
            catch_unwind(move || {
                let mut buf = [const { MaybeUninit::uninit() }; CAP];
                let at = AtomicUsize::new(0);
                let _ = buf.write_filled_owned(BadCloneFill::new(tx.clone(), &at, GOOD));
            }).unwrap_err().downcast_ref::<&'static str>(),
            Some(&"intentional panic"),
        );
        // when the iterator panics, the previously-returned values are correctly dropped.
        // the drop order is 1, 2, 0, because 0 is a parameter of filled, which will obviously only
        // be dropped last.
        let expected_log = (0..GOOD).map(DropLogEntry::Init)
            .chain((1..GOOD).map(DropLogEntry::Drop))
            .chain([DropLogEntry::Drop(0)]);
        assert!(rx.iter().eq(expected_log));
    }

    #[derive(Debug)]
    struct BadClone {
        drop_log: DropLog,
        do_panic: bool,
    }

    impl Clone for BadClone {
        fn clone(&self) -> Self {
            if self.do_panic {
                panic!("intentional panic");
            } else {
                Self {
                    drop_log: DropLog::new(self.drop_log.id, self.drop_log.log.clone()),
                    do_panic: self.do_panic,
                }
            }
        }
    }

    #[test]
    fn write_clone_of_slice_panic() {
        let prev_hook = take_hook();
        let new_hook = move |info: &PanicHookInfo<'_>| {
            // if we intentionally panic, do nothing.
            if let Some(v) = info.payload().downcast_ref::<&'static str>() && *v == "intentional panic" {
                return;
            }
            prev_hook(info);
        };
        set_hook(Box::new(new_hook));

        let (tx, rx) = channel();
        const CAP: usize = 10;
        const GOOD: usize = 3;
    
        let orig = (0..CAP)
            .map(|j| BadClone { drop_log: DropLog::new(Some(j), tx.clone()), do_panic: j == GOOD })
            .collect::<Vec<_>>();

        assert_eq!(
            catch_unwind(|| {
                let mut buf = [const { MaybeUninit::uninit() }; CAP];
                let _ = buf.write_clone_of_slice_owned(&orig);
            }).unwrap_err().downcast_ref::<&'static str>(),
            Some(&"intentional panic"),
        );
        drop(tx);
        drop(orig);
        // when the iterator panics, the previously-returned values are correctly dropped.
        let expected_log = (0..CAP).map(DropLogEntry::Init)
            .chain((0..GOOD).map(DropLogEntry::Init))
            .chain((0..GOOD).map(DropLogEntry::Drop))
            .chain((0..CAP).map(DropLogEntry::Drop));
        assert!(rx.iter().eq(expected_log));
    }
}
