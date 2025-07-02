# uninit_buffers &emsp; [![Build Status]][actions] [![latest version]][crates.io] [![docs passing]][docs.rs]

[Build Status]: https://img.shields.io/github/actions/workflow/status/ljtpetersen/uninit_buffers/rust.yml
[actions]: https://github.com/ljtpetersen/uninit_buffers/actions
[latest version]: https://img.shields.io/crates/v/uninit_buffers
[crates.io]: https://crates.io/crates/uninit_buffers
[docs passing]: https://img.shields.io/docsrs/uninit_buffers
[docs.rs]: https://docs.rs/homedir/latest/uninit_buffers/

This crate aims to fill a hole in the currently-unstable `MaybeUninit` slice-filling API: there is
a safe way to fill a slice, but there is no safe way to drop elements of the slice. For this purpose,
we introduce a wrapper type, `Initialized`, which will drop the initialized elements when it goes out of scope.

Regarding safety, we treat the `Initialized` structure as if it owns the elements thar are filled. It is instantiated
by using the `SliceExt` trait, which is implemented on all `MaybeUninit` slices.

## Usage
This crate is on [crates.io](https://crates.io/crates/uninit_buffers) and can beused
by executing `cargo add uninit_buffers` or by adding the following to the dependencies in your
`Cargo.toml` file.

```toml
[dependencies]
uninit_buffers = "0.1"
```

## Licensing
Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT License
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

Feel free to put a copyright header in your name in any files you contribute to.

## Copyright and Credits
Copyright (C) 2025 James Petersen <m@jamespetersen.ca>.

The `SliceExt` trait implementation and documentation on `MaybeUninit` slices is heavily influenced (mostly copied)
from the corresponding implementation in the Rust standard library, with minor adjustments where necessary to accomodate
the `Initialized` structure as well as any missing unstable features.
