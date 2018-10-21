IDLSet
======

IDLSet - Fast u64 integer set operations

IDLSet is a specialised library for fast logical set operations on
u64. For example, this means union (or), intersection (and) and not
operations on sets. In the best case, speed ups of 15x have been observed
with the general case performing approximately 4x faster that a Vec<u64>
based implementation.

These operations are heavily used in low-level implementations of databases
for their indexing logic, but has applications with statistical analysis and
other domains that require logical set operations.

This seems very specific to only use u64, but has been chosen for a good reason. On
64bit cpus, native 64bit operations are faster than 32/16. Additionally,
due to the design of the library, unsigned types are simpler to operate
on for the set operations.

Contributing
------------

Please open an issue, pr or contact me directly by email (see github)

