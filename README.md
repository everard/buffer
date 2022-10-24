# DESCRIPTION
This repository contains a small header-only C++20 library. The main feature of
this library is safe buffer type which is similar to std::array, but has some
compile-time-checked slice/join/view operations.

# INSTALLATION
Add the header from the `src` directory to your project. Note: A C++20-capable
compiler is required.

# USAGE
## BUFFER
*All non-qualified types mentioned in this section are located in the `rose`
namespace.*

In this library a _buffer_ is a type which contains (or references) an array of
some value type and size.

A _buffer_ also has a _tag type_. This _tag type_ helps to differentiate between
buffers of the same value types and sizes (see the
[BUFFER TAG TYPES](#buffer-tag-types) section for details).

There are 3 kinds of buffers:

 * `buffer`, `buffer_secure` - aggregates, the last one zeroes out the memory it
    owns upon its destruction;
 * `buffer_reference` - not an aggregate, can only be constructed from the other
    two buffer types, behaves as a reference to existing buffer objects.

### BUFFER CONCEPTS
This library defines the following concepts:

* `static_buffer` - matches any kind of buffer type;
* `static_buffer_reference` - matches any kind of buffer reference type;
* `static_byte_buffer` - matches any kind of buffer type whose value type is
  `unsigned char`;

### BUFFER REFERENCES, BUFFER VIEWING
Buffer reference type, `buffer_reference`, behaves exactly as a normal C++
reference, i.e. it has the same interface as the corresponding `buffer` or
`secure_buffer` types.

Its purpose is to avoid construction of temporary buffers. The
`buffer_reference` type contains only a single pointer, so it has zero overhead
over a normal C++ reference. In practice, the `buffer_reference` type should
_almost never be used directly_; the `ref`/`mut_ref` templates should be used
instead.

Note: All function templates which create `buffer_reference`s are _always_ safe
to use. Possible buffer overflows are checked at compile time.

Now, how can this reference type benefit your code? Let's consider the following
example.

Let's say we have a 256-byte packet which we know starts with two 32-byte
frames. We want to parse this packet and process these two starting frames.

The following code shows how this can be done using this library's type-safe
buffer interface.

```
#include "buffer.hh"

using rose::buffer;
using rose::ref;

// Note: rose::ref<T> behaves similarly to T const&.
// Note: rose::mut_ref<T> behaves similarly to T&.

using packet = buffer<256>;
using frame = buffer<32>;

void
process_frame(ref<frame> f) {
    // Do something with f.
}

void
parse(ref<packet> packet) {
    auto [f0, f1] = packet.view_as<frame, frame>();
    process_frame(f0);
    process_frame(f1);
}
```

If normal C++ reference was used as the parameter for the `process_frame`
function, then construction of temporary objects would have been necessary. But
in this example we essentially just take two pointers.

This example uses variadic `view_as` member function template which, depending
on the number of its template parameters, returns either an object of type
`buffer_reference`, or a `std::tuple` of `buffer_reference`s. This member
function template can also make buffer references starting from some offset:

```
// This makes buffer_references starting from packet's 16-th byte.
auto [f0, f1] = packet.view_as<16, frame, frame>();
```

All offset computations are performed at compile time, and buffer overflow can
never happen: if, for instance, the packet's size was 63 instead of 256, then
the compiler would have detected buffer overflow and the code wouldn't have
compiled at all.

There's another useful variadic function template which can create
`buffer_reference`s: `view_buffer_by_chunks`. It is a non-member function
template which takes one buffer as an input, let's call it `x`, and creates a
sequence of `buffer_reference`s of the specified size which reference successive
parts of `x` (these `buffer_reference`s all have the same value and tag types as
the initial buffer).

Let's take a look at the following example.

```
// packet is of type packet from example above.
auto chunks = view_buffer_by_chunks<32>(packet);
```

Here `chunks` is an 8-element (256 / 32 = 8) `buffer` of `buffer_reference`s,
each having their size equal to 32. The `view_buffer_by_chunks` function
template always checks that division does not produce a remainder: in the
example above it checks at compile time if (256 % 32 == 0).

### BUFFER JOINING
Buffers can be joined (concatenated) using the `join_buffers` and
`join_buffers_secure` non-member variadic function templates:

```
// In the following x0, x1, x2 are buffers or buffer_references.
auto y0 = join_buffers(x0, x1, x2);
auto y1 = join_buffers_secure(x0, x1, x2);

// Note: y0 is buffer, y1 is buffer_secure.
// Note: y0 has the same contents as y1, i.e. concatenation of x0, x1 and x2.
```

### BUFFER COPYING/FILLING/EXTRACTION
Note: All function templates which copy/fill/extract buffers are _always_ safe
to use. Possible buffer overflow is checked at compile time. Copy/fill
operations are also safe for overlapping buffers (this situation may happen when
`buffer_reference`s are used), since these operations use `std::copy_n`
algorithm which doesn't require non-overlapping ranges (unlike `std::copy` which
_requires_ non-overlapping source and destination ranges).

Contents of one buffer, let's call it `src`, can be copied to a non-empty list
of buffers using the `copy_into` variadic member function template:

```
// Copy elements from src successively into x0, x1, x2:
src.copy_into(x0, x1, x2);

// Copy elements from src, starting from offset 21 (in src), successively into
// x0, x1, x2:
src.copy_into<21>(x0, x1, x2);
```

Contents of one buffer, let's call it `dst`, can be filled from a non-empty list
of buffers using the `fill_from` variadic member function template:

```
// Copy elements from x0, x1, x2 successively into dst:
dst.fill_from(x0, x1, x2);

// Copy elements from x0, x1, x2 successively into dst, starting from offset 21
// (in dst):
dst.fill_from<21>(x0, x1, x2);
```

Contents of one buffer, let's call it `src`, can be extracted to another buffer,
or a `std::tuple` of buffers, using the `extract` variadic member function
template:

```
// Extract elements from src to b0 which will be a buffer size 21:
auto b0 = src.extract<buffer<21>>();

// Extract elements from src to std::tuple<buffer<15>, buffer<32>, buffer<11>>:
auto [b1, b2, b3] = src.extract<buffer<15>, buffer<32>, buffer<11>>();

// Extract elements from src, starting from offset 21, to b4 which will be a
// buffer of size 33:
auto b4 = src.extract<21, buffer<33>>();

// Extract elements from src to std::tuple<buffer<11>, buffer<1>, buffer<2>>,
// starting from offset 21:
auto [b5, b6, b7] = src.extract<21, buffer<11>, buffer<1>, buffer<2>>();
```

### BUFFER TO INTEGER CONVERSION
Any integer of type `T` can be converted to/from a buffer whose value type is
`unsigned char` and size equals to the number of bytes in value representation
of T (which equals to `sizeof(T)` if T has no padding bytes) using the
`int_to_buffer` and `buffer_to_int` non-member function templates:

```
using integer_container = buffer<rose::integer_size<unsigned>>;

unsigned i = 57;
auto b0 = rose::int_to_buffer(i);

auto b1 = integer_container{};
int_to_buffer(i, b1); // ADL should find this function template.

// Postcondition: b0 == b1

auto j = buffer_to_int<unsigned>(b0);
auto k = unsigned{};
buffer_to_int(b1, k);

// Postcondition: (i == j) && (j == k)
```

### BUFFER TAG TYPES
Here's an example from cryptography of when _tag types_ can be useful. Let's say
we have two key types: one for stream cipher and another one for MAC (message
authentication code). Both are buffers of `unsigned char`s of size 32:

```
using rose::buffer_secure;

using cipher_key = buffer_secure<32, unsigned char>;
using mac_key = buffer_secure<32, unsigned char>;
```

But this two types are in realty the same, i.e.
`buffer_secure<32, unsigned char>`, and using one key in place of the other is
totally fine from compiler's point of view, while actually doing so is usually
an error.

Tag types for the rescue:

```
struct cipher_key_tag {};
struct mac_key_tag {};

using cipher_key = buffer_secure<32, unsigned char, cipher_key_tag>;
using mac_key = buffer_secure<32, unsigned char, mac_key_tag>;
```

Now, `cipher_key` and `mac_key` are different types, and compiler will catch the
misuse of one key when the other one is required.

Note that `buffer_reference` also contains tag type, so using `ref` in function
parameters guarantees type safety:

```
void
encrypt(ref<cipher_key> k);

void
compute_mac(ref<mac_key> k);
```

# LICENSE
Copyright Nezametdinov E. Ildus 2022.
Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or copy at
https://www.boost.org/LICENSE_1_0.txt)
