// Copyright Nezametdinov E. Ildus 2023.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// https://www.boost.org/LICENSE_1_0.txt)
//
#ifndef H_639425CCA60E448B9BEB43186E06CA57
#define H_639425CCA60E448B9BEB43186E06CA57

#include <algorithm>
#include <concepts>
#include <ranges>

#include <climits>
#include <cstddef>
#include <limits>

#include <span>
#include <tuple>

#include <type_traits>
#include <utility>

namespace rose {

////////////////////////////////////////////////////////////////////////////////
// Standard integer type definitions.
////////////////////////////////////////////////////////////////////////////////

using std::ptrdiff_t;
using std::size_t;

////////////////////////////////////////////////////////////////////////////////
// Compile-time addition interface of values of unsigned type. Generates
// compilation error on wrap-around.
////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <std::unsigned_integral T, T X, T Y, T... Rest>
constexpr auto
static_add() noexcept -> T requires(static_cast<T>(X + Y) >= Y) {
    if constexpr(sizeof...(Rest) == 0) {
        return (X + Y);
    } else {
        return static_add<T, static_cast<T>(X + Y), Rest...>();
    }
}

} // namespace detail

template <size_t X, size_t Y, size_t... Rest>
constexpr auto static_sum = detail::static_add<size_t, X, Y, Rest...>();

////////////////////////////////////////////////////////////////////////////////
// Forward declarations of buffer interface and buffer storages.
////////////////////////////////////////////////////////////////////////////////

template <typename Storage>
struct buffer_interface;

template <size_t N, typename T, typename Tag>
struct buffer_storage_normal;

template <size_t N, typename T, typename Tag>
struct buffer_storage_secure;

template <size_t N, typename T, typename Tag>
struct buffer_storage_reference;

////////////////////////////////////////////////////////////////////////////////
// Buffer concepts.
////////////////////////////////////////////////////////////////////////////////

template <typename T>
concept static_buffer =
    std::ranges::sized_range<T> &&
    (std::derived_from<
         T, buffer_interface<buffer_storage_normal<
                T::static_size(), typename T::value_type, typename T::tag>>> ||
     std::derived_from<
         T, buffer_interface<buffer_storage_secure<
                T::static_size(), typename T::value_type, typename T::tag>>> ||
     std::derived_from<
         T, buffer_interface<buffer_storage_reference<
                T::static_size(), typename T::value_type, typename T::tag>>>);

template <typename T>
concept static_buffer_reference =
    std::ranges::sized_range<T> &&
    std::derived_from<
        T, buffer_interface<buffer_storage_reference<
               T::static_size(), typename T::value_type, typename T::tag>>>;

template <typename T>
concept static_byte_buffer =
    static_buffer<T> &&
    std::same_as<std::remove_cv_t<std::ranges::range_value_t<T>>,
                 unsigned char>;

////////////////////////////////////////////////////////////////////////////////
// Buffer principal value type definition.
////////////////////////////////////////////////////////////////////////////////

template <static_buffer Buffer>
using buffer_value_type = std::remove_cv_t<typename Buffer::value_type>;

////////////////////////////////////////////////////////////////////////////////
// Buffer predicates.
////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <static_buffer Buffer0, static_buffer Buffer1,
          static_buffer... Buffers>
constexpr auto
check_homogeneity() noexcept -> bool {
    auto f =
        std::same_as<buffer_value_type<Buffer0>, buffer_value_type<Buffer1>>;

    if constexpr(sizeof...(Buffers) == 0) {
        return f;
    } else {
        return f && check_homogeneity<Buffer0, Buffers...>();
    }
}

template <size_t Offset, static_buffer Buffer_src, static_buffer Buffer_dst0,
          static_buffer... Buffers_dst>
constexpr auto
check_buffer_overflow() noexcept -> bool {
    return (Buffer_src::static_size() >=
            static_sum<Offset, Buffer_dst0::static_size(),
                       Buffers_dst::static_size()...>);
}

} // namespace detail

// A predicate which shows if the given buffer types are homogeneous (have the
// same principal value types).
template <static_buffer Buffer0, static_buffer Buffer1,
          static_buffer... Buffers>
constexpr auto are_homogeneous_buffers =
    detail::check_homogeneity<Buffer0, Buffer1, Buffers...>();

// A predicate which shows if the given buffer types are comparable.
template <static_buffer Buffer0, static_buffer Buffer1>
constexpr auto are_comparable_buffers = //
    ((Buffer0::static_size() == Buffer1::static_size()) &&
     std::three_way_comparable_with<buffer_value_type<Buffer0>,
                                    buffer_value_type<Buffer1>>);

// A predicate which shows that buffer operation on the given buffer types is
// viable (i.e. buffer types are homogeneous), and will not produce an overflow.
template <size_t Offset, static_buffer Buffer_src, static_buffer Buffer_dst0,
          static_buffer... Buffers_dst>
constexpr auto no_buffer_overflow =
    (detail::check_homogeneity<Buffer_src, Buffer_dst0, Buffers_dst...>() &&
     detail::check_buffer_overflow<Offset, Buffer_src, Buffer_dst0,
                                   Buffers_dst...>());

// A predicate which shows that the given buffer types can be used as a target
// for extract operation (in other words, they must not be buffer reference
// types).
template <static_buffer... Buffers>
constexpr auto are_valid_extraction_target =
    (!static_buffer_reference<Buffers> && ...);

// A predicate which shows that extracting an object of source buffer type to a
// sequence of objects of destination buffer types is valid.
template <size_t Offset, static_buffer Buffer_src, static_buffer Buffer_dst0,
          static_buffer... Buffers_dst>
constexpr auto is_valid_extraction =
    (are_valid_extraction_target<Buffer_dst0, Buffers_dst...> &&
     no_buffer_overflow<Offset, Buffer_src, Buffer_dst0, Buffers_dst...>);

// Compound size of the given buffer types.
template <static_buffer Buffer, static_buffer... Buffers>
constexpr auto compound_size =
    ((sizeof...(Buffers) == 0)
         ? Buffer::static_size()
         : static_sum<Buffer::static_size(), Buffers::static_size()...>);

////////////////////////////////////////////////////////////////////////////////
// Tag type definitions.
////////////////////////////////////////////////////////////////////////////////

struct tag_default {};

template <typename T>
struct tag_wrapped {};

template <static_buffer Buffer, static_buffer... Buffers>
using tag_compound =
    std::conditional_t<(sizeof...(Buffers) == 0), typename Buffer::tag,
                       std::tuple<tag_wrapped<typename Buffer::tag>,
                                  tag_wrapped<typename Buffers::tag>...>>;

////////////////////////////////////////////////////////////////////////////////
// Buffer definitions.
////////////////////////////////////////////////////////////////////////////////

template <size_t N, typename T = unsigned char, typename Tag = tag_default>
using buffer = buffer_interface<buffer_storage_normal<N, T, Tag>>;

template <size_t N, typename T = unsigned char, typename Tag = tag_default>
using buffer_secure = buffer_interface<buffer_storage_secure<N, T, Tag>>;

template <static_buffer Buffer, static_buffer... Buffers>
using buffer_compound =
    buffer<compound_size<Buffer, Buffers...>, buffer_value_type<Buffer>,
           tag_compound<Buffer, Buffers...>>;

template <static_buffer Buffer, static_buffer... Buffers>
using buffer_compound_secure =
    buffer_secure<compound_size<Buffer, Buffers...>, buffer_value_type<Buffer>,
                  tag_compound<Buffer, Buffers...>>;

////////////////////////////////////////////////////////////////////////////////
// Buffer reference definitions.
////////////////////////////////////////////////////////////////////////////////

template <size_t N, typename T = unsigned char, typename Tag = tag_default>
using buffer_reference = buffer_interface<buffer_storage_reference<N, T, Tag>>;

template <static_buffer T>
using mut_ref =
    buffer_reference<T::static_size(), typename T::value_type, typename T::tag>;

template <static_buffer T>
using ref = buffer_reference<T::static_size(), typename T::value_type const,
                             typename T::tag>;

namespace detail {

template <static_buffer Buffer_src, static_buffer Buffer_dst>
using select_ref =
    std::conditional_t<(std::is_const_v<Buffer_src> ||
                        std::is_const_v<typename Buffer_src::value_type>),
                       ref<Buffer_dst>, mut_ref<Buffer_dst>>;

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Buffer interface definition.
////////////////////////////////////////////////////////////////////////////////

template <typename Storage>
struct buffer_interface : Storage {
    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = typename Storage::value_type;
    using tag = typename Storage::tag;

    ////////////////////////////////////////////////////////////////////////////
    // Corresponding buffer reference definitions.
    ////////////////////////////////////////////////////////////////////////////

    using mut_ref = buffer_reference<Storage::static_size, value_type, tag>;
    using ref = buffer_reference<Storage::static_size, value_type const, tag>;

    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr auto is_noexcept =
        std::is_nothrow_copy_constructible_v<value_type>;

    ////////////////////////////////////////////////////////////////////////////
    // Accessors.
    ////////////////////////////////////////////////////////////////////////////

    constexpr auto
    begin() noexcept {
        return this->data();
    }

    constexpr auto
    begin() const noexcept {
        return this->data();
    }

    constexpr auto
    end() noexcept {
        return this->data() + this->static_size();
    }

    constexpr auto
    end() const noexcept {
        return this->data() + this->static_size();
    }

    static constexpr auto
    size() noexcept -> size_t {
        return Storage::static_size;
    }

    static constexpr auto
    static_size() noexcept -> size_t {
        return Storage::static_size;
    }

    constexpr auto&
    operator[](size_t i) noexcept {
        return this->data()[i];
    }

    constexpr auto&
    operator[](size_t i) const noexcept {
        return this->data()[i];
    }

    constexpr auto
    data() noexcept {
        return this->data_;
    }

    constexpr auto
    data() const noexcept {
        return static_cast<std::add_const_t<value_type>*>(this->data_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Copy operations.
    ////////////////////////////////////////////////////////////////////////////

    template <size_t Offset, static_buffer Buffer, static_buffer... Buffers>
    constexpr void
    copy_into(Buffer& x, Buffers&... rest) const noexcept(is_noexcept) requires(
        no_buffer_overflow<Offset, buffer_interface, Buffer, Buffers...>) {
        this->copy_into_<Offset>(x, rest...);
    }

    template <static_buffer Buffer, static_buffer... Buffers>
    constexpr void
    copy_into(Buffer& x, Buffers&... rest) const noexcept(is_noexcept)
        requires(no_buffer_overflow<0, buffer_interface, Buffer, Buffers...>) {
        this->copy_into_<0>(x, rest...);
    }

    template <size_t Offset, static_buffer Buffer, static_buffer... Buffers>
    constexpr auto&
    fill_from(Buffer const& x, Buffers const&... rest) noexcept(is_noexcept)
        requires(
            no_buffer_overflow<Offset, buffer_interface, Buffer, Buffers...>) {
        return this->fill_from_<Offset>(x, rest...);
    }

    template <static_buffer Buffer, static_buffer... Buffers>
    constexpr auto&
    fill_from(Buffer const& x, Buffers const&... rest) noexcept(is_noexcept)
        requires(no_buffer_overflow<0, buffer_interface, Buffer, Buffers...>) {
        return this->fill_from_<0>(x, rest...);
    }

    template <size_t Offset, static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    extract() const noexcept(is_noexcept) -> decltype(auto) requires(
        is_valid_extraction<Offset, buffer_interface, Buffer, Buffers...>) {
        using result = //
            std::conditional_t<(sizeof...(Buffers) == 0), Buffer,
                               std::tuple<Buffer, Buffers...>>;

        result r;

        if constexpr(sizeof...(Buffers) == 0) {
            this->copy_into_<Offset>(r);
        } else {
            this->extract_tuple_<Offset, 0>(r);
        }

        return r;
    }

    template <static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    extract() const noexcept(is_noexcept) -> decltype(auto)
        requires(is_valid_extraction<0, buffer_interface, Buffer, Buffers...>) {
        return this->extract<0, Buffer, Buffers...>();
    }

    ////////////////////////////////////////////////////////////////////////////
    // View operations.
    ////////////////////////////////////////////////////////////////////////////

    template <size_t Offset, static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    view_as() noexcept -> decltype(auto) requires(
        no_buffer_overflow<Offset, buffer_interface, Buffer, Buffers...>) {
        return this->view_as_<Offset, buffer_interface, Buffer, Buffers...>(
            *this);
    }

    template <size_t Offset, static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    view_as() const noexcept -> decltype(auto) requires(
        no_buffer_overflow<Offset, buffer_interface, Buffer, Buffers...>) {
        return this
            ->view_as_<Offset, buffer_interface const, Buffer, Buffers...>(
                *this);
    }

    template <static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    view_as() noexcept -> decltype(auto)
        requires(no_buffer_overflow<0, buffer_interface, Buffer, Buffers...>) {
        return this->view_as<0, Buffer, Buffers...>();
    }

    template <static_buffer Buffer, static_buffer... Buffers>
    constexpr auto
    view_as() const noexcept -> decltype(auto)
        requires(no_buffer_overflow<0, buffer_interface, Buffer, Buffers...>) {
        return this->view_as<0, Buffer, Buffers...>();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Conversion operators.
    ////////////////////////////////////////////////////////////////////////////

    operator mut_ref() noexcept {
        return {this->data()};
    }

    operator ref() const noexcept {
        return {this->data()};
    }

    operator std::span<value_type, Storage::static_size>() noexcept {
        return {this->data(), this->size()};
    }

    operator std::span<std::add_const_t<value_type>, Storage::static_size>()
        const noexcept {
        return {this->data(), this->size()};
    }

    ////////////////////////////////////////////////////////////////////////////
    // Comparison operators.
    ////////////////////////////////////////////////////////////////////////////

    template <static_buffer Buffer>
    auto
    operator<=>(Buffer const& other) const -> decltype(auto)
        requires(are_comparable_buffers<buffer_interface, Buffer>) {
        return std::lexicographical_compare_three_way(
            this->begin(), this->end(), other.begin(), other.end());
    }

    template <static_buffer Buffer>
    auto
    operator==(Buffer const& other) const -> decltype(auto)
        requires(are_comparable_buffers<buffer_interface, Buffer>) {
        return std::ranges::equal(*this, other);
    }

    template <static_buffer Buffer>
    auto
    operator!=(Buffer const& other) const -> decltype(auto)
        requires(are_comparable_buffers<buffer_interface, Buffer>) {
        return !(std::ranges::equal(*this, other));
    }

private:
    ////////////////////////////////////////////////////////////////////////////
    // Private utility functions.
    ////////////////////////////////////////////////////////////////////////////

    template <size_t Offset, typename Buffer, typename... Buffers>
    constexpr void
    copy_into_(Buffer& x, Buffers&... rest) const noexcept(is_noexcept) {
        std::copy_n(this->data() + Offset, x.static_size(), x.data());

        if constexpr(sizeof...(Buffers) != 0) {
            this->copy_into_<Offset + Buffer::static_size()>(rest...);
        }
    }

    template <size_t Offset, typename Buffer, typename... Buffers>
    constexpr auto&
    fill_from_(Buffer const& x, Buffers const&... rest) noexcept(is_noexcept) {
        std::copy_n(x.data(), x.static_size(), this->data() + Offset);

        if constexpr(sizeof...(Buffers) != 0) {
            this->fill_from_<Offset + Buffer::static_size()>(rest...);
        }

        return *this;
    }

    template <size_t Offset, typename Buffer_src, typename Buffer_dst0,
              typename... Buffers_dst>
    static constexpr auto
    view_as_(Buffer_src& self) noexcept -> decltype(auto) {
        if constexpr(sizeof...(Buffers_dst) == 0) {
            return detail::select_ref<Buffer_src, Buffer_dst0>{self.data() +
                                                               Offset};
        } else {
            auto r =
                std::tuple<detail::select_ref<Buffer_src, Buffer_dst0>,
                           detail::select_ref<Buffer_src, Buffers_dst>...>{};

            view_as_tuple_<Offset, 0>(self, r);
            return r;
        }
    }

    template <size_t Offset, size_t I, typename Tuple>
    constexpr void
    extract_tuple_(Tuple& t) const noexcept(is_noexcept) {
        this->copy_into_<Offset>(std::get<I>(t));

        if constexpr((I + 1) < std::tuple_size_v<Tuple>) {
            this->extract_tuple_<
                Offset + std::tuple_element_t<I, Tuple>::static_size(), I + 1>(
                t);
        }
    }

    template <size_t Offset, size_t I, typename Buffer, typename Tuple>
    static constexpr void
    view_as_tuple_(Buffer& self, Tuple& t) noexcept {
        std::get<I>(t).data_ = self.data() + Offset;

        if constexpr((I + 1) < std::tuple_size_v<Tuple>) {
            view_as_tuple_<
                Offset + std::tuple_element_t<I, Tuple>::static_size(), I + 1>(
                self, t);
        }
    }

    template <typename T>
    friend struct buffer_interface;
};

////////////////////////////////////////////////////////////////////////////////
// Buffer normal storage definition.
////////////////////////////////////////////////////////////////////////////////

template <size_t N, typename T, typename Tag>
struct buffer_storage_normal {
    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr size_t static_size = N;

    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = T;
    using tag = Tag;

    ////////////////////////////////////////////////////////////////////////////
    // Data members.
    ////////////////////////////////////////////////////////////////////////////

    T data_[N];
};

template <typename T, typename Tag>
struct buffer_storage_normal<0, T, Tag> {
    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr size_t static_size = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = T;
    using tag = Tag;

    ////////////////////////////////////////////////////////////////////////////
    // Data members.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr auto data_ = static_cast<value_type*>(nullptr);
};

////////////////////////////////////////////////////////////////////////////////
// Buffer secure storage definition. Upon its destruction fills the data with
// statically initialized objects.
////////////////////////////////////////////////////////////////////////////////

template <size_t N, typename T, typename Tag>
struct buffer_storage_secure {
    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr size_t static_size = N;

    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = T;
    using tag = Tag;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/destruction.
    ////////////////////////////////////////////////////////////////////////////

    ~buffer_storage_secure() {
        auto volatile ptr = this->data_;
        std::fill_n(ptr, N, T{});
    }

    ////////////////////////////////////////////////////////////////////////////
    // Data members.
    ////////////////////////////////////////////////////////////////////////////

    T data_[N];
};

template <typename T, typename Tag>
struct buffer_storage_secure<0, T, Tag> {
    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr size_t static_size = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = T;
    using tag = Tag;

    ////////////////////////////////////////////////////////////////////////////
    // Data members.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr auto data_ = static_cast<value_type*>(nullptr);
};

////////////////////////////////////////////////////////////////////////////////
// Buffer reference storage definition. Holds a pointer to elements which belong
// to another container.
////////////////////////////////////////////////////////////////////////////////

template <size_t N, typename T, typename Tag>
struct buffer_storage_reference {
    ////////////////////////////////////////////////////////////////////////////
    // Constants.
    ////////////////////////////////////////////////////////////////////////////

    static constexpr size_t static_size = N;

    ////////////////////////////////////////////////////////////////////////////
    // Value and tag type definitions.
    ////////////////////////////////////////////////////////////////////////////

    using value_type = T;
    using tag = Tag;

protected:
    ////////////////////////////////////////////////////////////////////////////
    // Construction/destruction.
    ////////////////////////////////////////////////////////////////////////////

    buffer_storage_reference() noexcept : data_{} {
    }

    buffer_storage_reference(T* x) noexcept : data_{x} {
    }

    template <typename Storage>
    friend struct buffer_interface;

    ////////////////////////////////////////////////////////////////////////////
    // Data members.
    ////////////////////////////////////////////////////////////////////////////

    T* data_;
};

////////////////////////////////////////////////////////////////////////////////
// Buffer viewing and joining interface.
////////////////////////////////////////////////////////////////////////////////

// Constructs buffer references of the given size to sequential regions of the
// given buffer.
template <size_t Chunk_size, static_buffer Buffer>
constexpr auto
view_buffer_by_chunks(Buffer& x) noexcept -> decltype(auto)
    requires((Chunk_size != 0) && ((Buffer::static_size() % Chunk_size) == 0)) {
    using view = detail::select_ref<
        Buffer,
        buffer<Chunk_size, typename Buffer::value_type, typename Buffer::tag>>;

    return ([&x]<size_t... Indices>(std::index_sequence<Indices...>) {
        return buffer<sizeof...(Indices), view>{
            x.template view_as<Indices * Chunk_size, view>()...};
    })(std::make_index_sequence<Buffer::static_size() / Chunk_size>{});
}

template <static_buffer Buffer, static_buffer... Buffers>
constexpr auto
join_buffers( //
    Buffer const& x, Buffers const&... rest) noexcept(Buffer::is_noexcept)
    -> decltype(auto) {
    buffer_compound<Buffer, Buffers...> r;
    r.fill_from(x, rest...);

    return r;
}

template <static_buffer Buffer, static_buffer... Buffers>
constexpr auto
join_buffers_secure( //
    Buffer const& x, Buffers const&... rest) noexcept(Buffer::is_noexcept)
    -> decltype(auto) {
    buffer_compound_secure<Buffer, Buffers...> r;
    r.fill_from(x, rest...);

    return r;
}

////////////////////////////////////////////////////////////////////////////////
// Buffer conversion utilities.
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// A helper concept which models any integer type, except for boolean.
template <typename T>
concept integer = //
    std::integral<T> && (!std::same_as<std::remove_cv_t<T>, bool>);

} // namespace detail

// Size of integer type's value representation.
template <detail::integer T>
constexpr auto integer_size = static_sum<
    ((std::numeric_limits<std::make_unsigned_t<T>>::digits / CHAR_BIT)),
    ((std::numeric_limits<std::make_unsigned_t<T>>::digits % CHAR_BIT) != 0)>;

// A predicate which shows that an object of the given integer type can be
// converted to/from an object of the buffer type.
template <detail::integer T, static_byte_buffer Buffer>
constexpr auto is_valid_buffer_conversion = //
    (Buffer::static_size() == integer_size<T>);

////////////////////////////////////////////////////////////////////////////////
// Buffer conversion interface.
////////////////////////////////////////////////////////////////////////////////

template <detail::integer T, static_byte_buffer Buffer>
constexpr void
int_to_buffer(T x, Buffer& buffer) noexcept
    requires(is_valid_buffer_conversion<T, Buffer>) {
    for(size_t i = 0; i < integer_size<T>; ++i) {
        buffer[i] = static_cast<unsigned char>(
            static_cast<std::make_unsigned_t<T>>(x) >> (i * CHAR_BIT));
    }
}

template <detail::integer T>
constexpr auto
int_to_buffer(T x) noexcept -> decltype(auto) {
    buffer<integer_size<T>> r;
    int_to_buffer(x, r);

    return r;
}

template <detail::integer T, static_byte_buffer Buffer>
constexpr void
buffer_to_int(Buffer const& buffer, T& x) noexcept
    requires(is_valid_buffer_conversion<T, Buffer>) {
    using unsigned_type = std::make_unsigned_t<T>;

    auto r = unsigned_type{};
    for(size_t i = 0; i < integer_size<T>; ++i) {
        r |= static_cast<unsigned_type>(buffer[i]) << (i * CHAR_BIT);
    }

    x = static_cast<T>(r);
}

template <detail::integer T, static_byte_buffer Buffer>
constexpr auto
buffer_to_int(Buffer const& buffer) noexcept -> T
    requires(is_valid_buffer_conversion<T, Buffer>) {
    T r;
    buffer_to_int(buffer, r);

    return r;
}

////////////////////////////////////////////////////////////////////////////////
// Byte sequence definitions.
////////////////////////////////////////////////////////////////////////////////

using byte_sequence = std::span<unsigned char const>;
using byte_sequence_mutable = std::span<unsigned char>;

////////////////////////////////////////////////////////////////////////////////
// Byte sequence construction interface.
////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N>
constexpr auto
as_byte_sequence_mutable(std::span<T, N> span) -> byte_sequence_mutable {
    return byte_sequence_mutable{
        reinterpret_cast<unsigned char*>(span.data()), span.size_bytes()};
}

template <typename T, size_t N>
constexpr auto
as_byte_sequence(std::span<T, N> span) -> byte_sequence {
    return byte_sequence{
        reinterpret_cast<unsigned char const*>(span.data()), span.size_bytes()};
}

} // namespace rose

#endif // H_639425CCA60E448B9BEB43186E06CA57
