#ifndef BOOST_TEST_LESS_THAN_CONSTEXPR_HPP
#define BOOST_TEST_LESS_THAN_CONSTEXPR_HPP

//  Copyright (c) 2019 Robert Ramey
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp> // BOOST_CLANG
#include <boost/safe_numerics/safe_integer.hpp>

#if BOOST_CLANG==1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-comparison"
#endif

template<class T1, class T2>
constexpr bool test_less_than_constexpr(
    T1 v1,
    T2 v2,
    char expected_result
){
    using namespace boost::safe_numerics;
    // if we don't expect the operation to pass, we can't
    // check the constexpr version of the calculation so
    // just return success.
    if(expected_result == 'x')
        return true;
    safe_t<T1>(v1) < v2;
    v1 < safe_t<T2>(v2);
    safe_t<T1>(v1) < safe_t<T2>(v2);
    return true; // correct result
}

#if BOOST_CLANG==1
#pragma GCC diagnostic pop
#endif

#endif // BOOST_TEST_LESS_THAN_CONSTEXPR_HPP
