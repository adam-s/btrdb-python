# tests.test_timez
# Testing for the btrdb.timez module
#
# Author:   PingThings
# Created:  Mon Dec 17 15:23:25 2018 -0500
#
# For license information, see LICENSE.txt
# ID: test_timez.py [] allen@pingthings.io $

"""
Testing for the btrdb.timez module
"""

##########################################################################
## Imports
##########################################################################

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytz

from btrdb.utils.timez import (
    currently_as_ns,
    datetime_to_ns,
    ns_delta,
    ns_to_datetime,
    to_nanoseconds,
)

##########################################################################
## Initialization Tests
##########################################################################


class TestCurrentlyAsNs(object):
    def test_currently_as_ns(self):
        """
        Assert currently_as_ns returns the clock's nanoseconds, unrounded
        """
        expected = 1514808000123456789
        with patch("time.time_ns", return_value=expected):
            assert currently_as_ns() == expected


class TestDatetimeToNs(object):
    def test_datetime_to_ns_naive(self):
        """
        Assert datetime_to_ns handles naive datetime
        """
        dt = datetime.datetime(2018, 1, 1, 12, 0, 0, 123457)
        expected = 1514808000123457000

        assert dt.tzinfo is None
        assert datetime_to_ns(dt) == expected

    def test_datetime_to_ns_aware(self):
        """
        Assert datetime_to_ns handles tz aware datetime
        """
        eastern = pytz.timezone("US/Eastern")
        dt = datetime.datetime(2018, 1, 1, 17, 0, 0, 123457, tzinfo=eastern)
        expected = 1514843760123457000

        assert dt.tzinfo is not None
        assert datetime_to_ns(dt) == expected
        assert (
            datetime_to_ns(pd.Timestamp("2018-01-01 12:00:00.123456789", tz="UTC"))
            == 1514808000123456789
        )


class TestNsToDatetime(object):
    def test_ns_to_datetime_is_utc(self):
        """
        Assert ns_to_datetime returns UTC aware datetime
        """
        dt = datetime.datetime.utcnow()
        ns = int(dt.timestamp() * 1e9)

        assert dt.tzinfo is None
        assert ns_to_datetime(ns).tzinfo == pytz.UTC

    def test_ns_to_datetime_is_correct(self):
        """
        Assert ns_to_datetime returns correct datetime
        """
        val = 1514808000123457501
        expected = datetime.datetime(2018, 1, 1, 12, 0, 0, 123458, tzinfo=pytz.UTC)

        assert ns_to_datetime(val) == expected


class TestToNanoseconds(object):
    def test_datetime_to_ns_naive(self):
        """
        Assert to_nanoseconds handles naive datetime
        """
        dt = datetime.datetime(2018, 1, 1, 12, 0, 0, 123457)
        expected = 1514808000123457000

        assert dt.tzinfo is None
        assert to_nanoseconds(dt) == expected

    def test_datetime_to_ns_aware(self):
        """
        Assert to_nanoseconds handles tz aware datetime
        """
        eastern = pytz.timezone("US/Eastern")
        dt = datetime.datetime(2018, 1, 1, 17, tzinfo=eastern)
        expected = int(dt.astimezone(pytz.utc).timestamp() * 1e9)

        assert dt.tzinfo is not None
        assert to_nanoseconds(dt) == expected

    def test_str(self):
        """
        Assert to_nanoseconds handles RFC3339 format
        """
        dt = datetime.datetime(2018, 1, 1, 12, tzinfo=pytz.utc)
        expected = int(dt.timestamp() * 1e9)

        dt_str = "2018-1-1 12:00:00.0-0000"
        assert dt.tzinfo is not None
        assert to_nanoseconds(dt_str) == expected

        dt_str = "2018-1-1 7:00:00.0-0500"
        dt = datetime.datetime(2018, 1, 1, 12, tzinfo=pytz.timezone("US/Eastern"))
        assert dt.tzinfo is not None
        assert to_nanoseconds(dt_str) == expected

        dt_str = "2018-01-15 07:32:49"
        dt = datetime.datetime(2018, 1, 15, 7, 32, 49, tzinfo=pytz.utc)
        expected = int(dt.timestamp() * 1e9)
        assert to_nanoseconds(dt_str) == expected

        assert to_nanoseconds("2018-01-01T12:00:00.123456789Z") == 1514808000123456789
        assert to_nanoseconds("2018-01-01 12:00:00.1234567") == 1514808000123456700
        dt_str = "2018-01-01T13:00:00.12345678+01:00"
        assert to_nanoseconds(dt_str) == 1514808000123456780
        dt_str = "2018-01-01 12:00:01.1234567+00:00:01.000001"
        assert to_nanoseconds(dt_str) == 1514808000123455700

    def test_str_midnight(self):
        """
        Test parse a date at midnight
        """
        expected = datetime.datetime(2019, 4, 7, tzinfo=pytz.utc)
        expected = int(expected.timestamp() * 1e9)
        assert to_nanoseconds("2019-04-07") == expected

    def test_str_raise_valueerror(self):
        """
        Assert to_nanoseconds raises on invalid str
        """
        dt_str = "01 Jan 2018 12:00:00 -0000"
        with pytest.raises(ValueError, match="RFC3339") as exc:
            to_nanoseconds(dt_str)
        with pytest.raises(ValueError, match="RFC3339"):
            to_nanoseconds("2018-01-01T12:00:00.1234567891Z")  # ten decimals
        with pytest.raises(ValueError, match="RFC3339"):
            to_nanoseconds("2018-01-01 12:00:00+00:00:01.1234567")  # a zone's decimals

    def test_int(self):
        """
        Assert to_nanoseconds handles int
        """
        assert 42 == to_nanoseconds(42)

    def test_float(self):
        """
        Assert to_nanoseconds handles float
        """
        assert 42 == to_nanoseconds(42.0)

    def test_float_raise_valueerror(self):
        """
        Assert to_nanoseconds raises on invalid float
        """
        with pytest.raises(ValueError) as exc:
            to_nanoseconds(42.5)
        assert "can only convert whole numbers" in str(exc)

    def test_datetime64(self):
        """
        Assert to_nanoseconds handles datetime64, to the nanosecond
        """
        dt = datetime.datetime(2018, 1, 1, 12, tzinfo=pytz.utc)
        expected = int(dt.timestamp() * 1e9)

        dt64 = np.datetime64("2018-01-01T12:00")
        assert expected == to_nanoseconds(dt64)
        assert (
            to_nanoseconds(np.datetime64("2018-01-01T12:00:00.123456789"))
            == 1514808000123456789
        )
        assert (
            to_nanoseconds(np.datetime64("2300-01-01T00:00:00.123457"))
            == 10413792000123457000
        )
        assert to_nanoseconds(np.datetime64("2018-01")) == 1514764800000000000
        assert to_nanoseconds(np.datetime64("2018")) == 1514764800000000000
        assert to_nanoseconds(np.datetime64("2018-01-04", "W")) == 1515024000000000000
        assert to_nanoseconds(np.datetime64("2018-01-01T12")) == 1514808000000000000
        assert (
            to_nanoseconds(np.datetime64("2018-01-01T12:00:00.123"))
            == 1514808000123000000
        )
        assert (
            to_nanoseconds(np.datetime64(151480800012, "10ms")) == 1514808000120000000
        )
        assert to_nanoseconds(np.datetime64(-1500, "ps")) == -2  # rounded down
        # the lowest counts, and a unit with a multiplier
        assert to_nanoseconds(np.datetime64(-(2**63) + 1, "ps")) == -9223372036854776
        assert to_nanoseconds(np.datetime64(-(2**63) + 1, "as")) == -9223372037
        assert to_nanoseconds(np.datetime64(2 * 10**18, "7fs")) == 14000000000000

    def test_datetime64_raise_typeerror(self):
        """
        Assert to_nanoseconds raises on a datetime64 that holds no time
        """
        no_unit = np.array([5], dtype=np.int64).view("M8")[0]
        for dt64 in (np.datetime64("NaT", "ns"), no_unit):
            with pytest.raises(TypeError):
                to_nanoseconds(dt64)


class TestToNsDelta(object):
    def test_ns_delta(self):
        """
        Assert ns_delta converts inputs properly
        """
        val = ns_delta(1, 2, 1, 3, 1, 23, 1)
        assert val == 93663001023001

    def test_ns_delta_precision(self):
        """
        Assert ns_delta deals with real inputs
        """
        val = ns_delta(days=365, minutes=0.5, nanoseconds=1)
        assert val == int(1e9 * 60 * 60 * 24 * 365) + int(1e9 * 30) + 1

    def test_returns_int(self):
        """
        Assert ns_delta returns int if floats used for arguments
        """
        val = ns_delta(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1)
        assert val == int(
            1 + 1e3 + 1e6 + 1e9 + (1e9 * 60) + (1e9 * 60 * 60) + (1e9 * 60 * 60 * 24)
        )
        assert isinstance(val, int)
