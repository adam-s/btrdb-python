# btrdb.utils.timez
# Conversion utilities for btrdb
#
# Author:   PingThings
# Created:  Wed Jan 02 17:00:49 2019 -0500
#
# Copyright (C) 2018 PingThings LLC
# For license information, see LICENSE.txt
#
# ID: timez.py [] allen@pingthings.io $

"""
Time related utilities
"""

##########################################################################
## Imports
##########################################################################

import re
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from operator import mul

import pytz

##########################################################################
## Module Variables
##########################################################################

EPOCH = datetime(1970, 1, 1, tzinfo=pytz.utc)
DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f%z",  # most common RFC3339 nanoseconds
    "%Y-%m-%d %H:%M:%S.%f",  # expects UTC default timezone
    "%Y-%m-%dT%H:%M:%S.%fZ",  # JSON encoding, UTC timezone
    "%Y-%m-%dT%H:%M:%SZ",  # JSON encoding, UTC timezone
    "%Y-%m-%dT%H:%M:%S.%f%z",  # less common JSON-ish encoding
    "%Y-%m-%dT%H:%M:%S.%f",  # for completeness, UTC default timezone
    "%Y-%m-%d %H:%M:%S%z",  # human readable date time with TZ
    "%Y-%m-%d %H:%M:%S",  # human readable date time UTC default
    "%Y-%m-%d",  # helper to get midnight on a particular date
)


##########################################################################
## Functions
##########################################################################


def currently_as_ns():
    """
    Returns the current UTC time as nanoseconds since epoch
    """
    return time.time_ns()


def ns_to_datetime(ns):
    """
    Converts nanoseconds to a naive datetime object (UTC+0)

    Parameters
    ----------
    ns : int
        nanoseconds since epoch

    Returns
    -------
    nanoseconds since epoch as a datetime object : datetime

    """
    return EPOCH + timedelta(microseconds=int(round(Fraction(ns) / 1000)))


def datetime_to_ns(dt):
    """
    Converts a datetime object to nanoseconds since epoch.  If a timezone aware
    object is received then it will be converted to UTC.

    Parameters
    ----------
    dt : datetime

    Returns
    -------
    nanoseconds : int

    """
    # from its fields, in integers, so nothing is rounded
    offset = dt.utcoffset()  # None when naive: the time is taken as UTC
    days = dt.toordinal() - EPOCH.toordinal()
    seconds = ((days * 24 + dt.hour) * 60 + dt.minute) * 60 + dt.second
    ns = seconds * 1_000_000_000 + dt.microsecond * 1000 + getattr(dt, "nanosecond", 0)
    return ns - offset // timedelta(microseconds=1) * 1000 if offset else ns


def to_nanoseconds(val):
    """
    Converts datetime, datetime64, float, str (RFC 2822) to nanoseconds.  If a
    datetime-like object is received then nanoseconds since epoch is returned.

    Parameters
    ----------
    val : datetime, datetime64, float, str
        an object to convert to nanoseconds

    Returns
    -------
    object converted to nanoseconds : int

    Notes
    ----
    The following string formats are supported for conversion.

    +--------------------------------+------------------------------------------+
    | Format String                  | Description                              |
    +================================+==========================================+
    | %Y-%m-%d %H:%M:%S.%f%z         | RFC3339 format                           |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%d %H:%M:%S.%f           | RFC3339 with UTC default timezone        |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%dT%H:%M:%S.%fZ          | JSON encoding, UTC timezone              |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%dT%H:%M:%SZ             | JSON encoding, UTC timezone, without μs  |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%dT%H:%M:%S.%f%z         | JSON-like encoding                       |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%dT%H:%M:%S.%f           | JSON-like encoding, UTC default timezone |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%d %H:%M:%S%z            | human readable date time with TZ         |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%d %H:%M:%S              | human readable date time UTC default     |
    +--------------------------------+------------------------------------------+
    | %Y-%m-%d                       | midnight at a particular date            |
    +--------------------------------+------------------------------------------+

    """
    if val is None or isinstance(val, int):
        return val

    try:
        import numpy as np

        # no unit (dtype M8, numpy's "generic") means no time: TypeError below
        if isinstance(val, np.datetime64) and val.dtype != "M8" and not np.isnat(val):
            # count with Python integers: numpy's own casts can wrap past their range
            unit, step = np.datetime_data(val.dtype)
            count = int(val.view(np.int64)) * step
            if unit in ("Y", "M"):  # the calendar repeats every 400 years (146097 days)
                year, month = divmod(count * (12 if unit == "Y" else 1) + 1970 * 12, 12)
                cycles, year = divmod(year - 1, 400)
                days = date(year + 1, month + 1, 1).toordinal() - EPOCH.toordinal()
                count, unit = days + cycles * 146097, "D"
            if unit in ("ps", "fs", "as"):  # below 1 ns, round down as numpy does
                return count // 10 ** {"ps": 3, "fs": 6, "as": 9}[unit]
            count *= {"W": 604800, "D": 86400, "h": 3600, "m": 60}.get(unit, 1)
            return count * 10 ** {"ms": 6, "us": 3, "ns": 0}.get(unit, 9)
    except ImportError:
        pass

    if isinstance(val, str):
        # handle int as string
        if val.isdigit():
            return int(val)

        # strptime reads six decimals of the seconds at most; the rest are nanoseconds
        m = re.fullmatch(r"(\d+-\d+-\d+[\sTt]+\d+:\d+:\d+\.\d{6})(\d{1,3})(\D.*)?", val)
        extra, val = (int(m[2].ljust(3, "0")), m[1] + (m[3] or "")) if m else (0, val)

        # handle datetime as string
        for format in DATETIME_FORMATS:
            try:
                val = datetime.strptime(val, format)
                break
            except ValueError:
                pass

        if isinstance(val, str):
            raise ValueError("unsupported string format, please use RFC3339")
        return datetime_to_ns(val) + extra

    if isinstance(val, datetime):
        return datetime_to_ns(val)

    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        else:
            raise ValueError("can only convert whole numbers to nanoseconds")

    raise TypeError("only int, float, str, datetime, and datetime64 are allowed")


def ns_delta(
    days=0, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0, nanoseconds=0
):
    """
    Similar to ``timedelta``, ``ns_delta`` represents a span of time but as
    the total number of nanoseconds.

    Parameters
    ----------
    days : int, float, decimal.Decimal
        days (as 24 hours) to convert to nanoseconds
    hours : int, float, decimal.Decimal
        hours to convert to nanoseconds
    minutes : int, float, decimal.Decimal
        minutes to convert to nanoseconds
    seconds : int, float, decimal.Decimal
        seconds to convert to nanoseconds
    milliseconds : int, float, decimal.Decimal
        milliseconds to convert to nanoseconds
    microseconds : int, float, decimal.Decimal
        microseconds to convert to nanoseconds
    nanoseconds : int
        nanoseconds to add to the time span

    Returns
    -------
    amount of time in nanoseconds : int


    Examples
    --------
    1 minute time delta should be 60 billion nanoseconds

    >>> deltaT = ns_delta(minutes=1)
    >>> deltaT == 1 * 60 * 10**9 # 1 minute * 60 seconds * 1billion nanoseconds / second
    True
    """
    MICROSECOND = 1000
    MILLISECOND = MICROSECOND * 1000
    SECOND = MILLISECOND * 1000
    MINUTE = SECOND * 60
    HOUR = MINUTE * 60
    DAY = HOUR * 24

    if not isinstance(nanoseconds, int):
        raise TypeError("nanoseconds argument must be an integer")

    units = []
    for unit in (days, hours, minutes, seconds, milliseconds, microseconds):
        if isinstance(unit, float):
            unit = Decimal(unit)
        units.append(unit)

    factors = [DAY, HOUR, MINUTE, SECOND, MILLISECOND, MICROSECOND]
    nanoseconds += sum(map(mul, units, factors))
    return int(nanoseconds)
