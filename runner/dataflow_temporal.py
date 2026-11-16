"""Temporal wire contract v1: exact integer physical values with typed schema.

Date = days since 1970-01-01; datetime = UTC microseconds since that epoch;
time = microseconds since midnight (no timezone). Datetime timezone belongs to
column metadata and changes display/extraction, never the represented instant.
No float timestamp conversion or machine-local timezone is used.
"""
from datetime import date, datetime, time, timedelta, timezone as dt_timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DTYPES = ('date', 'datetime', 'time')
DAY_US = 86_400_000_000
EPOCH_DATE = date(1970, 1, 1)
EPOCH = datetime(1970, 1, 1, tzinfo=dt_timezone.utc)
# Locale-independent numeric directives only. Week/calendar arithmetic is deferred.
DIRECTIVES = frozenset('YmdHMSfz%')
PARTS = {'date': ('year','month','day','weekday','dayOfYear'),
         'datetime': ('year','month','day','weekday','dayOfYear','hour','minute','second','microsecond','utcOffsetSeconds'),
         'time': ('hour','minute','second','microsecond')}

class PrecisionError(ValueError): pass
class LocalTimeError(ValueError): pass


def zone(name):
    if type(name) is not str or not name or len(name) > 128:
        raise ValueError('Choose an explicit IANA timezone (for example UTC)')
    try: return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError('Unknown IANA timezone') from error


def metadata(dtype, timezone='UTC'):
    if dtype not in DTYPES: raise ValueError('Choose Date, DateTime or Time')
    result = {'version': 1, 'unit': 'day' if dtype == 'date' else 'microsecond'}
    if dtype == 'datetime':
        zone(timezone); result['timezone'] = timezone
    return result


def validate_column(column):
    dtype = column.get('dtype')
    if dtype in DTYPES:
        supplied = column.get('temporal')
        if type(supplied) is not dict or type(supplied.get('version')) is not int or supplied != metadata(dtype, supplied.get('timezone', 'UTC')):
            raise ValueError('Temporal columns require canonical version1 precision/timezone metadata')
    elif 'temporal' in column: raise ValueError('Temporal metadata requires a temporal dtype')


def valid_value(value, dtype):
    if type(value) is not int: return False
    if dtype == 'date': return (date.min-EPOCH_DATE).days <= value <= (date.max-EPOCH_DATE).days
    if dtype == 'time': return 0 <= value < DAY_US
    if dtype == 'datetime':
        return delta_us(datetime.min.replace(tzinfo=dt_timezone.utc)-EPOCH) <= value <= delta_us(datetime.max.replace(tzinfo=dt_timezone.utc)-EPOCH)
    return False


def delta_us(delta):
    return delta.days*DAY_US + delta.seconds*1_000_000 + delta.microseconds


def checked_pattern(pattern):
    if pattern == 'iso': return pattern
    if type(pattern) is not str or not 1 <= len(pattern) <= 200:
        raise ValueError('Use ISO or a bounded explicit pattern')
    index = 0
    while index < len(pattern):
        if pattern[index] == '%':
            index += 1
            if index == len(pattern) or pattern[index] not in DIRECTIVES:
                raise ValueError('Use invariant numeric pattern directives: %Y %m %d %H %M %S %f %z %%')
        index += 1
    return pattern


def pattern_directives(pattern):
    """Tokenize directives; an escaped percent consumes both characters."""
    checked_pattern(pattern)
    return {code for code in re.findall(r'%(.)', pattern) if code != '%'}


def localize(value, timezone, ambiguous, nonexistent):
    tz = zone(timezone)
    candidates = []
    for fold in (0, 1):
        aware = value.replace(tzinfo=tz, fold=fold)
        utc = aware.astimezone(dt_timezone.utc)
        if utc.astimezone(tz).replace(tzinfo=None) == value and utc not in candidates:
            candidates.append(utc)
    if not candidates:
        if nonexistent == 'missing': return None
        raise LocalTimeError('The local time is nonexistent during the timezone clock change')
    if len(candidates) > 1:
        if ambiguous == 'missing': return None
        if ambiguous == 'stop': raise LocalTimeError('The local time is ambiguous; choose earliest/latest or missing')
        return min(candidates) if ambiguous == 'earliest' else max(candidates)
    return candidates[0]


def parse(value, dtype, *, pattern='iso', source_timezone='', result_timezone='UTC', invalid='stop', ambiguous='stop', nonexistent='stop'):
    metadata(dtype, result_timezone); checked_pattern(pattern)
    if invalid not in ('stop','missing') or ambiguous not in ('stop','earliest','latest','missing') or nonexistent not in ('stop','missing'):
        raise ValueError('Choose explicit invalid/ambiguous/nonexistent policies')
    if source_timezone: zone(source_timezone)
    if value is None: return None
    if type(value) is not str or len(value) > 16384: raise ValueError('Temporal Parse needs bounded text')
    # Python fromisoformat silently truncates extra fractional digits: prohibit it.
    if re.search(r'[.,][0-9]{7,}', value): raise PrecisionError('Temporal precision exceeds microseconds; no rounding was applied')
    try:
        if pattern == 'iso':
            if dtype == 'date':
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None: raise ValueError('Use ISO YYYY-MM-DD')
                parsed = date.fromisoformat(value)
            elif dtype == 'datetime':
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?', value) is None: raise ValueError('Use explicit ISO date and time with seconds')
                parsed = datetime.fromisoformat(value)
            else:
                if re.fullmatch(r'\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?', value) is None: raise ValueError('Use ISO time without an offset')
                parsed = time.fromisoformat(value)
        else:
            directives = pattern_directives(pattern)
            required = set('Ymd') if dtype == 'date' else set('HMS') if dtype == 'time' else set('YmdHMS')
            if not required <= directives:raise ValueError('Custom pattern must explicitly include every date/time component')
            parsed = datetime.strptime(value, pattern)
            if dtype == 'date':
                if directives & set('HMSfz'): raise ValueError('Date pattern cannot discard time or offset')
                parsed = parsed.date()
            elif dtype == 'time':
                if directives & set('Ymdz'): raise ValueError('Time pattern cannot discard date or offset')
                parsed = parsed.time()
        if dtype == 'date': return (parsed-EPOCH_DATE).days
        if dtype == 'time': return ((parsed.hour*60+parsed.minute)*60+parsed.second)*1_000_000+parsed.microsecond
        if parsed.tzinfo is None:
            if not source_timezone: raise LocalTimeError('Naive datetime needs an explicit source timezone')
            parsed = localize(parsed, source_timezone, ambiguous, nonexistent)
            if parsed is None: return None
        return delta_us(parsed.astimezone(dt_timezone.utc)-EPOCH)
    except (PrecisionError, LocalTimeError): raise
    except (ValueError, OverflowError) as error:
        if invalid == 'missing': return None
        raise ValueError(f'Invalid {dtype} value: {error}') from error


def decode(value, dtype, timezone='UTC'):
    if value is None: return None
    if not valid_value(value, dtype): raise ValueError('Invalid canonical temporal value')
    try:
        if dtype == 'date': return EPOCH_DATE+timedelta(days=value)
        if dtype == 'datetime': return (EPOCH+timedelta(microseconds=value)).astimezone(zone(timezone))
        hour, remainder = divmod(value, 3_600_000_000)
        minute, remainder = divmod(remainder, 60_000_000)
        second, microsecond = divmod(remainder, 1_000_000)
        return time(hour,minute,second,microsecond)
    except OverflowError as error: raise ValueError('Temporal timezone display exceeds supported calendar range') from error


def format_value(value, dtype, *, pattern='iso', timezone='UTC'):
    checked_pattern(pattern)
    if dtype == 'datetime': zone(timezone)
    parsed = decode(value,dtype,timezone)
    if parsed is None: return None
    if pattern == 'iso': return parsed.isoformat(timespec='microseconds') if dtype != 'date' else parsed.isoformat()
    # Refuse formatting nonexistent portions rather than emitting invented 1900/00 values.
    allowed = {'date':set('Ymd%'), 'time':set('HMSf%'), 'datetime':DIRECTIVES}[dtype]
    if any(code not in allowed for code in re.findall(r'%(.)',pattern)): raise ValueError('Format pattern references an unavailable temporal part')
    # Platform strftime may omit leading zeros for years below 1000.
    parts={'%':'%'}
    for code,attr,width in (('Y','year',4),('m','month',2),('d','day',2),('H','hour',2),('M','minute',2),('S','second',2),('f','microsecond',6)):
        if hasattr(parsed,attr):parts[code]=f'{getattr(parsed,attr):0{width}d}'
    if dtype=='datetime':parts['z']=parsed.strftime('%z')
    return re.sub(r'%(.)',lambda match:parts[match.group(1)],pattern)


def extract(value, dtype, part, *, timezone='UTC'):
    if part not in PARTS.get(dtype, ()): raise ValueError('Choose a part supported by the temporal type')
    parsed = decode(value,dtype,timezone)
    if parsed is None: return None
    if part == 'weekday': return parsed.isoweekday()
    if part == 'dayOfYear': return parsed.timetuple().tm_yday
    if part == 'utcOffsetSeconds': return delta_us(parsed.utcoffset())//1_000_000
    return getattr(parsed,part)


def difference(left, right, dtype, *, unit='seconds', representation='decimal'):
    units={'days':DAY_US,'hours':3_600_000_000,'minutes':60_000_000,'seconds':1_000_000,'milliseconds':1000,'microseconds':1}
    if dtype not in DTYPES or unit not in units or representation not in ('integer','decimal'):
        raise ValueError('Choose compatible temporal columns, explicit units and integer/decimal output')
    if left is None or right is None: return None
    if not valid_value(left,dtype) or not valid_value(right,dtype): raise ValueError('Invalid canonical temporal value')
    amount=(left-right)*(DAY_US if dtype=='date' else 1)
    divisor=units[unit]
    if representation=='integer':
        if amount%divisor: raise ValueError('Difference cannot be represented as an exact integer in these units')
        result=amount//divisor
        if not -(2**63)<=result<2**63: raise ValueError('Temporal difference overflows int64')
        return result
    result=amount/divisor
    # Decimal output explicitly requests float64; integer microseconds remains exact.
    return result
