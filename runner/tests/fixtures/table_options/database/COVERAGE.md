# Database option evidence

Synthetic selected-table fixtures only. `selection.sql` and `selection.json` provide independently written rows and expected selection results. PostgreSQL and MySQL start private disposable local servers through the existing test fixtures and are stopped afterward. TLS keys/certificates exist only under pytest temporary directories; no global trust, Keychain, production data or persisted service changes.

`test_table_options_database.OPTION_CASES` maps all 44 exposed option/backend combinations to executed tests: 12 SQLite settings and 16 each for PostgreSQL/MySQL. SQLite hides host, port, database and TLS in the native panel, so those four are intentionally excluded there.

| Option | Observable assertion |
|---|---|
| backend | Real SQLite, PostgreSQL and MySQL captures, separately selected by their backend |
| host / port / database | Real disposable endpoint/database success; different host, closed port and absent database refuse |
| schema | SQLite main/empty aliases and foreign-schema refusal; PostgreSQL actual second schema returns its distinct row, public/main/empty aliases; MySQL matching/empty/main aliases and mismatched-schema refusal |
| table | Alternative table returns distinct independently expected data; missing table refuses |
| columns | Ordered projection changes both names/order and values; missing column refuses |
| orderBy | Multiple ascending/descending keys produce exact expected row order; missing key refuses |
| rowLimit | Zero retains all four observations; nonzero limits to independently expected prefix |
| rowOffset | Skips independently expected rows, both alone and with a limit |
| fetchBatchRows | 1 produces four real IPC record batches; 8192 produces one; values remain identical |
| timeoutSeconds | SQLite busy-lock wait ends around one second; PostgreSQL statement and MySQL read/query timeout refuse an actually sleeping query within a finite bound |
| tlsMode | Both real remote engines accept verified TLS using temporary CA trusted only by the pytest process; untrusted CA and wrong hostname reject; literal-loopback plaintext works; nonliteral/nonloopback plaintext refuses |
| decimalMode | PostgreSQL/MySQL exact Decimal values retain trailing decimal zeros as text; error policy refuses; SQLite native numeric float remains float under either policy |
| dateTimeMode | PostgreSQL date/timestamp/time and MySQL date/datetime produce expected ISO8601 text; MySQL TIME(6) preserves signed durations, days, zero and microseconds as exact ISO8601 duration text; error policy refuses those typed values; SQLite date storage is already text and remains unchanged under either policy |
| binaryMode | All engines return exact independently expected hexadecimal/base64 strings for `00 ff 41`; default error refuses |

## Honest limits

- SQLite's timeout is currently the sqlite3 busy-lock wait. These tests do not establish a per-source CPU/query deadline for a large sort. Cancellation remains separately supported.
- SQLite's ordinary driver delivers DATE as string and NUMERIC as native float/int; the decoding controls do not reparse already typed values or manufacture Decimal/date objects.
- The MySQL duration fixture first failed on unsupported `datetime.timedelta`, recorded in `.local/table-options-database-duration-red.log`. Root repaired only MySQL TIME conversion using integer microseconds. PostgreSQL calendar INTERVAL remains explicitly refused, avoiding a driver approximation of months as fixed days.
- TLS success uses an environment variable scoped to the test process to trust the disposable CA. It does not establish a user-facing custom-CA control or change system trust. Native production helper does not inherit this fixture trust variable.
- These are execution/decoder fixtures. Root owns native configuration-panel visibility, editing/persistence and app/XPC acceptance; no claim of a new signed release is made.

## Fresh MySQL authentication regression

The twelfth test creates a unique SELECT-only `caching_sha2_password` role with a nonempty synthetic password and makes `database_source.capture` its first connection over explicit literal-loopback plaintext. It verifies returned values/scalar types, then independently verifies INSERT is denied. Credentials stay in memory; they are never passed in argv/environment or printed. OPTION_CASES maps this case to MySQL backend and TLS policy.

This first-auth fixture exposed PyMySQL 1.2.3's missing return after successful full RSA authentication. The server can authenticate and warm its cache before that client crash, so a successful retry cannot substitute for first-auth evidence. The verified immutable upstream fix passes the genuine first-connection test. RED receipt: `.local/table-options-database-fresh-rsa-red.log`; GREEN full suite: `.local/table-options-database-fresh-rsa-green.log` (12 passed).
