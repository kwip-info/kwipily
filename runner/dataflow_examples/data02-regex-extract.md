# Extract a domain into its own column

Keep each order and add the text after @ as a new email_domain column.

**Capture group:** a named part of a regex pattern whose text can be copied out. **Domain:** the part after @ in this example. **No match:** the pattern did not find the requested part; it is not a made-up empty domain.

1. Open Extract a domain into its own column. Select Formula, using Extract groups and the email column.
2. Select Formula. Choose the Extract groups guided editor. Its email pattern names the domain capture; the output grid writes that capture to a new email_domain text column.
3. Keep no-match as null and missing input as preserve. Predict A01, A03 and A05: a domain, no matching text, and a missing input. Run Flow.
4. Inspect the original email beside email_domain. Confirm all eight original order IDs remain. A capture changes columns, not the number of orders.
5. Try a capture name absent from the pattern. The app should explain the missing group. Restore domain; choose a unique output name unless replacing a column is deliberate.

## Inspect

Domains are example.com, example.org, null, north.example, null, example.com, example.org and example.com. The first match is used per row. Both unmatched and missing input are represented honestly as missing outputs under these chosen policies.

## Limits

Named and numbered groups are supported; group 0 means the whole match. One match is extracted per original row, with at most32 selected groups. The tool does not expand one row into many matches.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
