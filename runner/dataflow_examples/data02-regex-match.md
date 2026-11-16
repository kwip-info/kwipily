# Separate email-shaped text

Before sending contact data to another step, you may want to separate text that resembles an email address from text that does not.

The supplied canvas reads eight synthetic orders and routes them through Filter. The email column contains plausible addresses, invalid text and one missing cell. Matching and Nonmatching keep both groups available. This simple pattern checks text shape; it cannot establish that an address exists or belongs to anyone.

- **Regex:** a pattern describing characters and repetition.
- **Full match:** the entire cell must fit.
- **Search:** a matching part anywhere in the cell is enough.
- **Missing:** no text value is present.

## Step 1

1. Select Filter and open Configuration.
2. Find Text pattern mode, the email column and Full matching.
3. Follow both Matching and Nonmatching outputs.

The supplied pattern is already configured; inspect it before changing anything.

## Step 2

Compare an ordinary address-shaped value with invalid and the missing email in A05. Predict which should match the whole pattern.

A missing cell needs its own explicit policy; it is not an empty address that can be silently validated.

## Step 3

1. Keep the missing-cell policy set to Nonmatching.
2. Run Flow.
3. Inspect both captured tables and compare the order IDs, not just the counts.

## Step 4

1. Confirm six matching orders: A01, A02, A04, A06, A07 and A08.
2. Confirm A03 and A05 in Nonmatching.
3. Check that the two branches together cover the original eight orders once.

## Step 5

For an optional extension, save a copy of the source and add extra text around an address-shaped value. Select that copied source explicitly in Table Input. Compare Full with Search, applying and running each.

Search can find a matching substring even when the entire cell is unsuitable. Restore the supplied example with Start fresh if you want to continue from the original data.

## Inspect

Six orders match: A01, A02, A04, A06, A07 and A08. A03 and A05 are nonmatching. This simple pattern is an exercise, not a complete internet email validator.

## Limits

Patterns run in an isolated supervised worker with time, input and output limits. Expensive patterns are refused with guidance; cancellation stops the worker. The app never silently rewrites a pattern to make it succeed.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
