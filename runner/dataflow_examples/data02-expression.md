# Choose rows using two conditions

Suppose an order needs at least two items and each item must cost at least 10 to enter a review queue. One condition alone is not enough.

The supplied canvas reads eight synthetic orders, uses Columns to convert quantity and unit_price into numbers, then sends them through Filter. Its Matching and Nonmatching outputs retain the accepted and rejected orders separately. No classifier is trained.

- **Expression:** a written rule over column values.
- **AND:** both conditions must be true.
- **OR:** either condition can be true.
- **Null:** a missing cell, different from zero.
- **Branch:** a separate output connection.

## Step 1

1. Select Columns and inspect quantity and unit_price.
2. Follow its table into Filter.

Numeric comparisons need numeric values; a text value that looks like a number is not an automatic conversion.

## Step 2

1. Select Filter and inspect Expression mode.
2. Read [quantity] >= 2 AND [unit_price] >= 10.
3. Explain what each comparison accepts before combining them.

Square brackets identify an input column.

## Step 3

1. Predict the accepted order IDs. A01 has quantity 2 and unit_price 12.5; A03 has quantity 4 and unit_price 5.
2. Run Flow.
3. Inspect the Matching and Nonmatching tables and compare their IDs with your prediction.

## Step 4

1. Change AND to OR, Apply and run again.
2. Compare both branches with the previous result.
3. Restore AND, Apply and rerun.

OR accepts either condition; it does not require both.

## Step 5

Inspect the missing-value policy. An unknown final comparison with Error refuses a result. Use an explicit missing-value condition or policy rather than treating a missing value as zero.

Explain which branch a quantity-zero order would enter and why zero is different from missing.

## Inspect

Matching contains A01, A04 and A07. Nonmatching contains A02, A03, A05, A06 and A08. Together they contain all eight original orders, without duplication. A rejected order has not been deleted from its separate output.

## Limits

Arithmetic and comparisons use typed values, not automatic conversion. Use Columns first. Rules are bounded to 4096 characters and the documented operators; arbitrary functions and code are unsupported. Null comparisons stay unknown until handled explicitly.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
