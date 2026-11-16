# Create an order total

Add a new column by multiplying quantity by unit price, while keeping every original order.

**Calculated column:** a new value made from existing columns in the same row. **Whole number:** an integer such as 2. **Decimal number:** a number such as 12.5. **Replace:** deliberately overwrite a column rather than add another.

1. Open Create an order total. Follow the source into Columns: quantity is a whole number and unit price is a decimal number.
2. Select Formula. Its operation grid creates order_total using [quantity] * [unit_price], with type float64 and Write set to New column.
3. Predict the totals for Ana and Bo: 2 × 12.5 and 1 × 80. Apply the controls and Run Flow.
4. Inspect the table output. Confirm all eight order IDs remain in the original order and order_total appears alongside the original columns.
5. Try naming the output quantity with Write set to New column. The collision is refused. Choose a new name or explicitly choose Replace existing column, then restore order_total.

## Inspect

The totals in order are 25, 80, 20, 60, 10, 0, 60 and 6. A zero quantity produces a zero total; a missing quantity would require an explicit policy instead.

## Limits

The expression must produce the chosen output type. Numbers must remain finite and within the supported range. Missing values can be preserved explicitly; preserving them does not invent a total. No row grouping or cross-row totals are implied.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
