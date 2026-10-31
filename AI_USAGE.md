# AI usage

## Tools I used, and for what

I used ChatGPT to help me understand the task requirements, discuss possible approaches, and review parts of my implementation.

I used Python, pandas, NumPy, and scikit-learn to implement the history features, chronological split, preprocessing pipeline, and baseline comparison.

I used pytest to run the public tests and my additional tests.

## AI-suggested code I rejected or fixed in add_history_features or time_split

One approach I reviewed was sorting parcels by timestamp and using a normal cumulative count for each customer. This can be incorrect when multiple parcels from the same customer have the exact same timestamp, because one parcel could incorrectly count another parcel from that same timestamp as prior history.

I fixed this by grouping parcels by both customer_id and created_at, calculating cumulative totals for each timestamp group, and subtracting the current group. This ensures that only strictly earlier timestamps are counted.

I also checked that the result keeps the original row order and index.

## How I checked that my model has no leak

I excluded courier_notes, delivered_at, failed, and parcel_id from the model features. The model uses only information available when a parcel is created, together with customer history calculated from strictly earlier parcels. I evaluated the model on the last 20% of the data by time, rather than using a random split.