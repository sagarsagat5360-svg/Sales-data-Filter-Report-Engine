import numpy as np

# ============================================================
# PROJECT 3: SALES DATA FILTER & REPORT ENGINE
# NumPy concepts: arange, linspace, random, reshape, boolean
# masking, 2D slicing, fancy indexing, and reports.
# ============================================================

np.random.seed(42)  # Makes the sample data reproducible.

# STEP 1: Create month labels and sales data
months = np.arange(1, 13)
month_names = np.array([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])
store_names = np.array([f"Store_{i}" for i in range(1, 11)])

# Each row represents a store; each column represents a month.
# Sales values are in rupees.
sales = np.random.randint(20_000, 100_001, size=(10, 12))

print("=" * 65)
print("SALES DATA FILTER & REPORT ENGINE")
print("=" * 65)
print("\nSTEP 1: Sales matrix (rows = stores, columns = months)")
print("Shape:", sales.shape)
print("Months:", month_names)
print(sales)


# STEP 2: Slice quarters and individual stores using 2D slicing
print("\nSTEP 2: Quarterly sales for Store_1")
quarters = {
    "Q1 (Jan-Mar)": sales[:, 0:3],
    "Q2 (Apr-Jun)": sales[:, 3:6],
    "Q3 (Jul-Sep)": sales[:, 6:9],
    "Q4 (Oct-Dec)": sales[:, 9:12],
}

for quarter_name, quarter_data in quarters.items():
    print(f"{quarter_name} shape: {quarter_data.shape}")

store_1_sales = sales[0, :]
print("\nStore_1 monthly sales:")
for month, amount in zip(month_names, store_1_sales):
    print(f"{month}: ₹{amount:,}")

print("\nAll stores' Q1 sales (Jan-Mar):")
print(sales[:, 0:3])


# STEP 3: Boolean masking and combined conditions
monthly_total = sales.sum(axis=0)
monthly_average = sales.mean(axis=0)
store_total = sales.sum(axis=1)
store_average = sales.mean(axis=1)

# Create a target for each month: average sales + ₹5,000.
# This is an example target; adjust it for your use case.
monthly_target = monthly_average + 5_000

months_above_target = monthly_total > monthly_target
stores_below_average = store_average < store_average.mean()

print("\nSTEP 3: Boolean-mask queries")
print("\nMonthly totals:")
for month, total in zip(month_names, monthly_total):
    print(f"{month}: ₹{total:,.0f}")

print("\nMonths whose total sales exceed the monthly target:")
if np.any(months_above_target):
    for month, total, target in zip(
        month_names[months_above_target],
        monthly_total[months_above_target],
        monthly_target[months_above_target],
    ):
        print(f"{month}: sales ₹{total:,.0f} > target ₹{target:,.0f}")
else:
    print("No months exceeded the target.")

print("\nStores below the overall average store sales:")
if np.any(stores_below_average):
    for store, average in zip(
        store_names[stores_below_average],
        store_average[stores_below_average],
    ):
        print(f"{store}: average monthly sales ₹{average:,.0f}")
else:
    print("No stores are below the average.")

# Combined condition (&): months where total sales exceed their target
# AND the total is greater than the overall monthly-total average.
combined_and = (monthly_total > monthly_target) & (
    monthly_total > monthly_total.mean()
)
print("\nMonths matching combined AND condition:")
print(month_names[combined_and])

# Combined condition (|): months where sales exceed target OR are
# among the top three monthly totals.
top_month_indices = np.argsort(monthly_total)[-3:]
top_month_mask = np.zeros(12, dtype=bool)
top_month_mask[top_month_indices] = True
combined_or = (monthly_total > monthly_target) | top_month_mask
print("\nMonths matching combined OR condition:")
print(month_names[combined_or])

# (~) NOT mask: months that did not exceed target.
not_above_target = ~(monthly_total > monthly_target)
print("\nMonths that did not exceed target:")
print(month_names[not_above_target])


# STEP 4: Fancy indexing
selected_store_indices = np.array([1, 4, 7])  # Store_2, Store_5, Store_8
selected_stores = sales[selected_store_indices, :]

print("\nSTEP 4: Fancy indexing")
print("Selected stores:", store_names[selected_store_indices])
print("Their sales data:")
print(selected_stores)

# Pick the three best months by total sales.
best_month_indices = np.argsort(monthly_total)[-3:][::-1]
print("\nTop 3 months by total sales:")
for index in best_month_indices:
    print(f"{month_names[index]}: ₹{monthly_total[index]:,.0f}")

# Select the top 3 months for selected stores.
best_3_months_for_selected_stores = sales[
    np.ix_(selected_store_indices, best_month_indices)
]
print("\nSelected stores x best 3 months:")
print(best_3_months_for_selected_stores)


# STEP 5: Monthly and store-wise reports using np.linspace
print("\nSTEP 5A: Monthly report")
# Growth targets range from 3% to 8% across 12 months.
growth_targets = np.linspace(0.03, 0.08, 12)
monthly_growth_target = monthly_average * (1 + growth_targets)

print(f"{'Month':<8}{'Total Sales':>16}{'Avg/Store':>16}"
      f"{'Growth Target':>18}{'Status':>12}")
print("-" * 70)

for i, month in enumerate(month_names):
    status = "PASS" if monthly_total[i] >= monthly_growth_target[i] else "BELOW"
    print(
        f"{month:<8}₹{monthly_total[i]:>14,.0f}"
        f"₹{monthly_average[i]:>14,.0f}"
        f"₹{monthly_growth_target[i]:>16,.0f}{status:>12}"
    )

print("\nSTEP 5B: Store-wise report")
overall_store_average = store_average.mean()
print(f"{'Store':<12}{'Annual Total':>18}{'Monthly Avg':>18}{'Status':>12}")
print("-" * 60)

for i, store in enumerate(store_names):
    status = "ABOVE AVG" if store_average[i] >= overall_store_average else "BELOW AVG"
    print(
        f"{store:<12}₹{store_total[i]:>16,.0f}"
        f"₹{store_average[i]:>16,.0f}{status:>12}"
    )

# Summary metrics
best_store_index = np.argmax(store_total)
best_month_index = np.argmax(monthly_total)
print("\nFINAL SUMMARY")
print(f"Total annual sales across all stores: ₹{sales.sum():,}")
print(f"Best-performing store: {store_names[best_store_index]} "
      f"(₹{store_total[best_store_index]:,})")
print(f"Best-performing month: {month_names[best_month_index]} "
      f"(₹{monthly_total[best_month_index]:,})")
print(f"Overall average monthly sales per store: ₹{sales.mean():,.2f}")
