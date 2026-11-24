"""
Generate a synthetic, anonymised stand-in for the two input files used by
ocean_booking_forecast.py.  The real files (booking_data.csv, Sales_title.csv)
contain confidential seller data, so this script reproduces only their SCHEMA
and the statistical shape (sparse daily bookings, Q3/Q4 peaks, long-tailed
units, missing values, a few bad rows) -- no real seller appears here.

Output
------
data/booking_data.csv : one row per booking
    shipper_id, booking_id, date_str, region, product, shipment_type,
    sort_type, shipper_size_ttm, booking_units
data/Sales_title.csv  : one row per shipper
    shipper_id, sales_name, title
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

N_SHIPPERS = 180
START, END = "2021-01-01", "2026-06-30"
days = pd.date_range(START, END, freq="D")

# ---- shipper master -------------------------------------------------------
shipper_ids = [f"SHP{str(i).zfill(4)}" for i in range(1, N_SHIPPERS + 1)]
size_choices = ["Large", "Medium", "Small", None]
sizes = rng.choice(size_choices, N_SHIPPERS, p=[0.25, 0.35, 0.30, 0.10])
sales_names = [f"AM_{c}" for c in "ABCDEFGHIJ"]
titles = ["Sr. Account Manager", "Account Manager", "Business Development", None]

sales_title = pd.DataFrame({
    "shipper_id": shipper_ids,
    "sales_name": rng.choice(sales_names, N_SHIPPERS),
    "title": rng.choice(titles, N_SHIPPERS, p=[0.3, 0.4, 0.2, 0.1]),
})
sales_title.to_csv(OUT / "Sales_title.csv", index=False)

# ---- bookings -------------------------------------------------------------
rows = []
booking_no = 1
for sid, size in zip(shipper_ids, sizes):
    # each shipper ships on 1-3 "lanes" (region x product x shipment x sort)
    n_lanes = rng.integers(1, 4)
    base = {"Large": 25000, "Medium": 6000, "Small": 1200, None: 3000}[size]
    # some shippers churned: last order before 2024 -> removed by the pipeline
    churned = rng.random() < 0.15
    last_day = pd.Timestamp("2023-09-30") if churned else pd.Timestamp(END)
    first_day = pd.Timestamp(START) + pd.Timedelta(days=int(rng.integers(0, 500)))
    growth = rng.normal(0.12, 0.08)          # yearly trend
    for _ in range(n_lanes):
        region = rng.choice(["US", "EU"], p=[0.75, 0.25])
        product = rng.choice(["PREMIUM", "STANDARD", "STAR"], p=[0.45, 0.45, 0.10])
        shipment_type = rng.choice(["FCL", "LCL"], p=[0.6, 0.4])
        sort_type = rng.choice(["Sort", "NonSort", None], p=[0.6, 0.3, 0.1])
        p_book = rng.uniform(0.04, 0.18)      # probability of a booking per day
        lane_scale = rng.lognormal(0, 0.5)
        for d in days[(days >= first_day) & (days <= last_day)]:
            if rng.random() > p_book:
                continue
            years = (d - pd.Timestamp(START)).days / 365.25
            season = 1 + 0.35 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365.25)  # Q3/Q4 high
            if d.month in (7, 8):              # pre-holiday ocean peak
                season *= 1.25
            units = base * lane_scale * season * (1 + growth) ** years * rng.lognormal(0, 0.6)
            rows.append((sid, f"BK{booking_no:07d}", d.strftime("%Y-%m-%d"), region,
                         product, shipment_type, sort_type, size, round(units)))
            booking_no += 1

df = pd.DataFrame(rows, columns=[
    "shipper_id", "booking_id", "date_str", "region", "product",
    "shipment_type", "sort_type", "shipper_size_ttm", "booking_units"])

# inject dirty rows the cleaning step is designed to catch
dirty = rng.choice(df.index, 400, replace=False)
df.loc[dirty[:150], "booking_units"] = np.nan          # missing units
df.loc[dirty[150:250], "booking_units"] = 0            # zero bookings
df.loc[dirty[250:300], "booking_units"] = -500         # cancellations
df.loc[dirty[300:], "booking_units"] = 900000          # data-entry outliers

df.to_csv(OUT / "booking_data.csv", index=False)
print(f"booking_data.csv : {len(df):,} rows, {df.shipper_id.nunique()} shippers")
print(f"Sales_title.csv  : {len(sales_title):,} rows")
