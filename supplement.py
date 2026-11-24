# https://github.com/ziqinghang/ocean-booking-forecast
"""Ocean-freight booking forecast: seller x lane x week with LightGBM (Tweedie).
Context: Amazon Global Logistics internship. Each quarter the sales team needs, for every seller
and shipping lane, the units that will be booked next quarter. Input: one CSV row per booking.
Condensed from the production notebook to the pipeline that shipped."""
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib.pyplot as plt
from category_encoders.m_estimate import MEstimateEncoder

TARGET_YEAR, TARGET_Q, MAX_UNITS, WEEKS_PER_YEAR = 2026, 3, 300_000, 52.143

# ---- 1. Ingest: read CSV files, join, parse dates -------------------------------------------
bookings = pd.read_csv("data/booking_data.csv")   # shipper_id, date_str, region, product, ..., units
owners = pd.read_csv("data/Sales_title.csv")      # shipper_id -> account manager name, title
df = bookings.merge(owners, how="left", on="shipper_id")
df["date_str"] = pd.to_datetime(df["date_str"])   # string -> datetime64

# ---- 2. Clean: filter scope, handle missing values and data types ---------------------------
df = df[df["product"].isin(["PREMIUM", "STANDARD"])]                       # ocean products only
last_order = df.groupby("shipper_id")["date_str"].max()                    # keep active sellers
df = df[df["shipper_id"].isin(last_order[last_order >= "2024-01-01"].index)]
df = df[df["booking_units"].between(1, MAX_UNITS)]                         # drops NaN, 0, <0, outliers
fills = {"shipper_size_ttm": "Unknown", "sort_type": "Sort", "title": "Unknown", "sales_name": "Unknown"}
df = df.fillna(fills)                                                      # missing -> own category
df["lane"] = df[["region", "product", "shipment_type", "sort_type"]].agg("_".join, axis=1)

# ---- 3. Own functions: calendar maths, lag feature, evaluation metric -----------------------
def week_index(year: pd.Series, week: pd.Series) -> pd.Series:
    """Continuous week counter so lags can cross year boundaries: 2021-W01 = 1, 2022-W01 = 53."""
    return (year - 2021) * 52 + week

def to_monday(frame: pd.DataFrame) -> pd.Series:
    """ISO (year, week) -> the date of that week's Monday."""
    return pd.to_datetime(frame.year.astype(str) + "-W" + frame.week.astype(str) + "-1", format="%G-W%V-%u")

def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """month / quarter from the week's Monday; sin/cos put week-of-year on a circle so that
    week 52 and week 1 are neighbours (an integer week number cannot express that)."""
    monday = to_monday(frame)
    frame["month"], frame["quarter"] = monday.dt.month, monday.dt.quarter
    frame["week_sin"] = np.sin(2 * np.pi * frame.week / WEEKS_PER_YEAR)
    frame["week_cos"] = np.cos(2 * np.pi * frame.week / WEEKS_PER_YEAR)
    return frame

def add_lag(frame: pd.DataFrame, lag: int = 14) -> pd.DataFrame:
    """Units the same seller booked on the same lane `lag` weeks earlier. 14 > one quarter, so the
    value is known for every week of the target quarter at forecast time: no leakage."""
    s = frame[["shipper_id", "lane", "week_idx", "booking_units"]].assign(week_idx=lambda d: d.week_idx + lag)
    return frame.merge(s.rename(columns={"booking_units": f"lag_{lag}"}), how="left",
                       on=["shipper_id", "lane", "week_idx"])

def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error: a 1,000-unit miss matters for a small seller, not a large one."""
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def rmsle_lgbm(y_pred, data):
    """Same metric in the (name, value, higher_is_better) form LightGBM expects for early stopping."""
    return "rmsle", rmsle(data.get_label(), y_pred), False

# ---- 4. Wrangle: daily bookings -> weekly panel, append future rows, add lag -----------------
keys = ["shipper_id", "lane", "shipper_size_ttm", "sales_name", "title"]
iso = df["date_str"].dt.isocalendar()
df["year"], df["week"] = iso.year.astype(int), iso.week.astype(int)
weekly = df[df.year >= 2021].groupby(keys + ["year", "week"], as_index=False)["booking_units"].sum()

q_start = pd.Timestamp(f"{TARGET_YEAR}-{3 * TARGET_Q - 2:02d}-01")
fut_weeks = pd.date_range(q_start, q_start + pd.offsets.QuarterEnd(0), freq="W-MON").isocalendar().week.values
future = weekly[keys].drop_duplicates().merge(pd.DataFrame({"week": fut_weeks, "year": TARGET_YEAR}), how="cross")
future["booking_units"] = np.nan                                     # NaN marks "to be predicted"

panel = pd.concat([weekly, future], ignore_index=True)
panel["week_idx"] = week_index(panel.year, panel.week)
panel = add_lag(add_calendar(panel), lag=14)
assert not panel.duplicated(keys + ["year", "week"]).any(), "expect one row per seller x lane x week"

# ---- 5. Encode and train: chronological validation, naive baseline for comparison ------------
history = panel[panel.booking_units.notna()].dropna()                # known weeks that have a lag
to_predict = panel[panel.booking_units.isna()].dropna(subset=["lag_14"])
is_valid = history.week_idx > history.week_idx.quantile(0.85)         # last ~15% of weeks validate

def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot the small categoricals; drop the raw time index so the model cannot memorise it."""
    return pd.get_dummies(frame.drop(columns="week_idx"), columns=["shipper_size_ttm", "title"], dtype=int)

train, test = encode(history), encode(to_predict)
y = np.log1p(train.pop("booking_units"))                              # log target tames the long tail
ids = ["shipper_id", "lane", "sales_name"]                            # high-cardinality -> target encoding
enc = MEstimateEncoder().fit(train.loc[~is_valid, ids], y[~is_valid])  # fit on the training fold only
X = pd.concat([enc.transform(train[ids]), train.drop(columns=ids)], axis=1)
X_test = pd.concat([enc.transform(test[ids]), test.drop(columns=ids + ["booking_units"])], axis=1)[X.columns]
tr, va = X.index[~is_valid.values], X.index[is_valid.values]

params = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.03, feature_fraction=0.5,
              bagging_fraction=0.75, bagging_freq=1, lambda_l2=0.1, metric="custom", verbosity=-1)
model = lgb.train(params, lgb.Dataset(X.loc[tr], y.loc[tr]), num_boost_round=20000, feval=rmsle_lgbm,
        valid_sets=[lgb.Dataset(X.loc[va], y.loc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
va_true, va_pred = np.expm1(y.loc[va]), np.expm1(model.predict(X.loc[va]))
baseline = X.loc[va, "lag_14"]                                        # naive forecast: same as 14 weeks ago
print(f"best iteration {model.best_iteration}; validation RMSLE: LightGBM {rmsle(va_true, va_pred):.3f} "
      f"vs naive lag-14 baseline {rmsle(va_true, baseline):.3f}")

# ---- 6. Forecast the quarter and calibrate the level with a seasonal ratio ------------------
test["booking_units"] = np.expm1(model.predict(X_test))
quarterly = add_calendar(weekly.copy()).groupby(["year", "quarter"])["booking_units"].sum().unstack()
prev_q = TARGET_Q - 1 if TARGET_Q > 1 else 4
ratio = (quarterly[TARGET_Q] / quarterly[prev_q]).dropna().mean()   # historical Q(prev) -> Q(target) uplift
coef = quarterly.loc[TARGET_YEAR, prev_q] * ratio / test["booking_units"].sum()
test["booking_units"] *= coef                        # rescale the level only; keep the learned seller/lane mix
print(f"seasonal ratio {ratio:.2f}, calibration coefficient {coef:.2f}, "
      f"{TARGET_YEAR} Q{TARGET_Q} forecast = {test.booking_units.sum() / 1e6:.1f}M units")
test["region"] = test.lane.str.split("_").str[0]
out_cols = ["shipper_id", "region", "lane", "week", "booking_units"]
test[out_cols].to_csv(f"forecast_{TARGET_YEAR}Q{TARGET_Q}.csv", index=False)

# ---- 7. Visualise: validation fit, feature importance, history + forecast -------------------
def weekly_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum all sellers to one row per week and attach a plot date."""
    out = frame.groupby(["year", "week"], as_index=False)["booking_units"].sum()
    return out.assign(date=to_monday(out))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.0), gridspec_kw={"width_ratios": [1, 1, 2]})
ax1.scatter(va_true, va_pred, s=5, alpha=0.3)
ax1.plot([1e2, 1e6], [1e2, 1e6], "r", lw=1)
ax1.set(xscale="log", yscale="log", xlabel="actual weekly units", ylabel="predicted", title="Validation fit")
gain = pd.Series(model.feature_importance("gain"), index=X.columns).nlargest(8).sort_values()
ax2.barh(gain.index, gain / gain.sum()); ax2.tick_params(labelsize=7)
ax2.set(xlabel="share of total gain", title="Top-8 features")
hist, fcst = weekly_total(weekly).iloc[:-1], weekly_total(test)       # drop the incomplete last history week
ax3.plot(hist.date, hist.booking_units / 1e6, lw=1, label="actual")
ax3.plot(fcst.date, fcst.booking_units / 1e6, "--", lw=2, label=f"forecast {TARGET_YEAR} Q{TARGET_Q}")
ax3.axvspan(fcst.date.min(), fcst.date.max(), color="grey", alpha=0.15)
ax3.set(ylabel="weekly units (M)", title="Weekly ocean booking units, all sellers"); ax3.legend()
fig.tight_layout(); fig.savefig("supplement_figure.png", dpi=160)
