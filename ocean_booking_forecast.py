"""
Ocean Freight Booking Forecast -- seller x lane x week, LightGBM
=================================================================

Business question
-----------------
Amazon Global Logistics (AGL) sells ocean freight to cross-border sellers.
Each quarter the sales team needs a forecast of how many *units* every seller
will book on every lane (region x product x shipment type x sort type) in the
coming quarter, so that capacity can be planned and under-performing sellers
can be followed up.

Pipeline (one section per numbered block below)
-----------------------------------------------
 1. Imports and configuration
 2. Helper functions (holiday calendar, metrics, calendar maths)
 3. Load the raw booking and sales-owner CSV files and join them
 4. Clean: product scope, active sellers, date window, unit range, missing values
 5. Exploratory plot: distribution of booking units
 6. Aggregate daily bookings to seller x lane x ISO-week and build calendar
    features (continuous week index, month, quarter, sin/cos seasonality)
 7. Append the empty "future" rows for the target quarter that we will predict
 8. Lag feature: units booked 14 weeks earlier on the same seller x lane
 9. Split history (train) from future (test) and encode categorical columns
10. Train LightGBM with a Tweedie objective and a custom RMSLE metric
11. Predict the future quarter week by week and restore readable columns
12. Calibrate the forecast total with a seasonal ratio ("live coefficient")
13. Visualise: validation fit, feature importance, history + forecast
14. Appendix: experiments that were tried but not used in the final model

Inputs  : data/booking_data.csv, data/Sales_title.csv  (see make_synthetic_data.py)
Outputs : output/*.csv forecast files, figures/*.png
"""

# =============================================================================
# 1. Imports and configuration
# =============================================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless: write PNGs, no window
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from category_encoders.m_estimate import MEstimateEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).parent
DATA_DIR, OUT_DIR, FIG_DIR = ROOT / "data", ROOT / "output", ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

BOOKING_PATH = DATA_DIR / "booking_data.csv"
SALES_TITLE_PATH = DATA_DIR / "Sales_title.csv"

HISTORY_START, HISTORY_END = "2021-01-01", "2026-06-30"   # training window
ACTIVE_SINCE = "2024-01-01"       # a seller must have booked on/after this date
MAX_UNITS = 300_000               # anything above is treated as a data error
TARGET_YEAR, TARGET_Q = 2026, 3   # the quarter we forecast
RUN_EXPERIMENTS = True            # section 14 (slower, not needed for forecast)

WEEKS_PER_YEAR = 52.143           # 365.25 / 7, used for the seasonal sin/cos


# =============================================================================
# 2. Helper functions
# =============================================================================
class ChinaHolidaysCalendar(AbstractHolidayCalendar):
    """Holiday / peak-season calendar for a China-origin ocean business.

    pandas' AbstractHolidayCalendar expects one Holiday rule per date, so
    `date_range_holiday` expands a start-end range into daily rules.  Besides
    public holidays (Spring Festival, National Day) it also marks the retail
    peaks that drive ocean demand: Prime Day stocking, Black Friday, Christmas.
    """
    rules = []

    def date_range_holiday(self, holiday_name, start_date, end_date):
        for d in pd.date_range(start=start_date, end=end_date):
            self.rules.append(Holiday(holiday_name, month=d.month, day=d.day, year=d.year))


def build_holidays(start, end):
    """Return a DatetimeIndex of all holiday / peak dates between start and end."""
    cal = ChinaHolidaysCalendar()
    spring_festival = {2021: ("02-11", "02-19"), 2022: ("01-31", "02-08"),
                       2023: ("01-21", "01-30"), 2024: ("02-09", "02-19"),
                       2025: ("01-28", "02-04"), 2026: ("02-15", "02-23")}
    for year in range(2021, 2027):
        sf_start, sf_end = spring_festival[year]
        cal.date_range_holiday(f"{year} Spring Festival", f"{year}-{sf_start}", f"{year}-{sf_end}")
        cal.date_range_holiday(f"{year} National Day",    f"{year}-10-01", f"{year}-10-07")
        cal.date_range_holiday(f"{year} Prime Day stock", f"{year}-05-15", f"{year}-06-15")
        cal.date_range_holiday(f"{year} Black Friday",    f"{year}-11-22", f"{year}-11-25")
        cal.date_range_holiday(f"{year} Christmas",       f"{year}-12-01", f"{year}-12-31")
    return cal.holidays(pd.Timestamp(start), pd.Timestamp(end))


def rmsle(actual, predicted):
    """Root Mean Squared Log Error (x1000 for readability).

    Log error is used because booking units span several orders of magnitude:
    being 1,000 units off matters for a small seller but not for a large one.
    np.abs guards against negative predictions before the log.
    """
    log_pred = np.log(np.abs(np.asarray(predicted) + 1.0))
    log_true = np.log(np.abs(np.asarray(actual) + 1.0))
    return 1000 * np.sqrt(np.mean((log_true - log_pred) ** 2))


def remove_negatives(pred, real):
    """Drop positions where either prediction or truth is negative (log undefined)."""
    bad = np.union1d(np.where(pred < 0)[0], np.where(real < 0)[0])
    return np.delete(pred, bad), np.delete(real, bad)


def msle(real, pred):
    """sklearn MSLE that survives negative predictions."""
    try:
        return mean_squared_log_error(real, pred)
    except ValueError:
        pred_nn, real_nn = remove_negatives(pred, real)
        return mean_squared_log_error(real_nn, pred_nn)


def rmsle_lgbm(y_pred, data):
    """Custom evaluation metric in the signature LightGBM expects.

    Returns (name, value, is_higher_better).  Because the target is already
    log1p-transformed, we log1p the predictions too so both sides match.
    """
    y_true = np.asarray(data.get_label())
    score = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    return "rmsle", score, False


def week_cumsum(year, week):
    """Continuous week counter: 2021-W01 = 1, 2022-W01 = 53, ... (52 weeks/year)."""
    return (year - 2021) * 52 + week


def add_calendar_features(frame):
    """Derive month / quarter / seasonal sin-cos from (year, week).

    ISO week -> Monday date -> month & quarter.  The sin/cos pair encodes the
    week-of-year as a point on a circle so that week 52 and week 1 are close,
    which a plain integer cannot express.
    """
    monday = pd.to_datetime(frame["year"].astype(str) + "-W" + frame["week"].astype(str) + "-1",
                            format="%G-W%V-%u")
    frame["month"] = monday.dt.month
    frame["quarter"] = monday.dt.quarter
    frame["week_sin"] = np.sin(2 * np.pi * frame["week"].astype(float) / WEEKS_PER_YEAR)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week"].astype(float) / WEEKS_PER_YEAR)
    return frame


def one_hot_and_cast(frame, one_hot_cols=("shipper_size_ttm", "title")):
    """One-hot encode low-cardinality columns; cast ids to pandas category."""
    dummies = pd.get_dummies(frame[list(one_hot_cols)]).astype(int)
    frame = pd.concat([frame.drop(columns=list(one_hot_cols)), dummies], axis=1)
    return frame.astype({"shipper_id": "category", "overall_type": "category", "week": int})


# =============================================================================
# 3. Load and join the raw data
# =============================================================================
booking_data = pd.read_csv(BOOKING_PATH)      # one row per booking
sales_title = pd.read_csv(SALES_TITLE_PATH)   # seller -> account manager & title
print(f"[3] bookings: {booking_data.shape}, sales_title: {sales_title.shape}")

df = booking_data.merge(sales_title, how="left", on="shipper_id")
df["date_str"] = pd.to_datetime(df["date_str"])          # string -> datetime


# =============================================================================
# 4. Cleaning
# =============================================================================
# 4a. Product scope: only the two ocean products (fast ocean + standard ocean).
df = df[df["product"].isin(["PREMIUM", "STANDARD"])].reset_index(drop=True)

# 4b. Active sellers only: drop sellers whose LAST booking is before ACTIVE_SINCE.
order_time = (df.groupby("shipper_id")["date_str"].agg(["min", "max"])
                .rename(columns={"min": "first_order_time", "max": "latest_order_time"}))
active_shipper = order_time.index[order_time["latest_order_time"] >= ACTIVE_SINCE]
df = df[df["shipper_id"].isin(active_shipper)].reset_index(drop=True)

# 4c. Date window.
df = df.set_index("date_str").sort_index().loc[HISTORY_START:HISTORY_END].reset_index()

# 4d. Unit range: remove missing, zero/negative (cancellations) and implausible outliers.
df = df[df["booking_units"].notnull()]
df = df[(df["booking_units"] > 0) & (df["booking_units"] <= MAX_UNITS)].reset_index(drop=True)
print("[4] booking_units after cleaning:\n", df["booking_units"].describe().round(0).to_string())

# 4e. Missing categorical values -> explicit labels so they become their own level.
df["shipper_size_ttm"] = df["shipper_size_ttm"].fillna("Unknown")
df["sort_type"] = df["sort_type"].fillna("Sort")
df["title"] = df["title"].fillna("Unknown")
df["sales_name"] = df["sales_name"].fillna("Unknown")

# 4f. Collapse the four lane dimensions into one key, e.g. "US_PREMIUM_FCL_Sort".
df["overall_type"] = df["region"] + "_" + df["product"] + "_" + df["shipment_type"] + "_" + df["sort_type"]
df = df.drop(columns=["region", "product", "shipment_type", "sort_type", "booking_id"])


# =============================================================================
# 5. Exploratory plot: how are booking sizes distributed?
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(df, x="booking_units", hue="title", palette="light:m_r",
             edgecolor=".3", linewidth=.5, log_scale=True, ax=ax)
ax.set_title("Booking units per booking (log scale), by account-manager title")
ax.set_xlabel("booking_units")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_units_distribution.png", dpi=150)
plt.close(fig)


# =============================================================================
# 6. Weekly aggregation and calendar features
# =============================================================================
# Daily bookings -> one row per seller x lane x day (several bookings a day add up)
df = df.groupby(["shipper_id", "overall_type", "date_str", "shipper_size_ttm",
                 "sales_name", "title"], as_index=False)["booking_units"].sum()

# ISO calendar: week 1..53 and the ISO year (Dec 31 can belong to next year's W01)
iso = df["date_str"].dt.isocalendar()
df["week"], df["year"] = iso["week"].astype(int), iso["year"].astype(int)
df = df[df["year"] != 2020].reset_index(drop=True)     # 2021-01-01..03 are ISO 2020-W53
df["week_cumsum"] = week_cumsum(df["year"], df["week"])

# Day -> week: one row per seller x lane x week
df = df.groupby(["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title",
                 "year", "week_cumsum", "week"], as_index=False)["booking_units"].sum()
df = add_calendar_features(df)
print(f"[6] weekly rows: {len(df):,}  sellers: {df.shipper_id.nunique()}  lanes: {df.overall_type.nunique()}")


# =============================================================================
# 7. Future rows for the target quarter
# =============================================================================
# For every seller x lane seen in history, create one empty row per week of the
# target quarter.  booking_units = NaN marks these as "to be predicted".
future_weeks = pd.date_range(f"{TARGET_YEAR}-{3 * TARGET_Q - 2:02d}-01",
                             pd.Timestamp(f"{TARGET_YEAR}-{3 * TARGET_Q:02d}-01") + pd.offsets.MonthEnd(0),
                             freq="W-MON").isocalendar()["week"].values

keys = df[["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title"]].drop_duplicates()
future = keys.loc[keys.index.repeat(len(future_weeks))].reset_index(drop=True)
future["week"] = np.tile(future_weeks, len(keys))
future["year"] = TARGET_YEAR
future["week_cumsum"] = week_cumsum(future["year"], future["week"])
future["booking_units"] = np.nan
future = add_calendar_features(future)

df = pd.concat([df, future]).reset_index(drop=True)
df[["week", "week_cumsum"]] = df[["week", "week_cumsum"]].astype(int)
print(f"[7] future rows added: {len(future):,} (weeks {future_weeks.min()}-{future_weeks.max()})")


# =============================================================================
# 8. Lag feature
# =============================================================================
# lag_14 = units the same seller booked on the same lane 14 weeks earlier.
# 14 weeks (> one quarter) is the smallest lag that is fully known for every
# week of the target quarter at forecast time, so no future information leaks.
shifted = df[["shipper_id", "overall_type", "shipper_size_ttm", "week_cumsum", "booking_units"]].copy()
shifted["week_cumsum"] += 14
shifted = shifted.rename(columns={"booking_units": "lag_14"})
df_state = df.merge(shifted, how="left",
                    on=["shipper_id", "overall_type", "shipper_size_ttm", "week_cumsum"])
df_state = df_state.drop(columns=["week_cumsum"])   # the model should not see the raw time index


# =============================================================================
# 9. Train / test split and categorical encoding
# =============================================================================
train_df = df_state[df_state["booking_units"].notnull()].dropna().reset_index(drop=True)
test_df = df_state[df_state["booking_units"].isnull()].dropna(subset=["lag_14"]).reset_index(drop=True)
print(f"[9] train rows: {len(train_df):,}   future rows with a lag: {len(test_df):,}")

train = one_hot_and_cast(train_df)
X = train.drop(columns="booking_units")
Y = np.log1p(train["booking_units"])          # log target: tames the heavy right tail

# High-cardinality ids are target-encoded (M-estimate = smoothed mean of the
# log target per level) instead of one-hot, which would create 100s of columns.
category_list = ["shipper_id", "overall_type", "sales_name"]
encoder = MEstimateEncoder().fit(X[category_list], Y)
X = pd.concat([encoder.transform(X[category_list], Y), X.drop(columns=category_list)], axis=1)


# =============================================================================
# 10. LightGBM training
# =============================================================================
params = {
    "objective": "tweedie",            # right-skewed, non-negative target with many small values
    "tweedie_variance_power": 1.1,     # close to Poisson
    "force_row_wise": True,
    "learning_rate": 0.03,
    "sub_row": 0.75, "bagging_freq": 1,  # row subsampling each iteration
    "feature_fraction": 0.5,           # column subsampling
    "lambda_l2": 0.1,
    "metric": "custom",                # we pass rmsle_lgbm as feval
    "num_iterations": 20000,           # upper bound; early stopping decides
    "verbosity": -1,
}

np.random.seed(42)
valid_idx = np.random.choice(X.index.values, min(5000, len(X) // 5), replace=False)
train_idx = np.setdiff1d(X.index.values, valid_idx)
train_set = lgb.Dataset(X.loc[train_idx], label=Y.loc[train_idx], free_raw_data=False)
valid_set = lgb.Dataset(X.loc[valid_idx], label=Y.loc[valid_idx], free_raw_data=False)

evals = {}
m_lgb = lgb.train(params, train_set, valid_sets=[valid_set], valid_names=["valid"], feval=rmsle_lgbm,
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(evals)])
print(f"[10] best iteration: {m_lgb.best_iteration}, valid rmsle (log space): "
      f"{evals['valid']['rmsle'][m_lgb.best_iteration - 1]:.4f}")

# Back-transform validation predictions to units for a business-readable check
valid_pred_units = np.expm1(m_lgb.predict(X.loc[valid_idx], num_iteration=m_lgb.best_iteration))
valid_true_units = np.expm1(Y.loc[valid_idx])
print(f"     valid RMSLE x1000 on units: {rmsle(valid_true_units, valid_pred_units):.1f}")


# =============================================================================
# 11. Predict the future quarter
# =============================================================================
test_info = test_df[["shipper_id", "overall_type", "sales_name"]].copy()   # keep readable ids
test = one_hot_and_cast(test_df)
test = pd.concat([encoder.transform(test[category_list]), test.drop(columns=category_list)], axis=1)
test = test.reindex(columns=X.columns.tolist() + ["booking_units"], fill_value=0)  # align dummy columns

for wk in sorted(test["week"].unique()):              # predict one target week at a time
    rows = test["week"] == wk
    test.loc[rows, "booking_units"] = m_lgb.predict(test.loc[rows, X.columns], num_iteration=m_lgb.best_iteration)
test["booking_units"] = np.round(np.expm1(test["booking_units"]))   # back to units

test[["shipper_id", "overall_type", "sales_name"]] = test_info.values
test[["region", "product", "shipment_type", "sort_type"]] = test["overall_type"].str.split("_", expand=True)
validation_df = test[["shipper_id", "region", "product", "shipment_type", "sort_type", "week", "booking_units"]].copy()
validation_df.to_csv(OUT_DIR / f"forecast_raw_{TARGET_YEAR}Q{TARGET_Q}.csv", index=False)
print(f"[11] raw forecast total {TARGET_YEAR} Q{TARGET_Q}: {validation_df.booking_units.sum():,.0f}")


# =============================================================================
# 12. Seasonal calibration ("live coefficient")
# =============================================================================
# Why: the lag_14 model is trained on history and tends to under-shoot a
# growing business.  Instead of hand-picking a fudge factor each quarter, we
# derive it from data:
#   target_total = last quarter's ACTUAL total x average historical ratio
#                  (target quarter / previous quarter)
#   coefficient  = target_total / model's raw total
# The coefficient scales every row equally, so the seller / lane / region MIX
# the model learned is preserved; only the level is corrected.
hist = pd.read_csv(BOOKING_PATH)
hist["date_str"] = pd.to_datetime(hist["date_str"])
hist = hist[(hist["booking_units"] > 0) & (hist["booking_units"] <= MAX_UNITS)]
hist = hist[hist["product"].isin(["PREMIUM", "STANDARD"])]
quarter_totals = (hist.assign(y=hist["date_str"].dt.year, q=hist["date_str"].dt.quarter)
                      .groupby(["y", "q"])["booking_units"].sum().unstack())

PREV_YEAR, PREV_Q = (TARGET_YEAR - 1, 4) if TARGET_Q == 1 else (TARGET_YEAR, TARGET_Q - 1)
prev_actual = quarter_totals.loc[PREV_YEAR, PREV_Q]

MIN_COMPLETE_QUARTER = quarter_totals.stack().median() * 0.3   # skip partial / missing quarters
ratios = []
for y in quarter_totals.index:
    ty = y if TARGET_Q > PREV_Q else y + 1
    if ty not in quarter_totals.index:
        continue
    a, c = quarter_totals.loc[y, PREV_Q], quarter_totals.loc[ty, TARGET_Q]
    if pd.notna(a) and pd.notna(c) and a > MIN_COMPLETE_QUARTER and c > MIN_COMPLETE_QUARTER:
        ratios.append(float(c / a))
season_ratio = float(np.mean(ratios))

target_total = prev_actual * season_ratio
raw_total = validation_df["booking_units"].sum()
CALIB_COEF = target_total / raw_total

print(f"[12] previous quarter {PREV_YEAR} Q{PREV_Q} actual : {prev_actual:,.0f}")
print(f"     historical Q{PREV_Q}->Q{TARGET_Q} ratios      : {[round(r, 2) for r in ratios]} -> mean {season_ratio:.3f}")
print(f"     target total = {target_total:,.0f}; model raw = {raw_total:,.0f}; coefficient = {CALIB_COEF:.3f}")

validation_df["booking_units"] = validation_df["booking_units"] * CALIB_COEF
validation_df.to_csv(OUT_DIR / f"forecast_calibrated_{TARGET_YEAR}Q{TARGET_Q}.csv", index=False)
calibrated_total = validation_df["booking_units"].sum()
print(f"     calibrated total: {calibrated_total:,.0f}")
for region, g in validation_df.groupby("region")["booking_units"].sum().sort_values(ascending=False).items():
    print(f"       {region}: {g:,.0f} ({g / calibrated_total:.1%})")


# =============================================================================
# 13. Visualisation
# =============================================================================
# 13a. Training curve: validation RMSLE per boosting round
fig, ax = plt.subplots(figsize=(7, 4))
curve = evals["valid"]["rmsle"]
ax.plot(range(1, len(curve) + 1), curve, lw=1.5)
ax.axvline(m_lgb.best_iteration, color="grey", ls="--", lw=1, label=f"best iteration = {m_lgb.best_iteration}")
ax.set_xlabel("boosting round"); ax.set_ylabel("validation RMSLE (log1p space)")
ax.set_title("LightGBM training curve with early stopping"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "02_training_curve.png", dpi=150); plt.close(fig)

# 13b. Validation fit: predicted vs actual units (log-log)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(valid_true_units, valid_pred_units, s=6, alpha=0.35)
lim = [max(1, min(valid_true_units.min(), valid_pred_units.min())), max(valid_true_units.max(), valid_pred_units.max())]
ax.plot(lim, lim, color="red", lw=1, label="perfect prediction")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("actual weekly units"); ax.set_ylabel("predicted weekly units")
ax.set_title(f"Validation set: predicted vs actual\nRMSLE x1000 = {rmsle(valid_true_units, valid_pred_units):.1f}")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "03_valid_pred_vs_actual.png", dpi=150); plt.close(fig)

# 13c. Feature importance (gain)
imp = (pd.Series(m_lgb.feature_importance(importance_type="gain"), index=X.columns)
         .sort_values(ascending=True))
imp = imp / imp.sum()
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("share of total gain"); ax.set_title("Feature importance (LightGBM gain)")
fig.tight_layout(); fig.savefig(FIG_DIR / "04_feature_importance.png", dpi=150); plt.close(fig)

# 13d. Weekly history by region + the calibrated forecast for the target quarter
hist_weekly = (df[df["booking_units"].notnull()]
               .assign(region=lambda d: d["overall_type"].str.split("_").str[0])
               .groupby(["region", "year", "week"], as_index=False)["booking_units"].sum())
hist_weekly["date"] = pd.to_datetime(hist_weekly["year"].astype(str) + "-W" + hist_weekly["week"].astype(str) + "-1",
                                     format="%G-W%V-%u")
# drop the last, incomplete week (HISTORY_END falls mid-week) so the line does not dive to zero
hist_weekly = hist_weekly[hist_weekly["date"] + pd.Timedelta(days=6) <= pd.Timestamp(HISTORY_END)]
fc_weekly = validation_df.groupby(["region", "week"], as_index=False)["booking_units"].sum()
fc_weekly["date"] = pd.to_datetime(f"{TARGET_YEAR}-W" + fc_weekly["week"].astype(str) + "-1", format="%G-W%V-%u")

fig, ax = plt.subplots(figsize=(11, 4.5))
for region, color in zip(["US", "EU"], ["#1f77b4", "#ff7f0e"]):
    h = hist_weekly[hist_weekly["region"] == region]
    f = fc_weekly[fc_weekly["region"] == region]
    ax.plot(h["date"], h["booking_units"] / 1e3, color=color, lw=1.2, label=f"{region} actual")
    ax.plot(f["date"], f["booking_units"] / 1e3, color=color, lw=2, ls="--", label=f"{region} forecast")
ax.axvspan(fc_weekly["date"].min(), fc_weekly["date"].max(), color="grey", alpha=0.12)
ax.set_ylabel("weekly booking units (thousands)")
ax.set_title(f"Weekly ocean booking units by region: history and calibrated {TARGET_YEAR} Q{TARGET_Q} forecast")
ax.legend(ncol=4, fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "05_history_and_forecast.png", dpi=150); plt.close(fig)

# 13e. Forecast mix: target quarter by lane
lane_mix = (validation_df.groupby(["region", "product", "shipment_type"])["booking_units"].sum()
                         .sort_values(ascending=True) / 1e3)
fig, ax = plt.subplots(figsize=(7, 4.5))
lane_mix.plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("forecast units (thousands)"); ax.set_title(f"{TARGET_YEAR} Q{TARGET_Q} forecast by lane")
fig.tight_layout(); fig.savefig(FIG_DIR / "06_forecast_by_lane.png", dpi=150); plt.close(fig)
print(f"[13] figures written to {FIG_DIR}")



# =============================================================================
# 14. Appendix: every disabled (raw) cell of the original notebook, runnable
# =============================================================================
# While developing the pipeline a number of ideas were tried and then switched
# off (converted to "raw" cells) in the notebook.  They are reproduced here one
# by one, tagged with the original cell number, each with the reason it was
# not kept.  They run on copies of the data and never change the forecast above.
if RUN_EXPERIMENTS:
    print("[14] appendix: disabled cells of the original notebook")
    hist_df = df[df["booking_units"].notnull()].copy()          # weekly history, final features

    # ---- 14a. [cell 12] restrict to Large sellers only --------------------------
    # Tried: model only the "Large" TTM size band.  Dropped: too few rows per lane
    # and the small/medium sellers are exactly the ones sales wants a forecast for.
    large_only = hist_df[hist_df["shipper_size_ttm"] == "Large"]
    print(f"     [c12] Large-only subset: {len(large_only):,} of {len(hist_df):,} weekly rows")

    # ---- 14b. [cell 22] drop "negative" sellers ---------------------------------
    # Tried: remove sellers whose LAST order fell in the first quarter of the
    # active window AND who had fewer than 5 bookings overall (one-off trials).
    # Dropped: the active-seller rule in section 4b already removes most of them.
    neg_window_end = (pd.Timestamp(ACTIVE_SINCE) + pd.offsets.QuarterEnd(0)).strftime("%Y-%m-%d")
    negative_seller = order_time[(order_time["latest_order_time"] >= ACTIVE_SINCE)
                                 & (order_time["latest_order_time"] <= neg_window_end)].index
    n_bookings = booking_data.groupby("shipper_id").size()
    few_bookings = n_bookings[n_bookings < 5].index
    intersection_seller = set(negative_seller) & set(few_bookings)
    print(f"     [c22] negative sellers (last order {ACTIVE_SINCE}..{neg_window_end} and <5 bookings): "
          f"{len(intersection_seller)}")

    # ---- 14c. [cell 24 / 30] keep the four lane dimensions as separate columns ---
    # Tried: aggregate by region/product/shipment_type/sort_type separately and
    # save a wide "source.csv".  Dropped in favour of the single overall_type key,
    # which keeps the model's categorical handling simpler.
    lane_cols = hist_df["overall_type"].str.split("_", expand=True)
    lane_cols.columns = ["region", "product", "shipment_type", "sort_type"]
    source = pd.concat([hist_df, lane_cols], axis=1).groupby(
        ["shipper_id", "region", "product", "shipment_type", "sort_type",
         "shipper_size_ttm", "sales_name", "title", "year", "week"], as_index=False)["booking_units"].sum()
    source = add_calendar_features(source)
    source.to_csv(OUT_DIR / "source_wide.csv", index=False)
    print(f"     [c24/30] wide source table: {source.shape}")

    # ---- 14d. [cell 28] daily holiday flag -------------------------------------
    # Tried: flag each booking DAY that falls in the holiday / peak calendar.
    # Dropped: after weekly aggregation the flag collapses, and the sin/cos
    # seasonality already captures the same yearly pattern.
    holidays = build_holidays(HISTORY_START, HISTORY_END)
    daily = booking_data.assign(date_str=pd.to_datetime(booking_data["date_str"]))
    daily["is_holidays"] = daily["date_str"].isin(holidays).astype(int)
    hol_iso = pd.Series(holidays).dt.isocalendar()
    hol_weeks = set(zip(hol_iso["year"].astype(int), hol_iso["week"].astype(int)))
    hist_df["is_holiday_week"] = [int((y, w) in hol_weeks) for y, w in zip(hist_df["year"], hist_df["week"])]
    print(f"     [c28] holiday days share: {daily['is_holidays'].mean():.1%}; "
          f"holiday weeks share: {hist_df['is_holiday_week'].mean():.1%}")

    # ---- 14e. [cell 29] which Large lanes have a long enough history? ----------
    # Tried: count weekly points per seller x lane and keep those with >= 70
    # weeks (US PREMIUM only) for a per-lane time-series model.
    # Dropped: too few lanes qualify; a single pooled model generalises better.
    lane_len = pd.concat([hist_df, lane_cols], axis=1)
    lane_len = lane_len[lane_len["shipper_size_ttm"] == "Large"].groupby(
        ["shipper_id", "region", "product", "shipment_type", "sort_type"], as_index=False).size()
    long_lanes = lane_len[(lane_len["size"] >= 70) & (lane_len["region"] == "US") & (lane_len["product"] == "PREMIUM")]
    print(f"     [c29] Large US PREMIUM lanes with >=70 weeks: {len(long_lanes)}")

    # ---- 14f. [cell 34 / 35] fill the missing weeks of every lane with zeros ---
    # Tried: re-index each seller x lane to a continuous week range so that
    # weeks without bookings appear as explicit 0 rows (and most_year() maps the
    # week counter back to a year).  Dropped: it multiplies the rows by ~4 with
    # zeros, which pushes the Tweedie model toward predicting zero everywhere.
    def most_year(week_cumsum_value):
        """Inverse of week_cumsum(): continuous week index -> calendar year."""
        return 2021 + int((week_cumsum_value - 1) // 52)

    def expand_lane(group):
        """Reindex one seller x lane to every week between its first and last booking."""
        group = group.set_index("week_cumsum")
        idx = pd.Index(range(int(group.index.min()), int(group.index.max()) + 1), name="week_cumsum")
        group = group.reindex(idx).reset_index()
        group["week"] = ((group["week_cumsum"] - 1) % 52 + 1).astype(int)
        group["year"] = group["week_cumsum"].apply(most_year)
        constant_cols = ["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title"]
        group[constant_cols] = group[constant_cols].ffill().bfill()
        group["booking_units"] = group["booking_units"].fillna(0)
        return group

    expanded = pd.concat([expand_lane(g) for _, g in hist_df.groupby(["shipper_id", "overall_type"])])
    expanded = add_calendar_features(expanded)
    print(f"     [c34/35] rows after zero-filling missing weeks: {len(expanded):,} "
          f"({(expanded['booking_units'] == 0).mean():.0%} zeros) vs {len(hist_df):,} original")

    # ---- 14g. [cell 40] fixed column order for the concatenated frame ----------
    column_order = ["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title", "year", "quarter",
                    "month", "week", "week_sin", "week_cos", "week_cumsum", "booking_units"]
    ordered = hist_df.reindex(columns=column_order)
    print(f"     [c40] re-ordered frame: {ordered.shape}")

    # ---- 14h. [cells 42-54] previous-quarter and previous-month totals ----------
    # Tried: for every seller x lane compute last quarter's / last month's total
    # units and attach them to the following quarter / month as features.
    # Dropped: at forecast time the "previous month" of months 2 and 3 of the
    # target quarter is itself unknown, so the feature cannot be built.
    def update_year_quarter(group):
        """Shift (year, quarter) one quarter forward so Q(n) totals attach to Q(n+1)."""
        group.loc[group["quarter"] == 4, "year"] += 1
        group["quarter"] = group["quarter"] % 4 + 1
        return group

    def update_year_month(group):
        """Shift (year, month) one month forward so December totals attach to January."""
        group.loc[group["month"] == 12, "year"] += 1
        group["month"] = group["month"] % 12 + 1
        return group

    group_cols = ["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title"]
    df1 = (hist_df.groupby(group_cols + ["year", "quarter"], as_index=False)["booking_units"].sum()
                  .sort_values(group_cols + ["year", "quarter"]))
    df1["prev_quarter_units"] = df1.groupby(["shipper_id", "overall_type"])["booking_units"].shift()
    df1["QoQ_growth"] = (df1["booking_units"] - df1["prev_quarter_units"]) / df1["prev_quarter_units"]
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    df1_next = update_year_quarter(df1.copy())                    # attach to the NEXT quarter

    df2 = (hist_df.groupby(group_cols + ["year", "month"], as_index=False)["booking_units"].sum()
                  .sort_values(group_cols + ["year", "month"]))
    df2["prev_month_units"] = df2.groupby(["shipper_id", "overall_type"])["booking_units"].shift()
    df2["MoM_growth"] = (df2["booking_units"] - df2["prev_month_units"]) / df2["prev_month_units"]
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
    df2_next = update_year_month(df2.copy())

    df_state_exp = hist_df.merge(df1_next[group_cols + ["year", "quarter", "prev_quarter_units"]],
                                 how="left", on=group_cols + ["year", "quarter"])
    df_state_exp = df_state_exp.merge(df2_next[group_cols + ["year", "month", "prev_month_units"]],
                                      how="left", on=group_cols + ["year", "month"])
    print(f"     [c42-54] rows with both prev_quarter and prev_month: "
          f"{df_state_exp.dropna(subset=['prev_quarter_units', 'prev_month_units']).shape[0]:,}")

    # ---- 14i. [cell 50] week-over-week growth ----------------------------------
    df3 = (hist_df.groupby(["shipper_id", "overall_type", "year", "week_cumsum"], as_index=False)["booking_units"].sum()
                  .sort_values(["shipper_id", "overall_type", "year", "week_cumsum"]))
    df3["prev_week_units"] = df3.groupby(["shipper_id", "overall_type"])["booking_units"].shift()
    df3["WoW_growth"] = (df3["booking_units"] - df3["prev_week_units"]) / df3["prev_week_units"]
    df3 = df3.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"     [c50] WoW growth: median {df3['WoW_growth'].median():+.1%}, "
          f"IQR {df3['WoW_growth'].quantile(.25):+.1%} .. {df3['WoW_growth'].quantile(.75):+.1%}")

    # ---- 14j. [cell 63 / 64] quarterly sanity check for one seller --------------
    df_shifted_q = hist_df.groupby(["shipper_id", "overall_type", "year", "quarter", "week"],
                                   as_index=False)["booking_units"].sum()
    sample_seller = hist_df["shipper_id"].iloc[0]
    check = hist_df[(hist_df["shipper_id"] == sample_seller) & (hist_df["year"] == 2023)
                    & (hist_df["quarter"] == 2) & hist_df["overall_type"].str.contains("US")]
    print(f"     [c63/64] {sample_seller} 2023 Q2 US units: {check['booking_units'].sum():,.0f}")

    # ---- 14k. [cell 67] mean of weeks t+14 .. t+18 (first lag idea) -----------
    # Tried: for each row average the units 14-18 weeks later as a target proxy.
    # Dropped: replaced by the simpler, leak-free lag_14 merge in section 8.
    df_shifted = hist_df.copy()
    df_shifted["past_14week"] = df_shifted["week_cumsum"] + 14
    df_shifted["past_18week"] = df_shifted["week_cumsum"] + 18
    first_group = next(iter(df_shifted.groupby(group_cols)))[1].reset_index(drop=True)
    row = first_group.iloc[0]
    past_delta = first_group[(first_group["week_cumsum"] >= row["past_14week"])
                             & (first_group["week_cumsum"] <= row["past_18week"])]
    print(f"     [c67] mean units in weeks +14..+18 for the first lane/row: {past_delta['booking_units'].mean():,.0f}")

    # ---- 14l. [cell 71 / 78] lag_4 and 3-week rolling mean ---------------------
    # Tried: a 4-week lag and a rolling mean -- both correlate better with the
    # target than lag_14 (see printout).  Dropped: they are not known for weeks
    # 5+ of the target quarter, so they would force recursive forecasting.
    for lag in (14, 4):
        s = hist_df[["shipper_id", "overall_type", "shipper_size_ttm", "week_cumsum", "booking_units"]].copy()
        s["week_cumsum"] += lag
        s = s.rename(columns={"booking_units": f"lag_{lag}"})
        hist_df = hist_df.merge(s, how="left", on=["shipper_id", "overall_type", "shipper_size_ttm", "week_cumsum"])
    hist_df = hist_df.sort_values(["shipper_id", "overall_type", "week_cumsum"])
    hist_df["rolling_mean"] = (hist_df.groupby(group_cols)["booking_units"]
                                      .transform(lambda s: s.rolling(3).mean()))
    hist_df = hist_df.merge(df_state_exp[["shipper_id", "overall_type", "year", "week", "prev_quarter_units",
                                          "prev_month_units"]].drop_duplicates(),
                            how="left", on=["shipper_id", "overall_type", "year", "week"])
    corr = (hist_df[["booking_units", "lag_14", "lag_4", "rolling_mean", "prev_quarter_units", "prev_month_units"]]
                .corr()["booking_units"].drop("booking_units"))
    print("     [c71/78] correlation of candidate features with booking_units:\n       "
          + corr.round(3).to_string().replace("\n", "\n       "))

    # ---- 14m. [cell 75] simple exponential smoothing toy ------------------------
    # Tried: smooth one lane's recent weeks with SES and extrapolate the quarter.
    # Dropped: no seasonality, no cross-seller information; kept as a baseline.
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        toy = pd.Series([47633.9931, 183252, 400630, 159391, 321969, 134000],
                        index=pd.Index([137, 138, 139, 140, 141, 142], name="week"))
        ses_toy = SimpleExpSmoothing(toy.values.astype(float), initialization_method="estimated").fit()
        print(f"     [c75] SES toy forecast sum for 13 weeks: {ses_toy.forecast(13).sum():,.0f}")
        lane_total = hist_df.groupby("week_cumsum")["booking_units"].sum()
        ses_all = SimpleExpSmoothing(lane_total.values.astype(float), initialization_method="estimated").fit()
        print(f"     [c75] SES on total weekly units, next {len(future_weeks)} weeks: "
              f"{ses_all.forecast(len(future_weeks)).sum():,.0f}")
    except ImportError:
        print("     [c75] statsmodels not installed; SES skipped")

    # ---- 14n. [cell 92] seasonality on the continuous week counter -------------
    # Tried: sin/cos of week_cumsum instead of week-of-year.  Mathematically the
    # same angle (52.143 period) but drifts by 0.143 week per year; dropped.
    hist_df["week_sin_cum"] = np.sin(2 * np.pi * hist_df["week_cumsum"] / WEEKS_PER_YEAR)
    hist_df["week_cos_cum"] = np.cos(2 * np.pi * hist_df["week_cumsum"] / WEEKS_PER_YEAR)
    print(f"     [c92] max |week_sin - week_sin_cum| = {(hist_df['week_sin'] - hist_df['week_sin_cum']).abs().max():.3f}")

    # ---- 14o. [cells 131-133] time-split comparison: RandomForest vs LightGBM ---
    # Tried: a chronological split (train on earlier weeks, validate on later)
    # with the sklearn APIs and the hyper-parameters below.  Kept as a check:
    # LightGBM (lgb.train with Tweedie) in section 10 remained the final model.
    exp = hist_df.dropna(subset=["lag_14"])[["shipper_id", "overall_type", "shipper_size_ttm", "sales_name", "title",
                                              "year", "quarter", "month", "week", "week_sin", "week_cos",
                                              "week_cumsum", "lag_14", "booking_units"]]
    exp = one_hot_and_cast(exp)
    Xe, Ye = exp.drop(columns="booking_units"), np.log1p(exp["booking_units"])
    enc_e = MEstimateEncoder().fit(Xe[category_list], Ye)
    Xe = pd.concat([enc_e.transform(Xe[category_list], Ye), Xe.drop(columns=category_list)], axis=1)
    split_week = int(Xe["week_cumsum"].quantile(0.85))      # ~ first 85% of weeks train, rest validate
    x_train = Xe[Xe["week_cumsum"] <= split_week].drop(columns="week_cumsum"); y_train = Ye[x_train.index]
    x_valid = Xe[Xe["week_cumsum"] > split_week].drop(columns="week_cumsum");  y_valid = Ye[x_valid.index]

    rf_base = RandomForestRegressor(n_jobs=-1, random_state=42).fit(x_train, y_train)
    rf_tuned = RandomForestRegressor(bootstrap=True, criterion="squared_error", max_depth=30, max_features="sqrt",
                                     min_samples_leaf=1, min_samples_split=10, n_estimators=600, oob_score=True,
                                     n_jobs=-1, random_state=42).fit(x_train, y_train)
    model_lgb_base = lgb.LGBMRegressor(objective="regression", verbose=-1).fit(x_train, y_train)
    model_lgb_tuned = lgb.LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=200, max_depth=30,
                                        num_leaves=1200, min_child_samples=20, min_child_weight=30,
                                        min_data_in_leaf=70, min_split_gain=0.0001, feature_fraction=0.5,
                                        bagging_fraction=0.8, bagging_freq=4, verbose=-1).fit(x_train, y_train)
    print(f"     [c131-133] chronological split at week_cumsum {split_week}: "
          f"train {len(x_train):,} / valid {len(x_valid):,}")
    for name, model in [("RF base", rf_base), ("RF tuned", rf_tuned),
                        ("LGBM base", model_lgb_base), ("LGBM tuned", model_lgb_tuned)]:
        pred = np.expm1(model.predict(x_valid))
        print(f"       {name:10s} RMSLE x1000 = {rmsle(np.expm1(y_valid), pred):6.1f}   "
              f"MSLE = {msle(np.expm1(y_valid).values, pred):.4f}")

print("done.")
