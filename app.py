# app.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ----------------------------
# Page & layout
# ----------------------------
st.set_page_config(page_title="Precipitation Bias Correction (QDM & FD)", layout="wide")
st.title("Precipitation Bias Correction and Diagnostics")
st.caption("Upload Observed, Historical (simulated present), and Futuristic (simulated future) Excel files to run QDM, Linear Scaling, diagnostics, and frequency-distribution corrections.")

# ----------------------------
# Helpers: IO and normalization
# ----------------------------
DATE_LIKE = ["date", "time", "day", "timestamp"]
PR_LIKE = ["pr", "precip", "precipitation", "rain", "ppt"]

def _find_date_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in DATE_LIKE:
        if c in low:
            return low[c]
    # try datetime dtype
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try parse-able
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.8:
                return c
        except Exception:
            continue
    raise ValueError("No suitable date column found (expected one of: date/time/day/timestamp or datetime-like).")

def _find_pr_col(df: pd.DataFrame, exclude: str) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in PR_LIKE:
        if c in low and pd.api.types.is_numeric_dtype(df[low[c]]):
            return low[c]
    # first numeric column not excluded
    for c in df.columns:
        if c != exclude and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric precipitation column found (expected: pr/precip/precipitation/rain/ppt).")

def _read_excel(uploaded_file, preferred_sheet: str | None = None) -> pd.DataFrame:
    # Read preferred sheet if available; else first sheet
    xls = pd.ExcelFile(uploaded_file)
    sheet = None
    if preferred_sheet:
        for s in xls.sheet_names:
            if s.lower().strip() == preferred_sheet.lower().strip():
                sheet = s
                break
    if sheet is None:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    return df

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = _find_date_col(df)
    pr_col = _find_pr_col(df, exclude=date_col)
    out = df[[date_col, pr_col]].copy()
    out.columns = ["Date", "pr"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["pr"] = pd.to_numeric(out["pr"], errors="coerce")
    out = out.dropna(subset=["pr"]).sort_values("Date").reset_index(drop=True)
    return out

def _fmt_dt_for_metric(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return "N/A"
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(x)

# ----------------------------
# Bias correction methods
# ----------------------------
def quantile_delta_mapping(obs_hist: np.ndarray, mod_hist: np.ndarray, mod_fut: np.ndarray, obs_fut: np.ndarray) -> np.ndarray:
    """
    Quantile Delta Mapping (QDM) for precipitation:
    - Empirical quantiles via ranks
    - Multiplicative for wet quantiles, additive for dry/near-zero
    - Non-negative enforcement
    - Caps extreme corrected values using 98th percentile of obs_fut (max 50 mm/day)
    - Aligns zeros proportion with obs_fut
    """
    n_f = len(mod_fut)
    jitter = np.random.uniform(0, 1e-6, size=n_f)
    mod_fut_j = mod_fut + jitter

    ranks = rankdata(mod_fut_j)
    tau_f = (ranks - 0.5) / n_f

    # Use np.quantile with linear method (np>=1.22 uses 'method' instead of 'interpolation')
    try:
        mod_hist_q = np.quantile(mod_hist, tau_f, method="linear")
        obs_hist_q = np.quantile(obs_hist, tau_f, method="linear")
    except TypeError:
        mod_hist_q = np.quantile(mod_hist, tau_f, interpolation="linear")
        obs_hist_q = np.quantile(obs_hist, tau_f, interpolation="linear")

    corrected = np.zeros(n_f)
    threshold = 1e-6
    multiplicative = mod_hist_q > threshold

    delta_m = np.zeros_like(mod_fut)
    delta_m[multiplicative] = mod_fut[multiplicative] / np.maximum(mod_hist_q[multiplicative], threshold)
    corrected[multiplicative] = obs_hist_q[multiplicative] * delta_m[multiplicative]

    additive = ~multiplicative
    delta_a = mod_fut[additive] - mod_hist_q[additive]
    corrected[additive] = obs_hist_q[additive] + delta_a

    corrected[corrected < 0] = 0

    # Cap outliers
    if len(obs_fut) > 0:
        max_threshold = min(np.percentile(obs_fut, 98), 50.0)
    else:
        max_threshold = 50.0
    corrected = np.minimum(corrected, max_threshold)

    # Match zero proportion
    if len(obs_fut) > 0:
        zero_prop_obs = np.mean(obs_fut == 0)
        zero_prop_corr = np.mean(corrected == 0)
        if zero_prop_corr > zero_prop_obs:
            n_zeros_to_remove = int((zero_prop_corr - zero_prop_obs) * n_f)
            zero_idx = np.where(corrected == 0)[0]
            if len(zero_idx) > 0:
                k = min(n_zeros_to_remove, len(zero_idx))
                if k > 0:
                    idx_change = np.random.choice(zero_idx, size=k, replace=False)
                    min_nonzero = np.min(obs_hist[obs_hist > 0]) if np.any(obs_hist > 0) else 0.1
                    corrected[idx_change] = min_nonzero
        elif zero_prop_corr < zero_prop_obs:
            n_zeros_to_add = int((zero_prop_obs - zero_prop_corr) * n_f)
            non_zero_idx = np.where(corrected > 0)[0]
            if len(non_zero_idx) > 0:
                k = min(n_zeros_to_add, len(non_zero_idx))
                if k > 0:
                    idx_zero = np.random.choice(non_zero_idx, size=k, replace=False)
                    corrected[idx_zero] = 0

    return corrected

def linear_scaling(obs_hist: np.ndarray, mod_hist: np.ndarray, mod_fut: np.ndarray) -> np.ndarray:
    mean_mod = np.mean(mod_hist)
    mean_obs = np.mean(obs_hist)
    scaling = (mean_obs / mean_mod) if mean_mod > 0 else 1.0
    corrected = mod_fut * scaling
    corrected[corrected < 0] = 0
    return corrected

# ----------------------------
# Frequency distribution analysis and correction
# ----------------------------
def compute_frequency_distribution(df: pd.DataFrame, precip_col: str, bins: np.ndarray) -> pd.DataFrame:
    d = df[[precip_col]].copy()
    d = d[d[precip_col] > 0.1]  # remove <=0.1 mm
    d["precip_rounded"] = d[precip_col].round(1)
    freq = d["precip_rounded"].value_counts().sort_index()
    freq_df = pd.DataFrame({"Precipitation (mm)": bins})
    freq_df["Frequency"] = freq_df["Precipitation (mm)"].map(freq).fillna(0.0)
    total = freq_df["Frequency"].sum()
    if total > 0:
        freq_df["Frequency"] = freq_df["Frequency"] / total
    return freq_df

def correct_frequency_distributions(obs_freq_df: pd.DataFrame, hist_freq_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        obs_freq_df, hist_freq_df, on="Precipitation (mm)", suffixes=("_obs", "_hist")
    ).fillna(0.0)
    if merged.empty:
        raise ValueError("Merged frequency dataset is empty. Check precipitation bins.")
    # Simple scaling
    mean_obs = merged["Frequency_obs"].mean()
    mean_hist = merged["Frequency_hist"].mean()
    scaling_factor = (mean_obs / mean_hist) if mean_hist != 0 else 1.0
    merged["Simple_Scaling"] = np.round(merged["Frequency_hist"] * scaling_factor, 6)

    # Variance scaling
    std_obs = merged["Frequency_obs"].std()
    std_hist = merged["Frequency_hist"].std()
    std_ratio = (std_obs / std_hist) if std_hist != 0 else 1.0
    merged["Variance_Scaling"] = np.round(merged["Frequency_hist"] * std_ratio, 6)

    # Enhanced quantile mapping (by re-mapping hist freq distribution to obs)
    obs_sorted = np.sort(merged["Frequency_obs"].values)
    hist_sorted = np.sort(merged["Frequency_hist"].values)
    if len(obs_sorted) > 0 and len(hist_sorted) > 0 and np.sum(obs_sorted) > 0 and np.sum(hist_sorted) > 0:
        # Normalize to CDF
        obs_cdf = np.cumsum(obs_sorted) / np.sum(obs_sorted)
        hist_cdf = np.cumsum(hist_sorted) / np.sum(hist_sorted)
        # Interpolate historical frequencies to observed scale using value-space mapping
        merged["Quantile_Mapping"] = np.interp(
            merged["Frequency_hist"].values, hist_sorted, obs_sorted
        )
    else:
        merged["Quantile_Mapping"] = np.nan

    # Non-negative and renormalize
    for col in ["Simple_Scaling", "Variance_Scaling", "Quantile_Mapping"]:
        merged[col] = merged[col].clip(lower=0.0)
        tot = merged[col].sum()
        if tot > 0:
            merged[col] = merged[col] / tot

    return merged

def apply_fd_to_future(future_freq_df: pd.DataFrame, merged_hist_obs: pd.DataFrame) -> pd.DataFrame:
    # Use scaling factors derived earlier
    mean_obs = merged_hist_obs["Frequency_obs"].mean()
    mean_hist = merged_hist_obs["Frequency_hist"].mean()
    scaling_factor = (mean_obs / mean_hist) if mean_hist != 0 else 1.0

    std_obs = merged_hist_obs["Frequency_obs"].std()
    std_hist = merged_hist_obs["Frequency_hist"].std()
    std_ratio = (std_obs / std_hist) if std_hist != 0 else 1.0

    hist_sorted = np.sort(merged_hist_obs["Frequency_hist"].values)
    obs_sorted = np.sort(merged_hist_obs["Frequency_obs"].values)

    out = future_freq_df.copy()
    out["Simple_Scaling"] = np.round(out["Frequency"] * scaling_factor, 6)
    out["Variance_Scaling"] = np.round(out["Frequency"] * std_ratio, 6)
    if len(hist_sorted) > 0 and len(obs_sorted) > 0:
        out["Quantile_Mapping"] = np.interp(out["Frequency"].values, hist_sorted, obs_sorted)
    else:
        out["Quantile_Mapping"] = np.nan

    for col in ["Simple_Scaling", "Variance_Scaling", "Quantile_Mapping"]:
        out[col] = out[col].clip(lower=0.0)
        tot = out[col].sum()
        if tot > 0:
            out[col] = out[col] / tot
    return out

# ----------------------------
# Sidebar: Uploaders
# ----------------------------
st.sidebar.header("Inputs")
obs_up = st.sidebar.file_uploader(
    "Observed (.xlsx)",
    type=["xlsx", "xls"],
    help="Observed precipitation workbook with a date column and a precipitation column.",  # st.file_uploader behavior per docs
)  # [web:4]
hist_up = st.sidebar.file_uploader(
    "Historical (simulated present) (.xlsx)",
    type=["xlsx", "xls"],
    help="Historical model precipitation (baseline/present).",  # uploader usage from docs
)  # [web:4]
fut_up = st.sidebar.file_uploader(
    "Futuristic (simulated future) (.xlsx)",
    type=["xlsx", "xls"],
    help="Futuristic/Scenario model precipitation.",  # uploader usage from docs
)  # [web:4]

# Optional sheet hints
with st.sidebar.expander("Advanced: sheet names"):
    obs_sheet = st.text_input("Observed sheet name (optional)", value="observeddata")
    hist_sheet = st.text_input("Historical sheet name (optional)", value="")
    fut_sheet = st.text_input("Futuristic sheet name (optional)", value="")

# ----------------------------
# Load and normalize data
# ----------------------------
obs_df = hist_df = fut_df = None
load_msgs = []

if obs_up is not None:
    try:
        df_raw = _read_excel(obs_up, preferred_sheet=obs_sheet if obs_sheet.strip() else None)
        obs_df = _normalize_df(df_raw)
    except Exception as e:
        load_msgs.append(f"Observed: {e}")

if hist_up is not None:
    try:
        df_raw = _read_excel(hist_up, preferred_sheet=hist_sheet if hist_sheet.strip() else None)
        hist_df = _normalize_df(df_raw)
    except Exception as e:
        load_msgs.append(f"Historical: {e}")

if fut_up is not None:
    try:
        df_raw = _read_excel(fut_up, preferred_sheet=fut_sheet if fut_sheet.strip() else None)
        fut_df = _normalize_df(df_raw)
    except Exception as e:
        load_msgs.append(f"Futuristic: {e}")

if load_msgs:
    st.warning(" · ".join(load_msgs))

# ----------------------------
# Tabs for workflow
# ----------------------------
tab_upload, tab_timeseries, tab_bias, tab_freq, tab_cdfdemo = st.tabs(
    ["Upload & overview", "Time series", "Bias correction (QDM/LS)", "Freq. distributions", "CDF delta demo"]
)  # organize content with tabs per docs [web:56]

with tab_upload:
    st.subheader("Overview")
    cols = st.columns(3)
    for i, (name, df) in enumerate([("Observed", obs_df), ("Historical", hist_df), ("Futuristic", fut_df)]):
        with cols[i]:
            st.markdown(f"#### {name}")
            if df is not None and not df.empty:
                st.write(df.head())
                st.metric("Rows", f"{len(df):,}")  # numeric/string allowed in metric [web:60]
                st.metric("From", _fmt_dt_for_metric(df["Date"].min()))
                st.metric("To", _fmt_dt_for_metric(df["Date"].max()))
            else:
                st.info(f"Upload {name.lower()} file to preview.")

with tab_timeseries:
    st.subheader("Daily, Monthly, Yearly series")
    # Date filter
    min_dt = None
    max_dt = None
    for d in [obs_df, hist_df, fut_df]:
        if d is not None and not d.empty:
            min_dt = d["Date"].min() if min_dt is None else min(min_dt, d["Date"].min())
            max_dt = d["Date"].max() if max_dt is None else max(max_dt, d["Date"].max())
    if min_dt is None:
        st.info("Upload data to view time series.")
    else:
        d1, d2 = st.date_input(
            "Filter window", value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(), max_value=max_dt.date()
        )
        start_ts = pd.to_datetime(d1)
        end_ts = pd.to_datetime(d2)

        def _clip(df):
            if df is None or df.empty:
                return df
            return df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

        obs_v = _clip(obs_df)
        hist_v = _clip(hist_df)
        fut_v = _clip(fut_df)

        # Daily
        st.markdown("### Daily")
        daily_cols = st.columns(3)
        if obs_v is not None and not obs_v.empty:
            with daily_cols[0]:
                st.caption("Observed")
                st.line_chart(obs_v.set_index("Date")["pr"])
        if hist_v is not None and not hist_v.empty:
            with daily_cols[1]:
                st.caption("Historical")
                st.line_chart(hist_v.set_index("Date")["pr"])
        if fut_v is not None and not fut_v.empty:
            with daily_cols[2]:
                st.caption("Futuristic")
                st.line_chart(fut_v.set_index("Date")["pr"])

        # Monthly
        st.markdown("### Monthly")
        monthly_cols = st.columns(3)
        def _monthly(df):
            if df is None or df.empty:
                return None
            m = df.resample("MS", on="Date")["pr"].sum().reset_index()
            return m
        if obs_v is not None and not obs_v.empty:
            with monthly_cols[0]:
                st.caption("Observed monthly")
                m = _monthly(obs_v)
                st.line_chart(m.set_index("Date")["pr"])
        if hist_v is not None and not hist_v.empty:
            with monthly_cols[1]:
                st.caption("Historical monthly")
                m = _monthly(hist_v)
                st.line_chart(m.set_index("Date")["pr"])
        if fut_v is not None and not fut_v.empty:
            with monthly_cols[2]:
                st.caption("Futuristic monthly")
                m = _monthly(fut_v)
                st.line_chart(m.set_index("Date")["pr"])

        # Yearly
        st.markdown("### Yearly")
        yearly_cols = st.columns(3)
        def _yearly(df):
            if df is None or df.empty:
                return None
            y = df.resample("YS", on="Date")["pr"].sum().reset_index()
            return y
        if obs_v is not None and not obs_v.empty:
            with yearly_cols[0]:
                st.caption("Observed yearly")
                y = _yearly(obs_v)
                st.bar_chart(y.set_index("Date")["pr"])
        if hist_v is not None and not hist_v.empty:
            with yearly_cols[1]:
                st.caption("Historical yearly")
                y = _yearly(hist_v)
                st.bar_chart(y.set_index("Date")["pr"])
        if fut_v is not None and not fut_v.empty:
            with yearly_cols[2]:
                st.caption("Futuristic yearly")
                y = _yearly(fut_v)
                st.bar_chart(y.set_index("Date")["pr"])

with tab_bias:
    st.subheader("Bias correction (QDM/LS) — your exact workflow")

    # Guard: need observed and historical loaded
    if obs_df is None or hist_df is None or obs_df.empty or hist_df.empty:
        st.error("Upload Observed and Historical files to run this section.")
    else:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        def quantile_delta_mapping(obs_hist, mod_hist, mod_fut, obs_fut,
                                   kind='multiplicative', max_ratio=4.0, clip_quantile=0.99):
            """
            Bias-correct future model data using Quantile Delta Mapping (QDM).
            obs_hist : 1D array of observed values (historical)
            mod_hist : 1D array of model historical values
            mod_fut  : 1D array of model future values
            obs_fut  : 1D array of observed future values (for diagnostics)
            kind : 'multiplicative' or 'additive'
            max_ratio : upper cap on multiplicative ratio
            clip_quantile : upper quantile for capping corrected values
            """
            n_f = len(mod_fut)
            jitter = np.random.uniform(0, 1e-6, size=n_f)
            mod_fut_j = mod_fut + jitter
            ranks = rankdata(mod_fut_j)
            tau_f = (ranks - 0.5) / float(n_f)

            try:
                mod_hist_q = np.quantile(mod_hist, tau_f, method='linear')
                obs_hist_q = np.quantile(obs_hist, tau_f, method='linear')
            except TypeError:
                mod_hist_q = np.quantile(mod_hist, tau_f, interpolation='linear')
                obs_hist_q = np.quantile(obs_hist, tau_f, interpolation='linear')

            corrected = np.zeros(n_f, dtype=float)
            threshold = 1e-6

            if kind == 'multiplicative':
                multiplicative = mod_hist_q > threshold
                ratio = np.ones(n_f, dtype=float)
                ratio[multiplicative] = mod_fut[multiplicative] / mod_hist_q[multiplicative]
                # Clip the ratio to prevent runaway
                ratio[multiplicative] = np.minimum(ratio[multiplicative], max_ratio)
                corrected[multiplicative] = obs_hist_q[multiplicative] * ratio[multiplicative]

                additive = ~multiplicative
                delta_a = mod_fut[additive] - mod_hist_q[additive]
                corrected[additive] = obs_hist_q[additive] + delta_a
            else:
                delta = mod_fut - mod_hist_q
                corrected = obs_hist_q + delta

            corrected[corrected < 0] = 0.0

            # Cap extremes based on observed future percentiles
            cap_value = np.percentile(obs_fut, clip_quantile * 100.0)
            outlier_mask = corrected > cap_value
            if np.any(outlier_mask):
                st.text(f"QDM: capping {np.sum(outlier_mask)} values above {cap_value:.2f}")
                corrected[outlier_mask] = cap_value

            return corrected

        def linear_scaling(obs_hist, mod_hist, mod_fut):
            scaling_factor = np.mean(obs_hist) / np.mean(mod_hist) if np.mean(mod_hist) > 0 else 1.0
            corrected = mod_fut * scaling_factor
            corrected[corrected < 0] = 0
            return corrected

        # Reading & alignment logic
        obs_work = obs_df.copy()
        mod_work = hist_df.copy()

        merged = obs_work.merge(mod_work, on="Date", how="inner", suffixes=("_obs", "_mod"))
        if merged.empty:
            st.error("No overlapping dates between Observed and Historical; please adjust inputs/date ranges.")
        else:
            obs_df_local = merged[["Date", "pr_obs"]].rename(columns={"pr_obs": "pr"})
            mod_df_local = merged[["Date", "pr_mod"]].rename(columns={"pr_mod": "pr"})

            obs_df_local["datetime"] = obs_df_local["Date"]
            mod_df_local["datetime"] = mod_df_local["Date"]

            obs_pr = obs_df_local["pr"].values
            mod_pr = mod_df_local["pr"].values

            n = len(obs_pr)
            split_idx = n // 2
            hist_period = slice(0, split_idx)
            fut_period = slice(split_idx, None)

            obs_hist = obs_pr[hist_period]
            mod_hist = mod_pr[hist_period]
            mod_fut = mod_pr[fut_period]
            obs_fut = obs_pr[fut_period]

            corrected_fut_qdm = quantile_delta_mapping(obs_hist, mod_hist, mod_fut, obs_fut,
                                                        kind='multiplicative', max_ratio=4.0, clip_quantile=0.99)
            corrected_fut_linear = linear_scaling(obs_hist, mod_hist, mod_fut)

            # Metrics
            r2_qdm = r2_score(obs_fut, corrected_fut_qdm) if len(obs_fut) else np.nan
            mae_qdm = mean_absolute_error(obs_fut, corrected_fut_qdm) if len(obs_fut) else np.nan
            rmse_qdm = np.sqrt(mean_squared_error(obs_fut, corrected_fut_qdm)) if len(obs_fut) else np.nan

            r2_linear = r2_score(obs_fut, corrected_fut_linear) if len(obs_fut) else np.nan
            mae_linear = mean_absolute_error(obs_fut, corrected_fut_linear) if len(obs_fut) else np.nan
            rmse_linear = np.sqrt(mean_squared_error(obs_fut, corrected_fut_linear)) if len(obs_fut) else np.nan

            st.markdown("#### QDM Metrics")
            st.text(f"R-squared (corrected vs observed future): {r2_qdm:.4f}")
            st.text(f"Mean Absolute Error (MAE): {mae_qdm:.4f}")
            st.text(f"Root Mean Squared Error (RMSE): {rmse_qdm:.4f}")

            st.markdown("#### Linear Scaling Metrics")
            st.text(f"R-squared (corrected vs observed future): {r2_linear:.4f}")
            st.text(f"Mean Absolute Error (MAE): {mae_linear:.4f}")
            st.text(f"Root Mean Squared Error (RMSE): {rmse_linear:.4f}")

            scaling_factor_raw = np.mean(mod_fut) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan
            scaling_factor_qdm = np.mean(corrected_fut_qdm) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan
            scaling_factor_linear = np.mean(corrected_fut_linear) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan

            st.markdown("#### Scaling Factors")
            st.text(f"Scaling factor (raw model / obs): {scaling_factor_raw:.4f}")
            st.text(f"Scaling factor (corrected QDM / obs): {scaling_factor_qdm:.4f}")
            st.text(f"Scaling factor (corrected linear / obs): {scaling_factor_linear:.4f}")

            variance_ratio_raw = np.var(mod_fut) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan
            variance_ratio_qdm = np.var(corrected_fut_qdm) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan
            variance_ratio_linear = np.var(corrected_fut_linear) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan

            st.markdown("#### Variance Ratios")
            st.text(f"Variance ratio (raw model / obs): {variance_ratio_raw:.4f}")
            st.text(f"Variance ratio (corrected QDM / obs): {variance_ratio_qdm:.4f}")
            st.text(f"Variance ratio (corrected linear / obs): {variance_ratio_linear:.4f}")

            residuals_qdm = obs_fut - corrected_fut_qdm
            residuals_linear = obs_fut - corrected_fut_linear

            st.markdown("#### Proportion of Zeros")
            st.text(f"obs_hist: {np.mean(obs_hist == 0):.4f}")
            st.text(f"mod_hist: {np.mean(mod_hist == 0):.4f}")
            st.text(f"obs_fut: {np.mean(obs_fut == 0):.4f}")
            st.text(f"mod_fut: {np.mean(mod_fut == 0):.4f}")
            st.text(f"corrected_fut_qdm: {np.mean(corrected_fut_qdm == 0):.4f}")
            st.text(f"corrected_fut_linear: {np.mean(corrected_fut_linear == 0):.4f}")

            st.markdown("#### Extreme Values")
            st.text(f"Max obs_hist: {np.max(obs_hist):.4f}")
            st.text(f"Max mod_hist: {np.max(mod_hist):.4f}")
            st.text(f"Max obs_fut: {np.max(obs_fut):.4f}")
            st.text(f"Max mod_fut: {np.max(mod_fut):.4f}")
            st.text(f"Max corrected_fut_qdm: {np.max(corrected_fut_qdm):.4f}")
            st.text(f"Max corrected_fut_linear: {np.max(corrected_fut_linear):.4f}")

            st.markdown("#### Residual Quantiles (QDM)")
            st.text(f"Min: {np.min(residuals_qdm):.4f}, 25th: {np.percentile(residuals_qdm,25):.4f}, "
                    f"Median: {np.median(residuals_qdm):.4f}, 75th: {np.percentile(residuals_qdm,75):.4f}, Max: {np.max(residuals_qdm):.4f}")

            st.markdown("#### Residual Quantiles (Linear)")
            st.text(f"Min: {np.min(residuals_linear):.4f}, 25th: {np.percentile(residuals_linear,25):.4f}, "
                    f"Median: {np.median(residuals_linear):.4f}, 75th: {np.percentile(residuals_linear,75):.4f}, Max: {np.max(residuals_linear):.4f}")

            # Plotting
            fut_dates = obs_df_local["datetime"].values[fut_period]

            st.markdown("#### Time series (future period)")
            fig1, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(fut_dates, obs_fut, label='Observed', alpha=0.7)
            ax1.plot(fut_dates, mod_fut, label='Raw Model', alpha=0.7)
            ax1.plot(fut_dates, corrected_fut_qdm, label='Corrected (QDM)', alpha=0.7)
            ax1.plot(fut_dates, corrected_fut_linear, label='Corrected (Linear)', alpha=0.7)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Precipitation (mm)')
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)

            st.markdown("#### QQ Plot")
            sorted_obs = np.sort(obs_fut)
            sorted_mod = np.sort(mod_fut)
            sorted_cor_qdm = np.sort(corrected_fut_qdm)
            sorted_cor_linear = np.sort(corrected_fut_linear)
            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.scatter(sorted_obs, sorted_mod, label='Raw Model', alpha=0.5)
            ax2.scatter(sorted_obs, sorted_cor_qdm, label='Corrected (QDM)', alpha=0.5)
            ax2.scatter(sorted_obs, sorted_cor_linear, label='Corrected (Linear)', alpha=0.5)
            if len(sorted_obs):
                ax2.plot([0, max(sorted_obs)], [0, max(sorted_obs)], 'r--', label='1:1 Line')
            ax2.set_xlabel('Observed Quantiles')
            ax2.set_ylabel('Model/Corrected Quantiles')
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

            st.markdown("#### Distribution of All Precipitation Data")
            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.hist(obs_hist, bins=50, alpha=0.3, label='Obs Hist', density=True)
            ax3.hist(mod_hist, bins=50, alpha=0.3, label='Mod Hist', density=True)
            ax3.hist(obs_fut, bins=50, alpha=0.3, label='Obs Fut', density=True)
            ax3.hist(mod_fut, bins=50, alpha=0.3, label='Mod Fut', density=True)
            ax3.hist(corrected_fut_qdm, bins=50, alpha=0.3, label='Corrected QDM', density=True)
            ax3.hist(corrected_fut_linear, bins=50, alpha=0.3, label='Corrected Linear', density=True)
            ax3.set_xlabel('Precipitation (mm)')
            ax3.set_ylabel('Density')
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3)

            st.markdown('#### Distribution of Wet Day Precipitation ("Future" Period)')
            fig4, ax4 = plt.subplots(figsize=(10,4))
            ax4.hist(obs_fut[obs_fut > 0], bins=50, alpha=0.5, label='Observed (wet days)', density=True)
            ax4.hist(mod_fut[mod_fut > 0], bins=50, alpha=0.5, label='Raw Model (wet days)', density=True)
            ax4.hist(corrected_fut_qdm[corrected_fut_qdm > 0], bins=50, alpha=0.5, label='Corrected QDM (wet days)', density=True)
            ax4.hist(corrected_fut_linear[corrected_fut_linear > 0], bins=50, alpha=0.5, label='Corrected Linear (wet days)', density=True)
            ax4.set_ylim(0, 0.22)
            ax4.set_xlabel('Precipitation (mm)')
            ax4.set_ylabel('Density')
            ax4.grid(True)
            ax4.legend()
            st.pyplot(fig4)

            st.markdown("#### Residuals of Corrected vs Observed Precipitation")
            fig5, ax5 = plt.subplots(figsize=(10,4))
            ax5.scatter(fut_dates, residuals_qdm, alpha=0.5, label='QDM Residuals')
            ax5.scatter(fut_dates, residuals_linear, alpha=0.5, label='Linear Residuals')
            ax5.axhline(0, color='red', linestyle='--')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Residuals (Observed - Corrected)')
            ax5.grid(True)
            ax5.legend()
            st.pyplot(fig5)

            large_residual_threshold = st.number_input("Large residual threshold (mm)", min_value=0.0, value=10.0, step=1.0)
            large_residual_indices_qdm = np.where(np.abs(residuals_qdm) > large_residual_threshold)[0]
            large_residual_indices_linear = np.where(np.abs(residuals_linear) > large_residual_threshold)[0]
            st.markdown("#### Large Residuals (QDM)")
            st.text(f"Number: {len(large_residual_indices_qdm)}")
            if len(large_residual_indices_qdm) > 0:
                st.text(f"Indices: {large_residual_indices_qdm}")
                st.text(f"Example values: {residuals_qdm[large_residual_indices_qdm][:5]}")
            st.markdown("#### Large Residuals (Linear)")
            st.text(f"Number: {len(large_residual_indices_linear)}")
            if len(large_residual_indices_linear) > 0:
                st.text(f"Indices: {large_residual_indices_linear}")
                st.text(f"Example values: {residuals_linear[large_residual_indices_linear][:5]}")


# ----------------------------
# New: CDF delta demo tab (add this new tab content)
# ----------------------------
with tab_cdfdemo:
    st.subheader("CDF-delta (real data)")

    # Require all three inputs
    if not (obs_df is not None and hist_df is not None and fut_df is not None):
        st.error("Upload Observed, Historical, and Futuristic files to run CDF-delta.")
    else:
        # ---------- Calendar alignment ----------
        st.markdown("##### Calendar alignment")

        # Normalize and get coverage
        def _prep_cov(df: pd.DataFrame, name: str):
            d = df.copy()
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.floor("D")
            d = d.dropna(subset=["Date"]).sort_values("Date")
            cov = dict(name=name, min=d["Date"].min(), max=d["Date"].max(), n=len(d))
            return d, cov

        o, cov_o = _prep_cov(obs_df, "Observed")
        h, cov_h = _prep_cov(hist_df, "Historic")
        f, cov_f = _prep_cov(fut_df, "Futuristic")

        c1, c2, c3 = st.columns(3)
        c1.caption(f"Observed: {cov_o['min'].date() if pd.notna(cov_o['min']) else '—'} → {cov_o['max'].date() if pd.notna(cov_o['max']) else '—'} | {cov_o['n']:,} rows")
        c2.caption(f"Historic: {cov_h['min'].date() if pd.notna(cov_h['min']) else '—'} → {cov_h['max'].date() if pd.notna(cov_h['max']) else '—'} | {cov_h['n']:,} rows")
        c3.caption(f"Futuristic: {cov_f['min'].date() if pd.notna(cov_f['min']) else '—'} → {cov_f['max'].date() if pd.notna(cov_f['max']) else '—'} | {cov_f['n']:,} rows")

        # Union bounds for selector
        mins = [c["min"] for c in [cov_o, cov_h, cov_f] if pd.notna(c["min"])]
        maxs = [c["max"] for c in [cov_o, cov_h, cov_f] if pd.notna(c["max"])]
        if not mins or not maxs:
            st.error("At least one series has no valid dates. Verify Date column parsing.")
            st.stop()

        union_start = min(mins).date()
        union_end = max(maxs).date()

        w1, w2 = st.columns(2)
        with w1:
            sel_start = st.date_input("Analysis start", value=union_start, min_value=union_start, max_value=union_end)
        with w2:
            sel_end = st.date_input("Analysis end", value=union_end, min_value=union_start, max_value=union_end)

        if sel_start > sel_end:
            st.error("Start date is after end date. Adjust the analysis window.")
            st.stop()

        start_ts = pd.to_datetime(sel_start)
        end_ts = pd.to_datetime(sel_end)

        # Clip to window
        def _clip(df):
            return df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

        o_win = _clip(o).rename(columns={"pr": "pr_obs"})
        h_win = _clip(h).rename(columns={"pr": "pr_hist"})
        f_win = _clip(f).rename(columns={"pr": "pr_fut"})

        # Matching mode
        st.markdown("##### Matching mode")
        mcol1, mcol2 = st.columns([2, 1])
        with mcol1:
            match_mode = st.selectbox("Match mode", ["Exact day", "Nearest within tolerance", "Monthly"], index=1)
        with mcol2:
            tol_days = st.slider("Tolerance (days)", 0, 14, 1, help="Used in 'Nearest within tolerance' mode")

        # Build daily grid over selected window
        daily_grid = pd.date_range(start=start_ts, end=end_ts, freq="D")

        def _reindex_daily(df, col):
            if df.empty:
                return pd.Series(index=daily_grid, dtype=float)
            s = df.set_index("Date")[col].astype(float)
            return s.reindex(daily_grid)

        # Prepare aligned frame
        active_mode = None
        if match_mode == "Exact day":
            so = _reindex_daily(o_win, "pr_obs")
            sh = _reindex_daily(h_win, "pr_hist")
            sf = _reindex_daily(f_win, "pr_fut")
            aligned = pd.DataFrame({
                "Date": daily_grid,
                "pr_obs": so.values,
                "pr_hist": sh.values,
                "pr_fut": sf.values
            }).dropna(subset=["pr_obs", "pr_hist"], how="any")
            active_mode = "Exact daily"
            if aligned.empty:
                st.info("No exact daily overlap in selected window; switching to nearest-match with tolerance.")
                match_mode = "Nearest within tolerance"

        if match_mode == "Nearest within tolerance":
            so = _reindex_daily(o_win, "pr_obs")
            sh = _reindex_daily(h_win, "pr_hist")
            sf = _reindex_daily(f_win, "pr_fut")

            def _nearest_map(source_s, target_s):
                src = source_s.dropna().reset_index()
                src.columns = ["Date", "val_src"]
                tgt = target_s.dropna().reset_index()
                tgt.columns = ["Date", "val_tgt"]
                if src.empty or tgt.empty:
                    return pd.Series(index=daily_grid, dtype=float)
                m = pd.merge_asof(
                    src.sort_values("Date"),
                    tgt.sort_values("Date"),
                    on="Date",
                    direction="nearest",
                    tolerance=pd.Timedelta(days=int(tol_days)),
                )
                out = pd.Series(index=src["Date"], data=m["val_tgt"].values)
                return out.reindex(daily_grid)

            sh_near = _nearest_map(so, sh)
            sf_near = _nearest_map(so, sf)
            aligned = pd.DataFrame({
                "Date": daily_grid,
                "pr_obs": so.values,
                "pr_hist": sh_near.values,
                "pr_fut": sf_near.values
            }).dropna(subset=["pr_obs", "pr_hist"], how="any")
            active_mode = f"Nearest (≤{tol_days}d)"
            if aligned.empty:
                st.info("Nearest-day alignment found no matches; switching to monthly mode.")
                match_mode = "Monthly"

        if match_mode == "Monthly":
            def _monthly_sum(df, name_col):
                if df.empty:
                    return pd.Series(dtype=float, name=name_col)
                d = df.copy()
                d["MS"] = d["Date"].values.astype("datetime64[M]")
                return d.groupby("MS")[name_col].sum(min_count=1)

            mo = _monthly_sum(o_win.rename(columns={"pr_obs": "pr"}), "pr")
            mh = _monthly_sum(h_win.rename(columns={"pr_hist": "pr"}), "pr")
            mf = _monthly_sum(f_win.rename(columns={"pr_fut": "pr"}), "pr")
            mo.name, mh.name, mf.name = "pr_obs", "pr_hist", "pr_fut"
            m_aligned = pd.concat([mo, mh, mf], axis=1)
            aligned = m_aligned.reset_index().rename(columns={"MS": "Date"})
            aligned = aligned.dropna(subset=["pr_obs", "pr_hist"], how="any")
            active_mode = "Monthly sums"

        if aligned.empty:
            st.error("Still no overlap after trying all modes. Widen the window or verify the files.")
            st.stop()

        st.success(f"Aligned {len(aligned):,} samples using mode: {active_mode}")

        # ---------- Windows and target ----------
        st.markdown("##### Windows and target series")
        mode = st.radio("Pick split mode", ["Half-split (demo)", "By date range"], horizontal=True)
        if mode == "Half-split (demo)":
            n = len(aligned)
            split_idx = n // 2
            hist_slice = slice(0, split_idx)
            fut_slice = slice(split_idx, None)
            work = aligned.reset_index(drop=True)
        else:
            dmin, dmax = aligned["Date"].min(), aligned["Date"].max()
            # Support both daily and monthly stamps
            dmin_date = pd.to_datetime(dmin).date()
            dmax_date = pd.to_datetime(dmax).date()
            cA, cB = st.columns(2)
            with cA:
                hist_to = st.date_input("Historical period end", value=dmin_date, min_value=dmin_date, max_value=dmax_date)
            with cB:
                fut_from = st.date_input("Future period start", value=dmax_date, min_value=dmin_date, max_value=dmax_date)
            work = aligned.copy()
            work["Date_ts"] = pd.to_datetime(work["Date"])
            hist_mask = work["Date_ts"] <= pd.to_datetime(hist_to)
            fut_mask = work["Date_ts"] >= pd.to_datetime(fut_from)
            hist_idx = np.where(hist_mask.values)[0]
            fut_idx = np.where(fut_mask.values)[0]
            if hist_idx.size == 0 or fut_idx.size == 0:
                st.error("Chosen windows have no data. Adjust dates.")
                st.stop()
            hist_slice = slice(hist_idx.min(), hist_idx.max() + 1)
            fut_slice = slice(fut_idx.min(), fut_idx.max() + 1)

        # Arrays
        dates = work["Date"].values
        obs_hist = work["pr_obs"].values[hist_slice]
        mod_hist = work["pr_hist"].values[hist_slice]
        obs_fut = work["pr_obs"].values[fut_slice]
        fut_dates = dates[fut_slice]

        target_label = st.selectbox("Future series to correct", ["Historical model (pr_hist)", "Scenario model (pr_fut)"])
        mod_fut_full = work["pr_hist"].values if target_label.startswith("Historical") else work["pr_fut"].values
        mod_fut = mod_fut_full[fut_slice]

        # ---------- CDF-delta computation ----------
        q_list = st.multiselect(
            "Quantiles for delta mapping",
            options=[5, 10, 25, 50, 75, 90, 95, 99],
            default=[50, 90, 95],
        )
        q_arr = np.array(sorted(set(q_list))) / 100.0 if len(q_list) else np.array([0.5, 0.9, 0.95])

        def _qtiles(x, q):
            try:
                return np.quantile(x, q, method="linear")
            except TypeError:
                return np.quantile(x, q, interpolation="linear")

        obs_hist_q = _qtiles(obs_hist, q_arr)
        mod_hist_q = _qtiles(mod_hist, q_arr)
        delta_q = obs_hist_q - mod_hist_q

        threshold = 1e-6
        ratio_q = np.divide(obs_hist_q, np.maximum(mod_hist_q, threshold))
        ratio_q = np.where(np.isfinite(ratio_q), ratio_q, 1.0)

        order = np.argsort(mod_hist_q)
        ax = mod_hist_q[order]
        ay_add = delta_q[order]
        ay_mul = ratio_q[order]

        add_field = np.interp(mod_fut, ax, ay_add, left=ay_add[0], right=ay_add[-1])
        mul_field = np.interp(mod_fut, ax, ay_mul, left=ay_mul[0], right=ay_mul[-1])

        wet_mask = mod_fut > threshold
        cor_cdf_delta = np.zeros_like(mod_fut, dtype=float)
        cor_cdf_delta[~wet_mask] = np.maximum(mod_fut[~wet_mask] + add_field[~wet_mask], 0.0)
        cor_cdf_delta[wet_mask] = np.maximum(mod_fut[wet_mask] * mul_field[wet_mask], 0.0)

        max_thr = min(np.percentile(obs_fut, 98), 50.0) if len(obs_fut) else 50.0
        cor_cdf_delta = np.minimum(cor_cdf_delta, max_thr)

        if len(obs_fut):
            z_obs = np.mean(obs_fut == 0)
            z_cor = np.mean(cor_cdf_delta == 0)
            n_f = len(cor_cdf_delta)
            if z_cor > z_obs:
                k = int((z_cor - z_obs) * n_f)
                zero_idx = np.where(cor_cdf_delta == 0)[0]
                k = min(k, len(zero_idx))
                if k > 0:
                    min_nz = np.min(obs_hist[obs_hist > 0]) if np.any(obs_hist > 0) else 0.1
                    idx = np.random.choice(zero_idx, size=k, replace=False)
                    cor_cdf_delta[idx] = min_nz
            elif z_cor < z_obs:
                k = int((z_obs - z_cor) * n_f)
                nz_idx = np.where(cor_cdf_delta > 0)[0]
                k = min(k, len(nz_idx))
                if k > 0:
                    idx = np.random.choice(nz_idx, size=k, replace=False)
                    cor_cdf_delta[idx] = 0.0

        # ---------- Metrics ----------
        def _metric_safe(y, x, func):
            return func(y, x) if len(y) == len(x) and len(y) > 1 else np.nan

        r2_cdf = _metric_safe(obs_fut, cor_cdf_delta, r2_score)
        mae_cdf = _metric_safe(obs_fut, cor_cdf_delta, mean_absolute_error)
        rmse_cdf = (
            np.sqrt(mean_squared_error(obs_fut, cor_cdf_delta))
            if len(obs_fut) == len(cor_cdf_delta) and len(obs_fut) > 1 else np.nan
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("CDF-delta R²", "N/A" if np.isnan(r2_cdf) else f"{r2_cdf:.3f}")
        m2.metric("CDF-delta MAE", "N/A" if np.isnan(mae_cdf) else f"{mae_cdf:.3f}")
        m3.metric("CDF-delta RMSE", "N/A" if np.isnan(rmse_cdf) else f"{rmse_cdf:.3f}")

        # ---------- Plots ----------
        st.markdown("#### Historical empirical CDFs and Δ annotations")
        xo = np.sort(obs_hist)
        xm = np.sort(mod_hist)
        Fo = np.arange(1, len(xo) + 1) / len(xo) if len(xo) else np.array([])
        Fm = np.arange(1, len(xm) + 1) / len(xm) if len(xm) else np.array([])

        fig_cdf, axcdf = plt.subplots(figsize=(9, 5))
        if len(xo):
            axcdf.plot(xo, Fo, label="Observed (hist)", color="blue", linewidth=2)
        if len(xm):
            axcdf.plot(xm, Fm, label="Model (hist)", color="red", linewidth=2)
        for q, oh, mh, dq in zip(q_arr, obs_hist_q, mod_hist_q, delta_q):
            axcdf.axvline(x=oh, color="green", linestyle="--", alpha=0.5)
            axcdf.axvline(x=mh, color="green", linestyle="--", alpha=0.5)
            axcdf.annotate(
                f"q{int(q*100)} Δ={dq:.2f}",
                xy=(oh, q),
                xytext=(oh + 0.05 * max(1.0, np.nanmax(xo) if len(xo) else 1.0), min(1.0, q + 0.05)),
                color="black",
            )
        axcdf.set_xlabel("Precipitation (mm)")
        axcdf.set_ylabel("CDF")
        axcdf.set_title("Historical empirical CDFs with selected Δ")
        axcdf.grid(True, linestyle="--", alpha=0.6)
        axcdf.legend()
        st.pyplot(fig_cdf)

        st.markdown("#### Time series (future window)")
        fig_cmp, axcmp = plt.subplots(figsize=(10, 4))
        axcmp.plot(fut_dates, obs_fut, label="Observed", alpha=0.7)
        axcmp.plot(fut_dates, mod_fut, label="Raw model", alpha=0.7)
        axcmp.plot(fut_dates, cor_cdf_delta, label="Corrected (CDF-delta)", alpha=0.8)
        axcmp.set_ylabel("Precipitation (mm)")
        axcmp.grid(True, ls="--", alpha=0.5)
        axcmp.legend()
        st.pyplot(fig_cmp)

        st.markdown("#### QQ plot vs observed-future")
        so = np.sort(obs_fut)
        sm = np.sort(mod_fut)
        sc = np.sort(cor_cdf_delta)
        fig_qqc, axqqc = plt.subplots(figsize=(6, 6))
        axqqc.scatter(so, sm, alpha=0.5, label="Raw")
        axqqc.scatter(so, sc, alpha=0.5, label="CDF-delta")
        lim = max(so.max() if so.size else 0, sm.max() if sm.size else 0, sc.max() if sc.size else 0)
        axqqc.plot([0, lim], [0, lim], "r--", label="1:1")
        axqqc.set_xlabel("Observed quantiles")
        axqqc.set_ylabel("Model/Corrected quantiles")
        axqqc.grid(True, ls="--", alpha=0.5)
        axqqc.legend()
        st.pyplot(fig_qqc)

        st.markdown("#### Residuals (Observed - CDF-delta)")
        res_cdf = obs_fut - cor_cdf_delta
        fig_rc, axrc = plt.subplots(figsize=(10, 3.5))
        axrc.scatter(fut_dates, res_cdf, alpha=0.5, label="CDF-delta")
        axrc.axhline(0, color="red", ls="--")
        axrc.set_ylabel("Residual (mm)")
        axrc.grid(True, ls="--", alpha=0.5)
        axrc.legend()
        st.pyplot(fig_rc)

        # ---------- Download ----------
        out_cdf = pd.DataFrame({
            "Date": fut_dates,
            "obs_fut": obs_fut,
            "mod_fut": mod_fut,
            "cor_cdf_delta": cor_cdf_delta,
            "res_cdf": res_cdf
        })
        st.download_button(
            "Download CDF-delta corrected (CSV)",
            data=out_cdf.to_csv(index=False).encode("utf-8"),
            file_name="cdf_delta_corrected_future.csv",
            mime="text/csv",
        )

with tab_freq:
    st.subheader("Frequency distributions and corrections (improved)")

    if not (obs_df is not None and hist_df is not None and fut_df is not None):
        st.info("Upload all three files to analyze frequency distributions.")
    else:
        import io
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score, mean_squared_error
        from math import ceil

        # --- Additional imports for xclim+Kappa tail handling ---
        try:
            import xarray as xr
            from xclim import sdba
            from xclim.sdba import Grouper
            from scipy.stats import kappa4
            XC_AVAILABLE = True
        except Exception:
            XC_AVAILABLE = False

        # -------------------------
        # Utilities & best-practice params
        # -------------------------
        WET_DAY_THRESHOLD = 0.1        # mm — threshold defining wet day
        HYBRID_TAIL_QUANTILE = 0.99   # used earlier in freq routines
        MAX_UPPER_SCALE = 4.0         # safety cap for tail linearization (if used)
        # Parameters for xclim+Kappa hybrid tail handling
        TAIL_QUANTILE = 0.99          # above this, use Kappa tail mapping
        MIN_KAPPA_SAMPLES = 50        # minimum wet-day samples to fit Kappa reliably
        KAPPA_FIT_WARN = True         # show fit warnings in streamlit

        # Freedman–Diaconis bins
        def freedman_diaconis_bins(data, minimum_bins=10, fallback_bins=50):
            data = np.asarray(data)
            data = data[np.isfinite(data)]
            if data.size < 2:
                return fallback_bins
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            n = data.size
            if iqr <= 0:
                return max(minimum_bins, int(ceil(np.sqrt(n))))
            h = 2 * iqr / (n ** (1/3))
            if h <= 0:
                return fallback_bins
            nbins = int(ceil((data.max() - data.min()) / h))
            if nbins < minimum_bins:
                nbins = minimum_bins
            return nbins

        def prepare_df_with_dates(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()
            d["Date"] = pd.to_datetime(d["Date"] if "Date" in d.columns else d.get("date"), errors="coerce").dt.floor("D")
            d = d.sort_values("Date").reset_index(drop=True)
            d["pr"] = pd.to_numeric(d["pr"], errors="coerce").clip(lower=0)
            d = d.dropna(subset=["Date"])
            return d

        obs_df_prep = prepare_df_with_dates(obs_df)
        hist_df_prep = prepare_df_with_dates(hist_df)
        fut_df_prep = prepare_df_with_dates(fut_df)

        # -------------------------
        # Frequency helpers (kept)
        # -------------------------
        def compute_frequency_distribution(df: pd.DataFrame, prcol: str, bins_arr: np.ndarray):
            pr = df[prcol].dropna().values
            counts, edges = np.histogram(pr, bins=bins_arr)
            mids = (edges[:-1] + edges[1:]) / 2.0
            total = counts.sum()
            freq = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
            out = pd.DataFrame({
                "Precipitation (mm)": np.round(mids, 3),
                "Count": counts,
                "Frequency": freq
            })
            return out

        def correct_frequency_distributions(obs_freq_df, hist_freq_df, bins_edges):
            obs = obs_freq_df.set_index("Precipitation (mm)")["Frequency"].rename("Frequency_obs")
            hist = hist_freq_df.set_index("Precipitation (mm)")["Frequency"].rename("Frequency_hist")
            merged = pd.concat([obs, hist], axis=1).fillna(0)

            # Simple scaling
            mean_obs = merged["Frequency_obs"].mean()
            mean_hist = merged["Frequency_hist"].mean() if merged["Frequency_hist"].mean() > 0 else 1.0
            simple_scaled = merged["Frequency_hist"] * (mean_obs / mean_hist)

            # Variance scaling (heuristic)
            cdf_hist = np.cumsum(merged["Frequency_hist"].values)
            cdf_obs = np.cumsum(merged["Frequency_obs"].values)
            var_obs = merged["Frequency_obs"].var()
            var_hist = merged["Frequency_hist"].var()
            if var_hist > 0 and var_obs > 0:
                scale_v = np.sqrt(var_obs / var_hist)
                q_idx = np.arange(1, len(cdf_hist)+1) / float(len(cdf_hist))
                q_idx_scaled = np.clip((q_idx - 0.5) * scale_v + 0.5, 0, 1)
                interp_obs_cdf = np.interp(q_idx_scaled, q_idx, cdf_obs)
                pdf_var_scaled = np.diff(np.concatenate(([0.0], interp_obs_cdf)))
                variance_scaled = pdf_var_scaled if pdf_var_scaled.size == merged.shape[0] else simple_scaled.values
            else:
                variance_scaled = simple_scaled.values

            # Quantile-mapping-like redistribution on bins (hybrid)
            hist_pdf = merged["Frequency_hist"].values
            obs_pdf = merged["Frequency_obs"].values
            hist_cdf = np.cumsum(hist_pdf)
            obs_cdf = np.cumsum(obs_pdf)

            n = len(hist_pdf)
            qm_pdf = np.zeros_like(hist_pdf)
            for i in range(n):
                q_h = hist_cdf[i]
                target_pos = np.searchsorted(obs_cdf, q_h, side="right")
                if target_pos <= 0:
                    target_bin = 0
                elif target_pos >= n:
                    target_bin = n-1
                else:
                    target_bin = target_pos - 1
                qm_pdf[target_bin] += hist_pdf[i]

            if qm_pdf.sum() > 0:
                qm_pdf = qm_pdf / qm_pdf.sum() * merged["Frequency_hist"].sum()

            # Tail smoothing / blending like earlier
            bin_quantiles = (np.arange(1, n+1) / float(n))
            tail_mask = bin_quantiles >= HYBRID_TAIL_QUANTILE
            if tail_mask.any():
                hist_tail_mean = hist_pdf[tail_mask].mean() if hist_pdf[tail_mask].size else 0
                obs_tail_mean = obs_pdf[tail_mask].mean() if obs_pdf[tail_mask].size else 0
                linear_tail = hist_pdf.copy()
                if hist_tail_mean > 0:
                    lin_scale = obs_tail_mean / hist_tail_mean if hist_tail_mean > 0 else 1.0
                    linear_tail[tail_mask] = np.minimum(hist_pdf[tail_mask] * lin_scale,
                                                         hist_pdf[tail_mask] * MAX_UPPER_SCALE)
                else:
                    linear_tail[tail_mask] = qm_pdf[tail_mask]
                blend_alpha = 0.6
                final_qm = qm_pdf.copy()
                final_qm[tail_mask] = blend_alpha * qm_pdf[tail_mask] + (1 - blend_alpha) * linear_tail[tail_mask]
            else:
                final_qm = qm_pdf

            final_qm = np.clip(final_qm, 0, None)
            if final_qm.sum() > 0:
                final_qm = final_qm / final_qm.sum() * merged["Frequency_hist"].sum()

            out = merged.reset_index().rename(columns={"index": "Precipitation (mm)"})
            out["Simple_Scaling"] = simple_scaled.values
            out["Variance_Scaling"] = variance_scaled
            out["Quantile_Mapping"] = final_qm

            for col in ["Simple_Scaling", "Variance_Scaling", "Quantile_Mapping"]:
                vals = out[col].values
                vals = np.clip(vals, 0, None)
                s = vals.sum()
                if s > 0:
                    out[col] = vals / s * out["Frequency_hist"].sum()
                else:
                    out[col] = np.zeros_like(vals)

            out["Precipitation (mm)"] = np.round(out["Precipitation (mm)"], 3)
            return out

        # -------------------------
        # Compute bins
        # -------------------------
        pooled = np.concatenate([
            obs_df_prep["pr"].values,
            hist_df_prep["pr"].values,
            fut_df_prep["pr"].values
        ])
        pooled = pooled[np.isfinite(pooled)]
        wet_only = pooled[pooled > WET_DAY_THRESHOLD]
        if wet_only.size > 0:
            nbins = freedman_diaconis_bins(wet_only, minimum_bins=20, fallback_bins=50)
        else:
            nbins = 30
        max_precip = pooled.max() if pooled.size else 0.0
        if max_precip <= 0:
            st.info("Insufficient precipitation data to compute frequency distributions.")
        else:
            zero_edge = 0.0
            wet_min = max(WET_DAY_THRESHOLD, 0.001)
            edges_wet = np.linspace(wet_min, max_precip, nbins)
            bins_edges = np.concatenate(([0.0, wet_min], edges_wet[1:] if len(edges_wet) > 1 else edges_wet))

            obs_freq = compute_frequency_distribution(obs_df_prep.rename(columns={"pr": "pr_obs"}), "pr_obs", bins_edges)
            hist_freq = compute_frequency_distribution(hist_df_prep, "pr", bins_edges)
            fut_freq = compute_frequency_distribution(fut_df_prep, "pr", bins_edges)

            merged_hist_obs = correct_frequency_distributions(obs_freq, hist_freq, bins_edges)

            # Diagnostics (kept)
            st.markdown("#### Diagnostics (Observed vs Historical)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("Frequency (head)")
                st.dataframe(merged_hist_obs.head(12), use_container_width=True)
            with c2:
                st.write("Distribution stats")
                st.write({
                    "Observed mean (pdf)": float(merged_hist_obs["Frequency_obs"].mean()),
                    "Historical mean (pdf)": float(merged_hist_obs["Frequency_hist"].mean()),
                    "Observed std (pdf)": float(merged_hist_obs["Frequency_obs"].std()),
                    "Historical std (pdf)": float(merged_hist_obs["Frequency_hist"].std())
                })
            with c3:
                st.write("Time-series preview (first 5 rows)")
                st.write("Observed:")
                st.write(obs_df_prep[["Date", "pr"]].head())
                st.write("Historical:")
                st.write(hist_df_prep[["Date", "pr"]].head())

            logy = st.checkbox("Log-scale Y axis", value=False, key="freq_logy_final")

            st.markdown("#### Corrected distributions vs Observed (bar charts)")
            for method in ["Simple_Scaling", "Variance_Scaling", "Quantile_Mapping"]:
                fig, ax = plt.subplots(figsize=(9, 4))
                width = (bins_edges[1] - bins_edges[0]) * 0.8 if len(bins_edges) > 1 else 0.4
                ax.bar(merged_hist_obs["Precipitation (mm)"], merged_hist_obs[method],
                       width=width, alpha=0.7, label=method, edgecolor="black")
                ax.bar(merged_hist_obs["Precipitation (mm)"], merged_hist_obs["Frequency_obs"],
                       width=width*0.5, alpha=0.5, label="Observed", edgecolor="black")
                ax.set_xlabel("Precipitation (mm)")
                ax.set_ylabel("Probability (PDF)")
                if logy:
                    ax.set_yscale("log")
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                ax.legend()
                st.pyplot(fig)

            def safe_r2_rmse(y_true, y_pred):
                try:
                    r2 = r2_score(y_true, y_pred)
                except:
                    r2 = np.nan
                try:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                except:
                    rmse = np.nan
                return r2, rmse

            r2_simple, rmse_simple = safe_r2_rmse(merged_hist_obs["Frequency_obs"], merged_hist_obs["Simple_Scaling"])
            r2_var, rmse_var = safe_r2_rmse(merged_hist_obs["Frequency_obs"], merged_hist_obs["Variance_Scaling"])
            r2_qm, rmse_qm = safe_r2_rmse(merged_hist_obs["Frequency_obs"], merged_hist_obs["Quantile_Mapping"])

            st.markdown("#### Fit scores (Observed PDF vs corrected Historical PDF)")
            st.write({
                "R2 Simple": float(r2_simple) if not np.isnan(r2_simple) else None,
                "RMSE Simple": float(rmse_simple) if not np.isnan(rmse_simple) else None,
                "R2 Variance": float(r2_var) if not np.isnan(r2_var) else None,
                "RMSE Variance": float(rmse_var) if not np.isnan(rmse_var) else None,
                "R2 QuantileMap": float(r2_qm) if not np.isnan(r2_qm) else None,
                "RMSE QuantileMap": float(rmse_qm) if not np.isnan(rmse_qm) else None
            })

            # Apply simple scaling to timeseries (kept)
            obs_mean_freq = merged_hist_obs["Frequency_obs"].mean()
            hist_mean_freq = merged_hist_obs["Frequency_hist"].mean() if merged_hist_obs["Frequency_hist"].mean() > 0 else 1.0
            scaling_factor_simple = obs_mean_freq / hist_mean_freq

            hist_corrected_simple = hist_df_prep.copy()
            hist_corrected_simple["pr_simple_scaled"] = hist_corrected_simple["pr"] * scaling_factor_simple
            fut_corrected_simple = fut_df_prep.copy()
            fut_corrected_simple["pr_simple_scaled"] = fut_corrected_simple["pr"] * scaling_factor_simple

            # Downloads (kept)
            hist_out = hist_corrected_simple[["Date", "pr", "pr_simple_scaled"]].copy().rename(columns={"pr":"pr_original"})
            buf_hist = io.BytesIO()
            with pd.ExcelWriter(buf_hist, engine="openpyxl") as writer:
                hist_out.to_excel(writer, sheet_name="corrected_hist_timeseries", index=False)
            st.download_button("Download corrected Historical time-series (XLSX)",
                               data=buf_hist.getvalue(),
                               file_name="corrected_historical_timeseries.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            fut_out = fut_corrected_simple[["Date", "pr", "pr_simple_scaled"]].copy().rename(columns={"pr":"pr_original"})
            buf_fut = io.BytesIO()
            with pd.ExcelWriter(buf_fut, engine="openpyxl") as writer:
                fut_out.to_excel(writer, sheet_name="corrected_future_timeseries", index=False)
            st.download_button("Download corrected Future time-series (XLSX)",
                               data=buf_fut.getvalue(),
                               file_name="corrected_future_timeseries.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            buf_freq = io.BytesIO()
            with pd.ExcelWriter(buf_freq, engine="openpyxl") as writer:
                merged_hist_obs.to_excel(writer, sheet_name="freq_corrections", index=False)
            st.download_button("Download frequency-correction summary (XLSX)",
                               data=buf_freq.getvalue(),
                               file_name="frequency_corrections_summary.xlsx",
                               mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")
                        # -------------------------
            # Combined corrected dataset (Historical + Future)
            # -------------------------
            try:
                combined_corrected = pd.concat([
                    hist_out.assign(Source="Historical"),
                    fut_out.assign(Source="Future")
                ], ignore_index=True)

                # Sort by date (optional, ensures chronological order)
                combined_corrected = combined_corrected.sort_values("Date").reset_index(drop=True)

                buf_combined = io.BytesIO()
                with pd.ExcelWriter(buf_combined, engine="openpyxl") as writer:
                    # Write combined merged data
                    combined_corrected.to_excel(writer, sheet_name="combined_corrected", index=False)
                    # Also keep individual sheets for clarity
                    hist_out.to_excel(writer, sheet_name="historical_corrected", index=False)
                    fut_out.to_excel(writer, sheet_name="future_corrected", index=False)
                    merged_hist_obs.to_excel(writer, sheet_name="freq_corrections_summary", index=False)

                st.download_button(
                    "📘 Download FULL Combined Corrected Data (Historical + Future)",
                    data=buf_combined.getvalue(),
                    file_name="combined_corrected_precipitation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Error creating combined corrected file: {e}")

