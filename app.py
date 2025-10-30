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
tab_upload, tab_timeseries, tab_bias, tab_freq, tab_cdfdemo, tab_downloads = st.tabs(
    ["Upload & overview", "Time series", "Bias correction (QDM/LS)", "Freq. distributions", "CDF delta demo", "Downloads"]
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
        # 1) Bring your functions verbatim
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        def quantile_delta_mapping(obs_hist, mod_hist, mod_fut, obs_fut):
            n_f = len(mod_fut)
            jitter = np.random.uniform(0, 1e-6, size=n_f)
            mod_fut_j = mod_fut + jitter
            ranks = rankdata(mod_fut_j)
            tau_f = (ranks - 0.5) / n_f
            # NumPy API compatibility shim
            try:
                mod_hist_q = np.quantile(mod_hist, tau_f, method='linear')
                obs_hist_q = np.quantile(obs_hist, tau_f, method='linear')
            except TypeError:
                mod_hist_q = np.quantile(mod_hist, tau_f, interpolation='linear')
                obs_hist_q = np.quantile(obs_hist, tau_f, interpolation='linear')

            corrected = np.zeros(n_f)
            threshold = 1e-6
            multiplicative = mod_hist_q > threshold
            delta_m = mod_fut[multiplicative] / mod_hist_q[multiplicative]
            corrected[multiplicative] = obs_hist_q[multiplicative] * delta_m

            additive = ~multiplicative
            delta_a = mod_fut[additive] - mod_hist_q[additive]
            corrected[additive] = obs_hist_q[additive] + delta_a

            corrected[corrected < 0] = 0
            max_threshold = min(np.percentile(obs_fut, 98), 50.0)
            outlier_mask = corrected > max_threshold
            corrected[outlier_mask] = max_threshold
            st.text(f"Number of outliers capped (> {max_threshold:.2f} mm): {np.sum(outlier_mask)}")

            zero_prop_obs = np.mean(obs_fut == 0)
            zero_prop_corrected = np.mean(corrected == 0)
            if zero_prop_corrected > zero_prop_obs:
                n_zeros_to_remove = int((zero_prop_corrected - zero_prop_obs) * n_f)
                zero_indices = np.where(corrected == 0)[0]
                if len(zero_indices) > 0:
                    indices_to_change = np.random.choice(zero_indices, size=min(n_zeros_to_remove, len(zero_indices)), replace=False)
                    min_nonzero = np.min(obs_hist[obs_hist > 0]) if np.any(obs_hist > 0) else 0.1
                    corrected[indices_to_change] = min_nonzero
            elif zero_prop_corrected < zero_prop_obs:
                n_zeros_to_add = int((zero_prop_obs - zero_prop_corrected) * n_f)
                non_zero_indices = np.where(corrected > 0)[0]
                if len(non_zero_indices) > 0:
                    indices_to_zero = np.random.choice(non_zero_indices, size=min(n_zeros_to_add, len(non_zero_indices)), replace=False)
                    corrected[indices_to_zero] = 0
            return corrected

        def linear_scaling(obs_hist, mod_hist, mod_fut):
            scaling_factor = np.mean(obs_hist) / np.mean(mod_hist) if np.mean(mod_hist) > 0 else 1.0
            corrected = mod_fut * scaling_factor
            corrected[corrected < 0] = 0
            return corrected

        # 2) Recreate your reading and alignment logic inside the app using the already loaded dataframes
        # The provided files have a sheet named 'observeddata' for observed and a plain two-column layout for historical [observeddata].
        # Here, obs_df and hist_df are already normalized with Date (datetime64[ns]) and pr (float) columns.
        obs_work = obs_df.copy()
        mod_work = hist_df.copy()

        # Optional: let users override sheet names in sidebar in the main app; here we proceed with normalized frames.
        # Ensure same length for the scripted split demo: align on inner join
        merged = obs_work.merge(mod_work, on="Date", how="inner", suffixes=("_obs", "_mod"))
        if merged.empty:
            st.error("No overlapping dates between Observed and Historical; please adjust inputs/date ranges.")
        else:
            # Restore your script’s variable names
            obs_df_local = merged[["Date", "pr_obs"]].rename(columns={"pr_obs": "pr"})
            mod_df_local = merged[["Date", "pr_mod"]].rename(columns={"pr_mod": "pr"})

            # Your script expects these columns
            obs_df_local["datetime"] = obs_df_local["Date"]
            mod_df_local["datetime"] = mod_df_local["Date"]

            obs_pr = obs_df_local["pr"].values
            mod_pr = mod_df_local["pr"].values

            # Split midpoint
            n = len(obs_pr)
            split_idx = n // 2
            hist_period = slice(0, split_idx)
            fut_period = slice(split_idx, None)

            obs_hist = obs_pr[hist_period]
            mod_hist = mod_pr[hist_period]
            mod_fut = mod_pr[fut_period]
            obs_fut = obs_pr[fut_period]

            # Run QDM and Linear Scaling per your code
            corrected_fut_qdm = quantile_delta_mapping(obs_hist, mod_hist, mod_fut, obs_fut)
            corrected_fut_linear = linear_scaling(obs_hist, mod_hist, mod_fut)

            # Metrics
            r2_qdm = r2_score(obs_fut, corrected_fut_qdm) if len(obs_fut) and len(corrected_fut_qdm) else np.nan
            mae_qdm = mean_absolute_error(obs_fut, corrected_fut_qdm) if len(obs_fut) and len(corrected_fut_qdm) else np.nan
            rmse_qdm = np.sqrt(mean_squared_error(obs_fut, corrected_fut_qdm)) if len(obs_fut) and len(corrected_fut_qdm) else np.nan

            r2_linear = r2_score(obs_fut, corrected_fut_linear) if len(obs_fut) and len(corrected_fut_linear) else np.nan
            mae_linear = mean_absolute_error(obs_fut, corrected_fut_linear) if len(obs_fut) and len(corrected_fut_linear) else np.nan
            rmse_linear = np.sqrt(mean_squared_error(obs_fut, corrected_fut_linear)) if len(obs_fut) and len(corrected_fut_linear) else np.nan

            st.markdown("#### QDM Metrics")
            st.text(f"R-squared (corrected vs observed future): {r2_qdm:.4f}")
            st.text(f"Mean Absolute Error (MAE): {mae_qdm:.4f}")
            st.text(f"Root Mean Squared Error (RMSE): {rmse_qdm:.4f}")

            st.markdown("#### Linear Scaling Metrics")
            st.text(f"R-squared (corrected vs observed future): {r2_linear:.4f}")
            st.text(f"Mean Absolute Error (MAE): {mae_linear:.4f}")
            st.text(f"Root Mean Squared Error (RMSE): {rmse_linear:.4f}")

            # Scaling factors and variance ratios
            scaling_factor_raw = np.mean(mod_fut) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan
            scaling_factor_corrected_qdm = np.mean(corrected_fut_qdm) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan
            scaling_factor_corrected_linear = np.mean(corrected_fut_linear) / np.mean(obs_fut) if np.mean(obs_fut) > 0 else np.nan

            st.markdown("#### Scaling Factors")
            st.text(f"Scaling factor (raw model / obs): {scaling_factor_raw:.4f}")
            st.text(f"Scaling factor (corrected QDM / obs): {scaling_factor_corrected_qdm:.4f}")
            st.text(f"Scaling factor (corrected linear / obs): {scaling_factor_corrected_linear:.4f}")

            variance_ratio_raw = np.var(mod_fut) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan
            variance_ratio_corrected_qdm = np.var(corrected_fut_qdm) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan
            variance_ratio_corrected_linear = np.var(corrected_fut_linear) / np.var(obs_fut) if np.var(obs_fut) > 0 else np.nan

            st.markdown("#### Variance Ratios")
            st.text(f"Variance ratio (raw model / obs): {variance_ratio_raw:.4f}")
            st.text(f"Variance ratio (corrected QDM / obs): {variance_ratio_corrected_qdm:.4f}")
            st.text(f"Variance ratio (corrected linear / obs): {variance_ratio_corrected_linear:.4f}")

            # Residuals
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
            st.text(f"Min: {np.min(residuals_qdm):.4f}, 25th: {np.percentile(residuals_qdm, 25):.4f}, "
                    f"Median: {np.median(residuals_qdm):.4f}, 75th: {np.percentile(residuals_qdm, 75):.4f}, "
                    f"Max: {np.max(residuals_qdm):.4f}")

            st.markdown("#### Residual Quantiles (Linear)")
            st.text(f"Min: {np.min(residuals_linear):.4f}, 25th: {np.percentile(residuals_linear, 25):.4f}, "
                    f"Median: {np.median(residuals_linear):.4f}, 75th: {np.percentile(residuals_linear, 75):.4f}, "
                    f"Max: {np.max(residuals_linear):.4f}")

            # 3) Plotting within Streamlit (use st.pyplot instead of plt.show and avoid saving to disk)
            fut_dates = obs_df_local["datetime"].values[fut_period]

            st.markdown("#### Time series (future period)")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
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
            fig2, ax2 = plt.subplots(figsize=(6, 6))
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
            fig3, ax3 = plt.subplots(figsize=(10, 4))
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
            fig4, ax4 = plt.subplots(figsize=(10, 4))
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
            fig5, ax5 = plt.subplots(figsize=(10, 4))
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
    st.subheader("CDF delta demo with annotated quantiles")
    st.caption("Synthetic example to visualize delta adjustments at q50 and q95 using normal distributions.")
    # Parameters
    u1, s1 = 100, 20
    bias_d = st.number_input("Bias shift (Δ mean)", value=2.0, step=0.5)
    n = 100
    q_ = 20

    dist1 = norm(loc=u1, scale=s1)
    dist2 = norm(loc=u1 + bias_d, scale=s1)

    q_dist1 = np.array([dist1.ppf(x * 0.05) for x in range(1, q_)])
    q_dist2 = np.array([dist2.ppf(x * 0.05) for x in range(1, q_)])

    ref_dataset = np.random.normal(u1, s1, n).round(2)
    model_present = (q_dist1 - np.random.normal(20, 2, q_ - 1)).round(2)

    observed_sorted = np.sort(ref_dataset)
    historical_sorted = np.sort(model_present)

    observed_cdf = np.arange(1, len(observed_sorted) + 1) / len(observed_sorted)
    historical_cdf = np.arange(1, len(historical_sorted) + 1) / len(historical_sorted)

    q50_observed = np.percentile(observed_sorted, 50)
    q50_historical = np.percentile(historical_sorted, 50)
    delta_q50 = q50_observed - q50_historical

    q95_observed = np.percentile(observed_sorted, 95)
    q95_historical = np.percentile(historical_sorted, 95)
    delta_q95 = q95_observed - q95_historical

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(observed_sorted, observed_cdf, label="observed", color="blue", linewidth=2)
    ax.plot(historical_sorted, historical_cdf, label="simulated", color="red", linewidth=2)
    # annotate q50
    ax.axvline(x=q50_observed, color="green", linestyle="--", alpha=0.6)
    ax.axvline(x=q50_historical, color="green", linestyle="--", alpha=0.6)
    ax.annotate("Delta adjust", xy=(q50_observed, 0.5), xytext=(q50_observed - 10, 0.6),
                arrowprops=dict(facecolor="green", shrink=0.05), color="green")
    ax.annotate(f"q50 (Δ={delta_q50:.2f})", xy=(q50_observed, 0.5), xytext=(q50_observed + 5, 0.45), color="black")
    ax.annotate("q50", xy=(q50_historical, 0.5), xytext=(q50_historical + 5, 0.55), color="black")
    # annotate q95
    ax.axvline(x=q95_observed, color="green", linestyle="--", alpha=0.6)
    ax.axvline(x=q95_historical, color="green", linestyle="--", alpha=0.6)
    ax.annotate("Delta adjust", xy=(q95_observed, 0.95), xytext=(q95_observed - 10, 1.0),
                arrowprops=dict(facecolor="green", shrink=0.05), color="green")
    ax.annotate(f"q95 (Δ={delta_q95:.2f})", xy=(q95_observed, 0.95), xytext=(q95_observed + 5, 0.9), color="black")
    ax.annotate("q95", xy=(q95_historical, 0.95), xytext=(q95_historical + 5, 1.0), color="black")

    ax.set_xlabel("Quantiles")
    ax.set_ylabel("CDF")
    ax.set_title("Observed vs Simulated CDF with Delta Annotations")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    st.pyplot(fig)
    
with tab_freq:
    st.subheader("Frequency distributions and corrections")
    if not (obs_df is not None and hist_df is not None and fut_df is not None):
        st.info("Upload all three files to analyze frequency distributions.")
    else:
        # Prepare bins across all three datasets
        max_precip = max(
            obs_df["pr"].max() if not obs_df.empty else 0,
            hist_df["pr"].max() if not hist_df.empty else 0,
            fut_df["pr"].max() if not fut_df.empty else 0
        )
        step = st.number_input("Bin step (mm)", min_value=0.1, value=0.1, step=0.1)
        bins = np.round(np.arange(0.2, max_precip + step, step), 1) if max_precip > 0 else np.array([])

        if bins.size == 0:
            st.info("Insufficient data to build frequency bins.")
        else:
            obs_freq = compute_frequency_distribution(obs_df.rename(columns={"pr": "pr_obs"}), "pr_obs", bins)
            hist_freq = compute_frequency_distribution(hist_df, "pr", bins)
            fut_freq = compute_frequency_distribution(fut_df, "pr", bins)

            merged_hist_obs = correct_frequency_distributions(obs_freq, hist_freq)

            # Diagnostics
            st.markdown("#### Diagnostics (Observed vs Historical)")
            diag_cols = st.columns(2)
            with diag_cols[0]:
                st.write("Merged head")
                st.write(merged_hist_obs.head())
            with diag_cols[1]:
                st.write("Stats")
                st.write({
                    "Observed mean": float(merged_hist_obs["Frequency_obs"].mean()),
                    "Historical mean": float(merged_hist_obs["Frequency_hist"].mean()),
                    "Observed std": float(merged_hist_obs["Frequency_obs"].std()),
                    "Historical std": float(merged_hist_obs["Frequency_hist"].std())
                })

            # Plots
            st.markdown("#### Corrected distributions vs Observed")
            for method in ["Simple_Scaling", "Variance_Scaling", "Quantile_Mapping"]:
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(merged_hist_obs["Precipitation (mm)"], merged_hist_obs[method],
                       color="purple", edgecolor="black", width=0.4, alpha=0.7, label=method)
                ax.bar(merged_hist_obs["Precipitation (mm)"], merged_hist_obs["Frequency_obs"],
                       color="royalblue", edgecolor="black", width=0.2, alpha=0.5, label="Observed")
                ax.set_xlabel("Precipitation (mm)")
                ax.set_ylabel("Probability")
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                ax.legend()
                st.pyplot(fig)

            # R2 scores
            st.markdown("#### R² scores (Observed vs corrected Historical)")
            r2_simple = r2_score(merged_hist_obs["Frequency_obs"], merged_hist_obs["Simple_Scaling"])
            r2_var = r2_score(merged_hist_obs["Frequency_obs"], merged_hist_obs["Variance_Scaling"])
            if merged_hist_obs["Quantile_Mapping"].notna().any():
                r2_qm = r2_score(merged_hist_obs["Frequency_obs"], merged_hist_obs["Quantile_Mapping"])
            else:
                r2_qm = np.nan
            st.write({"R2 Simple": float(r2_simple), "R2 Variance": float(r2_var), "R2 QuantileMap": float(r2_qm) if not np.isnan(r2_qm) else None})

            # Apply to future freq
            fut_corr = apply_fd_to_future(fut_freq, merged_hist_obs)

            # Downloads
            xls_buf_hist = io.BytesIO()
            with pd.ExcelWriter(xls_buf_hist, engine="xlsxwriter") as writer:
                merged_hist_obs.to_excel(writer, sheet_name="corrected_hist_freq", index=False)
            st.download_button("Download corrected Historical frequency (XLSX)",
                               data=xls_buf_hist.getvalue(),
                               file_name="corrected_historical_freq.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  # download_button [web:60]

            xls_buf_fut = io.BytesIO()
            with pd.ExcelWriter(xls_buf_fut, engine="xlsxwriter") as writer:
                fut_corr.to_excel(writer, sheet_name="corrected_future_freq", index=False)
            st.download_button("Download corrected Futuristic frequency (XLSX)",
                               data=xls_buf_fut.getvalue(),
                               file_name="corrected_future_freq.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  # download_button [web:60]

with tab_downloads:
    st.subheader("Export options")
    st.write("Use the download buttons embedded in each tab to export CSV/XLSX outputs.")  # general guidance
