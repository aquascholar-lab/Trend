# ============================================================
# Hydrological & Climate Data Trend Analysis Web App
# Streamlit Version
# Developed for hydrological year analysis: Oct–Sep
# Ladakh seasons:
#   Winter: Oct–Mar
#   Spring: Apr–Jun
#   Summer: Jul–Sep
# ============================================================

import calendar
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import pymannkendall as mk
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Hydrological & Climate Trend Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS for beautiful UI
# ============================================================

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 45%, #e3f2fd 100%);
    }

    /* Hide default Streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main hero box */
    .hero-box {
        background: linear-gradient(135deg, #004e92, #000428);
        padding: 35px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 12px 35px rgba(0, 78, 146, 0.35);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 18px;
        opacity: 0.95;
        margin-bottom: 8px;
    }

    .developer-text {
        font-size: 15px;
        opacity: 0.9;
        margin-top: 18px;
    }

    /* Card style */
    .custom-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(0, 105, 180, 0.12);
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0, 80, 130, 0.10);
        margin-bottom: 18px;
    }

    .section-title {
        color: #005a9c;
        font-weight: 800;
        font-size: 26px;
        margin-bottom: 10px;
    }

    .small-muted {
        color: #5f6c7b;
        font-size: 15px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(0, 105, 180, 0.12);
        box-shadow: 0 6px 18px rgba(0, 80, 130, 0.08);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd, #ffffff);
        border-right: 1px solid rgba(0, 105, 180, 0.12);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 700;
    }

    /* Dataframe styling container */
    .dataframe-title {
        color: #004e92;
        font-weight: 700;
        font-size: 20px;
        margin-top: 10px;
    }

    /* Info badges */
    .badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: #e3f2fd;
        color: #004e92;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(0, 78, 146, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Helper functions
# ============================================================

def get_ladakh_season(month: int) -> str:
    """
    Ladakh season definition:
    Winter: Oct–Mar
    Spring: Apr–Jun
    Summer: Jul–Sep
    """
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6]:
        return "Spring"
    else:
        return "Summer"


def hydro_year_start(date_value) -> int:
    """
    Hydrological year starts in October and ends in September.

    Example:
    2000-10 to 2001-09 is labelled as hydrological year 2000.
    """
    if date_value.month >= 10:
        return date_value.year
    return date_value.year - 1


def hydro_year_label(year: int) -> str:
    """Convert hydrological year start into a display label."""
    return f"{year}–{year + 1}"


def robust_parse_dates(series: pd.Series) -> pd.Series:
    """
    Robust date parser for formats such as:
    YYYY-MM, YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY, etc.
    """
    parsed = pd.to_datetime(series, errors="coerce")

    # If too many failed, try dayfirst=True
    if parsed.notna().sum() < max(1, int(0.5 * len(series))):
        parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed_dayfirst.notna().sum() > parsed.notna().sum():
            parsed = parsed_dayfirst

    # For newer pandas versions, try mixed format if needed
    try:
        parsed_mixed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed_mixed.notna().sum() > parsed.notna().sum():
            parsed = parsed_mixed
    except Exception:
        pass

    return parsed


def detect_date_column(df: pd.DataFrame):
    """Detect date column by case-insensitive match."""
    for col in df.columns:
        if str(col).strip().lower() == "date":
            return col

    # Fallback: detect columns containing date keyword
    for col in df.columns:
        if "date" in str(col).strip().lower():
            return col

    return None


def detect_numeric_parameter_columns(df: pd.DataFrame, date_col: str):
    """
    Detect valid numeric data columns.
    Ignores:
    - Date column
    - Columns containing trend / trendline
    - Fully non-numeric columns
    """
    parameter_cols = []

    for col in df.columns:
        if col == date_col:
            continue

        col_lower = str(col).lower()

        # Ignore precomputed trendline/helper columns
        if "trend" in col_lower or "trendline" in col_lower:
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")

        if numeric_series.notna().sum() >= 3:
            parameter_cols.append(col)

    return parameter_cols


def safe_mk_test(values):
    """
    Safely run Mann-Kendall test.
    Requires at least 3 valid values.
    """
    series = pd.Series(values).dropna()

    if len(series) < 3:
        return {
            "Trend": "Insufficient data",
            "h (reject H0?)": "NA",
            "p-value": np.nan,
            "Z statistic": np.nan,
            "Tau": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
        }

    try:
        result = mk.original_test(series.values)
        return {
            "Trend": result.trend,
            "h (reject H0?)": result.h,
            "p-value": result.p,
            "Z statistic": result.z,
            "Tau": result.Tau,
            "Slope": result.slope,
            "Intercept": result.intercept,
        }
    except Exception:
        return {
            "Trend": "Test failed",
            "h (reject H0?)": "NA",
            "p-value": np.nan,
            "Z statistic": np.nan,
            "Tau": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
        }


def trend_dict_to_df(trend_dict: dict) -> pd.DataFrame:
    """Convert MK result dictionary to dataframe."""
    return pd.DataFrame(
        list(trend_dict.items()),
        columns=["Metric", "Value"]
    )


def apply_plot_style(fig, height=500):
    """Apply consistent Plotly style."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title={
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "family": "Arial", "color": "#004e92"}
        },
        margin=dict(l=30, r=30, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0.98)",
        paper_bgcolor="rgba(255,255,255,0)"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 78, 146, 0.08)",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 78, 146, 0.08)",
        zeroline=False
    )

    return fig


def prepare_analysis(df: pd.DataFrame, date_col: str, parameter_col: str, start_year: int, end_year: int):
    """
    Clean data and prepare all analysis datasets.
    """
    working_df = df.copy()

    # Parse date
    working_df[date_col] = robust_parse_dates(working_df[date_col])
    working_df = working_df.dropna(subset=[date_col])

    if working_df.empty:
        raise ValueError("No valid dates found after parsing.")

    # Sort and index
    working_df = working_df.sort_values(date_col)
    working_df = working_df.set_index(date_col)

    # Convert selected parameter to numeric
    working_df[parameter_col] = pd.to_numeric(working_df[parameter_col], errors="coerce")
    working_df = working_df.dropna(subset=[parameter_col])

    if working_df.empty:
        raise ValueError(f"No valid numeric values found for selected parameter: {parameter_col}")

    # Add hydrological year and Ladakh season
    working_df["HydroYear"] = working_df.index.to_series().apply(hydro_year_start)
    working_df["Season"] = working_df.index.month.map(get_ladakh_season)

    # Filter by hydrological year range
    working_df = working_df[
        (working_df["HydroYear"] >= start_year) &
        (working_df["HydroYear"] <= end_year)
    ]

    if working_df.empty:
        raise ValueError("No data available for the selected hydrological year range.")

    # Main time series
    original_series = working_df[parameter_col].copy()

    # Monthly mean
    monthly_series = original_series.resample("M").mean().dropna()

    # Hydrological-year mean
    yearly_series = working_df.groupby("HydroYear")[parameter_col].mean()

    # Seasonal mean by hydrological year
    seasonal_df = (
        working_df
        .groupby(["HydroYear", "Season"])[parameter_col]
        .mean()
        .unstack()
    )

    # Enforce season order
    season_order = ["Winter", "Spring", "Summer"]
    seasonal_df = seasonal_df.reindex(columns=[s for s in season_order if s in seasonal_df.columns])

    # Monthly by calendar month within hydrological years
    monthly_by_month = (
        working_df
        .groupby(["HydroYear", working_df.index.month])[parameter_col]
        .mean()
        .unstack()
    )

    if not monthly_by_month.empty:
        monthly_by_month.columns = [calendar.month_name[int(m)] for m in monthly_by_month.columns]

        # Put Oct, Nov, Dec first for hydrological-year logic, then Jan–Sep
        hydro_month_order = [
            "October", "November", "December",
            "January", "February", "March",
            "April", "May", "June",
            "July", "August", "September"
        ]
        monthly_by_month = monthly_by_month.reindex(
            columns=[m for m in hydro_month_order if m in monthly_by_month.columns]
        )

    return {
        "df": working_df,
        "original_series": original_series,
        "monthly_series": monthly_series,
        "yearly_series": yearly_series,
        "seasonal_df": seasonal_df,
        "monthly_by_month": monthly_by_month
    }


def build_monthly_mk_table(original_series: pd.Series) -> pd.DataFrame:
    """MK test for each month."""
    rows = []

    hydro_month_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for m in hydro_month_order:
        month_name = calendar.month_name[m]
        month_series = original_series[original_series.index.month == m].dropna()
        result = safe_mk_test(month_series.values)
        row = {"Month": month_name, "Records": len(month_series)}
        row.update(result)
        rows.append(row)

    return pd.DataFrame(rows)


def build_seasonal_mk_table(seasonal_df: pd.DataFrame) -> pd.DataFrame:
    """MK test for each Ladakh season."""
    rows = []

    for season in seasonal_df.columns:
        season_series = seasonal_df[season].dropna()
        result = safe_mk_test(season_series.values)
        row = {"Season": season, "Records": len(season_series)}
        row.update(result)
        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes."""
    return df.to_csv(index=True).encode("utf-8")


# ============================================================
# Header / Hero section
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🌊 Hydrological & Climate Data Trend Analysis</div>
        <div class="hero-subtitle">
            Interactive Streamlit web application for hydrological-year based trend analysis using
            Mann–Kendall statistics, seasonal aggregation and Plotly visual analytics.
        </div>
        <div>
            <span class="badge">Hydrological year: Oct–Sep</span>
            <span class="badge">Ladakh seasons</span>
            <span class="badge">Mann–Kendall trend test</span>
            <span class="badge">Interactive Plotly charts</span>
        </div>
        <div class="developer-text">
            Developed by <b>Dr. Sachchidanand Singh</b>, Scientist, Western Himalayan Regional Centre, Jammu
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Sidebar documentation and upload
# ============================================================

with st.sidebar:
    st.markdown("## 📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain one Date column and at least one numeric parameter column."
    )

    st.markdown("---")

    st.markdown("## 📌 Required Format")
    st.info(
        """
        Your CSV should contain:
        - One `Date` column
        - One or more numeric data columns
        - Extra columns like `Trendline` will be ignored
        """
    )

    sample_df = pd.DataFrame({
        "date": pd.date_range("2000-10-01", periods=12, freq="M").strftime("%Y-%m"),
        "Rainfall": [12, 18, 22, 35, 40, 55, 20, 12, 8, 4, 3, 10],
        "Temperature": [-5, -8, -10, -7, -2, 3, 8, 10, 12, 15, 14, 6],
        "Trendline": [np.nan] * 12
    })

    st.download_button(
        label="⬇️ Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_hydro_climate_data.csv",
        mime="text/csv"
    )

    st.markdown("---")

    with st.expander("📖 App documentation"):
        st.markdown(
            """
            This Streamlit app performs climate and hydrological trend analysis.

            **Main steps:**
            1. Upload CSV data.
            2. Select the parameter column.
            3. Select hydrological year range.
            4. Run trend analysis.
            5. View interactive charts and Mann–Kendall statistics.

            **Hydrological year definition:**
            October to September.

            **Ladakh seasonal definition:**
            - Winter: October to March
            - Spring: April to June
            - Summer: July to September
            """
        )


# ============================================================
# Main app logic
# ============================================================

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">📤 Upload</div>
                <div class="small-muted">
                    Upload a CSV file containing a Date column and one or more hydrological or climate variables.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">⚙️ Select</div>
                <div class="small-muted">
                    Choose the parameter and hydrological year range directly from the interface.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">📈 Analyze</div>
                <div class="small-muted">
                    Generate trend charts, seasonal summaries and Mann–Kendall statistics.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning("Please upload a CSV file from the sidebar to begin analysis.")

else:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">✅ Dataset Uploaded Successfully</div>
                <div class="small-muted">
                    The file has been read successfully. Review the detected structure below and select analysis options.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Detect date column
        date_col = detect_date_column(raw_df)

        if date_col is None:
            st.error("❌ No Date column found. Please ensure your CSV contains a column named Date/date/DATE.")
            st.stop()

        # Parse dates for year detection
        parsed_dates = robust_parse_dates(raw_df[date_col])
        valid_dates_mask = parsed_dates.notna()

        if valid_dates_mask.sum() == 0:
            st.error("❌ Date column was detected, but no valid dates could be parsed.")
            st.stop()

        temp_df = raw_df.loc[valid_dates_mask].copy()
        temp_df[date_col] = parsed_dates.loc[valid_dates_mask]
        temp_df = temp_df.sort_values(date_col).set_index(date_col)

        # Detect numeric parameter columns
        parameter_options = detect_numeric_parameter_columns(raw_df, date_col)

        if len(parameter_options) == 0:
            st.error(
                "❌ No valid numeric parameter column was found. "
                "Please add at least one numeric variable column besides Date."
            )
            st.stop()

        # Hydrological years available
        hydro_years = sorted(temp_df.index.to_series().apply(hydro_year_start).unique())

        if len(hydro_years) == 0:
            st.error("❌ No hydrological years could be detected.")
            st.stop()

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(raw_df):,}")
        m2.metric("Valid dates", f"{valid_dates_mask.sum():,}")
        m3.metric("Parameters", len(parameter_options))
        m4.metric("Hydro years", len(hydro_years))

        with st.expander("🔍 Preview uploaded data", expanded=False):
            st.dataframe(raw_df.head(50), use_container_width=True)

        # ============================================================
        # Selection controls
        # ============================================================

        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">⚙️ Analysis Settings</div>
                <div class="small-muted">
                    Select the data column and hydrological year period for analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([1.4, 1, 1])

        with c1:
            selected_parameter = st.selectbox(
                "Select parameter to analyze",
                parameter_options,
                help="If multiple columns are available, select the variable to process."
            )

        with c2:
            start_year = st.selectbox(
                "Start hydrological year",
                hydro_years,
                index=0,
                format_func=hydro_year_label
            )

        with c3:
            end_year_candidates = [y for y in hydro_years if y >= start_year]

            end_year = st.selectbox(
                "End hydrological year",
                end_year_candidates,
                index=len(end_year_candidates) - 1,
                format_func=hydro_year_label
            )

        run_analysis = st.button(
            "🚀 Run Hydrological Trend Analysis",
            type="primary",
            use_container_width=True
        )

        if run_analysis:
            results = prepare_analysis(
                df=raw_df,
                date_col=date_col,
                parameter_col=selected_parameter,
                start_year=start_year,
                end_year=end_year
            )

            df = results["df"]
            original_series = results["original_series"]
            monthly_series = results["monthly_series"]
            yearly_series = results["yearly_series"]
            seasonal_df = results["seasonal_df"]
            monthly_by_month = results["monthly_by_month"]

            hydro_year_min = int(df["HydroYear"].min())
            hydro_year_max = int(df["HydroYear"].max())

            st.success(
                f"Analysis completed for {selected_parameter} "
                f"from hydrological year {hydro_year_label(hydro_year_min)} "
                f"to {hydro_year_label(hydro_year_max)}."
            )

            # ============================================================
            # Summary metrics
            # ============================================================

            st.markdown("## 📊 Summary")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Parameter", selected_parameter)
            s2.metric("Mean", f"{df[selected_parameter].mean():.3f}")
            s3.metric("Minimum", f"{df[selected_parameter].min():.3f}")
            s4.metric("Maximum", f"{df[selected_parameter].max():.3f}")
            s5.metric("Records used", f"{len(df):,}")

            # ============================================================
            # Mann-Kendall analysis
            # ============================================================

            daily_mk_df = trend_dict_to_df(safe_mk_test(original_series.values))
            yearly_mk_df = trend_dict_to_df(safe_mk_test(yearly_series.values))
            monthly_mk_df = build_monthly_mk_table(original_series)
            seasonal_mk_df = build_seasonal_mk_table(seasonal_df)

            # ============================================================
            # Charts
            # ============================================================

            fig_original = px.line(
                x=original_series.index,
                y=original_series.values,
                title=f"Original Time Series of {selected_parameter}",
                labels={"x": "Date", "y": selected_parameter}
            )
            fig_original = apply_plot_style(fig_original)

            fig_monthly = px.line(
                x=monthly_series.index,
                y=monthly_series.values,
                title=f"Monthly Average {selected_parameter}",
                labels={"x": "Date", "y": selected_parameter}
            )
            fig_monthly = apply_plot_style(fig_monthly)

            fig_yearly = px.line(
                x=yearly_series.index.astype(str),
                y=yearly_series.values,
                markers=True,
                title=f"Hydrological-Year Average {selected_parameter} (Oct–Sep)",
                labels={"x": "Hydrological year start", "y": selected_parameter}
            )
            fig_yearly = apply_plot_style(fig_yearly)

            fig_monthly_all = px.line(
                monthly_by_month,
                markers=True,
                title=f"Monthly Trend by Calendar Month ({selected_parameter})",
                labels={"HydroYear": "Hydrological year", "value": selected_parameter}
            )
            fig_monthly_all = apply_plot_style(fig_monthly_all, height=560)

            fig_seasonal = px.line(
                seasonal_df,
                markers=True,
                title=(
                    f"Ladakh Seasonal {selected_parameter} "
                    "(Winter: Oct–Mar, Spring: Apr–Jun, Summer: Jul–Sep)"
                ),
                labels={"HydroYear": "Hydrological year", "value": selected_parameter}
            )
            fig_seasonal = apply_plot_style(fig_seasonal, height=540)

            # ============================================================
            # Results tabs
            # ============================================================

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "📈 Time Series",
                    "📅 Monthly",
                    "🏔️ Seasonal",
                    "📆 Hydro-Year",
                    "📋 Statistics",
                    "⬇️ Downloads"
                ]
            )

            with tab1:
                st.markdown("### Original / Input Time Series")
                st.plotly_chart(fig_original, use_container_width=True)

                st.markdown("### Mann–Kendall Trend Test: Original Series")
                st.dataframe(daily_mk_df, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("### Monthly Average Time Series")
                st.plotly_chart(fig_monthly, use_container_width=True)

                st.markdown("### Monthly Trend by Calendar Month")
                st.plotly_chart(fig_monthly_all, use_container_width=True)

                st.markdown("### Mann–Kendall Trend Test for Each Month")
                st.dataframe(monthly_mk_df, use_container_width=True, hide_index=True)

            with tab3:
                st.markdown("### Ladakh Seasonal Trend")
                st.plotly_chart(fig_seasonal, use_container_width=True)

                st.markdown("### Mann–Kendall Trend Test for Ladakh Seasons")
                st.dataframe(seasonal_mk_df, use_container_width=True, hide_index=True)

            with tab4:
                st.markdown("### Hydrological-Year Trend")
                st.plotly_chart(fig_yearly, use_container_width=True)

                st.markdown("### Mann–Kendall Trend Test: Hydrological-Year Series")
                st.dataframe(yearly_mk_df, use_container_width=True, hide_index=True)

            with tab5:
                st.markdown("### Descriptive Statistics")
                stats_df = df[selected_parameter].describe().to_frame(name=selected_parameter)
                st.dataframe(stats_df, use_container_width=True)

                st.markdown("### Processed Dataset")
                st.dataframe(df, use_container_width=True)

            with tab6:
                st.markdown("### Download Results")

                cleaned_download_df = df.copy()
                cleaned_download_df.index.name = "Date"

                st.download_button(
                    label="⬇️ Download processed dataset",
                    data=cleaned_download_df.to_csv().encode("utf-8"),
                    file_name=f"processed_{selected_parameter}_hydro_year.csv",
                    mime="text/csv"
                )

                st.download_button(
                    label="⬇️ Download monthly MK table",
                    data=monthly_mk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"monthly_mk_{selected_parameter}.csv",
                    mime="text/csv"
                )

                st.download_button(
                    label="⬇️ Download seasonal MK table",
                    data=seasonal_mk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"seasonal_mk_{selected_parameter}.csv",
                    mime="text/csv"
                )

                combined_summary = {
                    "Original MK": daily_mk_df,
                    "Hydrological Year MK": yearly_mk_df,
                    "Monthly MK": monthly_mk_df,
                    "Seasonal MK": seasonal_mk_df
                }

                # Excel export
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    cleaned_download_df.to_excel(writer, sheet_name="Processed_Data")
                    daily_mk_df.to_excel(writer, sheet_name="Original_MK", index=False)
                    yearly_mk_df.to_excel(writer, sheet_name="HydroYear_MK", index=False)
                    monthly_mk_df.to_excel(writer, sheet_name="Monthly_MK", index=False)
                    seasonal_mk_df.to_excel(writer, sheet_name="Seasonal_MK", index=False)
                    stats_df.to_excel(writer, sheet_name="Statistics")

                st.download_button(
                    label="⬇️ Download complete Excel report",
                    data=excel_buffer.getvalue(),
                    file_name=f"trend_analysis_report_{selected_parameter}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error("❌ Something went wrong while processing the file.")
        st.exception(e)
