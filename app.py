import calendar
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pymannkendall as mk
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hydrological & Climate Trend Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS - CLEAN BLUE-WHITE UI WITH READABLE SIDEBAR
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eaf8ff 0%, #ffffff 45%, #dff3ff 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f5ff 0%, #ffffff 100%) !important;
        border-right: 1px solid rgba(0, 91, 150, 0.18);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #0b3558 !important;
    }

    .hero-box {
        background: linear-gradient(135deg, #004e92 0%, #001b4d 100%);
        padding: 36px;
        border-radius: 26px;
        color: #ffffff !important;
        box-shadow: 0 14px 36px rgba(0, 70, 130, 0.32);
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 12px;
        color: #ffffff !important;
    }

    .hero-subtitle {
        font-size: 18px;
        line-height: 1.7;
        color: #eaf6ff !important;
        margin-bottom: 18px;
    }

    .developer-text {
        margin-top: 20px;
        font-size: 15px;
        color: #dbeeff !important;
    }

    .badge-pill {
        display: inline-block;
        padding: 9px 15px;
        border-radius: 999px;
        background: #dff3ff;
        color: #004e92 !important;
        font-weight: 750;
        margin: 5px 6px 5px 0;
        border: 1px solid rgba(0, 78, 146, 0.18);
    }

    .custom-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(0, 91, 150, 0.13);
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 80, 130, 0.10);
        margin-bottom: 20px;
    }

    .section-title {
        color: #005a9c !important;
        font-weight: 850;
        font-size: 26px;
        margin-bottom: 8px;
    }

    .small-muted {
        color: #526d82 !important;
        font-size: 15px;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(0, 91, 150, 0.14);
        box-shadow: 0 7px 20px rgba(0, 80, 130, 0.08);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #003f6b !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #0077b6, #023e8a) !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 800;
        color: #004e92;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

IGNORE_COLUMN_KEYWORDS = ("trend", "trendline")


def get_ladakh_season(month: int) -> str:
    """
    Ladakh seasons:
    Winter: Oct-Mar
    Spring: Apr-Jun
    Summer: Jul-Sep
    """
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6]:
        return "Spring"
    else:
        return "Summer"


def hydro_year_start(date_value) -> int:
    """
    Hydrological year: October to September.
    Label is based on start year.

    Example:
    2000-10 to 2001-09 = Hydrological year 2000.
    """
    return date_value.year if date_value.month >= 10 else date_value.year - 1


def hydro_year_label(year: int) -> str:
    return f"{year}–{year + 1}"


def robust_parse_dates(series: pd.Series) -> pd.Series:
    """
    Handles dates such as:
    1979-01
    1979-01-01
    01-01-1979
    01/01/1979
    """
    parsed = pd.to_datetime(series, errors="coerce")

    if parsed.notna().sum() < max(1, int(0.5 * len(series))):
        parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed_dayfirst.notna().sum() > parsed.notna().sum():
            parsed = parsed_dayfirst

    try:
        parsed_mixed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed_mixed.notna().sum() > parsed.notna().sum():
            parsed = parsed_mixed
    except Exception:
        pass

    return parsed


def detect_date_column(df: pd.DataFrame):
    for col in df.columns:
        if str(col).strip().lower() == "date":
            return col

    for col in df.columns:
        if "date" in str(col).strip().lower():
            return col

    return None


def is_ignored_column(col_name: str) -> bool:
    col_lower = str(col_name).strip().lower()
    return any(keyword in col_lower for keyword in IGNORE_COLUMN_KEYWORDS)


def detect_numeric_parameter_columns(df: pd.DataFrame, date_col: str):
    """
    Detect numeric parameter columns.
    Trendline / trend columns are ignored completely.
    """
    parameter_cols = []

    for col in df.columns:
        if col == date_col:
            continue

        if is_ignored_column(col):
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")

        if numeric_series.notna().sum() >= 3:
            parameter_cols.append(col)

    return parameter_cols


def safe_mk_test(values):
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
    return pd.DataFrame(list(trend_dict.items()), columns=["Metric", "Value"])


def apply_plot_style(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title={
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "family": "Arial", "color": "#004e92"},
        },
        margin=dict(l=35, r=35, t=85, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0.98)",
        paper_bgcolor="rgba(255,255,255,0)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 78, 146, 0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 78, 146, 0.08)",
        zeroline=False,
    )

    return fig


def prepare_analysis(
    df: pd.DataFrame,
    date_col: str,
    parameter_col: str,
    start_year: int,
    end_year: int,
):
    working_df = df.copy()

    # Keep Date + selected parameter only.
    # This ensures Trendline is not used.
    working_df = working_df[[date_col, parameter_col]].copy()

    working_df[date_col] = robust_parse_dates(working_df[date_col])
    working_df = working_df.dropna(subset=[date_col])

    if working_df.empty:
        raise ValueError("No valid dates found after parsing.")

    working_df = working_df.sort_values(date_col)
    working_df = working_df.set_index(date_col)

    working_df[parameter_col] = pd.to_numeric(working_df[parameter_col], errors="coerce")
    working_df = working_df.dropna(subset=[parameter_col])

    if working_df.empty:
        raise ValueError(f"No valid numeric values found for selected parameter: {parameter_col}")

    working_df["HydroYear"] = working_df.index.to_series().apply(hydro_year_start)
    working_df["Season"] = working_df.index.month.map(get_ladakh_season)

    working_df = working_df[
        (working_df["HydroYear"] >= start_year)
        & (working_df["HydroYear"] <= end_year)
    ]

    if working_df.empty:
        raise ValueError("No data available for the selected hydrological year range.")

    original_series = working_df[parameter_col].copy()

    # Use MS instead of M to avoid pandas freq='M' error in hosted environments.
    monthly_series = original_series.resample("MS").mean().dropna()

    yearly_series = working_df.groupby("HydroYear")[parameter_col].mean()

    seasonal_df = (
        working_df.groupby(["HydroYear", "Season"])[parameter_col]
        .mean()
        .unstack()
    )

    season_order = ["Winter", "Spring", "Summer"]
    seasonal_df = seasonal_df.reindex(
        columns=[s for s in season_order if s in seasonal_df.columns]
    )

    monthly_by_month = (
        working_df.groupby(["HydroYear", working_df.index.month])[parameter_col]
        .mean()
        .unstack()
    )

    if not monthly_by_month.empty:
        monthly_by_month.columns = [
            calendar.month_name[int(m)] for m in monthly_by_month.columns
        ]

        hydro_month_order = [
            "October",
            "November",
            "December",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
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
        "monthly_by_month": monthly_by_month,
    }


def build_monthly_mk_table(original_series: pd.Series) -> pd.DataFrame:
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
    rows = []

    for season in seasonal_df.columns:
        season_series = seasonal_df[season].dropna()
        result = safe_mk_test(season_series.values)
        row = {"Season": season, "Records": len(season_series)}
        row.update(result)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🌊 Hydrological & Climate Data Trend Analysis</div>
        <div class="hero-subtitle">
            Interactive Streamlit web application for hydrological-year based trend analysis using
            Mann–Kendall statistics, Ladakh seasonal aggregation and Plotly visual analytics.
        </div>
        <div>
            <span class="badge-pill">Hydrological year: Oct–Sep</span>
            <span class="badge-pill">Ladakh seasons</span>
            <span class="badge-pill">Mann–Kendall trend test</span>
            <span class="badge-pill">Trendline column ignored</span>
        </div>
        <div class="developer-text">
            Developed by <b>Dr. Sachchidanand Singh</b>, Scientist,
            Western Himalayan Regional Centre, Jammu
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain one Date column and at least one numeric parameter column.",
    )

    st.markdown("---")

    st.markdown("## 📌 Required Format")
    st.markdown(
        """
        Your CSV should contain:

        - One `date` / `Date` column  
        - One or more numeric data columns  
        - Columns such as `Trendline` will be ignored automatically  
        """
    )

    # No pd.date_range used here to avoid hosted pandas freq error.
    sample_df = pd.DataFrame(
        {
            "date": [
                "2000-10",
                "2000-11",
                "2000-12",
                "2001-01",
                "2001-02",
                "2001-03",
                "2001-04",
                "2001-05",
                "2001-06",
                "2001-07",
                "2001-08",
                "2001-09",
            ],
            "Data": [12, 18, 22, 35, 40, 55, 20, 12, 8, 4, 3, 10],
            "Trendline": [np.nan] * 12,
        }
    )

    st.download_button(
        label="⬇️ Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_hydro_climate_data.csv",
        mime="text/csv",
    )

    st.markdown("---")

    with st.expander("📖 App documentation"):
        st.markdown(
            """
            This app performs climate and hydrological trend analysis.

            **Workflow**
            1. Upload CSV data.
            2. Select the parameter column.
            3. Select hydrological year range.
            4. Run trend analysis.
            5. View charts and Mann–Kendall statistics.

            **Hydrological year**
            October to September.

            **Ladakh seasons**
            - Winter: October to March
            - Spring: April to June
            - Summer: July to September

            **Important**
            Columns containing the word `trend` or `trendline`
            are ignored automatically.
            """
        )


# ============================================================
# MAIN PAGE BEFORE UPLOAD
# ============================================================

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">📤 Upload</div>
                <div class="small-muted">
                    Upload a CSV file containing a date column and hydrological or climate variables.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">⚙️ Configure</div>
                <div class="small-muted">
                    Select the parameter and hydrological year range for analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">📈 Analyze</div>
                <div class="small-muted">
                    Generate Plotly charts and Mann–Kendall trend statistics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("Please upload a CSV file from the sidebar to begin analysis.")

else:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">✅ Dataset Uploaded Successfully</div>
                <div class="small-muted">
                    The file has been read successfully. Trendline/helper columns will be ignored.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        date_col = detect_date_column(raw_df)

        if date_col is None:
            st.error("No Date column found. Please ensure your CSV contains a column named date or Date.")
            st.stop()

        parsed_dates = robust_parse_dates(raw_df[date_col])
        valid_dates_mask = parsed_dates.notna()

        if valid_dates_mask.sum() == 0:
            st.error("Date column was detected, but no valid dates could be parsed.")
            st.stop()

        temp_df = raw_df.loc[valid_dates_mask].copy()
        temp_df[date_col] = parsed_dates.loc[valid_dates_mask]
        temp_df = temp_df.sort_values(date_col).set_index(date_col)

        parameter_options = detect_numeric_parameter_columns(raw_df, date_col)

        if len(parameter_options) == 0:
            st.error(
                "No valid numeric parameter column was found. "
                "Trendline columns are ignored. Please include at least one numeric data column."
            )
            st.stop()

        hydro_years = sorted(temp_df.index.to_series().apply(hydro_year_start).unique())

        if len(hydro_years) == 0:
            st.error("No hydrological years could be detected.")
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows in file", f"{len(raw_df):,}")
        m2.metric("Valid dates", f"{valid_dates_mask.sum():,}")
        m3.metric("Usable parameters", len(parameter_options))
        m4.metric("Hydrological years", len(hydro_years))

        with st.expander("🔍 Preview uploaded CSV", expanded=False):
            st.dataframe(raw_df.head(50), use_container_width=True)

        ignored_cols = [
            c for c in raw_df.columns
            if c != date_col and is_ignored_column(c)
        ]

        if ignored_cols:
            st.info(f"Ignored helper/trend column(s): {', '.join(ignored_cols)}")

        st.markdown(
            """
            <div class="custom-card">
                <div class="section-title">⚙️ Analysis Settings</div>
                <div class="small-muted">
                    Select the column to process and the hydrological year period.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1.4, 1, 1])

        with c1:
            selected_parameter = st.selectbox(
                "Select parameter to analyze",
                parameter_options,
                help="Trendline columns are excluded automatically.",
            )

        with c2:
            start_year = st.selectbox(
                "Start hydrological year",
                hydro_years,
                index=0,
                format_func=hydro_year_label,
            )

        with c3:
            end_year_candidates = [y for y in hydro_years if y >= start_year]
            end_year = st.selectbox(
                "End hydrological year",
                end_year_candidates,
                index=len(end_year_candidates) - 1,
                format_func=hydro_year_label,
            )

        run_analysis = st.button(
            "🚀 Run Hydrological Trend Analysis",
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            results = prepare_analysis(
                df=raw_df,
                date_col=date_col,
                parameter_col=selected_parameter,
                start_year=start_year,
                end_year=end_year,
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
                f"Analysis completed for {selected_parameter} from "
                f"{hydro_year_label(hydro_year_min)} to {hydro_year_label(hydro_year_max)}."
            )

            st.markdown("## 📊 Summary")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Parameter", selected_parameter)
            s2.metric("Mean", f"{df[selected_parameter].mean():.3f}")
            s3.metric("Minimum", f"{df[selected_parameter].min():.3f}")
            s4.metric("Maximum", f"{df[selected_parameter].max():.3f}")
            s5.metric("Records used", f"{len(df):,}")

            daily_mk_df = trend_dict_to_df(safe_mk_test(original_series.values))
            yearly_mk_df = trend_dict_to_df(safe_mk_test(yearly_series.values))
            monthly_mk_df = build_monthly_mk_table(original_series)
            seasonal_mk_df = build_seasonal_mk_table(seasonal_df)

            fig_original = px.line(
                x=original_series.index,
                y=original_series.values,
                title=f"Original Time Series of {selected_parameter}",
                labels={"x": "Date", "y": selected_parameter},
            )
            fig_original = apply_plot_style(fig_original)

            fig_monthly = px.line(
                x=monthly_series.index,
                y=monthly_series.values,
                title=f"Monthly Average {selected_parameter}",
                labels={"x": "Date", "y": selected_parameter},
            )
            fig_monthly = apply_plot_style(fig_monthly)

            fig_yearly = px.line(
                x=[hydro_year_label(int(y)) for y in yearly_series.index],
                y=yearly_series.values,
                markers=True,
                title=f"Hydrological-Year Average {selected_parameter} (Oct–Sep)",
                labels={"x": "Hydrological year", "y": selected_parameter},
            )
            fig_yearly = apply_plot_style(fig_yearly)

            fig_monthly_all = px.line(
                monthly_by_month,
                markers=True,
                title=f"Monthly Trend by Calendar Month ({selected_parameter})",
                labels={"value": selected_parameter, "HydroYear": "Hydrological year"},
            )
            fig_monthly_all = apply_plot_style(fig_monthly_all, height=570)

            fig_seasonal = px.line(
                seasonal_df,
                markers=True,
                title=(
                    f"Ladakh Seasonal {selected_parameter} "
                    "(Winter: Oct–Mar, Spring: Apr–Jun, Summer: Jul–Sep)"
                ),
                labels={"value": selected_parameter, "HydroYear": "Hydrological year"},
            )
            fig_seasonal = apply_plot_style(fig_seasonal, height=550)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "📈 Time Series",
                    "📅 Monthly",
                    "🏔️ Seasonal",
                    "📆 Hydro-Year",
                    "📋 Statistics",
                    "⬇️ Downloads",
                ]
            )

            with tab1:
                st.subheader("Original / Input Time Series")
                st.plotly_chart(fig_original, use_container_width=True)

                st.subheader("Mann–Kendall Trend Test: Original Series")
                st.dataframe(daily_mk_df, use_container_width=True, hide_index=True)

            with tab2:
                st.subheader("Monthly Average Time Series")
                st.plotly_chart(fig_monthly, use_container_width=True)

                st.subheader("Monthly Trend by Calendar Month")
                st.plotly_chart(fig_monthly_all, use_container_width=True)

                st.subheader("Mann–Kendall Trend Test for Each Month")
                st.dataframe(monthly_mk_df, use_container_width=True, hide_index=True)

            with tab3:
                st.subheader("Ladakh Seasonal Trend")
                st.plotly_chart(fig_seasonal, use_container_width=True)

                st.subheader("Mann–Kendall Trend Test for Ladakh Seasons")
                st.dataframe(seasonal_mk_df, use_container_width=True, hide_index=True)

            with tab4:
                st.subheader("Hydrological-Year Trend")
                st.plotly_chart(fig_yearly, use_container_width=True)

                st.subheader("Mann–Kendall Trend Test: Hydrological-Year Series")
                st.dataframe(yearly_mk_df, use_container_width=True, hide_index=True)

            with tab5:
                st.subheader("Descriptive Statistics")
                stats_df = df[selected_parameter].describe().to_frame(name=selected_parameter)
                st.dataframe(stats_df, use_container_width=True)

                st.subheader("Processed Dataset")
                st.dataframe(df, use_container_width=True)

            with tab6:
                st.subheader("Download Results")

                cleaned_download_df = df.copy()
                cleaned_download_df.index.name = "Date"

                st.download_button(
                    label="⬇️ Download processed dataset",
                    data=cleaned_download_df.to_csv().encode("utf-8"),
                    file_name=f"processed_{selected_parameter}_hydro_year.csv",
                    mime="text/csv",
                )

                st.download_button(
                    label="⬇️ Download monthly MK table",
                    data=monthly_mk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"monthly_mk_{selected_parameter}.csv",
                    mime="text/csv",
                )

                st.download_button(
                    label="⬇️ Download seasonal MK table",
                    data=seasonal_mk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"seasonal_mk_{selected_parameter}.csv",
                    mime="text/csv",
                )

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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    except Exception as e:
        st.error("Something went wrong while processing the file.")
        st.exception(e)
