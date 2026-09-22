"""FPL Value Model dashboard.

Reads only ``data/predictions.parquet`` -- the committed artefact refreshed
by the daily GitHub Actions job (see ``.github/workflows/daily.yml``). No
network calls, no model loading, so this runs the same way locally and
deployed.

Invoke with ``streamlit run app/streamlit_app.py`` (or ``make app``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PREDICTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.parquet"

_STATUS_LABELS: dict[str, str] = {
    "a": "Available",
    "d": "Doubtful",
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
}


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    """Load the predictions parquet, cached for the session."""
    frame = pd.read_parquet(path)
    frame["status_label"] = frame["status"].map(_STATUS_LABELS).fillna("Unknown")
    return frame


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render the filter controls and return the filtered frame."""
    st.sidebar.header("Filters")

    positions = sorted(df["position"].unique())
    picked_positions = st.sidebar.multiselect("Position", positions, default=positions)

    price_min, price_max = float(df["price_m"].min()), float(df["price_m"].max())
    picked_price = st.sidebar.slider(
        "Price (GBP m)", price_min, price_max, (price_min, price_max), step=0.1
    )

    available_only = st.sidebar.checkbox("Available only", value=True)
    trusted_only = st.sidebar.checkbox(
        "Own history only",
        value=True,
        help="Hide players priced on a position average (new signings, "
        "promoted-team players, academy graduates) -- speculative, not a "
        "real recommendation.",
    )
    search = st.sidebar.text_input("Search by name")

    out = df[
        df["position"].isin(picked_positions) & df["price_m"].between(*picked_price)
    ]
    if available_only:
        out = out[out["available"]]
    if trusted_only:
        out = out[out["has_history"]]
    if search:
        out = out[out["name"].str.contains(search, case=False, na=False)]
    return out


def render_summary(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    """Render the top-line metric row."""
    cols = st.columns(4)
    cols[0].metric("Players shown", len(filtered))
    cols[1].metric("Total pool", len(df))
    if len(filtered):
        cols[2].metric("Most underpriced", f"{filtered['value_m'].max():.2f}m")
        cols[3].metric("Most overpriced", f"{filtered['value_m'].min():.2f}m")


def render_table(filtered: pd.DataFrame) -> None:
    """Render the sortable player table."""
    display_cols = {
        "name": "Player",
        "team_name": "Team",
        "position": "Pos",
        "price_m": "Price (m)",
        "pred_m": "Predicted (m)",
        "value_m": "Value",
        "points_per_m": "Pts/m",
        "total_points": "Points",
        "status_label": "Status",
        "has_history": "Own history",
    }
    table = filtered[list(display_cols)].rename(columns=display_cols)
    st.dataframe(
        table.sort_values("Value", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price (m)": st.column_config.NumberColumn(format="£%.1fm"),
            "Predicted (m)": st.column_config.NumberColumn(format="£%.2fm"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Pts/m": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_scatter(filtered: pd.DataFrame) -> None:
    """Render predicted vs. actual price, coloured by position."""
    st.subheader("Predicted vs. actual price")
    st.caption(
        "Points above the diagonal are players the model prices higher "
        "than the market: underpriced picks."
    )
    chart_df = filtered.rename(
        columns={"price_m": "Actual price (m)", "pred_m": "Predicted price (m)"}
    )
    st.scatter_chart(
        chart_df,
        x="Actual price (m)",
        y="Predicted price (m)",
        color="position",
    )


def main() -> None:
    """Render the dashboard page."""
    st.set_page_config(page_title="FPL Value Model", layout="wide")
    st.title("FPL Value Model")
    st.caption(
        "Predicted fair price vs. actual FPL price. Positive value = the "
        "model thinks the market is underpricing this player."
    )

    if not PREDICTIONS_PATH.exists():
        st.error(
            f"No predictions found at `{PREDICTIONS_PATH}`. Run "
            "`poetry run python -m fpl_value_model.pipeline --refresh` first."
        )
        return

    df = load_predictions(PREDICTIONS_PATH)
    filtered = render_sidebar(df)

    render_summary(df, filtered)
    st.divider()
    render_table(filtered)
    st.divider()
    render_scatter(filtered)


if __name__ == "__main__":
    main()
