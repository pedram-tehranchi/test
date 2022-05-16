import altair as alt

def get_chart(data):
    hover = alt.selection_single(
        fields=["hour_time"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Ontario Demand Prediction")
        .mark_line()
        .encode(
            x="hour_time",
            # x=alt.X('time:T', axis=alt.Axis(format='%H:%M')),
            y="demand:Q",
            color="symbol",
            strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="hour_time",
            y="demand:Q",
            color="symbol",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("demand", title="Demand (MW)"),
                alt.Tooltip("symbol", title="Source Name"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()