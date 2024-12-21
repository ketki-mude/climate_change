from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

### STREAMLIT CONFIG ###
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .main {
        padding: 20px 4px 4px 4px !important;
    }
    .block-container {
        padding: 20px 8px 8px 8px !important;
    }
    </style>
    """, unsafe_allow_html=True
)
### STREAMLIT CONFIG ###

### INIT ###
ELECTRICITY_DATA_FILE_PATH = './data/elec-fossil-nuclear-renewables.csv'
GHG_DATA_FILE_PATH = './data/total-ghg-emissions.csv'
TEMP_DATA_FILE_PATH = './data/contribution-temp-rise-degrees.csv'

ENTITY = "entity"
YEAR = "year"
CODE = "code"

ELECTRICITY_ENTITY = 'electricity_entity'
ELECTRICITY_CODE = 'electricity_code'
ELECTRICITY_YEAR = 'electricity_year'
ELECTRICITY_RENEWABLE = 'electricity_renewable'
ELECTRICITY_NUCLEAR = 'electricity_nuclear'
ELECTRICITY_FOSSIL = 'electricity_fossil'

GHG_ENTITY = 'ghg_entity'
GHG_CODE = 'ghg_code'
GHG_YEAR = 'ghg_year'
GHG_EMISSION = 'ghg_emission'

TEMP_ENTITY = 'temp_entity'
TEMP_CODE = 'temp_code'
TEMP_YEAR = 'temp_year'
TEMP_CHANGE = 'temp_change'

elecricity_columns = {
    'Entity': ENTITY,
    'Code': CODE,
    'Year': YEAR,
    'Electricity from renewables - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)': ELECTRICITY_RENEWABLE,
    'Electricity from nuclear - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)': ELECTRICITY_NUCLEAR,
    'Electricity from fossil fuels - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)': ELECTRICITY_FOSSIL
}
ghg_columns = {
    'Entity': ENTITY,
    'Code': CODE,
    'Year': YEAR,
    'Annual greenhouse gas emissions in CO₂ equivalents': GHG_EMISSION
}

temp_change_columns = {
    'Entity': ENTITY,
    'Code': CODE,
    'Year': YEAR,
    'Change in global mean surface temperature caused by greenhouse gas emissions': TEMP_CHANGE
}

electricity_org_df = pd.read_csv(ELECTRICITY_DATA_FILE_PATH)
ghg_org_df = pd.read_csv(GHG_DATA_FILE_PATH)
temp_org_df = pd.read_csv(TEMP_DATA_FILE_PATH)

PARENT_HEADER = "Electricity Generation vs Harmful Effects on Earth"


### INIT END ###

###UTILS###
def show_header(text):
    centered_header = f'<div style="text-align: center;"><h2>{text}</h2></div>'
    st.markdown(centered_header, unsafe_allow_html=True)


def show_visualization(heading, fig):
    st.markdown(heading, unsafe_allow_html=True)
    st.pyplot(fig)


def show_slider(text, start, end, current):
    return st.slider(text, min=start, max=end, value=current)


###UTILS END###

###DATA SCIENCE###
def preprocess_data():
    # Step 3: Rename columns in Elect_fossil_nuclear_renew_df
    electricity_org_df.rename(columns=elecricity_columns, inplace=True)

    ghg_org_df.rename(columns=ghg_columns, inplace=True)

    # Step 5: Rename columns in annual_temp_anomalies_df
    temp_org_df.rename(columns=temp_change_columns, inplace=True)

    # Clean and standardize
    for df in [electricity_org_df, ghg_org_df, temp_org_df]:
        df[ENTITY] = df[ENTITY].str.strip().str.capitalize()

    # Step 7: Merge all and green_house_gas_emission_df
    datasets = [electricity_org_df, ghg_org_df,
                temp_org_df]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=[ENTITY, CODE, YEAR], how='inner'), datasets)
    return merged_df


def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = data.copy()
    columns_to_normalize = [ELECTRICITY_RENEWABLE, ELECTRICITY_FOSSIL, ELECTRICITY_NUCLEAR, GHG_EMISSION,
                            TEMP_CHANGE]
    normalized_data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return normalized_data


def filter_data(data, year_range, selected_countries):
    # Filter the data based on year range and selected countries
    filtered = data[
        (data[YEAR] >= year_range[0]) &
        (data[YEAR] <= year_range[1]) &
        (data[ENTITY].isin(selected_countries))
        ]

    # Maintain the order of selected_countries
    filtered[ENTITY] = pd.Categorical(filtered[ENTITY], categories=selected_countries, ordered=True)

    # Sort by the ordered categories to reflect the selected_countries order
    filtered = filtered.sort_values(by=ENTITY)

    return filtered


def show_graph_set(data):
    st.markdown(
        f"These graphs displays the electricity production by source (renewables, fossil fuels, and nuclear energy) for selected countries ({', '.join(selected_countries).capitalize()})")

    # Layout for side-by-side graphs
    col1, col2, col3 = st.columns(3)

    # Electricity Generation Graphs
    with col1:
        st.markdown("### Energy Source Contribution to Electricity Generation (TWh)")
        energy_sources = [ELECTRICITY_RENEWABLE, ELECTRICITY_FOSSIL, ELECTRICITY_NUCLEAR]
        source_colors = {"Renewable": "green", "Fossil": "red", "Nuclear": "blue"}

        legend_mapping = {
            ELECTRICITY_RENEWABLE: 'Renewable',
            ELECTRICITY_NUCLEAR: 'Nuclear',
            ELECTRICITY_FOSSIL: 'Fossil'
        }

        for country in data[ENTITY].unique():
            country_data = data[data[ENTITY] == country]
            if not country_data.empty:
                melted_data = country_data.melt(
                    id_vars=[YEAR, ENTITY],
                    value_vars=energy_sources,
                    var_name="Energy Source",
                    value_name="Value"
                )
                melted_data["Energy Source"] = melted_data["Energy Source"].map(legend_mapping)
                fig = px.area(
                    melted_data,
                    x=YEAR,
                    y="Value",
                    color="Energy Source",
                    color_discrete_map=source_colors,
                    labels={
                        "Value": "Electricity (TWh)",
                        "year": "Year",
                        "entity": "Country",
                        "Energy Source": "Energy Source"
                    },
                    title=f"{country.capitalize()} - Electricity Production",
                    hover_data={YEAR: True, "Value": True, "Energy Source": True}
                )
                fig.update_layout(legend=dict(
                    orientation="h",
                    entrywidth=50,
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="right",
                    x=1
                ))
                st.plotly_chart(fig, use_container_width=True)

    # GHG Emissions Graphs
    with col2:
        st.markdown("### Impact of Energy Production on GHG Emissions")
        for country in data[ENTITY].unique():
            country_data = data[data[ENTITY] == country]
            if not country_data.empty:
                fig = px.area(
                    country_data,
                    x=YEAR,
                    y=GHG_EMISSION,
                    color_discrete_sequence=["purple"],
                    labels={
                        GHG_EMISSION: "GHG Emissions (MtCO₂)",
                        YEAR: "Year",
                        ENTITY: "Country"
                    },
                    title=f"{country.capitalize()} - Greenhouse Gas Emissions",
                    hover_data={YEAR: True, "ghg_emission": True}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Temperature Anomalies Graphs
    with col3:
        st.markdown("### Temperature Changes Caused by GHG Emissions")
        for country in data[ENTITY].unique():
            country_data = data[data[ENTITY] == country]
            if not country_data.empty:
                fig = px.area(
                    country_data,
                    x=YEAR,
                    y=TEMP_CHANGE,
                    color_discrete_sequence=["orange"],
                    labels={
                        TEMP_CHANGE: "Temperature Change (°C)",
                        YEAR: "Year",
                        ENTITY: "Country"
                    },
                    title=f"{country.capitalize()} - Temperature Change",
                    hover_data={YEAR: True, TEMP_CHANGE: True}
                )
                st.plotly_chart(fig, use_container_width=True)


def analyze_hypothesis(data):
    st.subheader("Comparative Visualizations of Energy Sources and Emissions")
    st.markdown(
        "This section examines the relationship between energy usage, greenhouse gas emissions, and temperature changes."
    )

    # Features and target variable for GHG emissions
    X = data[[ELECTRICITY_RENEWABLE, ELECTRICITY_FOSSIL, ELECTRICITY_NUCLEAR]]
    y_ghg = data[GHG_EMISSION]

    # Train polynomial regression model for GHG emissions
    poly_ghg_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_ghg_model.fit(X, y_ghg)
    y_ghg_pred = poly_ghg_model.predict(X)

    # Metrics for GHG emissions model
    st.write("### Overall GHG Emissions Model Metrics")
    overall_r2 = poly_ghg_model.score(X, y_ghg)
    st.write(f"Overall R² Score: {overall_r2:.2f}")

    # R² scores for individual energy sources
    # energy_sources = {
    #     "Renewable Energy": ELECTRICITY_RENEWABLE,
    #     "Fossil Energy": ELECTRICITY_FOSSIL,
    #     "Nuclear Energy": ELECTRICITY_NUCLEAR
    # }

    # st.write("### R² Scores for Each Energy Source")
    # for source_name, feature in energy_sources.items():
    #     # Train polynomial regression for each source individually
    #     X_single = data[[feature]]
    #     poly_model_single = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    #     poly_model_single.fit(X_single, y_ghg)
    #     r2_single = poly_model_single.score(X_single, y_ghg)
    #     st.write(f"{source_name}: R² Score: {r2_single:.2f}")

    # Display coefficients
    # poly_features = poly_ghg_model.named_steps['polynomialfeatures'].get_feature_names_out(X.columns)
    # coefficients = poly_ghg_model.named_steps['linearregression'].coef_
    # st.write("### GHG Emissions Model Coefficients")
    # st.table(pd.DataFrame({'Feature': poly_features, 'Coefficient': coefficients}))

    fossil_col, renewable_col, nuclear_col = st.columns(3)

    # Visualization: Fossil Fuels vs GHG Emissions
    with fossil_col:
        fig_fossil = px.scatter(
            data,
            x=ELECTRICITY_FOSSIL,
            y=GHG_EMISSION,
            trendline="ols",
            labels={ELECTRICITY_FOSSIL: "Fossil Fuels (TWh)", GHG_EMISSION: "GHG Emissions (MtCO₂)"},
            title="Fossil Fuels and Their Impact on GHG Emissions"
        )
        st.plotly_chart(fig_fossil, use_container_width=True)

    # Visualization: Renewable Energy vs GHG Emissions
    with renewable_col:
        fig_renewable = px.scatter(
            data,
            x=ELECTRICITY_RENEWABLE,
            y=GHG_EMISSION,
            trendline="ols",
            labels={ELECTRICITY_RENEWABLE: "Renewable Energy (TWh)", GHG_EMISSION: "GHG Emissions (MtCO₂)"},
            title="Renewable Energy and Their Impact on GHG Emissions"
        )
        st.plotly_chart(fig_renewable, use_container_width=True)

    # Visualization: Nuclear Energy vs GHG Emissions
    with nuclear_col:
        fig_nuclear = px.scatter(
            data,
            x=ELECTRICITY_NUCLEAR,
            y=GHG_EMISSION,
            trendline="ols",
            labels={ELECTRICITY_NUCLEAR: "Nuclear Energy (TWh)", GHG_EMISSION: "GHG Emissions (MtCO₂)"},
            title="Nuclear Energy  and Their Impact on GHG Emissions"
        )
        st.plotly_chart(fig_nuclear, use_container_width=True)

    st.subheader("Analysis of GHG Emissions vs Temperature Rise")

    # Features and target variable
    X = data[[GHG_EMISSION]]
    y = data[TEMP_CHANGE]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train polynomial regression model
    degree = 2  # You can change this to a higher degree if needed
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Metrics")
    st.write(f"R² Score: {r2:.2f}")

    # Coefficients
    # coefficients = model.named_steps['linearregression'].coef_
    # poly_features = model.named_steps['polynomialfeatures'].get_feature_names_out()

    # st.write("### Model Coefficients")
    # coef_df = pd.DataFrame({
    #     "Feature": poly_features,
    #     "Coefficient": coefficients
    # })
    # st.table(coef_df)

    # Visualization: GHG Emissions vs Temperature Rise
    # Generate predictions for the full range of data
    X_range = pd.DataFrame(np.linspace(X.min(), X.max(), 100), columns=[GHG_EMISSION])
    y_range_pred = model.predict(X_range)

    # Plot actual data and the polynomial regression line
    fig = px.scatter(
        data,
        x=GHG_EMISSION,
        y=TEMP_CHANGE,
        labels={GHG_EMISSION: "GHG Emissions (MtCO₂)", TEMP_CHANGE: "Temperature Rise (°C)"},
        title="GHG Emissions and Temperature Rise: A Critical Link"
    )
    fig.add_scatter(x=X_range[GHG_EMISSION], y=y_range_pred.flatten(), mode='lines', name='Polynomial Fit')
    st.plotly_chart(fig, use_container_width=True)


def train_polynomial_model(X, y, degree=3):
    """Train a polynomial regression model with the specified degree."""
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    return model


def predict_ghg_and_temp(df):
    st.subheader("Interactive Predictions and Country-Specific")
    st.markdown("Use the input fields below to enter the year and country for prediction:")

    # Input for year and country
    country_predict_col, year_predict_col = st.columns(2)
    with country_predict_col:
        unique_countries = df[ENTITY].unique()
        default_country = "World" if "World" in unique_countries else unique_countries[0]
        country = st.selectbox("Select Country", unique_countries, index=list(unique_countries).index(default_country))
    with year_predict_col:
        year = st.number_input("Select Year for Prediction", min_value=int(df[YEAR].min()), step=1, value=2025)

    # Filter the dataset for the selected country
    country_data = df[df[ENTITY] == country]

    # Ensure we have enough data for training
    if len(country_data) < 5:
        st.error(f"Not enough data for {country} to make predictions.")
        return

    # Features and target variables
    X = country_data[[YEAR]]
    target_renewable = country_data[ELECTRICITY_RENEWABLE]
    target_fossil = country_data[ELECTRICITY_FOSSIL]
    target_nuclear = country_data[ELECTRICITY_NUCLEAR]
    target_ghg = country_data[GHG_EMISSION]
    target_temp = country_data[TEMP_CHANGE]

    # Train polynomial models
    model_renewable = train_polynomial_model(X, target_renewable)
    model_fossil = train_polynomial_model(X, target_fossil)
    model_nuclear = train_polynomial_model(X, target_nuclear)
    model_ghg = train_polynomial_model(X, target_ghg, degree=2)
    model_temp = train_polynomial_model(X, target_temp, degree=2)

    # Predict values for the selected year
    year_input = pd.DataFrame([[year]], columns=[YEAR])
    predicted_renewable = model_renewable.predict(year_input)[0]
    predicted_fossil = model_fossil.predict(year_input)[0]
    predicted_nuclear = model_nuclear.predict(year_input)[0]

    # Enforce non-decreasing trend for renewable energy
    predicted_renewable = max(predicted_renewable, target_renewable.max())

    # Calculate predicted GHG emissions considering renewable energy's negative effect on GHG
    predicted_ghg = model_ghg.predict(year_input)[0] - 0.2 * predicted_renewable  # Reduce GHG by renewable impact
    predicted_ghg = max(predicted_ghg, 0)  # Ensure GHG emissions are non-negative

    predicted_temp = model_temp.predict(year_input)[0]

    # Extract latest data for comparison
    latest_data = country_data[country_data[YEAR] == country_data[YEAR].max()]
    renewable_latest = latest_data[ELECTRICITY_RENEWABLE].values[0]
    fossil_latest = latest_data[ELECTRICITY_FOSSIL].values[0]
    nuclear_latest = latest_data[ELECTRICITY_NUCLEAR].values[0]
    ghg_latest = latest_data[GHG_EMISSION].values[0]
    temp_latest = latest_data[TEMP_CHANGE].values[0]

    # Display predictions and differences
    table_data = {
        "Metric": ["Renewable Energy (TWh)", "Fossil Energy (TWh)", "Nuclear Energy (TWh)", "GHG Emissions (MtCO₂)",
                   "Temperature Change (°C)"],
        "Latest Actual": [f"{renewable_latest:.2f}", f"{fossil_latest:.2f}", f"{nuclear_latest:.2f}",
                          f"{ghg_latest:.2f}", f"{temp_latest:.2f}"],
        f"{year} Predicted": [f"{predicted_renewable:.2f}", f"{predicted_fossil:.2f}", f"{predicted_nuclear:.2f}",
                              f"{predicted_ghg:.2f}", f"{predicted_temp:.2f}"],
        "Difference": [f"{predicted_renewable - renewable_latest:.2f}", f"{predicted_fossil - fossil_latest:.2f}",
                       f"{predicted_nuclear - nuclear_latest:.2f}", f"{predicted_ghg - ghg_latest:.2f}",
                       f"{predicted_temp - temp_latest:.2f}"]
    }

    st.markdown(f"### Data Comparison for {country}")
    st.table(table_data)

    st.subheader("Exploring Predicted Impacts: Emissions and Temperature Trends")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        fig_ghg = px.bar(
            x=["Latest GHG Emissions", f"{year} Predicted GHG Emissions"],
            y=[ghg_latest, predicted_ghg],
            labels={"x": "Year", "y": "GHG Emissions (MtCO₂)"},
            title=f"GHG Emissions difference for {country}",
            text=[f"{ghg_latest:.2f} MtCO₂", f"{predicted_ghg:.2f} MtCO₂"]
        )
        fig_ghg.update_traces(textposition="outside")
        st.plotly_chart(fig_ghg, use_container_width=True)

    with col2:
        fig_temp = px.bar(
            x=["Latest Temperature Change", f"{year} Predicted Temperature Change"],
            y=[temp_latest, predicted_temp],
            labels={"x": "Year", "y": "Temperature Change (°C)"},
            title=f"Temperature Change difference for {country}",
            text=[f"{temp_latest:.5f} °C", f"{predicted_temp:.5f} °C"]
        )
        fig_temp.update_traces(textposition="outside")
        st.plotly_chart(fig_temp, use_container_width=True)


def conclusion(df, selected_countries):
    st.markdown(
        """
        <style>
        .conclusion-header {
            font-size: 28px;
            font-weight: bold;
            color: #2C3E50; /* Dark Slate Blue */
            text-align: center;
            padding: 12px;
            border: 2px solid #7F8C8D; /* Muted Grey Border */
            border-radius: 8px;
            background-color: #F4F6F7; /* Soft Light Grey Background */
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); /* Subtle Shadow */
        }
        </style>
        <div class="conclusion-header">Conclusions Based on Current Trends</div>
        """,
        unsafe_allow_html=True)

    # Input for selecting the target range of years for prediction
    year_range = st.slider(
        "Select Year Range for Conclusion",
        min_value=int(df[YEAR].min()),
        max_value=2100,
        value=(2025, 2070)
    )

    start_year, end_year = year_range

    # Create columns for the selected countries
    cols = st.columns(len(selected_countries))

    # Iterate through each selected country and display results in separate columns
    for col, country in zip(cols, selected_countries):
        with col:
            st.write(f"### Predictions for {country}")

            # Filter data for the selected country
            country_data = df[df[ENTITY] == country]

            # Ensure we have enough data to train models
            if len(country_data) < 5:
                st.error(f"Not enough data for {country} to make conclusions.")
                continue

            # Prepare features and targets
            X = country_data[[YEAR]]
            renewable = country_data[ELECTRICITY_RENEWABLE]
            fossil = country_data[ELECTRICITY_FOSSIL]
            ghg = country_data[GHG_EMISSION]
            temp = country_data[TEMP_CHANGE]

            # Train polynomial regression models
            model_renewable = train_polynomial_model(X, renewable, degree=2)
            model_fossil = train_polynomial_model(X, fossil, degree=2)
            model_ghg = train_polynomial_model(X, ghg, degree=2)
            model_temp = train_polynomial_model(X, temp, degree=2)

            # Generate predictions for the range of years
            years = pd.DataFrame(np.arange(start_year, end_year + 1), columns=[YEAR])
            predicted_renewable = model_renewable.predict(years)
            predicted_fossil = model_fossil.predict(years)
            predicted_ghg = model_ghg.predict(years)
            predicted_temp = model_temp.predict(years)

            # Convert predictions to a DataFrame for visualization
            predictions_df = pd.DataFrame({
                YEAR: years[YEAR],
                "Renewable Energy (TWh)": predicted_renewable,
                "Fossil Energy (TWh)": predicted_fossil,
                "GHG Emissions (MtCO₂)": predicted_ghg,
                "Temperature Rise (°C)": predicted_temp
            })

            # Conclusion 1: Check when renewable energy surpasses fossil fuel usage
            surpass_year = None
            for year, ren, fos in zip(predictions_df[YEAR], predicted_renewable, predicted_fossil):
                if ren > fos:
                    surpass_year = year
                    break

            if surpass_year:
                st.write(f"✅ **Renewable energy in {country} will surpass fossil fuel usage by {surpass_year}.**")
            else:
                st.write(
                    f"❗ **Renewable energy in {country} will not surpass fossil fuel usage between {start_year} and {end_year}.**")

            # Conclusion 2: GHG Emission and Temperature Trends
            st.write(f"**By {end_year}:**")
            st.write(f"- Predicted GHG Emissions: {predicted_ghg[-1]:.2f} MtCO₂")
            st.write(f"- Predicted Temperature Rise: {predicted_temp[-1]:.2f} °C")

            # Visualization of Predictions
            st.write("#### Renewable vs Fossil Energy")
            fig_energy = px.line(
                predictions_df,
                x=YEAR,
                y=["Renewable Energy (TWh)", "Fossil Energy (TWh)"],
                labels={"value": "Electricity Generation (TWh)", "variable": "Energy Source"},
                title=""
            )
            st.plotly_chart(fig_energy, use_container_width=True)

            st.write("#### GHG Emissions and Temperature Rise")
            fig_ghg_temp = px.line(
                predictions_df,
                x=YEAR,
                y=["GHG Emissions (MtCO₂)", "Temperature Rise (°C)"],
                labels={"value": "Value", "variable": "Metric"},
                title=""
            )
            st.plotly_chart(fig_ghg_temp, use_container_width=True)


###DATA SCIENCE END###


### DRIVER ###
show_header(PARENT_HEADER)

st.markdown("""
## Datasets Used in this Analysis

1. **[Change in Global Mean Surface Temperature](https://ourworldindata.org/grapher/contribution-temp-rise-degrees):**  
   This dataset illustrates the impact of greenhouse gas emissions on global mean surface temperature changes over time.

2. **[Electricity Production from Fossil Fuels, Nuclear, and Renewables](https://ourworldindata.org/grapher/elec-fossil-nuclear-renewables?showSelectionOnlyInTable=1):**  
   A comprehensive dataset covering electricity production trends globally, broken down by fossil fuels, nuclear energy, and renewables.

3. **[Greenhouse Gas Emissions](https://ourworldindata.org/greenhouse-gas-emissions):**  
   This dataset provides insights into greenhouse gas emissions across various regions and sectors.

---
""")

merged_df = preprocess_data()

st.write("### Filter Data")

country_select_col, spacer, year_range_col = st.columns([1, 0.2, 3])

default_select_countries = ["World", "United states", "China"]

with country_select_col:
    countries = merged_df[ENTITY].unique()
    selected_countries = st.multiselect(
        "Select up to 3 Countries",
        options=countries,
        default=default_select_countries,
        max_selections=3
    )

with year_range_col:
    year_range = st.slider(
        "Select Year Range",
        min_value=int(merged_df[YEAR].min()),
        max_value=int(merged_df[YEAR].max()),
        value=(2000, int(merged_df[YEAR].max()))
    )

filtered_df = filter_data(merged_df, year_range, selected_countries)

# Show visualization
if not filtered_df.empty:
    normalized_df = normalize_data(filtered_df)
    show_graph_set(filtered_df)
    analyze_hypothesis(normalized_df)
    predict_ghg_and_temp(merged_df)
    conclusion(merged_df, selected_countries)
else:
    st.write("No data available for the selected filters.")
