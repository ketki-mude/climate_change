# **Electricity Generation vs. Harmful Effects on Earth**

## **Project Overview**

This Streamlit app allows users to analyze and visualize the relationship between electricity generation sources, greenhouse gas (GHG) emissions, and temperature changes over time. The app supports interactive filtering, visualizations, predictions, and conclusions based on current trends.

---

## **Setup Instructions**

### **1. Create a Virtual Environment**

```bash
    python3 -m venv .venv
```

### **2. Activate the Virtual Environment**

- **On macOS and Linux**:

```bash
    source .venv/bin/activate
```

- **On Windows**:

```bash
    .venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
    pip install -r requirements.txt
```

---

## **Run the Streamlit App**

To start the app, run the following command:

```bash
    streamlit run main.py
```

---

## **App Features**

### **Input Fields**

1. **Filter Data Section**

   - **Country Selector**:  
     **Type**: Multi-select  
     **Label**: Select up to 3 Countries  
     **Default**: World, United States, China  
     **Description**: Allows selection of up to 3 countries for analysis.  

   - **Year Range Slider**:  
     **Type**: Slider  
     **Label**: Select Year Range  
     **Default**: 2000 to latest year in the dataset  
     **Description**: Filters the data based on the selected year range.  

2. **Prediction Section**

   - **Year Selector**:  
     **Type**: Number Input  
     **Label**: Select Year for Prediction  
     **Default**: 2025  
     **Description**: Specifies the year for predicting future GHG emissions and temperature changes.  

3. **Conclusion Section**

   - **Year Range Slider**:  
     **Type**: Slider  
     **Label**: Select Year Range for Conclusion  
     **Default**: 2025 to 2035  
     **Description**: Defines the range of years for generating conclusions and visualizations.  

---

### **App Sections**

1. **Header**  
   Displays the main title: **"Electricity Generation vs. Harmful Effects on Earth"**.  

2. **Filter Data**  
   Allows users to filter the dataset by selecting countries and a year range.  

3. **Visualizations**  
   - **Electricity Production by Source**:  
     Area charts for renewable, fossil, and nuclear energy production.  
   - **Impact of Energy Production on GHG Emissions**:  
     Area charts showing GHG emissions over time.  
   - **Temperature Changes Caused by GHG Emissions**:  
     Area charts depicting temperature changes.  

4. **Comparative Analysis**  
   - **Energy Sources and GHG Emissions**:  
     Scatter plots with regression lines for:  
     - Fossil Fuels vs. GHG Emissions  
     - Renewable Energy vs. GHG Emissions  
     - Nuclear Energy vs. GHG Emissions  

5. **Predictions**  
   Predicts future values for:  
   - Renewable Energy (TWh)  
   - Fossil Fuel Usage (TWh)  
   - Nuclear Energy (TWh)  
   - GHG Emissions (MtCO₂)  
   - Temperature Changes (°C)  
   Visualizes the predictions with bar charts.  

6. **Conclusions**  
   Provides insights based on current trends, such as:  
   - When renewable energy will surpass fossil fuel usage.  
   - Predicted GHG emissions and temperature rise by a selected year range.  
   Displays visualizations for:  
   - Renewable vs. Fossil Energy Predictions  
   - GHG Emissions and Temperature Rise Predictions  

---

## **Data Requirements**

Ensure the following CSV data files are placed in the `data` directory:  

1. **Electricity Data**: elec-fossil-nuclear-renewables.csv  
2. **GHG Emissions Data**: total-ghg-emissions.csv  
3. **Temperature Change Data**: contribution-temp-rise-degrees.csv  

---

## **Notes**

- The app supports analysis for a **maximum of three countries** at a time.  
- The visualizations and predictions help understand the impact of electricity generation on GHG emissions and global temperatures.

---