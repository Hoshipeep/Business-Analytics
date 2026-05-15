import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

print("Loading data and generating charts... This might take a few seconds!")

# --- 1. LOAD & CLEAN DATA ---
df = pd.read_csv("NYC Accidents 2020.csv")
df['BOROUGH'] = df['BOROUGH'].fillna('UNKNOWN')
df['CONTRIBUTING FACTOR VEHICLE 1'] = df['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified')
df['VEHICLE TYPE CODE 1'] = df['VEHICLE TYPE CODE 1'].fillna('UNKNOWN')
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M:%S', errors='coerce')

# --- 2. BUILD THE CHARTS ---

# Q1: Histogram (Hours)
df['CRASH HOUR'] = df['CRASH TIME'].dt.hour
fig1 = px.histogram(df, x='CRASH HOUR', nbins=24, title="Q1: Collisions Throughout the Day", color_discrete_sequence=['coral'])

# Q2: Time Series (COVID)
daily_crashes = df.groupby('CRASH DATE').size().reset_index(name='TOTAL CRASHES')
fig2 = px.line(daily_crashes, x='CRASH DATE', y='TOTAL CRASHES', title="Q2: Daily Crash Volume (2020)")
fig2.add_vline(x='2020-03-20', line_dash="dash", line_color="red")
fig2.add_annotation(x='2020-03-20', y=1, yref="paper", text="COVID Lockdown Starts", showarrow=False, font=dict(color="red"), xanchor="left", xshift=10)

# Q3: Heatmap (Borough vs Factor)
known_b = df[(df['BOROUGH'] != 'UNKNOWN') & (df['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
top_5 = known_b['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().nlargest(5).index
heat_df = known_b[known_b['CONTRIBUTING FACTOR VEHICLE 1'].isin(top_5)]
cross = pd.crosstab(heat_df['BOROUGH'], heat_df['CONTRIBUTING FACTOR VEHICLE 1'])
fig3 = px.imshow(cross, text_auto=True, aspect="auto", title="Q3: Top 5 Crash Factors by Borough", color_continuous_scale='YlOrRd')

# Q4: Grouped Bar (Vehicles vs Factors)
v_counts = heat_df.groupby(['CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1']).size().reset_index(name='COUNT')
top_v = v_counts.sort_values(['CONTRIBUTING FACTOR VEHICLE 1', 'COUNT'], ascending=[True, False]).groupby('CONTRIBUTING FACTOR VEHICLE 1').head(3)
fig4 = px.bar(top_v, x='CONTRIBUTING FACTOR VEHICLE 1', y='COUNT', color='VEHICLE TYPE CODE 1', barmode='group', title="Q4: Top Vehicles in Top Crash Factors")

# Q5: Scatter with Regression (Predictive)
daily_stats = df.groupby('CRASH DATE').agg(TOTAL_CRASHES=('CRASH DATE', 'count'), TOTAL_INJURIES=('NUMBER OF PERSONS INJURED', 'sum')).reset_index()
fig5 = px.scatter(daily_stats, x='TOTAL_CRASHES', y='TOTAL_INJURIES', trendline='ols', title="Q5: Predicting Injuries from Crash Volume", trendline_color_override="red")

# Q6: Bar Chart (Day of Week)
df['DAY OF WEEK'] = df['CRASH DATE'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['DAY OF WEEK'].value_counts().reindex(days_order).reset_index()
day_counts.columns = ['DAY OF WEEK', 'TOTAL CRASHES']
fig6 = px.bar(day_counts, x='DAY OF WEEK', y='TOTAL CRASHES', title="Q6: Collisions by Day of Week", color='DAY OF WEEK', color_discrete_sequence=px.colors.sequential.Magma)

# Q7: Horizontal Bar (Streets)
streets = df.dropna(subset=['ON STREET NAME'])
top_streets = streets['ON STREET NAME'].value_counts().head(10).reset_index()
top_streets.columns = ['STREET NAME', 'TOTAL CRASHES']
fig7 = px.bar(top_streets, x='TOTAL CRASHES', y='STREET NAME', orientation='h', title="Q7: Top 10 Most Dangerous Streets", color='TOTAL CRASHES', color_continuous_scale='Reds')
fig7.update_layout(yaxis={'categoryorder':'total ascending'})

# --- 3. SET UP THE WEB LAYOUT ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'font-family': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f9f9f9'}, children=[
    html.H1("NYC Motor Vehicle Collisions (2020) - Final Dashboard", style={'textAlign': 'center'}),
    html.P("Hover over the charts to see exact data points!", style={'textAlign': 'center', 'color': 'gray'}),
    
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig6),
    dcc.Graph(figure=fig7)
])

# --- 4. RUN THE SERVER ---
if __name__ == '__main__':
    print("Dashboard is ready! Open your web browser and go to: http://127.0.0.1:8050/")
    app.run(debug=True)