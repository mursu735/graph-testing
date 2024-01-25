import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import helpers
import textwrap

df = pd.read_csv("output/GPT/chapter_durations_fixed.csv", sep=";")
df["Start Date"] = pd.to_datetime(df["Start Date"])
df["End Date"] = pd.to_datetime(df["End Date"])
print(df)

start = df["Start Date"].min()
end = start + timedelta(days=80)
print("Start date:", start)

config = {'displayModeBar': False}

fig = go.Figure()

#for idx, row in df.iterrows():
#row = df.iloc[15]
country = "in"
country_text = country.split("_")[0] 
for idx, row in df.iterrows():
    countries = row["Country"].split(",")
    df.loc[idx,'Include'] = country in countries

print(df)

rows = df[df["Include"] == True]

print(rows)
row = {'Start Date': rows["Start Date"].min(), 'End Date': rows["End Date"].max(), "Chapter": f"From chapter {rows['Chapter'].min()} to chapter {rows['Chapter'].max()}"}
print(row)

#print(row["Start Date"], print(type(row["Start Date"])))
#print(asd, print(type(asd)))
#print(row["Chapter"])
text = f"Chapter start: {row['Start Date'].strftime('%d/%m/%Y')}, chapter end: {row['End Date'].strftime('%d/%m/%Y')}"
day = row['Start Date'] - start
print(day.days)

text = '<br>'.join(textwrap.wrap(text, width=30))
fig.add_trace(go.Line(x=[start, end], y=[0, 0], name="total", text=f"Country: {helpers.country_code_to_name[country_text]}, from chapter {rows['Chapter'].min()} to chapter {rows['Chapter'].max()}", hoverinfo='text', line={'width': 1, 'color': 'black'}))
fig.add_trace(go.Line(x=[row["Start Date"], row["End Date"]], y=[0, 0], text=text, hoverinfo='text', name=str(row["Chapter"]), line={'width': 100, 'color': 'blue'}))


fig.update_layout(
    width=300,
    height=200
    )

fig.update_xaxes(range=[start, end])
fig.update_yaxes(showticklabels=False, automargin=True)

fig.update_layout({
            'yaxis': {'fixedrange': True},
            'yaxis2': {'fixedrange': True},
            'showlegend': False,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'hoverlabel_namelength': len(text)
    })
fig.show(config=config)
