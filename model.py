import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
from datetime import datetime, timedelta
import requests
import numpy as np

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['DishName', 'Date'])

# -------------------------------
# Step 2: Create past-day feature
# -------------------------------
df['PrevQuantity'] = df.groupby('DishName')['QuantitySold'].shift(1)
df['PrevQuantity'] = df['PrevQuantity'].fillna(df['QuantitySold'].mean())

# -------------------------------
# Step 3: One-hot encode categorical features
# -------------------------------
df_encoded = pd.get_dummies(df, columns=['DishName', 'DayOfWeek', 'Weather'])

X = df_encoded.drop(columns=['Date', 'QuantitySold'])
y = df_encoded['QuantitySold']

# -------------------------------
# Step 4: Train the model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained!")

# -------------------------------
# Step 5: Fetch real weather for tomorrow
# -------------------------------
API_KEY = "8d33c71fa8957de4c26573cad5b830e6"
CITY = "Algiers,DZ"

tomorrow_date = datetime.now() + timedelta(days=1)
tomorrow_day = tomorrow_date.strftime("%A")
tomorrow_str = tomorrow_date.strftime("%Y-%m-%d")

url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()

# Default fallback
tomorrow_temp = 25
tomorrow_weather = "Sunny"  # Changed from Cloudy to Sunny

if 'list' in data:
    for item in data['list']:
        if item['dt_txt'].startswith(tomorrow_str):
            tomorrow_temp = item['main']['temp']
            ow_weather = item['weather'][0]['main']
            
            # Map to your dataset's weather categories: Rainy, Sunny, Hot
            if ow_weather in ['Rain', 'Drizzle', 'Thunderstorm']:
                tomorrow_weather = "Rainy"
            elif tomorrow_temp > 28:  # Consider it Hot if temperature is high
                tomorrow_weather = "Hot"
            else:  # Clear, Clouds, etc. become Sunny
                tomorrow_weather = "Sunny"
            break
else:
    print("⚠️ Error fetching weather data:", data)
    print("Using fallback temperature and weather.")

print(f"\n📅 Tomorrow: {tomorrow_str} ({tomorrow_day})")
print(f"🌤️ Weather: {tomorrow_weather} | 🌡️ {round(tomorrow_temp,1)}°C")

# -------------------------------
# Step 6: Previous day quantity per dish
# -------------------------------
prev_quantities = {}
dishes = df['DishName'].unique()
yesterday = tomorrow_date - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

for dish in dishes:
    try:
        # Convert to string comparison to avoid date format issues
        prev_qty = int(df[(df['Date'].dt.strftime('%Y-%m-%d') == yesterday_str) & 
                         (df['DishName'] == dish)]['QuantitySold'].iloc[0])
        prev_quantities[dish] = prev_qty
    except:
        # Use mean if yesterday's data not available
        prev_quantities[dish] = int(df[df['DishName'] == dish]['QuantitySold'].mean())

# -------------------------------
# Step 7: Build input for prediction
# -------------------------------
data_list = []
for dish in dishes:
    row = {'Temperature': tomorrow_temp, 'PrevQuantity': prev_quantities[dish]}
    
    # Dish encoding
    for d in dishes:
        row[f'DishName_{d}'] = 1 if d == dish else 0
    
    # Day of week encoding
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for day in days:
        row[f'DayOfWeek_{day}'] = 1 if day == tomorrow_day else 0
    
    # Weather encoding (only Rainy, Sunny, Hot)
    weathers = ['Weather_Rainy', 'Weather_Sunny', 'Weather_Hot']
    for w in weathers:
        row[w] = 1 if w == f"Weather_{tomorrow_weather}" else 0
    
    data_list.append(row)

X_tomorrow = pd.DataFrame(data_list)

# Align columns with training set
missing_cols = set(X.columns) - set(X_tomorrow.columns)
for col in missing_cols:
    X_tomorrow[col] = 0
X_tomorrow = X_tomorrow[X.columns]

# -------------------------------
# Step 8: Predict
# -------------------------------
predictions = model.predict(X_tomorrow)

dish_predictions = []
for dish, qty in zip(dishes, predictions):
    dish_predictions.append({
        "Dish": dish,
        "PredictedQuantity": int(qty)
    })

# Sort by predicted quantity in descending order
dish_predictions.sort(key=lambda x: x['PredictedQuantity'], reverse=True)

# Select top 5 dishes
top_5_dishes = dish_predictions[:5]

# If it's Friday, ensure couscous is the first recommendation
if tomorrow_day == "Friday":
    couscous_found = False
    couscous_index = -1
    
    # Check for couscous variations
    for i, dish in enumerate(top_5_dishes):
        if any(keyword in dish["Dish"].lower() for keyword in ["couscous", "cous cous", "كسكس"]):
            couscous_found = True
            couscous_index = i
            break
    
    if couscous_found:
        couscous_item = top_5_dishes.pop(couscous_index)
        top_5_dishes.insert(0, couscous_item)
    else:
        # Find couscous in all predictions
        for dish in dish_predictions:
            if any(keyword in dish["Dish"].lower() for keyword in ["couscous", "cous cous", "كسكس"]):
                top_5_dishes[-1] = dish
                break

# -------------------------------
# Step 9: Final output
# -------------------------------
output = []
for dish in top_5_dishes:
    output.append({
        "Dish": dish["Dish"],
        "PredictedQuantity": dish["PredictedQuantity"],
        "Date": tomorrow_str,
        "Day": tomorrow_day,
        "Weather": tomorrow_weather,
        "Temperature": round(tomorrow_temp, 1)
    })

print("\n🍽️ Top 5 Recommended Dishes:")
print(json.dumps(output, indent=4, ensure_ascii=False))

# Save to file
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
print("\n💾 Predictions saved to predictions.json")