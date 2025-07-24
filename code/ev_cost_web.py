from flask import Flask, render_template, request
import json
import os

app = Flask(__name__)

# Load electricity rates from JSON
def load_rates():
    with open('state_rates.json') as f:
        return json.load(f)

PUBLIC_CHARGING_MULTIPLIER = 1.5  # Public is 50% more expensive

@app.route('/', methods=['GET', 'POST'])
def index():
    cost_home = None
    cost_public = None
    battery_capacity = None
    selected_state = None
    error = None

    state_rates = load_rates()

    if request.method == 'POST':
        selected_state = request.form.get('state')
        battery_capacity = request.form.get('battery_capacity')

        try:
            if not selected_state or selected_state not in state_rates:
                error = "Please select a valid state."
            else:
                battery_capacity = float(battery_capacity)
                if battery_capacity <= 0:
                    raise ValueError

                rate = state_rates[selected_state]
                cost_home = round(rate * battery_capacity, 2)
                cost_public = round(cost_home * PUBLIC_CHARGING_MULTIPLIER, 2)

        except ValueError:
            error = "Please enter a valid positive number for battery capacity."

    return render_template(
        'index.html',
        states=state_rates.keys(),
        state=selected_state,
        battery_capacity=battery_capacity,
        cost_home=cost_home,
        cost_public=cost_public,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
