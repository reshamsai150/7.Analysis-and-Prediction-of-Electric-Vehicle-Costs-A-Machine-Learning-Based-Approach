import json
import os

def load_rates():
    # Load rates JSON file from same directory as script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rates_path = os.path.join(base_dir, 'state_rates.json')
    with open(rates_path, 'r') as file:
        rates = json.load(file)
    return rates

def calculate_cost(battery_capacity_kwh, state, rates):
    # Fetch rate for the selected state (case-insensitive match)
    state_key = None
    for key in rates:
        if key.lower() == state.lower():
            state_key = key
            break
    
    if state_key is None:
        raise ValueError(f"Electricity rate for state '{state}' not found.")
    
    rate_per_kwh = rates[state_key]
    if rate_per_kwh is None:
        raise ValueError(f"Electricity rate for state '{state}' is unknown.")
    
    cost = battery_capacity_kwh * rate_per_kwh
    return cost

if __name__ == "__main__":
    # List of all 28 states + 8 UTs of India
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", 
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", 
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", 
        "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu", "Delhi", "Jammu and Kashmir", 
        "Ladakh", "Lakshadweep", "Puducherry"
    ]

    print("Select your state/UT from the list below:")
    for idx, st in enumerate(states, 1):
        print(f"{idx}. {st}")

    # State selection with validation
    while True:
        try:
            choice = int(input("Enter number corresponding to your state/UT: "))
            if 1 <= choice <= len(states):
                selected_state = states[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(states)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Battery capacity input with validation
    while True:
        try:
            battery_capacity = float(input("Enter your EV battery capacity (in kWh): "))
            if battery_capacity <= 0:
                print("Battery capacity must be a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Load rates and calculate cost
    try:
        rates = load_rates()
        estimated_cost = calculate_cost(battery_capacity, selected_state, rates)
        print(f"\nEstimated EV Charging Cost in {selected_state}: ₹{estimated_cost:.2f}")
    except Exception as e:
        print(f"Error: {e}")
