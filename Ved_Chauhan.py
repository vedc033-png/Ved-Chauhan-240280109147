# Smart Fan Fuzzy Controller
# Python code extracted from Smart_Fan_Fuzzy_Controller_.ipynb
# Install dependencies if needed: pip install numpy scikit-fuzzy matplotlib pandas ipywidgets


# # Phase 3 – Python Implementation of a Fuzzy Controller
# 
# This notebook implements the **Smart Fan Speed Control** system using the **scikit-fuzzy** library. It follows the supplied Phase 3 design: three input variables (Distance, Temperature, Humidity), one output variable (Fan Speed), triangular membership functions, eight Mamdani IF–THEN rules, and centroid defuzzification.


# ## 1. Install and import libraries
# 
# If `scikit-fuzzy` is not already installed, run the installation command below.

# Run this only if scikit-fuzzy is not installed
# !pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# ## 2. Define linguistic variables and membership functions

# Input variables
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# Output variable
fan_speed = ctrl.Consequent(np.arange(0, 401, 1), 'fan_speed')

# Distance: 0–100 cm
distance['near'] = fuzz.trimf(distance.universe, [0, 0, 40])
distance['medium'] = fuzz.trimf(distance.universe, [20, 50, 80])
distance['far'] = fuzz.trimf(distance.universe, [60, 100, 100])

# Temperature: 0–50 °C
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['warm'] = fuzz.trimf(temperature.universe, [10, 25, 40])
temperature['hot'] = fuzz.trimf(temperature.universe, [30, 50, 50])

# Humidity: 0–100 %
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 40])
humidity['normal'] = fuzz.trimf(humidity.universe, [25, 50, 75])
humidity['humid'] = fuzz.trimf(humidity.universe, [60, 100, 100])

# Fan speed: 0–400 RPM
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 120])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [80, 180, 280])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [220, 320, 380])
fan_speed['maximum'] = fuzz.trimf(fan_speed.universe, [340, 400, 400])

# Visualize all membership functions
distance.view()
temperature.view()
humidity.view()
fan_speed.view()
plt.show()


# ## 3. Define the eight fuzzy IF–THEN rules

# R1: IF Distance is Near AND Temperature is Hot AND Humidity is Dry THEN Fan Speed is Maximum
rule1 = ctrl.Rule(distance['near'] & temperature['hot'] & humidity['dry'], fan_speed['maximum'])

# R2: IF Distance is Near AND Temperature is Warm THEN Fan Speed is Medium
rule2 = ctrl.Rule(distance['near'] & temperature['warm'], fan_speed['medium'])

# R3: IF Distance is Near AND Temperature is Cold THEN Fan Speed is Low
rule3 = ctrl.Rule(distance['near'] & temperature['cold'], fan_speed['low'])

# R4: IF Distance is Medium AND Temperature is Hot THEN Fan Speed is High
rule4 = ctrl.Rule(distance['medium'] & temperature['hot'], fan_speed['high'])

# R5: IF Distance is Medium AND Temperature is Warm THEN Fan Speed is Medium
rule5 = ctrl.Rule(distance['medium'] & temperature['warm'], fan_speed['medium'])

# R6: IF Distance is Far AND Temperature is Hot THEN Fan Speed is High
rule6 = ctrl.Rule(distance['far'] & temperature['hot'], fan_speed['high'])

# R7: IF Temperature is Hot AND Humidity is Humid THEN Fan Speed is Maximum
rule7 = ctrl.Rule(temperature['hot'] & humidity['humid'], fan_speed['maximum'])

# R8: IF Temperature is Cold THEN Fan Speed is Low
rule8 = ctrl.Rule(temperature['cold'], fan_speed['low'])


# ## 4. Create the Mamdani fuzzy control system

fan_control_system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4,
    rule5, rule6, rule7, rule8
])

fan_controller = ctrl.ControlSystemSimulation(fan_control_system)


# ## 5. Test the controller using the worked example
# 
# Inputs from the supplied design: **Distance = 30 cm, Temperature = 32 °C, Humidity = 40 %**.

# Sensor inputs
fan_controller.input['distance'] = 30
fan_controller.input['temperature'] = 32
fan_controller.input['humidity'] = 40

# Run fuzzy inference and centroid defuzzification
fan_controller.compute()

output_rpm = fan_controller.output['fan_speed']
print(f'Input Distance    : 30 cm')
print(f'Input Temperature : 32 °C')
print(f'Input Humidity    : 40 %')
print(f'Predicted Fan Speed: {output_rpm:.2f} RPM')

# Display the aggregated output and crisp fan-speed result
fan_speed.view(sim=fan_controller)
plt.title(f'Fuzzy Controller Output = {output_rpm:.2f} RPM')
plt.show()


# ## 6. Reusable prediction function
# 
# Use this function to calculate fan speed for any valid sensor readings.

def predict_fan_speed(distance_value, temperature_value, humidity_value):
    """Return the fuzzy-controller fan speed in RPM."""
    simulation = ctrl.ControlSystemSimulation(fan_control_system)
    simulation.input['distance'] = distance_value
    simulation.input['temperature'] = temperature_value
    simulation.input['humidity'] = humidity_value
    simulation.compute()
    return simulation.output['fan_speed']

# Example
rpm = predict_fan_speed(30, 32, 40)
print(f'Fan speed = {rpm:.2f} RPM')


# ## 7. Conclusion
# 
# The controller uses **Mamdani fuzzy inference** with minimum-based AND operations, maximum aggregation, and **centroid defuzzification** provided by `skfuzzy.control`. The resulting output is a smooth, continuous fan speed rather than a binary ON/OFF decision.


# ## Phase 4 – Testing
# The controller is tested with at least 10 different input combinations. The ranges follow the Phase 3 design: Distance 0–100 cm, Temperature 0–50 °C, Humidity 0–100%, and Fan Speed 0–400 RPM.

# Run this cell after running all Phase 3 cells

import pandas as pd

test_cases = [
    (10, 15, 20),
    (30, 32, 40),
    (20, 45, 30),
    (50, 35, 50),
    (70, 45, 40),
    (90, 40, 80),
    (30, 48, 80),
    (80, 15, 50),
    (50, 20, 70),
    (20, 10, 90),
]

results = []
for i, (d, t, h) in enumerate(test_cases, start=1):
    rpm = predict_fan_speed(d, t, h)
    results.append([i, d, t, h, round(rpm, 2)])

test_results = pd.DataFrame(
    results,
    columns=['Test', 'Distance (cm)', 'Temperature (°C)', 'Humidity (%)', 'Fan Speed (RPM)']
)
display(test_results)


# ### Testing observation
# The results show that fan speed generally increases when temperature becomes high. Cold conditions tend to produce low speed, while combinations that strongly activate the hot/maximum rules produce higher speed. The output is continuous because centroid defuzzification gives a crisp RPM value.


# ## Phase 5 – Analysis
# ### Controller behaviour
# - The controller combines Distance, Temperature and Humidity using fuzzy membership values.
# - The eight Mamdani rules determine the required fan-speed category.
# - Multiple rules can be active at the same time, so the final output can be between linguistic categories.
# - Centroid defuzzification produces a smooth numerical fan-speed value in RPM.


# ### Strengths
# 1. Easy to understand because the rules are written as IF–THEN statements.
# 2. Gives a continuous fan-speed output instead of only ON/OFF control.
# 3. Can handle gradual changes in temperature, humidity and distance.
# 4. Does not require a complicated mathematical model of the fan environment.


# ### Weaknesses
# 1. Only eight rules are used, so some combinations are not represented directly.
# 2. Membership-function limits are fixed and may not be optimal for every room or fan.
# 3. The present model does not include real sensor noise or hardware response.
# 4. The maximum fan speed is limited to the selected 0–400 RPM universe.


# ### Scope of improvement
# - Add more rules to cover missing combinations.
# - Tune membership functions using experimental sensor data.
# - Add real temperature, humidity and distance sensors.
# - Connect the controller to a microcontroller and actual fan driver.
# - Add data logging and a graph of inputs versus fan speed.
# - Add safety limits and sensor-error handling for practical deployment.


# ## Phase 6 – User Interface
# The UI below converts the fuzzy controller into an easy-to-use simulator. Enter Distance, Temperature and Humidity, then calculate the predicted fan speed. The code uses `ipywidgets`; if widgets are not enabled in the notebook environment, the same `predict_fan_speed()` function can still be used directly.

# Phase 6 – Interactive UI
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    distance_slider = widgets.FloatSlider(value=30, min=0, max=100, step=1, description='Distance:', continuous_update=False)
    temperature_slider = widgets.FloatSlider(value=32, min=0, max=50, step=1, description='Temp:', continuous_update=False)
    humidity_slider = widgets.FloatSlider(value=40, min=0, max=100, step=1, description='Humidity:', continuous_update=False)
    calculate_button = widgets.Button(description='Calculate Fan Speed', button_style='success')
    output_box = widgets.Output()

    def calculate_output(_):
        with output_box:
            clear_output()
            rpm = predict_fan_speed(distance_slider.value, temperature_slider.value, humidity_slider.value)
            print(f'Distance    : {distance_slider.value:.0f} cm')
            print(f'Temperature : {temperature_slider.value:.0f} °C')
            print(f'Humidity    : {humidity_slider.value:.0f} %')
            print(f'Fan Speed   : {rpm:.2f} RPM')

    calculate_button.on_click(calculate_output)
    display(widgets.VBox([distance_slider, temperature_slider, humidity_slider, calculate_button, output_box]))
except ImportError:
    print('ipywidgets is not installed. Install it with: pip install ipywidgets')

# Simple non-interactive simulator check
rpm = predict_fan_speed(30, 32, 40)
print(f'Distance = 30 cm, Temperature = 32 °C, Humidity = 40 %')
print(f'Predicted Fan Speed = {rpm:.2f} RPM')


# ## Final Conclusion
# The Phase 4 tests show that the fuzzy controller responds differently to different sensor conditions. The Phase 5 analysis shows that the main advantage is simple rule-based control with a smooth output, while the main limitation is the small rule set and fixed membership functions. In Phase 6, the controller is converted into a simple interactive UI so that a user can enter sensor values and view the predicted fan speed easily.