import streamlit as st
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DISCOUNT = 0.9
n_advertisers = 10
n_slots = 3  


advertisers_info = pd.read_csv(r"Advertisers_dataset.csv")  
auction_slots = pd.read_csv(r"Auction_Slots_corrected_dataset.csv")  
historical_data = pd.read_csv(r"Historical_Data_dataset.csv") 


st.title("Dynamic Auction Simulator")
st.sidebar.header("Auction Settings")
n_periods = st.sidebar.slider("Number of Auction Periods:", min_value=10, max_value=30, value=20)  

st.sidebar.header("Enter Advertiser Details")
user_data = {}
for i in range(1, n_advertisers + 1):
    bid_amount = st.sidebar.number_input(f'Advertiser {i} Bid Amount:', min_value=1.0, value=float(advertisers_info['Bid_Amount'][i-1]))
    prob_selling = st.sidebar.slider(f'Advertiser {i} Probability of Selling:', min_value=0.0, max_value=1.0, value=float(advertisers_info['Probability_of_Selling'][i-1]))
    items_for_sale = st.sidebar.number_input(f'Advertiser {i} Items for Sale:', min_value=1, value=int(advertisers_info['Items_for_Sale'][i-1]))
    
    user_data[i] = {'Bid_Amount': bid_amount, 'Probability_of_Selling': prob_selling, 'Items_for_Sale': items_for_sale}


def plot_results(results):
    """Plot auction results."""
    results_df = pd.DataFrame(results)

    
    plt.figure(figsize=(8, 12))  
    
    revenue_over_time = results_df.groupby('Time_Period').sum()['Was_sold'].reset_index()
    plt.subplot(2, 1, 1)
    sns.barplot(data=revenue_over_time, x='Time_Period', y='Was_sold', palette='viridis')
    plt.title('Total Items Sold Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('Total Items Sold')


    items_sold_by_advertiser = results_df.groupby('Advertiser_ID').sum()['Was_sold'].reset_index()
    plt.subplot(2, 1, 2)
    sns.barplot(data=items_sold_by_advertiser, x='Advertiser_ID', y='Was_sold', palette='plasma')
    plt.title('Items Sold by Advertiser')
    plt.xlabel('Advertiser ID')
    plt.ylabel('Total Items Sold')

    plt.tight_layout()
    st.pyplot(plt)

# Generate Advertiser Dataset or use Default
if st.sidebar.button("Simulate Auction"):
    if any(user_data.values()):
        st.write("Using user-provided data")
        advertisers = pd.DataFrame(user_data).T
        advertisers['Advertiser_ID'] = advertisers.index   # Add Advertiser_ID column
    else:
        st.write("Using default dataset")
        advertisers = advertisers_info.copy()
        advertisers['Advertiser_ID'] = advertisers.index
        

    slots = auction_slots.to_dict(orient='records')
    history = historical_data.to_dict(orient='records')
    
    # Initialize value function matrix (DP)
    V = np.zeros((n_periods + 1, n_advertisers + 1))

    # Function to calculate revenue
    def calculate_revenue(allocations, state, history):
        revenue = 0
        revenue_per_advertiser = np.zeros(len(advertisers))
        for slot, advertiser in enumerate(allocations):
            if advertiser is not None:
                advertiser_data = advertisers.iloc[advertiser]
                slot_quality = slots[slot]['Slot_Quality']

                # Adjust dynamic selling probability based on history
                advertiser_history = historical_data[historical_data['Advertiser_ID'] == advertiser + 1]
                total_bids = len(advertiser_history)
                successful_sales = advertiser_history['Previous_Sale_Status'].sum()

                dynamic_prob_sell = successful_sales / total_bids if total_bids > 0 else 0
                bid = advertiser_data['Bid_Amount']
                revenue += bid * dynamic_prob_sell * slot_quality
                revenue_per_advertiser[advertiser] = bid * dynamic_prob_sell * slot_quality
        return revenue_per_advertiser, revenue

    # Slot allocation using qν-rule
    def qnu_rule(advertisers, slots, state, V, t):
        qnu_values = []
        avg_bid = np.mean(advertisers['Bid_Amount'])
        for idx, advertiser in advertisers.iterrows():
            if advertiser['Items_for_Sale']<=0:
                    continue
            future_value = DISCOUNT * V[t + 1, int(advertiser['Advertiser_ID'])]
            for slot in slots:                
                qnu = ((advertiser['Bid_Amount'] / avg_bid) * advertiser['Probability_of_Selling'] * slot['Slot_Quality']) + future_value
                qnu_values.append((advertiser['Advertiser_ID'], slot['Slot_ID'], qnu))

        qnu_values = sorted(qnu_values, key=lambda x: x[2], reverse=True)
        allocations = [None] * len(slots)
        assigned_advertisers = set()
        assigned_slots = set()

        for adv_id, slot_id, _ in qnu_values:
            adv_id = int(adv_id)
            advertiser_data = advertisers.loc[advertisers['Advertiser_ID'] == adv_id]
            if advertiser_data['Items_for_Sale'].values[0] > 0 and adv_id<len(state) and adv_id not in assigned_advertisers and slot_id not in assigned_slots and state[adv_id-1] > 0:
                print()
                slot_index = slot_id - 1
                allocations[slot_index] = adv_id 
                assigned_advertisers.add(adv_id)
                assigned_slots.add(slot_id)
                if len(assigned_advertisers) == len(slots):
                    break
        return allocations

    # Auction simulation
    results = []
    total_revenue = 0

    for t in range(n_periods - 1, -1, -1):
        state = advertisers['Items_for_Sale'].tolist()
        
        allocations = qnu_rule(advertisers, slots, state, V, t)  
        revenue_per_advertiser, revenue = calculate_revenue(allocations, state, historical_data)
        total_revenue += revenue
        for adv in range(n_advertisers):
            future_value = DISCOUNT * V[t + 1, adv]
            V[t, adv] = revenue_per_advertiser[adv] + future_value
        
        for slot_index, adv_id in enumerate(allocations):
            if adv_id is not None:
                
                previous_sale_status = np.random.choice([0, 1], p=[1 - advertisers.at[adv_id,'Probability_of_Selling'], advertisers.at[adv_id,'Probability_of_Selling']])
                if previous_sale_status == 1:
                    advertisers.at[adv_id, 'Items_for_Sale'] = advertisers.at[adv_id, 'Items_for_Sale']-1
                    advertisers.at[adv_id, 'Bid_Amount'] = max(1, advertisers.at[adv_id, 'Bid_Amount'] * 1.1 * (1 + random.uniform(-0.02, 0.02)))
                    advertisers.at[adv_id, 'Probability_of_Selling'] = min(1.0, advertisers.at[adv_id, 'Probability_of_Selling'] * 1.05)
                else:
                    
                    advertisers.at[adv_id, 'Bid_Amount'] = max(1, advertisers.at[adv_id, 'Bid_Amount'] * 0.9 * (1 + random.uniform(-0.02, 0.02)))
                    advertisers.at[adv_id, 'Probability_of_Selling'] = max(0, advertisers.at[adv_id, 'Probability_of_Selling'] * 0.95)
                
                
                
                
                results.append({
                    'Time_Period': t,
                    'Advertiser_ID': adv_id ,
                    'Slot_ID': slot_index + 1,
                    'Was_sold': previous_sale_status,
                    'Remaining_items': advertisers.at[adv_id, 'Items_for_Sale']
                })

    # Display results and visualize
    st.write("Auction Results:")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)  # Display the results DataFrame
    st.write(f"Total Revenue: {total_revenue}")

    # Create a DataFrame for the Value Function and display it
    #V_df = pd.DataFrame(V[1:, 1:], columns=[f'Advertiser {i}' for i in range(1, n_advertisers + 1)], index=[f'Time {i}' for i in range(n_periods)])
    #st.write("Value Function Table:")
    #st.dataframe(V_df)  # Display the value function table"""

    # Call the plot function to display the visualizations
    plot_results(results)
