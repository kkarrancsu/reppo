import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from reppo.vetoken import (
    VeTokenomicsSimulation,
    SimulationParams,
    MarketState,
    VestingSchedule
)

st.set_page_config(layout="wide")

def create_simulation_inputs():
    st.sidebar.header("Simulation Parameters")
    
    with st.sidebar.expander("Vesting Schedule"):
        initial_amount = st.number_input("Initial Amount", value=1_000_000, step=100_000)
        cliff_epoch = st.number_input("Cliff Epoch", value=26, step=1)
        vesting_duration = st.number_input("Vesting Duration", value=104, step=1)
        release_frequency = st.number_input("Release Frequency", value=13, step=1)
    
    with st.sidebar.expander("Protocol Parameters"):
        gamma = st.slider("Gamma (veToken power)", 1.0, 4.0, 2.0, 0.1)
        alpha = st.slider("Alpha (Performance weight)", 0.0, 1.0, 0.5, 0.1)
        delta = st.slider("Delta (Base weight)", 0.0, 1.0, 0.2, 0.1)
        omega = st.slider("Omega (FCU generation)", 0.0, 1.0, 0.1, 0.1)
    
    with st.sidebar.expander("Pod Parameters"):
        base_fee_drift = st.slider("Base Fee Drift", 0.0, 0.2, 0.05, 0.01)
        max_lock_duration = st.number_input("Max Lock Duration", value=52, step=1)
        min_lock_duration = st.number_input("Min Lock Duration", value=4, step=1)
        num_pods = st.number_input("Number of Initial Pods", value=2, step=1)
    
    with st.sidebar.expander("Market Parameters"):
        base_fee_rate = st.slider("Base Fee Rate", 0.0, 0.5, 0.1, 0.01)
        growth_rate = st.slider("Growth Rate", -0.2, 0.2, 0.05, 0.01)
        volatility = st.slider("Volatility", 0.0, 0.5, 0.2, 0.01)
    
    with st.sidebar.expander("General Parameters"):
        base_stake_rate = st.slider("Base Stake Rate", 0.1, 0.9, 0.5, 0.1)
        initial_token_supply = st.number_input("Initial Token Supply", value=1_000_000, step=100_000)
        epochs = st.number_input("Epochs to Simulate", value=120, step=10)

    with st.sidebar.expander("Emission Parameters"):
        initial_emission = st.number_input("Initial Emission Rate", value=1000, step=100)
        decay_rate = st.slider("Emission Decay Rate", 0.0, 0.2, 0.05, 0.01)
            
    return {
        "vesting": VestingSchedule(
            initial_amount=initial_amount,
            cliff_epoch=cliff_epoch,
            vesting_duration=vesting_duration,
            release_frequency=release_frequency
        ),
        "protocol": {
            "gamma": gamma,
            "alpha": alpha,
            "delta": delta,
            "omega": omega,
        },
        "pods": {
            "base_fee_drift": base_fee_drift,
            "max_lock_duration": max_lock_duration,
            "min_lock_duration": min_lock_duration,
            "num_pods": num_pods
        },
        "market": MarketState(
            base_fee_rate=base_fee_rate,
            growth_rate=growth_rate,
            volatility=volatility
        ),
        "general": {
            "base_stake_rate": base_stake_rate,
            "initial_token_supply": initial_token_supply,
            "epochs": epochs
        },
        "emissions": {
            "initial_rate": initial_emission,
            "decay_rate": decay_rate
        }
    }

def create_simulation(inputs):
    def emission_schedule(state):
        return inputs["emissions"]["initial_rate"] * (
            (1 - inputs["emissions"]["decay_rate"]) ** state.epoch
        )
    
    params = SimulationParams(
        gamma=inputs["protocol"]["gamma"],
        alpha=inputs["protocol"]["alpha"],
        delta=inputs["protocol"]["delta"],
        omega=inputs["protocol"]["omega"],
        fee_volatility=0.1,
        base_stake_rate=inputs["general"]["base_stake_rate"],
        base_fee_drift=inputs["pods"]["base_fee_drift"],
        max_lock_duration=inputs["pods"]["max_lock_duration"],
        min_lock_duration=inputs["pods"]["min_lock_duration"],
        initial_pods=[f"pod{i+1}" for i in range(inputs["pods"]["num_pods"])],
        initial_token_supply=inputs["general"]["initial_token_supply"],
        epochs=inputs["general"]["epochs"],
        market=inputs["market"],
        vesting=inputs["vesting"],
        emission_schedule=emission_schedule
    )
    
    return VeTokenomicsSimulation(params)

def create_pod_metrics_tab(history):
    st.header("Pod Performance Metrics")
    
    df = pd.DataFrame(history)
    
    # Create a DataFrame for pod-specific metrics
    pod_data = []
    for epoch_data in history:
        metrics = epoch_data['metrics']
        for pod_name, emissions in epoch_data['pod_emissions'].items():
            pod_data.append({
                'epoch': epoch_data['epoch'],
                'pod': pod_name,
                'emissions': emissions,
                'votes': metrics.get('vote_distribution', {}).get(pod_name, 0),
                'fees': metrics.get('pod_fees', {}).get(pod_name, 0),
                'fcus': metrics.get('pod_fcus', {}).get(pod_name, 0),
                'cumulative_fees': metrics.get('cumulative_pod_fees', {}).get(pod_name, 0),
                'cumulative_fcus': metrics.get('cumulative_pod_fcus', {}).get(pod_name, 0),
                'fee_rate': metrics.get('fee_generation_rate', {}).get(pod_name, 0),
                'fcu_efficiency': metrics.get('fcu_efficiency', {}).get(pod_name, 0),
                'avg_vote_share': metrics.get('avg_vote_share', {}).get(pod_name, 0)
            })
    
    pod_df = pd.DataFrame(pod_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pod Fees Chart
        fees_chart = alt.Chart(pod_df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('fees:Q', title='Fees'),
            color=alt.Color('pod:N', title='Pod')
        ).properties(
            title='Pod Fee Generation',
            width=400,
            height=300
        )
        st.altair_chart(fees_chart, use_container_width=True)
        
        # Pod Vote Share Chart
        votes_chart = alt.Chart(pod_df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('votes:Q', title='Vote Share %'),
            color=alt.Color('pod:N', title='Pod')
        ).properties(
            title='Pod Vote Distribution',
            width=400,
            height=300
        )
        st.altair_chart(votes_chart, use_container_width=True)
    
    with col2:
        # Cumulative Metrics
        tabs = st.tabs(["Current", "Cumulative", "Efficiency"])
        
        with tabs[0]:
            # Pod Emissions Chart
            emissions_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('emissions:Q', title='Emissions'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='Pod Emissions',
                width=400,
                height=300
            )
            st.altair_chart(emissions_chart, use_container_width=True)
            
            # FCUs Chart
            fcus_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('fcus:Q', title='FCUs'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='Pod FCUs Issued',
                width=400,
                height=300
            )
            st.altair_chart(fcus_chart, use_container_width=True)
            
        with tabs[1]:
            # Cumulative Fees
            cum_fees_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('cumulative_fees:Q', title='Total Fees'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='Cumulative Pod Fees',
                width=400,
                height=300
            )
            st.altair_chart(cum_fees_chart, use_container_width=True)
            
            # Cumulative FCUs
            cum_fcus_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('cumulative_fcus:Q', title='Total FCUs'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='Cumulative Pod FCUs',
                width=400,
                height=300
            )
            st.altair_chart(cum_fcus_chart, use_container_width=True)
            
        with tabs[2]:
            # Fee Generation Rate
            fee_rate_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('fee_rate:Q', title='Fees/Vote'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='Fee Generation Efficiency',
                width=400,
                height=300
            )
            st.altair_chart(fee_rate_chart, use_container_width=True)
            
            # FCU Efficiency
            fcu_eff_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('fcu_efficiency:Q', title='FCUs/Fee'),
                color=alt.Color('pod:N', title='Pod')
            ).properties(
                title='FCU Generation Efficiency',
                width=400,
                height=300
            )
            st.altair_chart(fcu_eff_chart, use_container_width=True)

def create_macro_metrics_tab(history):
    st.header("Macroeconomic Indicators")
    
    df = pd.DataFrame(history)
    
    # Calculate derived metrics
    df['tvl'] = df['locked_tokens']
    df['fee_efficiency'] = df['metrics'].apply(lambda x: x['total_fees']) / df['tvl'].where(df['tvl'] > 0, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supply Metrics
        supply_chart = alt.Chart(df).transform_fold(
            ['total_supply', 'locked_tokens', 've_tokens'],
            as_=['metric', 'value']
        ).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('value:Q', title='Amount'),
            color=alt.Color('metric:N', title='Metric')
        ).properties(
            title='Token Supply Metrics',
            width=400,
            height=300
        )
        st.altair_chart(supply_chart, use_container_width=True)
        
        # System Participation
        participation_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('metrics.active_positions:Q', title='Active Positions'),
        ).properties(
            title='System Participation',
            width=400,
            height=300
        )
        st.altair_chart(participation_chart, use_container_width=True)
    
    with col2:
        # Fee Efficiency
        efficiency_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('fee_efficiency:Q', title='Fee/TVL Ratio'),
        ).properties(
            title='Fee Generation Efficiency',
            width=400,
            height=300
        )
        st.altair_chart(efficiency_chart, use_container_width=True)
        
        # Average Lock Duration
        duration_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('metrics.avg_lock_duration:Q', title='Epochs'),
        ).properties(
            title='Average Lock Duration',
            width=400,
            height=300
        )
        st.altair_chart(duration_chart, use_container_width=True)

def create_market_metrics_tab(history):
    st.header("Market Indicators")
    
    df = pd.DataFrame(history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Rate
        market_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('market_rate:Q', title='Rate'),
        ).properties(
            title='Market Base Fee Rate',
            width=400,
            height=300
        )
        st.altair_chart(market_chart, use_container_width=True)
        
        # Vote Entropy (Decentralization)
        entropy_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('metrics.vote_entropy:Q', title='Entropy'),
        ).properties(
            title='Vote Distribution Entropy',
            width=400,
            height=300
        )
        st.altair_chart(entropy_chart, use_container_width=True)
    
    with col2:
        # Prepare data for emissions vs fees comparison
        comparison_df = pd.DataFrame({
            'epoch': df['epoch'],
            'Total Fees': df['metrics'].apply(lambda x: x['total_fees']),
            'Total Emissions': df['total_emissions']
        }).melt(
            id_vars=['epoch'],
            value_vars=['Total Fees', 'Total Emissions'],
            var_name='metric',
            value_name='value'
        )
        
        comparison_chart = alt.Chart(comparison_df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('value:Q', title='Amount'),
            color=alt.Color('metric:N', title='Metric')
        ).properties(
            title='Total Emissions vs Fees Generated',
            width=400,
            height=300
        )
        st.altair_chart(comparison_chart, use_container_width=True)
        
        # Vesting Progress
        vesting_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('metrics.total_vested:Q', title='Amount'),
        ).properties(
            title='Cumulative Vested Tokens',
            width=400,
            height=300
        )
        st.altair_chart(vesting_chart, use_container_width=True)

def main():
    st.title("VeTokenomics Simulation")
    
    inputs = create_simulation_inputs()  # Assuming this function exists as before
    
    if st.sidebar.button("Run Simulation"):
        sim = create_simulation(inputs)  # Assuming this function exists as before
        
        # Create some initial locks
        sim.create_lock(amount=10000, duration=26)
        sim.create_lock(amount=20000, duration=52)
        
        # Run simulation
        states = sim.run()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Pod Metrics", "Macro Metrics", "Market Metrics"])
        
        with tab1:
            create_pod_metrics_tab(sim.history)
            
        with tab2:
            create_macro_metrics_tab(sim.history)
            
        with tab3:
            create_market_metrics_tab(sim.history)

if __name__ == "__main__":
    main()