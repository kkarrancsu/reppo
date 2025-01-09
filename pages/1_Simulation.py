import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from vetoken import (
    VeTokenomicsSimulation,
    SimulationParams,
    MarketState,
)
from vesting import (
    VestingSchedule,
    LinearVestingSchedule,
    CliffVestingSchedule
)

st.set_page_config(layout="wide")

def create_simulation_inputs():
    st.sidebar.header("Simulation Parameters")
    
    with st.sidebar.expander("Protocol Parameters"):
        gamma = st.slider(
            "Gamma (veToken power)", 
            1.0, 4.0, 2.0, 0.1,
            help="Controls how much additional voting power is granted for longer lock durations. "
                 "Higher values create stronger incentives for longer locks."
        )
        alpha = st.slider(
            "Alpha (Performance weight)", 
            0.0, 1.0, 0.5, 0.1,
            help="Weight given to pod performance when redistributing votes. "
                 "Higher values make vote distribution more sensitive to pod fee generation."
        )
        delta = st.slider(
            "Delta (Base weight)", 
            0.0, 1.0, 0.2, 0.1,
            help="Minimum base weight given to each pod regardless of performance. "
                 "Higher values promote more uniform vote distribution."
        )
        omega = st.slider(
            "Omega (FCU generation)", 
            0.0, 1.0, 0.1, 0.1,
            help="Rate at which Fee Claim Units (FCUs) are generated relative to fees. "
                 "Higher values create more FCUs per unit of fees generated."
        )
    
    with st.sidebar.expander("Pod Parameters"):
        base_fee_drift = st.slider(
            "Base Fee Drift", 
            0.0, 0.2, 0.05, 0.01,
            help="Base rate at which pods generate fees. "
                 "This is modified by votes and market conditions."
        )
        max_lock_duration = st.number_input(
            "Max Lock Duration", 
            value=52, 
            step=1,
            help="Maximum number of epochs users can lock their tokens. "
                 "Longer durations enable higher voting power but reduce liquidity."
        )
        min_lock_duration = st.number_input(
            "Min Lock Duration", 
            value=4, 
            step=1,
            help="Minimum number of epochs users must lock their tokens. "
                 "Higher minimums increase system stability but reduce flexibility."
        )
        num_pods = st.number_input(
            "Number of Initial Pods", 
            value=2, 
            step=1,
            help="Number of pods to simulate. Each pod represents a unit of intelligence "
                 "that can generate fees and receive vote allocations."
        )
    
    with st.sidebar.expander("Market Parameters"):
        base_fee_rate = st.slider(
            "Base Fee Rate", 
            0.0, 0.5, 0.1, 0.01,
            help="Base market rate for fee generation. "
                 "This represents the overall market activity level."
        )
        growth_rate = st.slider(
            "Growth Rate", 
            -0.2, 0.2, 0.05, 0.01,
            help="Market growth rate (can be negative). "
                 "Affects how the base fee rate evolves over time."
        )
        volatility = st.slider(
            "Volatility", 
            0.0, 0.5, 0.2, 0.01,
            help="Market volatility. Higher values create more random variation "
                 "in fee generation and market conditions."
        )
    
    with st.sidebar.expander("General Parameters"):
        base_stake_rate = st.slider(
            "Base Stake Rate", 
            0.1, 0.9, 0.5, 0.1,
            help="Base rate at which new staking events occur. "
                 "Modified by market conditions and system performance."
        )
        initial_token_supply = st.number_input(
            "Initial Token Supply", 
            value=1_000_000, 
            step=100_000,
            help="Initial total supply of tokens in the system."
        )
        epochs = st.number_input(
            "Epochs to Simulate", 
            value=120, 
            step=10,
            help="Number of time periods to simulate. Each epoch represents "
                 "a discrete time step where system state is updated."
        )

    vesting_duration = None
    with st.sidebar.expander("Emission Parameters"):
        initial_emission = st.number_input(
            "Initial Emission Rate", 
            value=1000, 
            step=100,
            help="Initial rate of new token emissions per epoch. "
                 "These tokens are distributed to pods based on votes."
        )
        decay_rate = st.slider(
            "Emission Decay Rate", 
            0.0, 0.2, 0.05, 0.01,
            help="Rate at which emissions decrease over time. "
                 "Higher values create faster reduction in new token issuance."
        )
        enable_vesting = st.checkbox(
            "Enable Emission Vesting", 
            value=False,
            help="When enabled, emissions are vested over time rather than "
                 "being immediately available."
        )
        if enable_vesting:
            vesting_duration = st.number_input(
                "Vesting Duration (epochs)", 
                value=13, 
                step=1, 
                help="Number of epochs over which emissions vest linearly. "
                     "Longer durations create smoother token distribution."
            )

    return {
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
            "decay_rate": decay_rate,
            "vesting_duration": vesting_duration
        }
    }

def create_simulation(inputs):
    def emission_schedule(state):
        return inputs["emissions"]["initial_rate"] * (
            (1 - inputs["emissions"]["decay_rate"]) ** state.epoch
        )
    
    initial_pods = [f"pod{i+1}" for i in range(inputs["pods"]["num_pods"])]
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
        initial_pods=initial_pods,
        initial_token_supply=inputs["general"]["initial_token_supply"],
        epochs=inputs["general"]["epochs"],
        market=inputs["market"],
        emission_vesting_duration=inputs["emissions"].get("vesting_duration", None),
        emission_schedule=emission_schedule,
        # TODO: these should be configurable
        fcu_duration=10,
        fcu_delay={pod: 0 for pod in initial_pods},
    )
    
    return VeTokenomicsSimulation(params)

def create_pod_metrics_tab(history):
    st.header("Pod Performance Metrics")
    
    df = pd.DataFrame(history)
    
    # Create a DataFrame for pod-specific metrics
    pod_data = []
    for epoch_data in history:
        metrics = epoch_data['metrics']
        total_epoch_emissions = metrics.get('emissions_this_epoch', 0)
        
        for pod_name, emissions in epoch_data['pod_emissions'].items():
            pod_fee_metrics = metrics.get('pod_fee_metrics', {}).get(pod_name, {})
            pod_data.append({
                'epoch': epoch_data['epoch'],
                'pod': pod_name,
                'emissions': emissions,
                'total_epoch_emissions': total_epoch_emissions,  # Add total emissions
                'votes': metrics.get('vote_distribution', {}).get(pod_name, 0),
                'fees': metrics.get('pod_fees', {}).get(pod_name, 0),
                'fcus': metrics.get('pod_fcus', {}).get(pod_name, 0),
                'cumulative_fees': metrics.get('cumulative_pod_fees', {}).get(pod_name, 0),
                'cumulative_fcus': metrics.get('cumulative_pod_fcus', {}).get(pod_name, 0),
                'fee_rate': metrics.get('fee_generation_rate', {}).get(pod_name, 0),
                'fcu_efficiency': metrics.get('fcu_efficiency', {}).get(pod_name, 0),
                'avg_vote_share': metrics.get('avg_vote_share', {}).get(pod_name, 0),
                'active_fcus': pod_fee_metrics.get('active_fcus', 0),
                'distributed_fees': pod_fee_metrics.get('distributed_fees', 0),
                'avg_fee_per_fcu': pod_fee_metrics.get('avg_fee_per_fcu', 0),
                'fee_distribution_ratio': pod_fee_metrics.get('fee_distribution_ratio', 0)
            })
    
    pod_df = pd.DataFrame(pod_data)
    
    # Create a separate DataFrame for total emissions
    total_emissions_df = pod_df.groupby('epoch')[['total_epoch_emissions']].first().reset_index()
    
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
        tabs = st.tabs(["Current", "Cumulative", "Efficiency", "Fee Distribution"])
        
        with tabs[0]:
            # Base Pod Emissions Chart
            emissions_chart = alt.Chart(pod_df).mark_line().encode(
                x=alt.X('epoch:Q', title='Epoch'),
                y=alt.Y('emissions:Q', title='Emissions'),
                color=alt.Color('pod:N', title='Pod')
            )
            
            # Total Emissions Line
            total_emissions_line = alt.Chart(total_emissions_df).mark_line(
                strokeDash=[5, 5],  # Create a dashed line
                color='red'
            ).encode(
                x='epoch:Q',
                y='total_epoch_emissions:Q'
            )
            
            # Combine the charts
            combined_emissions = (emissions_chart + total_emissions_line).properties(
                title='Pod Emissions (Red Dashed = Total Epoch Emissions)',
                width=400,
                height=300
            )
            st.altair_chart(combined_emissions, use_container_width=True)
            
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
            
        # Rest of the tabs remain the same
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

        with tabs[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Active FCUs Chart
                active_fcus_chart = alt.Chart(pod_df).mark_line().encode(
                    x=alt.X('epoch:Q', title='Epoch'),
                    y=alt.Y('active_fcus:Q', title='Active FCUs'),
                    color=alt.Color('pod:N', title='Pod')
                ).properties(
                    title='Active FCUs per Pod',
                    width=400,
                    height=300
                )
                st.altair_chart(active_fcus_chart, use_container_width=True)
                
                # Distributed Fees Chart
                distributed_fees_chart = alt.Chart(pod_df).mark_line().encode(
                    x=alt.X('epoch:Q', title='Epoch'),
                    y=alt.Y('distributed_fees:Q', title='Distributed Fees'),
                    color=alt.Color('pod:N', title='Pod')
                ).properties(
                    title='Distributed Fees per Pod',
                    width=400,
                    height=300
                )
                st.altair_chart(distributed_fees_chart, use_container_width=True)
                
            with col2:
                # Average Fee per FCU Chart
                fee_per_fcu_chart = alt.Chart(pod_df).mark_line().encode(
                    x=alt.X('epoch:Q', title='Epoch'),
                    y=alt.Y('avg_fee_per_fcu:Q', title='Average Fee per FCU'),
                    color=alt.Color('pod:N', title='Pod')
                ).properties(
                    title='Average Fee per FCU',
                    width=400,
                    height=300
                )
                st.altair_chart(fee_per_fcu_chart, use_container_width=True)
                
                # Fee Distribution Ratio Chart
                ratio_chart = alt.Chart(pod_df).mark_line().encode(
                    x=alt.X('epoch:Q', title='Epoch'),
                    y=alt.Y('fee_distribution_ratio:Q', 
                        title='Distribution Ratio',
                        scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('pod:N', title='Pod')
                ).properties(
                    title='Fee Distribution Ratio',
                    width=400,
                    height=300
                )
                st.altair_chart(ratio_chart, use_container_width=True)

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
            'Total Emissions': df['metrics'].apply(lambda x: x['total_emissions']),
            'Vested Emissions': df['metrics'].apply(lambda x: x['total_vested']),
            'Unvested Emissions': df['metrics'].apply(lambda x: x.get('unvested_emissions', 0))
        }).melt(
            id_vars=['epoch'],
            value_vars=['Total Fees', 'Total Emissions', 'Vested Emissions', 'Unvested Emissions'],
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

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        with st.form("auth_form"):
            st.text_input("Enter password:", type="password", key="password_input")
            if not st.secrets.get("APP_PASSWORD"):
                st.error("No password set in secrets. Please set the APP_PASSWORD in the secrets manager.")
                st.stop()
            
            submitted = st.form_submit_button("Login")
            if submitted:
                if st.session_state.password_input == st.secrets["APP_PASSWORD"]:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
                    st.stop()
        st.stop()

def main():
    check_password()

    st.title("VeTokenomics Simulation")
    
    inputs = create_simulation_inputs()
    
    if st.sidebar.button("Run Simulation"):
        sim = create_simulation(inputs)
        
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