import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from vetoken import (
    VeTokenomicsSimulation,
    SimulationParams,
    MarketState,
    DeterministicConfig,
    PodConfig,
    DeterministicSimulation,
    TokenBuyConfig
)
from vesting import (
    VestingSchedule,
    LinearVestingSchedule,
    CliffVestingSchedule
)

st.set_page_config(layout="wide")

def create_simulation_inputs():
    st.sidebar.header("Simulation Parameters")

    st.sidebar.header("Simulation Mode")
    simulation_type = st.sidebar.radio(
        "Select Simulation Type",
        ["Stochastic", "Deterministic"]
    )
    
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
        lock_duration = st.number_input(
            "Lock Duration",
            value=52,
            step=1,
            help="Duration of the lock in epochs."
        )
        lock_interval = st.number_input(
            "Lock Interval",
            value=3,
            step=1,
            help="Interval at which new locks can be created in epochs."
        )
        
    with st.sidebar.expander("Pod Parameters"):
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
        if simulation_type == "Stochastic":
            base_fee_drift = st.slider("Base Fee Drift", 0.0, 0.2, 0.05, 0.01)
        else:
            pod_configs = []
            for i in range(num_pods):
                st.write(f"\nPod {i+1} Configuration")
                # fee_growth = st.slider(
                #     f"Fee Growth Rate (Pod {i+1})",
                #     0.0, 0.2, 0.05, 0.01
                # )
                # initial_vote = st.slider(
                #     f"Initial Vote Share (Pod {i+1})",
                #     0.0, 1.0, 1.0/num_pods, 0.01
                # )
                fee_growth = 0.05 if i == 0 else 0.25  # just for testing
                initial_vote = 0.25 if i == 0 else 0.75  # just for testing
                # pod_fcu_rates[f"pod{i+1}"] = st.slider(
                #     f"FCU Generation Rate (Pod {i+1})",
                #     0.0, 1.0, 0.1, 0.01
                # )
                pod_configs.append({
                    "fee_growth": fee_growth,
                    "initial_vote_share": initial_vote
                })
    
    with st.sidebar.expander("Market Parameters"):
        base_fee_rate = st.slider(
            "Base Fee Rate", 
            0.0, 0.5, 0.1, 0.01,
            help="Base market rate for fee generation. "
                 "This represents the overall market activity level."
        )
        if simulation_type == "Stochastic":
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
        else:
            base_stake_rate = 0.0
            fixed_stake_amount = st.number_input(
                "Fixed Stake Amount per Epoch",
                value=1000,
                step=100
            )
            fixed_stake_duration = st.number_input(
                "Fixed Stake Duration",
                value=10,
                min_value=1,
                max_value=max_lock_duration
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
            value=10, 
            step=1,
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

    with st.sidebar.expander("Token Buy Parameters"):
        token_buy_rate = st.slider(
            "Base Buy Rate",
            0.0, 10000.0, 1000.0, 100.0,
            help="Base rate of token purchases per epoch"
        )
        market_sensitivity = st.slider(
            "Market Sensitivity",
            0.0, 1.0, 0.5, 0.1,
            help="How much market performance affects buying behavior"
        )
        randomization = st.slider(
            "Randomization Factor",
            0.0, 0.5, 0.2, 0.05,
            help="Random variation in purchase amounts"
        )

    return {
        "simulation_type": simulation_type,
        "protocol": {
            "gamma": gamma,
            "alpha": alpha,
            "delta": delta,
            "omega": omega,
        },
        "pods": {
            "num_pods": num_pods,
            # "max_lock_duration": max_lock_duration,
            # "min_lock_duration": min_lock_duration,
            "lock_duration": lock_duration,
            "lock_interval": lock_interval,
            "base_fee_drift": base_fee_drift if simulation_type == "Stochastic" else 0.0,
            "pod_configs": pod_configs if simulation_type == "Deterministic" else None,
            
        },
        "market": MarketState(
            base_fee_rate=base_fee_rate,
            growth_rate=growth_rate if simulation_type == "Stochastic" else 0.0,
            volatility=volatility if simulation_type == "Stochastic" else 0.0
        ),
        "general": {
            "initial_token_supply": initial_token_supply,
            "epochs": epochs
        },
        "emissions": {
            "initial_rate": initial_emission,
            "decay_rate": decay_rate,
            "vesting_duration": vesting_duration
        },
        "staking": {
            "base_stake_rate": base_stake_rate,
            "fixed_amount": fixed_stake_amount if simulation_type == "Deterministic" else 0.0,
            "fixed_duration": fixed_stake_duration if simulation_type == "Deterministic" else 0.0
        },
        "token_buy": {
            "base_buy_rate": token_buy_rate,
            "market_sensitivity": market_sensitivity,
            "randomization": randomization
        }
    }

def create_simulation(config):
    def emission_schedule(state):
        return config["emissions"]["initial_rate"] * (
            (1 - config["emissions"]["decay_rate"]) ** state.epoch
        )
    
    initial_pods = [f"pod{i+1}" for i in range(config["pods"]["num_pods"])]
    params = SimulationParams(
        gamma=config["protocol"]["gamma"],
        alpha=config["protocol"]["alpha"],
        delta=config["protocol"]["delta"],
        omega=config["protocol"]["omega"],
        fee_volatility=0.1,
        base_stake_rate=config["staking"]["base_stake_rate"],
        base_fee_drift=config["pods"]["base_fee_drift"],
        # max_lock_duration=config["pods"]["max_lock_duration"],
        # min_lock_duration=config["pods"]["min_lock_duration"],
        lock_duration=config["pods"]["lock_duration"],
        lock_interval=config["pods"]["lock_interval"],
        initial_pods=initial_pods,
        initial_token_supply=config["general"]["initial_token_supply"],
        epochs=config["general"]["epochs"],
        market=config["market"],
        emission_vesting_duration=config["emissions"].get("vesting_duration", None),
        emission_schedule=emission_schedule,
        # TODO: these should be configurable
        fcu_duration=10,
        fcu_delay={pod: 0 for pod in initial_pods},
        pod_fcu_rates={pod: 1.0 for pod in initial_pods}  # TODO: make this configurable
    )
    
    if config["simulation_type"] == "Deterministic":
        det_config = DeterministicConfig(
            staking_amount_per_interval=config["staking"]["fixed_amount"],
            pods={
                pod: PodConfig(
                    fee_growth=pod_cfg["fee_growth"],
                    initial_vote_share=pod_cfg["initial_vote_share"]
                )
                for pod, pod_cfg in zip(initial_pods, config["pods"]["pod_configs"])
            },
            market_growth=config["market"].growth_rate
        )
        sim = DeterministicSimulation(params, det_config)
    else:
        sim = VeTokenomicsSimulation(params)
    
    sim.token_buy_config = TokenBuyConfig(
        base_buy_rate=config["token_buys"]["base_buy_rate"],
        market_sensitivity=config["token_buys"]["market_sensitivity"],
        randomization_factor=config["token_buys"]["randomization_factor"]
    )
    return sim

def create_pod_metrics_tab(history):
    st.header("Pod Performance Metrics")
    
    df = pd.DataFrame(history)
    
    # Create a DataFrame for pod-specific metrics
    pod_data = []
    for epoch_data in history:
        metrics = epoch_data['metrics']
        current_emissions = metrics['emissions']['current']
        total_emissions = metrics['emissions']['total']
        vested_emissions = metrics['emissions'].get('vested', 0)  # Optional
        unvested_emissions = metrics['emissions'].get('unvested', 0)  # Optional
        
        for pod_name, pod_metrics in metrics['pods'].items():
            pod_data.append({
                'epoch': epoch_data['epoch'],
                'pod': pod_name,
                'emissions': pod_metrics['emissions']['current'],
                'cumulative_emissions': pod_metrics['emissions']['total'],
                'system_emissions': current_emissions,
                'system_total_emissions': total_emissions,
                'vested_emissions': vested_emissions,
                'unvested_emissions': unvested_emissions,
                'votes': pod_metrics['votes']['current'],
                'fees': pod_metrics['fees']['current'],
                'fcus': pod_metrics['fcus']['current'],
                'cumulative_fees': pod_metrics['fees']['total'],
                'cumulative_fcus': pod_metrics['fcus']['total'],
                'fee_rate': pod_metrics['efficiency']['fee_rate'],
                'fcu_efficiency': pod_metrics['efficiency']['fcu_rate'],
                'active_fcus': pod_metrics['fcus']['active'],
                'distributed_fees': pod_metrics['fees']['distributed']
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
        tabs = st.tabs(["Current", "Cumulative", "Efficiency", "Fee Distribution"])
        
        with tabs[0]:
            # Pod Emissions with System Total
            base_emissions = alt.Chart(pod_df).mark_line().encode(
                x='epoch:Q',
                y='emissions:Q',
                color=alt.Color('pod:N', title='Pod')
            )
            
            system_emissions = alt.Chart(pod_df.groupby('epoch')['system_emissions'].first().reset_index()).mark_line(
                strokeDash=[5, 5],
                color='red'
            ).encode(
                x='epoch:Q',
                y='system_emissions:Q'
            )
            
            emissions_chart = (base_emissions + system_emissions).properties(
                title='Pod Emissions (Red Dashed = Total System Emissions)',
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
            # Cumulative Pod Emissions with System Total
            base_cum_emissions = alt.Chart(pod_df).mark_line().encode(
                x='epoch:Q',
                y='cumulative_emissions:Q',
                color=alt.Color('pod:N', title='Pod')
            )
            
            system_cum_emissions = alt.Chart(pod_df.groupby('epoch')['system_total_emissions'].first().reset_index()).mark_line(
                strokeDash=[5, 5],
                color='red'
            ).encode(
                x='epoch:Q',
                y='system_total_emissions:Q'
            )
            
            if 'vested_emissions' in pod_df.columns:
                vested_emissions = alt.Chart(pod_df.groupby('epoch')['vested_emissions'].first().reset_index()).mark_line(
                    strokeDash=[2, 2],
                    color='green'
                ).encode(
                    x='epoch:Q',
                    y='vested_emissions:Q'
                )
                cum_emissions_chart = (base_cum_emissions + system_cum_emissions + vested_emissions).properties(
                    title='Cumulative Pod Emissions (Red = System Total, Green = Vested)',
                    width=400,
                    height=300
                )
            else:
                cum_emissions_chart = (base_cum_emissions + system_cum_emissions).properties(
                    title='Cumulative Pod Emissions (Red Dashed = System Total)',
                    width=400,
                    height=300
                )
            
            # st.altair_chart(cum_emissions_chart, use_container_width=True)
            st.altair_chart(base_cum_emissions + system_cum_emissions, use_container_width=True)
            
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
    # Calculate total fees across all pods
    df['fee_efficiency'] = df['metrics'].apply(
        lambda x: sum(pod['fees']['total'] for pod in x['pods'].values())
    ) / df['tvl'].where(df['tvl'] > 0, 1)
    
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
    
    with col2:
        comparison_df = pd.DataFrame({
            'epoch': df['epoch'],
            'Total Fees': df['metrics'].apply(lambda x: x['pods'].get('total_fees', 0)),
            'Total Emissions': df['metrics'].apply(lambda x: x['emissions']['total']),
            'Vested Emissions': df['metrics'].apply(lambda x: x['emissions']['vested']),
            'Unvested Emissions': df['metrics'].apply(lambda x: x['emissions']['unvested'])
        }).melt(
            id_vars=['epoch'],
            value_vars=['Total Fees', 'Total Emissions', 'Vested Emissions', 'Unvested Emissions'],
            var_name='metric',
            value_name='value'
        )
        
        vesting_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('metrics.emissions.vested:Q', title='Amount'),
        ).properties(
            title='Cumulative Vested Tokens',
            width=400,
            height=300
        )

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
    # check_password()
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