import streamlit as st
import TissDataScience as tds
import chart
from PIL import Image

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

image = Image.open('tiss.jpg')
col1, col2, col3 = st.columns([3,2,3])
with col2:
    st.image(image, width=150)
st.markdown("<h1 style='text-align: center; \
            color: #ac051b;'>Tiss</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; \
            color: #ac051b;'>Data Science Team</h5>", unsafe_allow_html=True)

space(2)

st.markdown("<h3 style='text-align: left; \
            color: black;'>Ontario Demand Chart</h3>", unsafe_allow_html=True)

temp = st.empty()
form = temp.form(key="login_form", clear_on_submit=True)
username = form.text_input('Username',max_chars=12, placeholder='your username')
password = form.text_input('Password',max_chars=8, type="password")
submit = form.form_submit_button('Login')
# return username, password, submit

# login_blocks = generate_login_block()
if username=='mehdi' and password=='1234567':
    temp.empty()
    
    op = tds.Operation()
    d = tds.Data()
    on = tds.Ontario_pipeline()
    
            
    @st.cache
    def results():
        daily_demand = op.daily_demand_prediction()
        # daily_demand.set_index('time', inplace=True)
        daily_demand['ieso_demands'] = daily_demand['ieso_demands'].shift(-1)
        return daily_demand
    
    
    @st.cache
    def pred_pick(df=None,x=None):
        peak_chance = op.predict_peak(df,x)
        return peak_chance
    
    @st.cache
    def history_results():
        data = d.merged_history_data()
        daily_demand = op.daily_demand_prediction(data)
        # peak_chance = op.predict_peak(daily_demand)
        # daily_demand.set_index('time', inplace=True)
        daily_demand['ieso_demands'] = daily_demand['ieso_demands'].shift(-1)
        return daily_demand
    
    def melt_df(data, label=None):
        if label == None:
            label = ['hour_time', 'ieso_demands', 'demand_predicted']
        temp = data[label].copy().set_index('hour_time')
        source = temp.reset_index().melt('hour_time',
                                         var_name='symbol', value_name='demand')
        return source
    
    
    start = d.toronto_time()
    
    history_demand = history_results().copy()
    history_demand['hour_time'] =history_demand['time'].dt.hour
    history_demand = history_demand[history_demand['time'].dt.day == start.day]
    history_demand.dropna(subset=['ieso_demands'], inplace=True)
    
    daily_demand = results().copy()
    daily_demand['hour_time'] =daily_demand['time'].dt.hour
    daily_demand = daily_demand[daily_demand['time'].dt.day == start.day]
    
    melted_daily = melt_df(daily_demand)
    
    melted_historical_data = melt_df(history_demand,
                                     ['hour_time', 'ieso_demands',
                                      'demand_predicted', 'Ontario Demand'])

    
    merge_data_melted = melted_daily.append(melted_historical_data)
    merge_data_melted.sort_values(by=['symbol', 'hour_time'], inplace=True)
    
    current_day_df = daily_demand.append(history_demand).sort_values(by=['time'])
    peak_chance = int(pred_pick(current_day_df))
    
    all_symbols = merge_data_melted.symbol.unique()
    symbols = st.multiselect("Choose source to visualize", all_symbols, all_symbols[:3])
    merge_data_melted = merge_data_melted[merge_data_melted.symbol.isin(symbols)]
    
    chart_merged = chart.get_chart(merge_data_melted)
    st.altair_chart(chart_merged, use_container_width=True)
        
    space(2)
    
    st.markdown("<h3 style='text-align: left; \
                color: black;'>Peak Window selector</h3>", unsafe_allow_html=True)
    
    option = st.selectbox(
         'please select a value for window',
         (1, 2, 3, 4),0)
    
    window = op.window(option, daily_demand)
    st.write(window)
    
    space(2)
    
    st.markdown("<h3 style='text-align: left; \
                color: black;'>Chance of Peak per Day</h3>", unsafe_allow_html=True)
                
    df1 = op.predict_peak(monthly=True)
    st.dataframe(df1)
    list_chances = list(df1.prob)
    list_chances = [round(num, 1) for num in list_chances]
    
    space(2)
    
    st.markdown("<h3 style='text-align: left; \
                color: black;'>Chance of Peak per Month</h3>", unsafe_allow_html=True)
    
    col1 , col2, col3,col4,col5 = st.columns(5)
    col1.metric(label="{}-{}-{}".format(start.year,start.month,start.day),
                value="{} %".format(list_chances[0]))
    col2.metric(label="{}-{}-{}".format(start.year,start.month,start.day+1),
                value="{} %".format(list_chances[1]), delta="{} %".format(round(list_chances[1]-list_chances[0],1)))
    col3.metric(label="{}-{}-{}".format(start.year,start.month,start.day+2),
                value="{} %".format(list_chances[2]), delta="{} %".format(round(list_chances[2]-list_chances[1],1)))
    
    col4.metric(label="{}-{}-{}".format(start.year,start.month,start.day+3),
                value="{} %".format(list_chances[3]), delta="{} %".format(round(list_chances[3]-list_chances[2],1)))
    col5.metric(label="{}-{}-{}".format(start.year,start.month,start.day+4),
                value="{} %".format(list_chances[4]), delta="{} %".format(round(list_chances[4]-list_chances[3],1)))
    
    @st.cache
    def pred_next_day():
        test = on.predict_next_days()
        return test
    
    pred_next = pred_next_day().copy()
    st.dataframe(pred_next)
    
    list_demand_yearly = list()
    for i in range(len(pred_next)):
        
        list_demand_yearly.append(int(pred_next['demand_predicted'][i]))

        
    list_chance_yearly = list()
    for i in range(len(list_demand_yearly)):
        list_chance_yearly.append(round(pred_pick(x=list_demand_yearly[i]),1))
        
    space(2)
    
    st.markdown("<h3 style='text-align: left; \
                color: black;'>Chance of Peak per Year</h3>", unsafe_allow_html=True)
    
    col1 , col2, col3,col4,col5 = st.columns(5)
    col1.metric(label="{}-{}-{}".format(start.year,start.month,start.day),
                value="{} %".format(list_chance_yearly[0]))
    col2.metric(label="{}-{}-{}".format(start.year,start.month,start.day+1),
                value="{} %".format(list_chance_yearly[1]), delta="{} %".format(round(list_chance_yearly[1]-list_chance_yearly[0],1)))
    col3.metric(label="{}-{}-{}".format(start.year,start.month,start.day+2),
                value="{} %".format(list_chance_yearly[2]), delta="{} %".format(round(list_chance_yearly[2]-list_chance_yearly[1],1)))
    
    col4.metric(label="{}-{}-{}".format(start.year,start.month,start.day+3),
                value="{} %".format(list_chance_yearly[3]), delta="{} %".format(round(list_chance_yearly[3]-list_chance_yearly[2],1)))
    col5.metric(label="{}-{}-{}".format(start.year,start.month,start.day+4),
                value="{} %".format(list_chance_yearly[4]), delta="{} %".format(round(list_chance_yearly[4]-list_chance_yearly[3],1)))

    
    space(2)
    
    st.markdown("<h3 style='text-align: left; \
                color: black;'>Raw Data</h3>", unsafe_allow_html=True)
    
    st.dataframe(current_day_df)
    
    
    
    





















