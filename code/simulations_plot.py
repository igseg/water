import pandas as pd
import numpy  as np
from data_tools import load_data
from Models.models import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

from parameters import *

# ts = load_data()
#
# ts['water_prices'] = np.zeros(ts.shape[0])
# ts.price_power = ts.price_power / 100
# ts = ts.set_index(pd.to_datetime(ts.index.values))
#
# prev_time = pd.Timestamp(ts.index.values[-1])
# print(ts.loc[prev_time,  'water_prices'])
#
# ##### Set Water Price Level #####
# ts.loc[prev_time,  'water_prices'] = 100 * 1000
#
#
#
# N = 10 # For the storage grid
# M = 10 # For the price grid
# W = 10 # For the water price grid
# L = 10 # For the outflow grid
# S = 12 # For the states
#
# correlation = 0

loaded_policy, loaded_V0, cashflow_e, cashflow_w = save_load_policy_V0(S,N,M,W, load_results = True)

time_0 = 0
last_date = ts.index.values[-1]

time_steps = 60

def simulating_paths(loaded_policy,
                     T = 10000,
                     t0 = time_0,
                     storage_0  = 0,
                     price_e_0  = 0,
                     inflow_0   = 0,
                     price_w_0  = 0,
                     cashflow_e = np.zeros((S,N,M,W)), cashflow_w = np.zeros((S,N,M,W)),
                     time_steps = 60):

    z_s = np.random.randn(T,time_steps)
    z_p_hat = np.random.randn(T,time_steps)
    z_p = correlation * z_s + np.sqrt(1-correlation**2) * z_p_hat
    z_w = np.random.randn(T,time_steps)

    storage = np.zeros((T,time_steps+1))
    storage[:,0] = storage_0

    inflow = np.zeros((T,time_steps+1))
    inflow[:,0] = inflow_0

    price_e = np.zeros((T,time_steps+1))
    price_e[:,0] = price_e_0

    price_w = np.zeros((T,time_steps+1))
    price_w[:,0] = price_w_0


    cashflow_sim_e = np.zeros((T,time_steps))
    cashflow_sim_w = np.zeros((T,time_steps))

    outflows_grid = np.zeros((T,time_steps))

    for s in range(time_steps):

        s_c = s % 12

        current_time = (time_0 + s) % 12

        storage_coord, _ = prepro_coord_array_numba(storage[:,s], storage_grid)
        price_e_coord, _ = prepro_coord_array_numba(price_e[:,s], price_grid_e)
        price_w_coord, _ = prepro_coord_array_numba(price_w[:,s], price_grid_w)

        storage_coord = coord_array_numba(storage_coord, storage_grid, step_storage)
        price_e_coord = coord_array_numba(price_e_coord, price_grid_e, step_price_e)
        price_w_coord = coord_array_numba(price_w_coord, price_grid_w, step_price_w)

        outflows = loaded_policy[s_c,:,:,:][storage_coord, price_e_coord, price_w_coord]
        outflows_grid[:,s] = outflows

        # input_next_storage = [model_inflow , model_evaporation, current_time,
                                              # storage[:,s], outflows, model_storage] # Outflow changed from IV loop


        cashflow_sim_w[:,s] = cashflow_w[s_c,:,:,:][storage_coord, price_e_coord, price_w_coord]
        cashflow_sim_e[:,s] = cashflow_e[s_c,:,:,:][storage_coord, price_e_coord, price_w_coord]


        # storage[:,s+1] = next_storage(*input_next_storage, z_s[:,s])
        # inflow[:, s+1] = model_inflow(inflow[:,s], current_time)
        # price_e[:,s+1] = next_price_2_rever_numba(current_time, price_e[:,s], z_p[:,s]) # adjust the price model
        # price_w[:,s+1] = next_water_price_numba(price_w[:,s], (current_time - 1) % 12, z_w[:,s])
        storage[:,s+1] = next_storage_numba(current_time+1, storage[:,s], outflows, z_s[:,s])
        inflow[:, s+1] = model_inflow_numba(current_time+1)
        price_e[:,s+1] = next_price_e(current_time, price_e[:,s], z_p[:,s]) # adjust the price model
        price_w[:,s+1] = next_water_price(price_w[:,s], z_w[:,s])

    # print(cashflow_e)
    # print(cashflow_sim_e)
    return storage, price_e, price_w, outflows_grid, inflow, cashflow_sim_e, cashflow_sim_w

def plot_conf_int_boot( l,
                        sample,
                        quantiles_2 = [0.005, 0.995],
                        quantiles = [0.05, 0.95],
                        var_name = 'noname',
                        time_steps = 60):

    T = sample.shape[0]
    S = sample.shape[1]

    i_5  = int(T*quantiles[0])
    i_95 = int(T*quantiles[1])
    i_50 = int(T*0.5)
    i_05 = int(T*quantiles_2[0])
    i_995= int(T*quantiles_2[1])

    S_bis = time_steps + 1
    toptop = np.zeros(S_bis)
    top = np.zeros(S_bis)
    med = np.zeros(S_bis)
    bot = np.zeros(S_bis)
    botbot = np.zeros(S_bis)

    for t in range(S_bis):

        sample_t = sorted(sample[:,t])

        toptop[t] = sample_t[i_995]
        top[t] = sample_t[i_95]
        med[t] = sample_t[i_50]
        bot[t] = sample_t[i_5]
        botbot[t] = sample_t[i_05]


    fig, (ax) = plt.subplots(1, 1, sharex=True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval= 6))

    ax.plot(l, med , color='black', label = 'Median')
    ax.fill_between(l, bot, top, where=top >= bot, facecolor='green', interpolate=True, alpha = 0.7, label= '5% - 95% confidence interval')
    ax.fill_between(l, botbot, bot, where=bot >= botbot, color='#70d567', interpolate=True, alpha = 0.7, label= '0.5% - 99.5% confidence interval')
    ax.fill_between(l, top, toptop, where=toptop >= top, color='#70d567', interpolate=True, alpha = 0.7)

    plt.gcf().autofmt_xdate()
    plt.legend()
    #plt.title(f'Forward simulation: {T} trajectories ',fontdict={'size':'18'})
    plt.ylabel(var_name,fontdict={'size':'16'})
    plt.xlabel('Months',fontdict={'size':'16'})
    plt.grid()
    #plt.tight_layout()

    return None

loaded_policy, loaded_V0, cashflow_e, cashflow_w = save_load_policy_V0(S, N, M , W, load_results = True)

storage_sim, price_e_sim, price_w_sim, outflows_sim, inflow, sim_cf_e, sim_cf_w = simulating_paths(loaded_policy,
                                                                                                   T = 10000,
                                                                                                   t0 = time_0,
                                                                                                   storage_0  = ts.loc[last_date, 'storage'],
                                                                                                   price_e_0  = ts.loc[last_date, 'price_power'],
                                                                                                   inflow_0   = ts.loc[last_date, 'inflow'],
                                                                                                   price_w_0  = ts.loc[last_date, 'water_prices'],
                                                                                                   cashflow_e = cashflow_e, cashflow_w = cashflow_w,
                                                                                                   time_steps = time_steps)

l = []
for s in range(time_steps + 1): # +1 because the also have the initial value, goto 50
    if s % 12 == 0:
        l.append(f'Dec{19+math.ceil(s/12)}')
    if s % 12 == 1:
        l.append(f'Jan{19+math.ceil(s/12)}')
    if s % 12 == 2:
        l.append(f'Feb{19+math.ceil(s/12)}')
    if s % 12 == 3:
        l.append(f'Mar{19+math.ceil(s/12)}')
    if s % 12 == 4:
        l.append(f'Apr{19+math.ceil(s/12)}')
    if s % 12 == 5:
        l.append(f'May{19+math.ceil(s/12)}')
    if s % 12 == 6:
        l.append(f'Jun{19+math.ceil(s/12)}')
    if s % 12 == 7:
        l.append(f'Jul{19+math.ceil(s/12)}')
    if s % 12 == 8:
        l.append(f'Aug{19+math.ceil(s/12)}')
    if s % 12 == 9:
        l.append(f'Sep{19+math.ceil(s/12)}')
    if s % 12 == 10:
        l.append(f'Oct{19+math.ceil(s/12)}')
    if s % 12 == 11:
        l.append(f'Nov{19+math.ceil(s/12)}')


cut = sim_cf_e.mean(axis= 0) / (sim_cf_e.mean(axis= 0) +   sim_cf_w.mean(axis= 0))

bot = np.zeros(len(cut))
top = np.ones(len(cut))

fig, (ax) = plt.subplots(1, 1, sharex=True)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

ax.fill_between(l[:-1], bot, cut, where=cut >= bot, facecolor='gold', interpolate = True, alpha = 0.7, label= 'electricity')
ax.fill_between(l[:-1], top, cut, where=top >= cut, facecolor='blue', interpolate = True, alpha = 0.7, label= 'water')

plt.legend()
plt.title('Average cashflow ratio of the simulation',fontdict={'size':'18'})

plt.gcf().autofmt_xdate()
plt.tight_layout()
# plt.savefig('Figures/paper/cf_ratios.pdf', format='pdf')

# plt.show()

plot_conf_int_boot(l, storage_sim, var_name = 'Inventory (MCM)', time_steps = time_steps)
plt.title('Facility Inventory simulation',fontdict={'size':'20'})
plt.xticks(range(time_steps + 1), labels=l)
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()

# if one_component:
#     if mr:
#         plt.savefig('Figures/paper/inventory_conf_int.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/inventory_conf_int_nonmr.pdf', format='pdf')
# else:
#     if mr:
#         plt.savefig('Figures/paper/inventory_conf_int_2.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/inventory_conf_int_nonmr_2.pdf', format='pdf')

plot_conf_int_boot(l, price_e_sim, var_name = 'Electricity Price ($/kwh)', time_steps = time_steps)
plt.title('Electricity price simulation',fontdict={'size':'20'})
plt.xticks(range(time_steps + 1), labels=l)
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()

# if one_component:
#     if mr:
#         plt.savefig('Figures/paper/price_e_conf_int.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/price_e_conf_int_nonmr.pdf', format='pdf')
# else:
#     if mr:
#         plt.savefig('Figures/paper/price_e_conf_int_2.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/price_e_conf_int_nonmr_2.pdf', format='pdf')


plot_conf_int_boot(l, price_w_sim, var_name = 'Water Price ($ / MCM)', time_steps = time_steps)
plt.title('Water price simulation',fontdict={'size':'20'})
plt.xticks(range(time_steps + 1), labels=l)
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()

# if one_component:
#     if mr:
#         plt.savefig('Figures/paper/price_w_conf_int.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/price_w_conf_int_nonmr.pdf', format='pdf')
# else:
#     if mr:
#         plt.savefig('Figures/paper/price_w_conf_int_2.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/price_w_conf_int_nonmr_2.pdf', format='pdf')

print(outflows_sim.shape)
plot_conf_int_boot(l[1:], outflows_sim, var_name = 'Outflow (MCM / month)', time_steps = time_steps - 1)
plt.title('Policy used (water outflow)',fontdict={'size':'20'})
plt.xticks(range(time_steps), labels=l[1:])
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()

# if one_component:
#     if mr:
#         plt.savefig('Figures/paper/outflow_conf_int.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/outflow_conf_int_nonmr.pdf', format='pdf')
# else:
#     if mr:
#         plt.savefig('Figures/paper/outflow_conf_int_2.pdf', format='pdf')
#     else:
#         plt.savefig('Figures/paper/outflow_conf_int_nonmr_2.pdf', format='pdf')
plt.show()
