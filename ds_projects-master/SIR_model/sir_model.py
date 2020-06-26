import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def plot_sir(time_grid, infectious, susceptible, recovered):
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(time_grid, susceptible, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(time_grid, infectious, 'r', alpha=0.5, lw=2, label='Infectious')
    ax.plot(time_grid, recovered, 'g', alpha=0.5, lw=2, label='Recovered')

    ax.set_xlabel('Days')
    ax.set_ylabel('Number of People')

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)


    plt.show()


def get_initial_susceptible(total_population, initial_infected, initial_recovered):
    return total_population - (initial_infected + initial_recovered)


def get_beta(avg_num_contacts_per_person, proba_disease_transmission):
    return avg_num_contacts_per_person * proba_disease_transmission


def get_mean_recov_rate(recovery_period_days):
    return 1.0 / recovery_period_days


def deriv_susceptible_wrt_time(beta, susceptible, infectious, total_population):
    '''  
    assumption: susceptible will always decrease assumed immunity
    '''
    return -beta * susceptible * (infectious / total_population)


def deriv_infected_wrt_time(beta, susceptible, infectious, total_population, mean_recovery_rate):
    return beta * susceptible * (infectious / total_population) - (mean_recovery_rate * infectious)


def deriv_recovered_wrt_time(mean_recovery_rate, infectious):
    ''' 
    assume recovered never decreases
    '''
    return mean_recovery_rate * infectious


def derivatives_helper(initial_conditions, time_grid, total_population, beta, mean_recovery_rate):
    ''' 
    facilitator for odeint
    notes: time_grid param is not explicitly used
        initial_recovered not explicitly used
    '''
    susceptible, infectious, recovered = initial_conditions 
    dSdt = deriv_susceptible_wrt_time(beta, susceptible, infectious, total_population)
    dIdt = deriv_infected_wrt_time(beta, susceptible, infectious, total_population, mean_recovery_rate)
    dRdt = deriv_recovered_wrt_time(mean_recovery_rate, infectious)

    return dSdt, dIdt, dRdt

if __name__ ==  "__main__":
    '''set params'''
    total_population = 100000
    days = 200
    initial_infected = 100
    initial_recovered = 0

    # params to build beta (transmission rate)
    avg_num_contacts_per_person = 5 #per day
    proba_disease_transmission = 0.04 

    # param to build gamma
    recovery_period_days= 10


    '''calculations'''
    initial_susceptible = get_initial_susceptible(total_population, initial_infected, initial_recovered)

    # transm rate
    beta = get_beta(avg_num_contacts_per_person, proba_disease_transmission)

    # gamma, transistion rate from infectious to recovered
    mean_recovery_rate = get_mean_recov_rate(recovery_period_days)

    # each deriv is a function of time, use numpy array for calculations
    time_grid = np.linspace(0, days, days)

    # set initial conditions for diffEQ solver
    initial_conditions = (initial_susceptible, initial_infected, initial_recovered)

    # integrate SIR equations over time grid
    integrated_functions = odeint(derivatives_helper, initial_conditions, time_grid, args=(total_population, beta, mean_recovery_rate))

    # unpack integrated functions array
    susceptible, infectious, recovered = integrated_functions.T 
    print(susceptible.shape)

    #plot data
    plot_sir(time_grid, susceptible, infectious, recovered)

    # '''Output'''
    # params_dict = {
    #     "total_population": total_population,
    #     "days": days,
    #     "initial_infected": initial_infected,
    #     "initial_recovered": initial_susceptible,
    #     "avg_num_contacts_per_person": avg_num_contacts_per_person,
    #     "proba_disease_transmission": proba_disease_transmission,
    #     "beta": beta,
    #     "recovery_period_days":recovery_period_days,
    #     "gamma": mean_recovery_rate,
    #     "time_grid" : len(time_grid)
    # }

    # for k, v in params_dict.items():
    #     print(f'{k}: {v}')

    # call driver()
    # driver()