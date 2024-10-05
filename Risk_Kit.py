import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
pd.options.display.float_format = '{:.4f}'.format

def sharpe_ratio(rets:pd.DataFrame,rfr=0.03,periods_per_year=12):
    if isinstance(rets,pd.Series):
        rets=rets.to_frame()
    Annualized_rets=((1+rets).prod())**(periods_per_year/rets.shape[0])-1
    Annualized_volatility=rets.std()*np.sqrt(periods_per_year)
    sharpe_ratio=(Annualized_rets-rfr)/Annualized_volatility
    return pd.DataFrame({"Annualized Returns":Annualized_rets,
                         "Annualized Volatility":Annualized_volatility,
                         "Sharpe Ratio":sharpe_ratio})

def drawdowns(rets:pd.Series,init_inv=1000):
    wealth_index=init_inv*((1+rets).cumprod())
    last_peak=wealth_index.cummax()
    drawdown=(wealth_index-last_peak)/last_peak
    return pd.DataFrame({"Wealth Index":wealth_index,
                         "Last Peak":last_peak,
                         "Drawdown":drawdown})

def magic_moments(rets:pd.DataFrame,moment):
    """
    An Alternate function to find skewness and kurtosis
    You can always use sci py stats functions like skew() and kurtosis()
    for skewness pass moment=3
    for kurtosis pass moment=4
    it wont give the excess kurtosis
    """
    if isinstance(rets,pd.Series):
        rets=rets.to_frame()
    demeaned_rets=rets-rets.mean()
    exp=(((demeaned_rets)**moment).mean())/(rets.std(ddof=0))**moment
    return exp

def is_normal(rets:pd.DataFrame,level=0.1):
    """
    Applying jarque-bera test and checking normal or not for the p-value of levels,default=0.1
    """
    statistics,p_value=scipy.stats.jarque_bera(rets)
    return p_value>level

def semi_deviation(rets:pd.DataFrame):
    neg_rets=rets<0
    return rets[neg_rets].std(ddof=0)

def var_historic(rets,level=5):
    """
    Estimate VaR historic for certain percent of level
    ie, returns fall below this for level percent of time
    or, returns are above for 100-level percent of the time
    """
    if isinstance(rets,pd.DataFrame):
        return rets.aggregate(var_historic,level=level)
    elif isinstance(rets,pd.Series):
        return -np.percentile(rets,level)
    else:
        raise TypeError("Expected a series or Dataframe")
        
def var_assumption(rets:pd.DataFrame,level=5,modified=False):
    """
    Estimating VaR gaussian or cornish-fisher for the given percent of risk
    """
    #from scipy.stats import norm
    z_scr=norm.ppf(level/100)
    if modified:
        s=magic_moments(rets,3)
        k=magic_moments(rets,4)
        z_scr=(z_scr +
                (z_scr**2 - 1)*s/6 +
                (z_scr**3 -3*z_scr)*(k-3)/24 -
                (2*z_scr**3 - 5*z_scr)*(s**2)/36
              )
               
    return -(rets.mean()+z_scr*rets.std(ddof=0))

def historic_cvar(rets,level=5):
    if isinstance(rets,pd.Series):
        is_beyond=rets<-(var_historic(rets,level=level))
        return rets[is_beyond].mean()
    elif isinstance(rets,pd.DataFrame):
        return rets.aggregate(historic_cvar,level=level)
    else:
        raise TypeError("Expected a series or Dataframe")
        
def annual_returns(rets,periods_per_year=12):
    gm_return=((1+rets).prod())**(1/rets.shape[0])-1
    ann_rets=(1+gm_return)**periods_per_year-1
    return ann_rets

def annual_volt(rets,periods_per_year=12):
    ann_volt=rets.std()*np.sqrt(periods_per_year)
    return ann_volt

def portfolio_return(weights,rets):
    """
    Calculates portfolio returns by weights and expected returns of each assets
    """
    return weights.T @ rets

def portfolio_volt(weights,cov):
    """
    Calculates portfolio volatility by weights and covariance matrix
    """
    return np.sqrt(weights.T @ cov @ weights)

# def portfolio_curve(df,n_periods=12,n_points=100):
#     er=annual_returns(df,periods_per_year=n_periods)
#     cov_mtx=df.cov()
#     weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
#     ptf_rets=[portfolio_return(w,er) for w in weights]
#     ptf_volt=[portfolio_volt(w,cov_mtx) for w in weights]
#     rets_volt=pd.DataFrame({"Portfolio Returns" : ptf_rets,"Portfolio Volatility" : ptf_volt})
#     return rets_volt.plot.scatter(x="Portfolio Volatility",y="Portfolio Returns")

def portfolio_curve(er,cov_mtx,n_points=100):
    """
    Plots two assets portfolio curve
    """
    if er.shape[0]!=2:
        raise ValueError("This function is meant to only calculate two assets portfolio curve")
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    ptf_rets=[portfolio_return(w,er) for w in weights]
    ptf_volt=[portfolio_volt(w,cov_mtx) for w in weights]
    rets_volt=pd.DataFrame({"Portfolio Returns" : ptf_rets,
                            "Portfolio Volatility" : ptf_volt})
    return rets_volt.plot.scatter(x="Portfolio Volatility",y="Portfolio Returns")

def optimal_weights(er,cov,n_points):
    """
    This function gives returns the optimal weight allocations for efficient frontier
    """
    weights=[minimum_volt(target_return,er,cov) for target_return in np.linspace(er.min(),er.max(),n_points)]
    return weights

def minimum_volt(target_return,er,cov):
    n=er.shape[0]
    init_alloc=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n #This will keep weight from lower limit to upper limit
    
    #setting the constraints
    return_is_target={"type" : "eq",
                     "args" :(er,),
                     "fun" : lambda weights,er : portfolio_return(weights,er)-target_return}
    weights_sum_to_1={"type" : "eq",
                      "fun" : lambda weights : np.sum(weights)-1}
    weights=minimize(portfolio_volt,init_alloc,
                    args=(cov,),
                    method="SLSQP",
                    options={'disp': False},
                    constraints=(return_is_target,weights_sum_to_1),
                    bounds=bounds)
    return weights.x


def efficient_frontier(er,cov,plt=True,n_points=100,):
    """
    This function expects the series of expected returns covariance matrix of the assets and number of allocations
    """
    weights=optimal_weights(er,cov,n_points)
    ptf_rets=[portfolio_return(w,er) for w in weights]
    ptf_volt=[portfolio_volt(w,cov) for w in weights]
    r_weights = [np.round(w, 2) for w in weights]
    eff_frntr=pd.DataFrame({"Weight Alloc" : r_weights,
                            "Returns" : ptf_rets,
                            "Volatility" : ptf_volt
    })
    eff_frntr=eff_frntr[eff_frntr["Returns"]>=eff_frntr["Returns"][eff_frntr["Volatility"].idxmin()]]
    eff_frntr.reset_index(drop=True,inplace=True)
    if plt:
        return eff_frntr.plot.scatter(x="Volatility",y="Returns",legend=False)
    else:
        return eff_frntr