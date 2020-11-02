import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import seaborn as sns
#--------------------------------------------------------------------------------------------------------------------
# Calculation returns
#--------------------------------------------------------------------------------------------------------------------
def compute_returns(s):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compute_returns)
    elif isinstance(s,pd.Series):
        return (s/s.shift(1) - 1)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
# calculation of annualized return
def annualize_return(s, periods_per_year):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_return, periods_per_year = periods_per_year)
    elif isinstance(s, pd.Series):
        ret = compute_returns(s).dropna()
        growth = (1+ret).prod()
        n_period_growth = ret.shape[0]
        return growth**(periods_per_year/n_period_growth) - 1
def compound(s):
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod() - 1
    # Note that this is equivalent to (but slower than)
    # return np.expm1( np.logp1(s).sum() )
def compound_returns(s, start=100):
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compound_returns, start=start )
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
def compute_logreturns(s):
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_logreturns )
    elif isinstance(s, pd.Series):
        return np.log( s / s.shift(1) )
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
def drawdown(rets: pd.Series, start=1000):
    wealth_index   = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
    df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
    return df
def annualize_vol(s, periods_per_year):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_vol, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        return s.std() * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s) * (periods_per_year)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (periods_per_year)**(0.5)
def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
    if isinstance(s, pd.DataFrame):
        return s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    elif isinstance(s, pd.Series):
        # convert the annual risk free rate to the period assuming that:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # now, annualize the excess return
        ann_ex_rets = annualize_rets(excess_return, periods_per_year)
        # compute annualized volatility
        ann_vol = annualize_vol(s, periods_per_year)
        return ann_ex_rets / ann_vol
    elif isinstance(s, (int,float)) and v is not None:
        # Portfolio case: s is supposed to be the single (already annnualized) 
        # return of the portfolio and v to be the single (already annualized) volatility. 
        return (s - risk_free_rate) / v
#---------------------------------------------------------------------------------------------------------------------------
# portfoilio optimization section
#---------------------------------------------------------------------------------------------------------------------------
# computing portfolio return
def portfolio_return(weights, vec_ret):
    return np.dot(weights, vec_ret)
#computing portfolio volatility
def portfolio_volatility(weights, cov_rets):
    return ( np.dot(weights.T, np.dot(cov_rets, weights)) )**(0.5) 
# computing portfolios random walk
def portfolios(df, n_portfolio, periods_per_year, n_assets, risk_free_rate):
    portfolios = pd.DataFrame(columns=["return", "volatility", "sharpe ratio"])
    for i in range(n_portfolio):
        weights = np.random.random(n_assets)
        weights/= np.sum(weights)
        # computing the return
        ret_ = compute_returns(df).dropna()
        ann_ret = annualize_return(df, periods_per_year)
        port_ret = portfolio_return(weights, ann_ret)
        # computing the volatility
        vol_ = annualize_vol(df, periods_per_year)
        cov_ret = ret_.cov()
        port_vol = portfolio_volatility(weights, cov_ret)
        ann_port_vol = annualize_vol(port_vol, periods_per_year)
        # computing sharp ratio
        portfolio_spr = sharpe_ratio(port_ret, risk_free_rate, periods_per_year, v=port_vol)
            # create dataframe 
        portfolios = portfolios.append( {"return":port_ret, 
                                     "volatility":ann_port_vol, 
                                     "sharpe ratio":portfolio_spr},ignore_index=True)
    return portfolios
def optimal_weights(n_points, rets, covmatrix, periods_per_year):
    #target_rets = np.linspace(rets.min(), rets.max(), n_points)
    target_rets = np.linspace(0.0, rets.max(), n_points) 
    weights = [minimize_volatility(rets, covmatrix, target) for target in target_rets]
    return weights
def minimize_volatility(rets, covmatrix, target_return=None):
    n_assets = rets.shape[0]    
    # initial guess weights
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constr = (return_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    result = minimize(portfolio_volatility, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets ) # bounds of each individual weight, i.e., w between 0 and 1
    return result.x
def efficient_front(df, risk_free_rate, periods_per_year):
    trets = np.linspace(0.0, 0.40, 50)
    tvols = []
    n_assets = df.shape[1]
    n_portfolios=50
    # computing the return
    weight = np.repeat(1/n_assets, n_assets)
    bnds = ((0.0,1.0),)*n_assets
    ret_ = compute_returns(df).dropna()
    ann_ret = annualize_return(df, periods_per_year)
    rets = np.array(ann_ret)
    ptr = portfolio_return(weight, rets)
    covmatrix = ret_.cov()
    ann_ret1 = np.linspace(0.0,0.40,50)
    weights = optimal_weights(n_portfolios, ann_ret, covmatrix, periods_per_year) 
    # in alternative, if only the portfolio consists of only two assets, the weights can be: 
    #weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

    # portfolio returns
    portfolio_ret = [portfolio_return(w, ann_ret) for w in weights]
    
    # portfolio volatility
    vols          = [portfolio_volatility(w, covmatrix) for w in weights] 
    portfolio_vol = [annualize_vol(v, periods_per_year) for v in vols]
    
    # portfolio sharpe ratio
    portfolio_spr = [sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vol)]
    dt = pd.DataFrame({"volatility": portfolio_vol,
                       "return": portfolio_ret,
                       "sharpe ratio": portfolio_spr})
    dt = pd.concat([dt, pd.DataFrame(weights)],axis=1)
    return dt
def plot_ef(port_df, eff):
    plt.figure(figsize=(12,7))
    plt.scatter(port_df["volatility"], port_df["return"], c=port_df["sharpe ratio"], s=20, edgecolor=None, cmap='seismic')
    plt.plot(eff["volatility"], eff["return"], color="coral", label="Efficient frontier")
    min_volp = eff[eff["volatility"]==min(eff["volatility"])]
    max_sharpe = eff[eff["sharpe ratio"]==max(eff["sharpe ratio"])]
    plt.plot(max_sharpe["volatility"], max_sharpe["return"], 'b*', markersize=10, label='Maximum sharp ratio' )
    plt.plot(min_volp["volatility"], min_volp['return'], 'go',markersize=10, label='Minimum volatility')
    plt.colorbar(label='Sharp Ratio')
    plt.grid(True)
#-----------------------------------------------------------------------------------------------------------------------------------
# Semi-volatility Risk measures
#-----------------------------------------------------------------------------------------------------------------------------------
def semivolatility(s):
    '''
    Returns the semivolatility of a series, i.e., the volatility of
    negative returns
    '''
    return s[s<0].std(ddof=0) 