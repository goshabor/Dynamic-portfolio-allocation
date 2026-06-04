import numpy as np
import pandas as pd

class OptionPricingFFT:
    def __init__(self, S0, r, T, VG_params, CarrMadan_params):
        self.S0 = S0
        self.r = r
        self.T = T
        
        self.sigma, self.theta, self.nu = VG_params
        self.eta, self.N_FFT, self.alpha = CarrMadan_params
        
        #static params
        self.epsilon = np.pi/self.eta
        self.lambda_ = 2*np.pi/(self.eta*self.N_FFT)

        self.u_grid = self.eta*np.arange(self.N_FFT)
        self.u_shifted = self.u_grid - 1j*(self.alpha + 1)

        self.k_grid = -self.epsilon + self.lambda_*np.arange(self.N_FFT)
        self.K_grid = np.exp(self.k_grid)

        self.phase = np.exp(1j * self.epsilon * self.u_grid)
        self.call_weight = np.exp(-self.alpha*self.k_grid)/np.pi

        j = np.arange(self.N_FFT)
        self.Simpson_weight = self.eta*(3 + (-1)**(j+1) - (j==0).astype(float))/3

    def characteristic_func_VG(self, u):
        omega = np.log(1-self.theta*self.nu - 0.5*self.nu*self.sigma**2)/self.nu
        drift = np.log(self.S0) + (self.r+omega)*self.T
        return np.exp(1j*u*drift) * (1 - 1j*self.theta*self.nu*u + 0.5*self.nu*(self.sigma*u)**2)**(-self.T/self.nu)
    
    def c_hat(self):
        nominator = np.exp(-self.r*self.T)*self.characteristic_func_VG(self.u_shifted)
        denominator = self.alpha**2 + self.alpha - self.u_grid**2 + 1j*self.u_grid*(2*self.alpha + 1)
        return nominator/denominator
    
    def price_options_FFT(self):
        FFT_input = self.phase * self.Simpson_weight * self.c_hat()
        FFT_output = np.fft.fft(FFT_input)
        C_grid = self.call_weight*np.real(FFT_output)
        return self.K_grid, C_grid

class Calibration:
    def __init__(self, df, S0, r, VG_params_init, CarrMadan_params):
        self.S0 = S0
        self.r = r

        self.VG_params_init = VG_params_init
        self.CarrMadan_params = CarrMadan_params

        #data pre-processing
        df = df.copy()

        df['time_to_maturity_years'] = pd.to_numeric(df['time_to_maturity_years'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['vega'] = pd.to_numeric(df['vega'], errors='coerce').dropna()

        # df = df[df['vega'] != 0]
        # df = df[df['strike'] > 1]

        maturities = np.sort(df['time_to_maturity_years'].unique())
        strike_grid = np.sort(df['strike'].unique()) #contains all unique strikes, needed to inialize a 3D-array containing all maturities, strikes and call prices available in the market data

        relevant_columns = ['price', 'vega']
        data_numpy = np.full((len(maturities), len(strike_grid), len(relevant_columns)+1), np.nan, dtype=float)
        data_numpy[:,:,0] = strike_grid[None,:]

        for i, T in enumerate(maturities):
            mask = np.isclose(df['time_to_maturity_years'].values, T, atol=1e-8, rtol=0)
            sub_data = df.loc[mask, ['strike']+relevant_columns]

            if(sub_data.empty): continue

            strikes = sub_data['strike'].to_numpy()

            column_idx = np.searchsorted(strike_grid, strikes)
            mask2 = (column_idx < len(strike_grid)) & np.isclose(strike_grid[column_idx], strikes, atol=1e-8, rtol=0)
            
            strikes = strikes[mask2]
            column_idx = column_idx[mask2]
            sub_data = sub_data.iloc[mask2]

            for j, col in enumerate(relevant_columns, start=1):
                data_numpy[i, column_idx, j] = sub_data[col].to_numpy()
            
        self.maturities = maturities
        self.data_numpy = data_numpy
        self.strikes = strike_grid
        
        #building one FFT pricer per maturity
        self.per_T = [] ##list of dicts: 'T':..., 'pricer':..., 'idx':..., 'w':..., 'price':...}
        for i, T in enumerate(maturities):
            pricer = OptionPricingFFT(S0=self.S0, r=self.r, T=T, VG_params=self.VG_params_init, CarrMadan_params=self.CarrMadan_params)

            #selecting market data available for this maturity
            data = self.data_numpy[i]

            cols_needed = [0, 1]  #strike, price
            rows_ok = ~np.isnan(data[:, cols_needed]).any(axis=1)
            data_clean = data[rows_ok]
            if data_clean.size == 0:
                continue

            strike_i = data_clean[:, 0]
            close_i  = data_clean[:, 1]
            vega_i  = data_clean[:, 2]    

            #interpolation mapping strike -> FFT grid
            k_grid_fft = pricer.k_grid
            k_market = np.log(strike_i)

            in_range = (k_market >= k_grid_fft[0]) & (k_market <= k_grid_fft[-1])
            if(not np.any(in_range)):
                continue

            k_use = k_market[in_range]

            left_idx = np.searchsorted(k_grid_fft, k_use) - 1
            left_idx = np.clip(left_idx, 0, len(k_grid_fft) - 2)

            k_left = k_grid_fft[left_idx]
            k_right = k_grid_fft[left_idx + 1]

            w = (k_use - k_left) / (k_right - k_left)

            self.per_T.append({
                'T': float(T),
                'pricer': pricer,
                'idx': left_idx.astype(int),
                'w': w.astype(float),
                'price': close_i[in_range].astype(float),
                'vega': vega_i[in_range].astype(float)
            })
    
    def objective_func(self, VG_params):
        BIG = 1e30
        sigma, theta, nu = VG_params

        total_error = 0
        for entry in self.per_T:
            pricer = entry['pricer']

            pricer.sigma = sigma
            pricer.theta = theta
            pricer.nu = nu

            _, C_grid = pricer.price_options_FFT()

            #interpolating to market strikes using pre-computed idx and weight
            idx = entry['idx']
            weight = entry['w']
            
            C_left = C_grid[idx]
            C_right = C_grid[idx+1]
            C_interpolated = (1-weight)*C_left + weight*C_right

            close = entry['price']
            vega = entry['vega']

            rel_error = (C_interpolated-close)/(close)
            penalty = 0.0

            # if(np.isclose(0.1, nu, rtol=0.0, atol=1e-2)):
            #     penalty = abs(nu-0.5)

            total_error += np.sum(rel_error**2) + penalty

            if(not np.isfinite(total_error) or np.isnan(total_error)):
                return BIG

        return total_error