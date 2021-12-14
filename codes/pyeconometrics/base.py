# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from pyeconometrics.utils import norm_cdf



class Results():
    def model_description(self):
        print('='*80)
        print('%-17s %30s %1s %-20s %8s' \
            % ('Dep. Variable:', self.output, ' ', 'Pseudo R-squ.:', 
                "%.5f" % (1-self.final_ll/self.init_ll)))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Model:', self.name, ' ', 'Log-Likelihood:', "%.3f" % self.final_ll))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Method:', 'MLE', ' ', 'LL-Null:', "%.3f" % self.init_ll))
        print('%-17s %30s %1s %-20s %8s' \
            % ('No. Observations:', self.nb_obs, ' ', 'LLR p-value:',
                "%.3f" % (1-st.chi2.cdf(2 * (self.final_ll - self.init_ll),
                len(self.beta)))))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Df Model:', len(self.beta)-1, ' ', ' ', ' '))
        print('%-17s %30s %1s %-20s %8s' \
            % ('Converged:', self.converged, ' ', ' ', ' '))
        print('='*80)

    def columns_header(self):
        print('%25s %8s %8s %8s %8s %18s' \
            % (' ', 'coef', 'std err', 't', 'P>|t|', '[95.0% Conf. Int.]'))
        print('-'*80)

    def beta_description(self):
        for i,var in enumerate([x for x in self.variables if x != self.output]):
            print('%-24s %8s %8s %8s %8s %9s %9s' \
                % (var, 
                   "%.4f" % self.beta[i], 
                   "%.3f" % self.beta_se[i], 
                   "%.3f" % (self.beta[i] / self.beta_se[i]), 
                   "%.3f" % (1-st.norm.cdf(abs(self.beta[i]) / self.beta_se[i])), 
                   "%.3f" % self.confidence_interval[i, 0], 
                   "%.3f" % self.confidence_interval[i, 1])
                )
        print('-'*80)

    def sigma_description(self):
        print('%-24s %8s %8s %8s %8s %9s %9s' \
            % ('/sigma', 
               "%.4f" % self.sigma, 
               "%.3f" % self.beta_se[-1], 
               "", 
               "", 
               "%.3f" % self.confidence_interval[-1, 0], 
               "%.3f" % self.confidence_interval[-1, 1])
            )
        print('-'*80)

    def censored_data_description(self):
        print('%27s %-52s' \
            % ('Obs. summary:',
                '%s censored observations' % self.nb_censored_obs))
        print('%27s %-52s' \
            % ('',
                '%s uncensored observations' % self.nb_uncensored_obs))


class BaseModel(Results):
    '''Base class inherited by other models
    Not intended to be used separately
    '''
    def input_data_preparation(self, X, drop_na=None, fill_value=None):
        X = self.handle_missing_values(X, drop_na, fill_value)
        return X

    def handle_missing_values(self, X, drop_na=None, fill_value=None):
        if drop_na is not None:
            if drop_na:
                X.dropna(inplace=True)
            elif fill_value == 'mean':
                X.fillna(X.mean(), inplace=True)
            elif fill_value == 'median':
                X.fillna(X.median(), inplace=True)
            elif fill_value is not None:
                for var in X.columns:
                    try:
                        X[var].fillna(fill_value.get(var), inplace=True)
                    except:
                        raise ValueError('\'fill_value\' argument must be in list ' \
                            + '[\'mean\', \'median\'] or of type dict. See docstring for more info.')

        return X

    def plot_trace_estimators(self):
        if self.beta is None:
            raise AttributeError('Fit method should be called before evaluating of the model')
            
        colors = ['b','g','r','c','m','y','k']
        for k in range(len(self.beta)):
            plt.plot(np.arange(1, len(self.beta_est)+1),
                     self.beta_est[:,k],
                     color=colors[(k-1) % len(colors)],
                     label="Beta_%s" % k)

        plt.xlim((1,len(self.beta_est)*1.2))
        plt.xlabel('Iterations')
        plt.ylabel('Estimators')
        plt.title('Trace plot of estimators of beta', size=16)
        plt.legend(loc='best')
        plt.show()

    def predict(self, X):
        if self.beta is None:
            raise AttributeError('Fit method should be called before evaluating the model.')

        X = self.input_data_preparation(X)
        X.insert(0, '_cons', 1)

        Z = self.response_function(X, self.beta)
        result = (np.sign(Z)+1)/2

        try:
            result = result.astype(int)
        except:
            raise ValueError('One or several data are missing.')

        return result
        
    def predict_proba(self, X):
        if self.beta is None:
            raise AttributeError('Fit method should be called before evaluating the model.')

        X = self.input_data_preparation(X)
        X.insert(0, '_cons', 1)

        Z = self.response_function(X,self.beta)
        return Z.apply(lambda x : norm_cdf(x))

    def summary(self):
        if self.beta is None:
            raise AttributeError('Fit method should be called before evaluating of the model.')

        self.model_description()
        self.columns_header()
        self.beta_description()



class PanelBaseModel(BaseModel):
    '''Base class inherited by other models
    Not intended to be used separately
    '''
    def input_data_preparation(self, X, drop_na=None, fill_value=None):
        try:
            X = X.to_frame()
        except:
            if len(X.index.names) != 2:
                raise ValueError("Only 2-level MultiIndex and Panel are supported.")

        X = self.handle_missing_values(X, drop_na, fill_value)
        return X


class CensoredBaseModel(BaseModel):
    '''Base class inherited by other models
    Not intended to be used separately
    '''
    def input_data_preparation(self, X, drop_na=None, fill_value=None):
        if self.output in X.columns:
            neg_values = X[X[self.output] <= 0]
            if len(neg_values[neg_values[self.output] < 0]) > 0:
                raise ValueError("Negative values where found in output variable." \
                    + "Please set all censored observations to 0 before fitting the model.")
            elif len(neg_values) == 0:
                raise ValueError("No censored observations were found." \
                    + "Please set output of all censored observations to 0 before fitting the model.")

        X = self.handle_missing_values(X, drop_na, fill_value)
        return X

    def summary(self):
        BaseModel.summary(self)
        self.sigma_description()
        self.censored_data_description()
     
        