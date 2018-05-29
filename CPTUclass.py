import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CPTU(object):

    def __init__(self, plik):
        self.df = pd.read_csv(plik, delimiter=',')
        self.df = self.df[['Depth(m)', 'qc', 'fs', 'u2', 'ta']]
        print('Zaimportowano pomyślnie')
        self.dd = 0.02
        self.a = 0.15
        self.Pa = 100.0
        self.Nkt = 14.0
        self.wlvl = 1.2

    def pokaz(self, wiersze=10):
        print(self.df.head(wiersze))

    def interpreter(self):
        self.df['qt'] = (self.df['qc'] + (self.df['u2'] / 1000) * (1 - self.a)) * 1000

        def friction_ratio(fs, qt):
            if float(fs) <= 0:
                return 0.0
            else:
                return (fs / qt) * 100

        self.df['Rf'] = self.df.apply(lambda x: friction_ratio(x['fs'], x['qt']), axis = 1)

        def soil_unit_weight(Rf, qt, Pa):
            if Rf == 0.0:
                return (0.27 * (-1) + 0.36 * (np.log10(qt / Pa)) + 1.236) * 9.81
            else:
                return (0.27 * (np.log10(Rf)) + 0.36 * (np.log10(qt / Pa)) + 1.236) * 9.81

        self.df['gamma'] = self.df.apply(lambda x: soil_unit_weight(x['Rf'], x['qt'], self.Pa), axis=1)

        self.df['sigma'] = (self.df['gamma'] * self.dd).cumsum()

        def preinsertion_pore_preasure(wlvl, depth):
            'funkcja oblicza ciśnienie wody w porach na podstawie wysokości słupa wody'
            if wlvl < depth:
                return (float(depth) - float(wlvl)) * 9.81
            else:
                return 0.0

        self.df['u0'] = self.df.apply(lambda x: preinsertion_pore_preasure(self.wlvl, x['Depth(m)']), axis=1)

        self.df['sigma_v0'] = self.df['sigma'] - self.df['u0']

        def norm_friction_r(fs, qt, sigma):
            if fs < 0:
                return (0 / (qt - sigma)) * 100
            else:
                return (fs / (qt - sigma)) * 100

        self.df['Fr'] = self.df.apply(lambda x: norm_friction_r(x['fs'], x['qt'], x['sigma']), axis=1)

        self.df['Qt'] = (self.df['qt'] - self.df['sigma']) / self.df['sigma_v0']

        self.df['Bq'] = (self.df['u2'] - self.df['u0']) / (self.df['qt'] - self.df['sigma'])

        def soil_behavior_type(Qt, Fr):
            if Fr < 0.1:
                return ((3.47 - np.log10(Qt)) ** 2 + (-1 + 1.22) ** 2) ** 0.5
            else:
                return ((3.47 - np.log10(Qt)) ** 2 + (np.log10(Fr) + 1.22) ** 2) ** 0.5

        def undrained_shear_str(qt, sigma, Ic, Nkt):
            if Ic > 2.6:
                return (qt - sigma) / Nkt
            else:
                return 0

        def undrained_shear_str_ratio(qt, sigma, sigmav0, Ic, Nkt):
            if Ic > 2.6:
                return ((qt - sigma) / sigmav0) * 1 / Nkt
            else:
                return 0

        self.df['Ic'] = self.df.apply(lambda x: soil_behavior_type(x['Qt'], x['Fr']), axis=1)

        self.df['su'] = self.df.apply(lambda x: undrained_shear_str(x['qt'], x['sigma'], x['Ic'], self.Nkt), axis=1)

        self.df['su2'] = self.df.apply(lambda x: undrained_shear_str_ratio(x['qt'], x['sigma'], x['sigma_v0'], x['Ic'], self.Nkt), axis=1)

        def cons_modulus_M(qt, Ic, sigma, Qt):
            if Ic < 2.2:
                alpha = 0.0188 * (10 ** (Ic * 0.55 + 1.68))
                return alpha * (qt - sigma)
            else:
                if Qt > 14:
                    return 14 * (qt - sigma)
                else:
                    return Qt * (qt - sigma)

        self.df['M'] = self.df.apply(lambda x: cons_modulus_M(x['qt'], x['Ic'], x['sigma'], x['Qt']),axis = 1)

        def overconsolidated_ratio(Ic, Qt):
            if Ic > 2.6:
                return 0.25 * Qt ** 1.25
            else:
                return 0

        self.df['OCR'] = self.df.apply(lambda x: overconsolidated_ratio(x['Ic'], x['Qt']), axis=1)

        def friction_angle(Ic, Qt):
            if Ic <= 2.6:
                return 17.6 + 11 * np.log10(Qt)
            else:
                return 0

        self.df['OCR'] = self.df.apply(lambda x: friction_ratio(x['Ic'], x['Qt']), axis=1)

        return self.df

    def eksport(self):
        self.df.to_csv(path_or_buf='test_CPTU_wynik.csv')

    def wykres(self, k1, k2, k3):

        ylim = (self.df['Depth(m)'].max(), self.df['Depth(m)'].min())

        f = plt.figure(figsize=(15,10))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)

        ax1.plot(self.df[k1], self.df['Depth(m)'], 'k-', linewidth=1.0)
        ax1.set_ylim(ylim)
        ax1.set(xlabel=k1, ylabel='Depth(m)')
        ax1.grid(True)


        ax2.plot(self.df[k2], self.df['Depth(m)'], 'k-', linewidth=1.0)
        ax2.set_ylim(ylim)
        ax2.set(xlabel=k2)
        ax2.grid(True)


        ax3.plot(self.df[k3], self.df['Depth(m)'], 'k-', linewidth=1.0)
        ax3.set_ylim(ylim)
        ax3.set(xlabel=k3)
        ax3.grid(True)

        plt.show()
